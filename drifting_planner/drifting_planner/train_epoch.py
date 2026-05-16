import argparse
from typing import Any

import torch
from timm.utils.model_ema import ModelEma
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from drifting_planner.drifting_loss import compute_scene_drifting_loss
from drifting_planner.utils import ddp
from drifting_planner.utils.data_augmentation import StatePerturbation
from drifting_planner.utils.train_utils import get_epoch_mean_loss


def heading_to_cos_sin(x: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            x[..., :2],
            x[..., 2:3].cos(),
            x[..., 2:3].sin(),
        ],
        dim=-1,
    )


def _repeat_batch_inputs(inputs: dict[str, Any], repeats: int, batch_size: int) -> dict[str, Any]:
    repeated: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value) and value.shape[0] == batch_size:
            repeated[key] = value.repeat_interleave(repeats, dim=0)
        else:
            repeated[key] = value
    return repeated


def _make_positive_samples(
    y_pos: torch.Tensor, num_positive_samples: int, positive_noise_std: float
) -> torch.Tensor:
    if y_pos.dim() == 2:
        y_pos = y_pos[:, None]
    if y_pos.shape[1] >= num_positive_samples:
        return y_pos[:, :num_positive_samples]

    repeats = (num_positive_samples + y_pos.shape[1] - 1) // y_pos.shape[1]
    y_pos = y_pos.repeat(1, repeats, 1)[:, :num_positive_samples].clone()
    if positive_noise_std > 0:
        y_pos = y_pos + torch.randn_like(y_pos) * positive_noise_std
    return y_pos


def train_epoch(
    data_loader: DataLoader | tqdm,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    ema: ModelEma | None,
    aug: StatePerturbation | None = None,
) -> tuple[dict[str, float], float]:
    epoch_loss: list[dict[str, Any]] = []

    model.train()

    if args.ddp:
        torch.cuda.synchronize()

    if ddp.get_rank() == 0:
        data_loader = tqdm(data_loader, desc="Training", unit="batch")

    temperatures: list[float] = args.drifting_temperatures
    num_generated_samples: int = args.drifting_num_samples
    num_positive_samples: int = args.drifting_num_positive_samples
    positive_noise_std: float = args.drifting_positive_noise_std

    if num_generated_samples < 2:
        raise ValueError("--drifting_num_samples must be at least 2.")
    if num_positive_samples < 1:
        raise ValueError("--drifting_num_positive_samples must be at least 1.")

    for inputs in data_loader:
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
        inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

        ego_future = inputs["ego_agent_future"]
        neighbors_future = inputs["neighbor_agents_future"]

        if aug is not None and (ego_future.dim() != 3 or neighbors_future.dim() != 4):
            raise ValueError("Data augmentation only supports one future sample per scene.")

        if aug is not None:
            inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

        ego_future = heading_to_cos_sin(ego_future)
        if ego_future.dim() == 3:
            ego_future_samples = ego_future[:, None]
        elif ego_future.dim() == 4:
            ego_future_samples = ego_future
        else:
            raise ValueError("Expected ego future shape [B, T, 3] or [B, M, T, 3].")

        neighbor_future_mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
        neighbors_future = heading_to_cos_sin(neighbors_future)
        neighbors_future = neighbors_future.masked_fill(neighbor_future_mask.unsqueeze(-1), 0.0)
        if neighbors_future.dim() == 4:
            neighbors_future_samples = neighbors_future[:, None]
            neighbor_future_mask_samples = neighbor_future_mask[:, None]
        elif neighbors_future.dim() == 5:
            neighbors_future_samples = neighbors_future
            neighbor_future_mask_samples = neighbor_future_mask
        else:
            raise ValueError("Expected neighbor futures shape [B, Pn, T, 3] or [B, M, Pn, T, 3].")

        B = inputs["ego_current_state"].shape[0]
        positive_data_samples = ego_future_samples.shape[1]
        if neighbors_future_samples.shape[1] != positive_data_samples:
            raise ValueError("Ego and neighbor futures must have the same positive sample count.")
        Pn = neighbors_future_samples.shape[2]
        P = 1 + Pn
        T = args.future_len

        ego_current_state_raw = inputs["ego_current_state"]
        ego_current_raw = ego_current_state_raw[:, :4]
        neighbors_current_raw = inputs["neighbor_agents_past"][:, :Pn, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current_raw[..., :4], 0), dim=-1) == 0

        inputs = args.observation_normalizer(inputs)

        optimizer.zero_grad()

        current_states_raw = torch.cat([ego_current_raw[:, None], neighbors_current_raw], dim=1)
        current_states = args.state_normalizer(current_states_raw[:, :, None, :])[:, :, 0, :]

        gt_future = torch.cat(
            [ego_future_samples[:, :, None, :, :], neighbors_future_samples],
            dim=2,
        )
        gt_future_delta_norm = args.trajectory_normalizer.normalize_future(
            gt_future, current_states_raw
        )
        current_states_samples = current_states[:, None, :, None, :].expand(
            -1, positive_data_samples, -1, -1, -1
        )
        all_gt_samples = torch.cat(
            [current_states_samples, gt_future_delta_norm],
            dim=3,
        )
        neighbor_mask = torch.cat(
            [
                neighbor_current_mask[:, None, :, None].expand(-1, positive_data_samples, -1, -1),
                neighbor_future_mask_samples,
            ],
            dim=-1,
        )
        neighbor_mask_full = torch.cat(
            [
                torch.zeros(
                    B,
                    positive_data_samples,
                    1,
                    T + 1,
                    dtype=torch.bool,
                    device=neighbor_future_mask_samples.device,
                ),
                neighbor_mask,
            ],
            dim=2,
        )
        all_gt_samples = all_gt_samples.masked_fill(neighbor_mask_full.unsqueeze(-1), 0.0)

        noise = torch.randn(B, num_generated_samples, P, T, 4, device=current_states.device)
        current_states_expanded = current_states[:, None, :, None, :].expand(
            -1, num_generated_samples, -1, -1, -1
        )
        noise_with_current = torch.cat([current_states_expanded, noise], dim=3)

        repeated_inputs = _repeat_batch_inputs(inputs, num_generated_samples, B)
        merged_inputs = {
            **repeated_inputs,
            "sampled_trajectories": noise_with_current.reshape(
                B * num_generated_samples, P, (1 + T) * 4
            ),
        }

        _, decoder_output = model(merged_inputs)

        model_output = decoder_output["model_output"].reshape(
            B, num_generated_samples, P, T + 1, 4
        )[:, :, :, 1:, :]

        x_generated = model_output.reshape(B, num_generated_samples, P, T, 4)

        valid_agent_mask = torch.cat(
            [
                torch.ones(B, 1, dtype=torch.bool, device=neighbor_future_mask_samples.device),
                (~neighbor_current_mask) & (~neighbor_future_mask_samples.all(dim=-1).all(dim=1)),
            ],
            dim=1,
        )

        sample_mask = valid_agent_mask[:, None, :, None, None]
        x_generated = x_generated.masked_fill(~sample_mask, 0.0)
        y_pos_scene = all_gt_samples[:, :, :, 1:, :].masked_fill(
            ~valid_agent_mask[:, None, :, None, None], 0.0
        )

        x_flat = x_generated.reshape(B, num_generated_samples, P * T * 4)
        y_pos = _make_positive_samples(
            y_pos_scene.reshape(B, positive_data_samples, P * T * 4),
            num_positive_samples,
            positive_noise_std,
        )

        drifting_loss_val, drift_norm = compute_scene_drifting_loss(
            x_flat,
            y_pos,
            temperatures,
        )

        loss: dict[str, Any] = {}
        loss["drifting_loss"] = drifting_loss_val
        loss["drift_norm"] = drift_norm
        loss["generated_sample_std"] = x_flat.detach().std(dim=1).mean()
        loss["total_loss"] = args.drifting_loss_weight * loss["drifting_loss"]

        loss["total_loss"].backward()

        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        if args.ddp:
            torch.cuda.synchronize()
        epoch_loss.append(loss)

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"{epoch_mean_loss['total_loss']=:.8f}")
        print(f"{epoch_mean_loss['drifting_loss']=:.8f}")
        print(f"{epoch_mean_loss['drift_norm']=:.8f}")
        print(f"{epoch_mean_loss['generated_sample_std']=:.8f}")

    return epoch_mean_loss, epoch_mean_loss["total_loss"]
