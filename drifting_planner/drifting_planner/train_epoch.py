import torch
from torch import nn
from tqdm import tqdm

from drifting_planner.drifting_loss import compute_grouped_drifting_loss
from drifting_planner.loss import loss_func, make_turn_indicator_gt
from drifting_planner.utils import ddp
from drifting_planner.utils.data_augmentation import StatePerturbation
from drifting_planner.utils.train_utils import get_epoch_mean_loss


def heading_to_cos_sin(x):
    return torch.cat(
        [
            x[..., :2],
            x[..., 2:3].cos(),
            x[..., 2:3].sin(),
        ],
        dim=-1,
    )


def train_epoch(data_loader, model, optimizer, args, ema, aug=None):
    epoch_loss = []

    model.train()

    if args.ddp:
        torch.cuda.synchronize()

    if ddp.get_rank() == 0:
        data_loader = tqdm(data_loader, desc="Training", unit="batch")

    temperatures = args.drifting_temperatures

    for inputs in data_loader:
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
        inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

        ego_future = inputs["ego_agent_future"]
        neighbors_future = inputs["neighbor_agents_future"]

        if aug is not None:
            inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

        ego_future = heading_to_cos_sin(ego_future)

        neighbor_future_mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
        neighbors_future = heading_to_cos_sin(neighbors_future)
        neighbors_future[neighbor_future_mask] = 0.0
        inputs = args.observation_normalizer(inputs)

        optimizer.zero_grad()

        B = inputs["ego_current_state"].shape[0]
        Pn = neighbors_future.shape[1]
        P = 1 + Pn
        T = args.future_len

        ego_current = inputs["ego_current_state"][:, :4]
        neighbors_current = inputs["neighbor_agents_past"][:, :Pn, -1, :4]

        current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)

        gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future], dim=1)
        all_gt = torch.cat([current_states[:, :, None, :], args.state_normalizer(gt_future)], dim=2)
        neighbor_mask_full = torch.cat(
            [torch.zeros(B, P, 1, dtype=torch.bool, device=neighbor_future_mask.device),
             torch.cat([torch.zeros(B, 1, T, dtype=torch.bool, device=neighbor_future_mask.device),
                        neighbor_future_mask], dim=1)],
            dim=2
        )
        all_gt[neighbor_mask_full.unsqueeze(-1).expand_as(all_gt)] = 0.0

        noise = torch.randn(B, P, T, 4, device=all_gt.device)
        noise_with_current = torch.cat([current_states[:, :, None, :], noise], dim=2)

        merged_inputs = {
            **inputs,
            "sampled_trajectories": noise_with_current.reshape(B, P, (1 + T) * 4),
            "gt_trajectories": all_gt,
        }

        encoder_outputs, decoder_output = model(merged_inputs)

        model_output = decoder_output["model_output"][:, :, 1:, :]

        x_generated = model_output.reshape(B, P, T, 4)

        y_pos = all_gt[:, :, 1:, :].reshape(B, P, T * 4)
        x_flat = x_generated.reshape(B, P, T * 4)
        y_neg = x_flat
        valid_agent_mask = torch.cat(
            [
                torch.ones(B, 1, dtype=torch.bool, device=neighbor_future_mask.device),
                ~neighbor_future_mask.any(dim=-1),
            ],
            dim=1,
        )

        drifting_loss_val, drift_norm = compute_grouped_drifting_loss(
            x_flat,
            y_pos,
            y_neg,
            valid_agent_mask,
            temperatures,
        )

        loss_dict = loss_func(x_generated, all_gt[:, :, 1:, :])
        position_lat_loss = loss_dict["position_lat_loss"]
        position_lon_loss = loss_dict["position_lon_loss"]
        heading_l2_loss = loss_dict["heading_l2_loss"]

        longitudinal_velocity = inputs["ego_current_state"][:, 4:5]
        velocity_weight = torch.abs(longitudinal_velocity) * args.coeff_velocity
        velocity_weight = torch.clamp_min(velocity_weight, 1.0).unsqueeze(-1)
        position_lon_loss = position_lon_loss / velocity_weight

        dpm_loss = (
            args.coeff_position_lat_loss * position_lat_loss
            + args.coeff_position_lon_loss * position_lon_loss
            + args.coeff_heading_l2_loss * heading_l2_loss
        )

        neighbors_future_valid = ~neighbor_future_mask
        masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

        loss = {}

        if masked_prediction_loss.numel() > 0:
            loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
        else:
            loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=dpm_loss.device)

        loss["ego_planning_loss"] = dpm_loss[:, 0, :args.ego_prediction_horizon].mean()
        loss["drifting_loss"] = drifting_loss_val
        loss["drift_norm"] = drift_norm

        turn_indicator_logit = decoder_output["turn_indicator_logit"]
        turn_indicator_gt = make_turn_indicator_gt(inputs["turn_indicators"])
        turn_indicator_loss = nn.functional.cross_entropy(
            turn_indicator_logit, turn_indicator_gt, reduction="none"
        )
        turn_indicator_change = inputs["turn_indicators"][:, -2] != inputs["turn_indicators"][:, -1]
        turn_indicator_coeff = torch.where(turn_indicator_change, 1.0, 0.05)
        turn_indicator_loss = (turn_indicator_loss * turn_indicator_coeff).mean()
        loss["turn_indicator_loss"] = turn_indicator_loss

        with torch.no_grad():
            loss["turn_indicator_accuracy"] = (
                (turn_indicator_logit.argmax(dim=-1) == turn_indicator_gt).float().mean()
            )

        loss["road_border_loss"] = torch.tensor(0.0, device=dpm_loss.device)
        loss["neighbor_collision_loss"] = torch.tensor(0.0, device=dpm_loss.device)

        loss["total_loss"] = (
            args.alpha_neighbor_loss * loss["neighbor_prediction_loss"]
            + args.alpha_planning_loss * loss["ego_planning_loss"]
            + args.drifting_loss_weight * loss["drifting_loss"]
            + loss["turn_indicator_loss"]
            + args.coeff_road_border_loss * loss["road_border_loss"]
            + args.coeff_neighbor_collision_loss * loss["neighbor_collision_loss"]
        )

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
        print(f"{epoch_mean_loss['total_loss']=:.4f}")
        print(f"{epoch_mean_loss['drifting_loss']=:.4f}")
        print(f"{epoch_mean_loss['drift_norm']=:.4f}")

    return epoch_mean_loss, epoch_mean_loss["total_loss"]
