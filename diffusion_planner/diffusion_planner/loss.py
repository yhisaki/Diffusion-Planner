from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm
from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_KEEP
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear


def make_turn_indicator_gt(
    turn_indicators: torch.Tensor,  # # [B, INPUT_T + 1]
) -> torch.Tensor:
    turn_indicators_gt = turn_indicators.long()  # [B, INPUT_T + 1]
    turn_indicators_gt_keep = turn_indicators_gt[:, -1] == turn_indicators_gt[:, -2]  # [B,]
    turn_indicators_gt = turn_indicators_gt[:, -1] * ~turn_indicators_gt_keep  # change to 0 if keep
    turn_indicators_gt = turn_indicators_gt + turn_indicators_gt_keep * TURN_INDICATOR_OUTPUT_KEEP
    return turn_indicators_gt


def snr_loss_weight(t: torch.Tensor) -> torch.Tensor:
    """Compute exp(2) * sigmoid(log-SNR - 2) for the linear VP SDE."""
    noise_schedule = dpm.NoiseScheduleVP()
    half_log_snr = noise_schedule.marginal_lambda(t)
    log_snr = 2.0 * half_log_snr
    return torch.exp(torch.ones((), device=t.device, dtype=t.dtype) * 2.0) * torch.sigmoid(
        log_snr - 2.0
    )


def compute_training_loss(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    futures: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: Namespace,
) -> dict[str, torch.Tensor]:
    norm = args.state_normalizer

    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask  # [B, Pn, T]

    B, Pn, T, _ = neighbors_future.shape
    P = 1 + Pn
    ego_current, neighbors_current = (
        inputs["ego_current_state"][:, :4],
        inputs["neighbor_agents_past"][:, :Pn, -1, :4],
    )
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat(
        (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1
    )

    gt_future = torch.cat(
        [ego_future[:, None, :, :], neighbors_future[..., :]], dim=1
    )  # [B, P, T, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)  # [B, P, 4]

    eps = 1e-3
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps  # [B,]
    t = t.view(B, 1, 1, 1)
    t = t.expand(B, P, T + 1, 1)
    z = torch.randn_like(gt_future, device=gt_future.device)  # [B, P, T, 4]

    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0

    mean, std = VPSDE_linear().marginal_prob(all_gt[..., 1:, :], t[..., 1:, :])
    # mean([B, P, T, D]), std([B, 1, T, 1]), z([B, P, T, D])
    xT = mean + std * z

    xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)

    merged_inputs = {
        **inputs,
        "gt_trajectories": all_gt,
        "sampled_trajectories": xT,
        "diffusion_time": t,
    }
    _, decoder_output = model(merged_inputs)  # [B, P, 1 + T, 4]
    model_output = decoder_output["model_output"][:, :, 1:, :]  # [B, P, T, 4]

    gt_target = all_gt[:, :, 1:, :]  # [B, P, T, 4]

    loss_dict = loss_func(model_output, gt_target)
    snr_weight = snr_loss_weight(t[..., 1:, 0])
    position_loss = loss_dict["position_loss"] * snr_weight  # [B, P, T]
    heading_loss = loss_dict["heading_loss"] * snr_weight  # [B, P, T]

    masked_neighbor_position_loss = position_loss[:, 1:, :][neighbors_future_valid]
    masked_neighbor_heading_loss = heading_loss[:, 1:, :][neighbors_future_valid]

    loss = {}

    if masked_neighbor_position_loss.numel() > 0:
        loss["neighbor_position_loss"] = masked_neighbor_position_loss.mean()
        loss["neighbor_heading_loss"] = masked_neighbor_heading_loss.mean()
    else:
        loss["neighbor_position_loss"] = torch.tensor(
            0.0, device=masked_neighbor_position_loss.device
        )
        loss["neighbor_heading_loss"] = torch.tensor(
            0.0, device=masked_neighbor_heading_loss.device
        )

    loss["ego_position_loss"] = position_loss[:, 0].mean()
    loss["ego_heading_loss"] = heading_loss[:, 0].mean()

    loss["ego_planning_loss"] = (
        args.coeff_pos_ego * loss["ego_position_loss"]
        + args.coeff_heading_ego * loss["ego_heading_loss"]
    )
    loss["neighbor_prediction_loss"] = (
        args.coeff_pos_neighbor * loss["neighbor_position_loss"]
        + args.coeff_heading_neighbor * loss["neighbor_heading_loss"]
    )

    assert not torch.isnan(position_loss).any(), f"position loss cannot be nan, z={z}"
    assert not torch.isnan(heading_loss).any(), f"heading loss cannot be nan, z={z}"

    turn_indicator_logit = decoder_output["turn_indicator_logit"]  # [B, TURN_INDICATOR_OUTPUT_KEEP]
    turn_indicator_gt = make_turn_indicator_gt(inputs["turn_indicators"])  # [B,]
    turn_indicator_loss = F.cross_entropy(turn_indicator_logit, turn_indicator_gt, reduction="none")
    turn_indicator_change = inputs["turn_indicators"][:, -2] != inputs["turn_indicators"][:, -1]
    turn_indicator_coeff = torch.where(turn_indicator_change, 1.0, 0.05)
    turn_indicator_loss = (turn_indicator_loss * turn_indicator_coeff).mean()
    loss["turn_indicator_loss"] = turn_indicator_loss

    with torch.no_grad():
        turn_indicator_accuracy = (
            (turn_indicator_logit.argmax(dim=-1) == turn_indicator_gt).float().mean()
        )
        loss["turn_indicator_accuracy"] = turn_indicator_accuracy

    return loss


def loss_func(
    trajectory_pred: torch.Tensor, trajectory_gt: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Calculate the loss between predicted and ground truth trajectories.

    Args:
        trajectory_pred (torch.Tensor): Predicted trajectory with shape [B, Pn + 1, T, D].
        trajectory_gt (torch.Tensor): Ground-truth trajectory with shape [B, Pn + 1, T, D].
            B is the batch size, Pn is the number of predicted neighbors, T is the
            prediction horizon, and D=4 represents (x, y, cos(heading), sin(heading)).

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the loss values.
            Each loss has shape [B, Pn + 1, T].
    """
    result_dict = {}

    position_error = trajectory_pred[..., :2] - trajectory_gt[..., :2]  # [B, Pn + 1, T, 2]
    dx = position_error[..., 0]  # [B, Pn + 1, T]
    dy = position_error[..., 1]  # [B, Pn + 1, T]

    cos_gt = trajectory_gt[..., 2]  # [B, Pn + 1, T]
    sin_gt = trajectory_gt[..., 3]  # [B, Pn + 1, T]

    longitudinal_error = dx * cos_gt + dy * sin_gt  # [B, Pn + 1, T]
    lateral_error = -dx * sin_gt + dy * cos_gt  # [B, Pn + 1, T]

    longitudinal_loss = F.huber_loss(
        longitudinal_error, torch.zeros_like(longitudinal_error), reduction="none"
    )  # [B, Pn + 1, T]
    lateral_loss = F.huber_loss(
        lateral_error, torch.zeros_like(lateral_error), reduction="none"
    )  # [B, Pn + 1, T]

    position_loss = longitudinal_loss + lateral_loss  # [B, Pn + 1, T]

    result_dict["longitudinal_loss"] = longitudinal_loss
    result_dict["lateral_loss"] = lateral_loss
    result_dict["position_loss"] = position_loss

    cos_pred = trajectory_pred[..., 2]  # [B, Pn + 1, T]
    sin_pred = trajectory_pred[..., 3]  # [B, Pn + 1, T]

    heading_loss = F.huber_loss(cos_pred, cos_gt, reduction="none") + F.huber_loss(
        sin_pred, sin_gt, reduction="none"
    )
    result_dict["heading_loss"] = heading_loss

    return result_dict
