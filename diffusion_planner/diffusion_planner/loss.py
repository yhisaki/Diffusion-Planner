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


def make_ego_stop_gt(
    ego_velocity_gt: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return stop labels where vx below 1e-3 is considered stopped."""
    return (ego_velocity_gt < 1e-3).to(dtype)


def ego_stop_transition_mask(
    ego_stop_current_gt: torch.Tensor,
    ego_stop_future_gt: torch.Tensor,
) -> torch.Tensor:
    """Return, per future step, whether the stop bool switched from the previous step.

    The previous step for the first future timestep is the current stop bool.
    """
    ego_stop_future_gt_bool = ego_stop_future_gt.bool()
    prev_stop_bool = torch.cat([ego_stop_current_gt.bool(), ego_stop_future_gt_bool[:, :-1]], dim=1)
    return ego_stop_future_gt_bool != prev_stop_bool


def weighted_ego_stop_future_loss(
    ego_stop_future_logit: torch.Tensor,
    ego_stop_future_gt: torch.Tensor,
    ego_stop_current_gt: torch.Tensor,
    transition_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute stop BCE loss, upweighting timesteps where the stop bool switches."""
    element_loss = F.binary_cross_entropy_with_logits(
        ego_stop_future_logit,
        ego_stop_future_gt,
        reduction="none",
    )
    transition_mask = ego_stop_transition_mask(ego_stop_current_gt, ego_stop_future_gt)
    element_weight = torch.where(
        transition_mask,
        torch.full_like(element_loss, transition_loss_weight),
        torch.ones_like(element_loss),
    )
    return (element_loss * element_weight).mean(), transition_mask


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

    model_output_physical = norm.inverse(model_output)
    gt_target_physical = norm.inverse(gt_target)
    loss_dict = loss_func(model_output_physical, gt_target_physical)
    snr_weight = snr_loss_weight(t[..., 1:, 0])
    position_loss = loss_dict["position_loss"]
    heading_loss = loss_dict["heading_loss"]  # [B, P, T]

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
    turn_indicator_coeff = torch.where(turn_indicator_change, 1.0, 0.2)
    turn_indicator_loss = (turn_indicator_loss * turn_indicator_coeff).mean()
    loss["turn_indicator_loss"] = turn_indicator_loss

    ego_velocity_future_prediction = decoder_output["ego_velocity_future_prediction"]
    ego_velocity_future_gt = inputs["ego_velocity_future"][..., 0:1]
    loss["ego_velocity_future_loss"] = F.huber_loss(
        ego_velocity_future_prediction,
        ego_velocity_future_gt,
    )
    ego_stop_future_logit = decoder_output["ego_stop_future_logit"]
    ego_stop_future_gt = make_ego_stop_gt(
        ego_velocity_future_gt,
        dtype=ego_stop_future_logit.dtype,
    )
    ego_velocity_current_gt = inputs["ego_current_state"][:, 4:5].unsqueeze(1)  # [B, 1, 1]
    ego_stop_current_gt = make_ego_stop_gt(
        ego_velocity_current_gt,
        dtype=ego_stop_future_logit.dtype,
    )
    loss["ego_stop_future_loss"], ego_stop_future_transition = weighted_ego_stop_future_loss(
        ego_stop_future_logit,
        ego_stop_future_gt,
        ego_stop_current_gt,
        transition_loss_weight=getattr(args, "stop_transition_loss_weight", 5.0),
    )
    loss["ego_speed_prediction_loss"] = (
        loss["ego_velocity_future_loss"]
        + getattr(args, "alpha_stop_loss", 1.0) * loss["ego_stop_future_loss"]
    )

    with torch.no_grad():
        turn_indicator_accuracy = (
            (turn_indicator_logit.argmax(dim=-1) == turn_indicator_gt).float().mean()
        )
        loss["turn_indicator_accuracy"] = turn_indicator_accuracy
        ego_stop_future_prediction = torch.sigmoid(ego_stop_future_logit) >= 0.5
        loss["ego_stop_future_accuracy"] = (
            (ego_stop_future_prediction == ego_stop_future_gt.bool()).float().mean()
        )
        loss["ego_stop_future_transition_ratio"] = ego_stop_future_transition.float().mean()

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
