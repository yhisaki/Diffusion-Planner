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
    neighbors_future_valid = ~neighbor_future_mask  # [B, Pn, V]

    B, Pn, T, _ = neighbors_future.shape
    P = 1 + Pn
    ego_current, neighbors_current = (
        inputs["ego_current_state"][:, :4],
        inputs["neighbor_agents_past"][:, :Pn, -1, :4],
    )
    longitudinal_velocity = inputs["ego_current_state"][:, 4:5]
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
    heading_l2_loss = loss_dict["heading_l2_loss"]  # [B, P, T]
    position_lat_loss = loss_dict["position_lat_loss"]  # [B, P, T]
    position_lon_loss = loss_dict["position_lon_loss"]  # [B, P, T]

    # velocity weight
    velocity_weight = longitudinal_velocity * args.coeff_velocity
    velocity_weight = torch.abs(velocity_weight)
    velocity_weight = torch.clamp_min(velocity_weight, 1.0)
    velocity_weight = velocity_weight.unsqueeze(-1)  # [B, 1, 1]
    position_lon_loss = position_lon_loss / velocity_weight

    # timestep weight
    timestep_weight = args.coeff_timestep
    assert T % len(timestep_weight) == 0, (
        f"Timestep {T} is not divisible by the number of timestep weights {len(timestep_weight)}"
    )
    unit = T // len(timestep_weight)
    for i in range(len(timestep_weight)):
        position_lat_loss[:, :, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]
        position_lon_loss[:, :, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]
        heading_l2_loss[:, :, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]

    dpm_loss = (
        args.coeff_position_lat_loss * position_lat_loss
        + args.coeff_position_lon_loss * position_lon_loss
        + args.coeff_heading_l2_loss * heading_l2_loss
    )  # [B, P, T]
    dpm_loss = dpm_loss * snr_loss_weight(t[..., 1:, 0])

    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

    loss = {}

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    loss["ego_planning_loss"] = dpm_loss[:, 0, : args.ego_prediction_horizon].mean()

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    turn_indicator_logit = decoder_output["turn_indicator_logit"]  # [B, TURN_INDICATOR_OUTPUT_KEEP]
    turn_indicator_gt = make_turn_indicator_gt(inputs["turn_indicators"])  # [B,]
    turn_indicator_loss = F.cross_entropy(
        turn_indicator_logit, turn_indicator_gt, reduction="none"
    )
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
        trajectory_pred (torch.Tensor): Predicted trajectory of shape [..., T, D].
        trajectory_gt (torch.Tensor): Ground truth trajectory of shape [..., T, D].
        where, D=4 (x, y, cos, sin).

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the loss values.
        where, each loss' shape is [..., T].
    """
    result_dict = {}

    ###################
    # Basic L2 Losses #
    ###################
    # simple L2 loss
    result_dict["simple_l2_loss"] = torch.mean((trajectory_pred - trajectory_gt) ** 2, dim=-1)

    # Position loss (x, y coordinates)
    position_pred = trajectory_pred[..., :2]  # [..., T, 2]
    position_gt = trajectory_gt[..., :2]  # [..., T, 2]

    # Calculate L2 distance for each time step
    position_diff = position_pred - position_gt  # [..., T, 2]
    position_error = torch.sum(position_diff**2, dim=-1)  # [..., T]
    result_dict["position_l2_loss"] = position_error

    # Heading loss (cos, sin components)
    cos_sin_pred = trajectory_pred[..., 2:]  # [..., T, 2]
    cos_sin_gt = trajectory_gt[..., 2:]  # [..., T, 2]

    # heading l2 loss
    heading_loss = torch.sum((cos_sin_pred - cos_sin_gt) ** 2, dim=-1)  # [..., T]
    result_dict["heading_l2_loss"] = heading_loss

    ######################
    # Specialized Losses #
    ######################
    # Lateral or longitudinal error (along vehicle direction)
    cos_gt = cos_sin_gt[..., 0]  # [..., T]
    sin_gt = cos_sin_gt[..., 1]  # [..., T]
    lon_diff = +position_diff[..., 0] * cos_gt + position_diff[..., 1] * sin_gt  # [..., T]
    lat_diff = -position_diff[..., 0] * sin_gt + position_diff[..., 1] * cos_gt  # [..., T]
    lat_error = torch.abs(lat_diff)  # [..., T]
    lon_error = torch.abs(lon_diff)  # [..., T]
    result_dict["position_lat_loss"] = lat_error
    result_dict["position_lon_loss"] = lon_error

    # Cosine similarity loss
    cosine_similarity = torch.sum(cos_sin_pred * cos_sin_gt, dim=-1)  # [..., T]
    result_dict["cosine_similarity_loss"] = 1.0 - cosine_similarity  # [..., T]

    return result_dict
