import torch
import torch.nn.functional as F


def loss_func(
    trajectory_pred: torch.Tensor, trajectory_gt: torch.Tensor
) -> dict[str, torch.Tensor]:
    result_dict = {}

    position_pred = trajectory_pred[..., :2]
    position_gt = trajectory_gt[..., :2]

    position_diff = position_pred - position_gt
    position_error = torch.sum(position_diff**2, dim=-1)
    result_dict["position_l2_loss"] = position_error

    cos_sin_pred = trajectory_pred[..., 2:]
    cos_sin_gt = trajectory_gt[..., 2:]

    heading_loss = torch.sum((cos_sin_pred - cos_sin_gt) ** 2, dim=-1)
    result_dict["heading_l2_loss"] = heading_loss

    cos_gt = cos_sin_gt[..., 0]
    sin_gt = cos_sin_gt[..., 1]
    lon_diff = +position_diff[..., 0] * cos_gt + position_diff[..., 1] * sin_gt
    lat_diff = -position_diff[..., 0] * sin_gt + position_diff[..., 1] * cos_gt
    lat_error = torch.abs(lat_diff)
    lon_error = torch.abs(lon_diff)
    result_dict["position_lat_loss"] = lat_error
    result_dict["position_lon_loss"] = lon_error

    return result_dict


def make_turn_indicator_gt(
    turn_indicators: torch.Tensor,
) -> torch.Tensor:
    from drifting_planner.dimensions import TURN_INDICATOR_OUTPUT_KEEP

    turn_indicators_gt = turn_indicators.long()
    turn_indicators_gt_keep = turn_indicators_gt[:, -1] == turn_indicators_gt[:, -2]
    turn_indicators_gt = turn_indicators_gt[:, -1] * ~turn_indicators_gt_keep
    turn_indicators_gt = turn_indicators_gt + turn_indicators_gt_keep * TURN_INDICATOR_OUTPUT_KEEP
    return turn_indicators_gt
