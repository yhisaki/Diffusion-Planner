import torch


def make_turn_indicator_gt(
    turn_indicators: torch.Tensor,
) -> torch.Tensor:
    from drifting_planner.dimensions import TURN_INDICATOR_OUTPUT_KEEP

    turn_indicators_gt = turn_indicators.long()
    turn_indicators_gt_keep = turn_indicators_gt[:, -1] == turn_indicators_gt[:, -2]
    turn_indicators_gt = turn_indicators_gt[:, -1] * ~turn_indicators_gt_keep
    turn_indicators_gt = turn_indicators_gt + turn_indicators_gt_keep * TURN_INDICATOR_OUTPUT_KEEP
    return turn_indicators_gt
