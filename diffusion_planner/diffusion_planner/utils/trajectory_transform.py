import torch


def transform_future_to_agent_frame(
    future: torch.Tensor,
    current: torch.Tensor,
    invalid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform future poses from ego frame to each agent's current frame.

    Args:
        future: [B, N, T, 4] poses in ego frame, with (x, y, cos, sin).
        current: [B, N, 4] current poses in ego frame, with (x, y, cos, sin).
        invalid_mask: optional [B, N, T] mask for padded future poses.

    Returns:
        Future poses in each agent-centric frame.
    """
    out = future.clone()
    cur_xy = current[..., None, :2]
    cur_cos = current[..., None, 2]
    cur_sin = current[..., None, 3]

    rel = future[..., :2] - cur_xy
    out[..., 0] = rel[..., 0] * cur_cos + rel[..., 1] * cur_sin
    out[..., 1] = -rel[..., 0] * cur_sin + rel[..., 1] * cur_cos

    fut_cos = future[..., 2]
    fut_sin = future[..., 3]
    out[..., 2] = fut_cos * cur_cos + fut_sin * cur_sin
    out[..., 3] = fut_sin * cur_cos - fut_cos * cur_sin

    if invalid_mask is not None:
        out[invalid_mask] = 0.0
    return out


def make_agent_centric_current_states(
    current_states: torch.Tensor,
    neighbor_invalid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return output-frame current states for ego and neighbor trajectories.

    Ego remains in the ego frame. Each valid neighbor is represented in its own
    current frame, so its current pose is (0, 0, 1, 0).
    """
    out = current_states.clone()
    if out.shape[1] <= 1:
        return out

    canonical = torch.zeros_like(out[:, 1:])
    canonical[..., 2] = 1.0
    canonical[neighbor_invalid_mask] = 0.0
    out[:, 1:] = canonical
    return out
