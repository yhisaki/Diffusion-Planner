"""Time-series data extraction helpers.

These functions extract past and future ego trajectories from NPZ data dicts
and return (time, values) tuples suitable for Plotly time-series charts.
"""

from __future__ import annotations

import numpy as np


def past_positions(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract ego past (x, y) positions joined with the current state.

    Args:
        data: NPZ data dict (requires ``"ego_agent_past"``).

    Returns:
        ``(t, positions)`` where ``t`` is a time-step vector (negative for
        past, 0 for current) and ``positions`` is ``(N, 2)``.  Returns
        ``None`` if past data is unavailable.
    """
    if "ego_agent_past" not in data:
        return None
    past = data["ego_agent_past"].reshape(-1, data["ego_agent_past"].shape[-1])
    ego_state = data["ego_current_state"].reshape(-1)
    positions = np.vstack([past[:, :2], [[ego_state[0], ego_state[1]]]])
    n = len(positions)
    t = np.arange(-n + 1, 1)
    return t, positions


def past_heading_cos_sin(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract past cos/sin of heading joined with the current state.

    Args:
        data: NPZ data dict (requires ``"ego_agent_past"``).

    Returns:
        ``(t, cos_vals, sin_vals)`` or ``None``.
    """
    if "ego_agent_past" not in data:
        return None
    past = data["ego_agent_past"].reshape(-1, data["ego_agent_past"].shape[-1])
    ego_state = data["ego_current_state"].reshape(-1)
    if past.shape[1] >= 4:
        cos_vals = past[:, 2]
        sin_vals = past[:, 3]
    elif past.shape[1] == 3:
        heading = past[:, 2]
        cos_vals = np.cos(heading)
        sin_vals = np.sin(heading)
    else:
        return None
    cos_vals = np.append(cos_vals, ego_state[2])
    sin_vals = np.append(sin_vals, ego_state[3])
    t = np.arange(-len(cos_vals) + 1, 1)
    return t, cos_vals, sin_vals


def future_heading_cos_sin(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract future cos/sin of heading starting from the current state.

    Args:
        data: NPZ data dict (requires ``"ego_agent_future"``).

    Returns:
        ``(t, cos_vals, sin_vals)`` or ``None``.
    """
    if "ego_agent_future" not in data:
        return None
    ego_state = data["ego_current_state"].reshape(-1)
    future = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
    if future.shape[1] >= 4:
        cos_vals = np.hstack([ego_state[2], future[:, 2]])
        sin_vals = np.hstack([ego_state[3], future[:, 3]])
    elif future.shape[1] >= 3:
        heading = np.hstack([np.arctan2(ego_state[3], ego_state[2]), future[:, 2]])
        cos_vals = np.cos(heading)
        sin_vals = np.sin(heading)
    else:
        return None
    t = np.arange(len(cos_vals))
    return t, cos_vals, sin_vals
