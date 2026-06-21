"""NPZ loading utilities — no dependency on other workspace packages."""

from __future__ import annotations

from pathlib import Path

import numpy as np

NPZ_KEYS = [
    "ego_agent_future",
    "ego_agent_past",
    "ego_current_state",
    "ego_shape",
    "goal_pose",
    "lanes",
    "lanes_has_speed_limit",
    "lanes_speed_limit",
    "line_strings",
    "neighbor_agents_future",
    "neighbor_agents_past",
    "polygons",
    "route_lanes",
    "route_lanes_has_speed_limit",
    "route_lanes_speed_limit",
    "static_objects",
    "turn_indicators",
]


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load an NPZ training data file and return a dict of numpy arrays.

    Skips scalar metadata keys (``version``, ``map_name``, ``token``, ``delay``)
    so only float arrays used by visualisation are returned.
    """
    with np.load(str(path), allow_pickle=True) as f:
        data: dict[str, np.ndarray] = {}
        for key in f.keys():
            if key in {"map_name", "token", "delay", "version"}:
                continue
            data[key] = f[key]
    return data


def discover_npz_files(directory: str | Path) -> list[Path]:
    """Return sorted list of ``*.npz`` paths under *directory*.

    Only top-level files are included (no recursion).
    """
    d = Path(directory).resolve()
    if not d.is_dir():
        raise FileNotFoundError(f"Not a directory: {d}")
    files = sorted(d.glob("*.npz"))
    return files
