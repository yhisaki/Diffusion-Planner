"""Data loading utilities for the Diffusion Planner GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from diffusion_planner.utils.dataset import add_current_state_to_ego_agent_future


def discover_npz_files(directory: str | Path) -> list[Path]:
    """Return sorted list of ``*.npz`` paths under *directory* (no recursion)."""
    d = Path(directory).resolve()
    if not d.is_dir():
        raise FileNotFoundError(f"Not a directory: {d}")
    return sorted(d.glob("*.npz"))


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load an NPZ training data file and return a dict of numpy arrays."""
    with np.load(str(path), allow_pickle=True) as f:
        data: dict[str, np.ndarray] = {}
        for key in f.keys():
            if key in {"map_name", "token", "delay", "version"}:
                continue
            data[key] = f[key]
    
    data = add_current_state_to_ego_agent_future(data)
    
    return data


def load_path_list(path_list_json: str | Path) -> list[Path]:
    """Load a JSON path list of NPZ paths from a file."""
    path = Path(path_list_json).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"path_list.json not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"path_list.json must contain a list: {path}")

    base_dir = path.parent
    paths: list[Path] = []
    for item in raw:
        item_path = _extract_path(item)
        if item_path is None:
            continue
        npz_path = Path(item_path).expanduser()
        if not npz_path.is_absolute():
            npz_path = base_dir / npz_path
        paths.append(npz_path.resolve())

    return paths


def _extract_path(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("npz_path", "npz", "path", "scene_path"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    return None


def resolve_data_path(path: str) -> list[Path]:
    """Resolve a data path to a list of NPZ files.

    Accepts:
    - A path_list.json file (*.json)
    - A single NPZ file (*.npz)
    - A directory containing NPZ files
    """
    p = Path(path).expanduser().resolve()

    if p.is_file():
        if p.suffix == ".json":
            return load_path_list(p)
        elif p.suffix == ".npz":
            return [p]

    if p.is_dir():
        return discover_npz_files(p)

    raise FileNotFoundError(f"Cannot resolve data path: {path}")
