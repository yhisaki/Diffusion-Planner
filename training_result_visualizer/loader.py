"""Path-list loading utilities for training result visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_path_list(path_list_json: str | Path) -> list[Path]:
    """Load a JSON path list.

    Accepts either a plain list of NPZ paths or a list of dicts containing one
    of the common path keys used in this workspace.
    """
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
