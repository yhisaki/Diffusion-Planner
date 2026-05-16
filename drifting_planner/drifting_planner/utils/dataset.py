import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset


def _collect_npz_files(path: str | Path) -> list[str]:
    """Collect .npz files from a path.

    If path is a directory, returns sorted list of all .npz files in it.
    If path is a JSON file, returns the list loaded from it.
    If path is a JSON string (content), parses it directly.
    """
    p = Path(path)

    if p.is_dir():
        return sorted([str(f) for f in p.glob("*.npz")])

    if p.suffix == ".json":
        with open(p, "r") as f:
            return json.load(f)

    raise ValueError(f"Unsupported path type: {p}. Must be a directory or .json file.")


class DriftingPlannerData(Dataset):
    def __init__(self, data_path: str) -> None:
        self.data_list: list[str] = _collect_npz_files(data_path)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        data = np.load(self.data_list[idx], allow_pickle=True)
        data = dict(data)
        return data
