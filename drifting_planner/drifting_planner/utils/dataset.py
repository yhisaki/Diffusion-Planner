import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


def _collect_npz_files(path):
    """Collect .npz files from a path.

    If path is a directory, returns sorted list of all .npz files in it.
    If path is a JSON file, returns the list loaded from it.
    If path is a JSON string (content), parses it directly.
    """
    path = Path(path)

    if path.is_dir():
        return sorted([str(p) for p in path.glob("*.npz")])

    if path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)

    raise ValueError(f"Unsupported path type: {path}. Must be a directory or .json file.")


class DiffusionPlannerData(Dataset):
    def __init__(self, data_path):
        self.data_list = _collect_npz_files(data_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle=True)
        data = dict(data)
        return data
