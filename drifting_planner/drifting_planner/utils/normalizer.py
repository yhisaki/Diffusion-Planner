from copy import copy
from importlib import resources

import torch


def _load_default_normalization():
    if hasattr(resources, "files"):
        ref = resources.files("drifting_planner").joinpath("normalization.json")
        with ref.open("r", encoding="utf-8") as f:
            return __import__("json").load(f)
    else:
        with resources.open_text("drifting_planner", "normalization.json") as f:
            return __import__("json").load(f)


class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    @classmethod
    def from_json(cls, args):
        data = _load_default_normalization()
        mean = [[data["ego"]["mean"]]] + [[data["neighbor"]["mean"]]] * args.predicted_neighbor_num
        std = [[data["ego"]["std"]]] + [[data["neighbor"]["std"]]] * args.predicted_neighbor_num
        return cls(mean, std)

    def __call__(self, data):
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse(self, data):
        return data * self.std.to(data.device) + self.mean.to(data.device)

    def to_dict(self):
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist(),
        }


class ObservationNormalizer:
    def __init__(self, normalization_dict):
        self._normalization_dict = normalization_dict

    @classmethod
    def from_json(cls, args):
        data = _load_default_normalization()
        ndt = {}
        for k, v in data.items():
            if k not in ["ego", "neighbor"]:
                ndt[k] = {
                    "mean": torch.tensor(v["mean"], dtype=torch.float32),
                    "std": torch.tensor(v["std"], dtype=torch.float32),
                }
        return cls(ndt)

    def __call__(self, data):
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = (data[k] - v["mean"].to(data[k].device)) / v["std"].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data

    def inverse(self, data):
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = data[k] * v["std"].to(data[k].device) + v["mean"].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data

    def to_dict(self):
        return {
            k: {kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()}
            for k, v in self._normalization_dict.items()
        }
