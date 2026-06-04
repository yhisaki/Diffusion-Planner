from copy import copy

import torch

from diffusion_planner.dimensions import OUTPUT_T
from diffusion_planner.utils.train_utils import openjson


class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    @classmethod
    def from_json(cls, args):
        data = openjson(args.normalization_file_path)
        horizon = getattr(args, "future_len", OUTPUT_T) + 1

        def expand_agent_stats(agent_stats):
            mean = torch.as_tensor(agent_stats["mean"], dtype=torch.float32)
            std = torch.as_tensor(agent_stats["std"], dtype=torch.float32)
            if mean.ndim == 1:
                mean = mean[None, :].expand(horizon, -1).clone()
                std = std[None, :].expand(horizon, -1).clone()
            if "xy_mean_by_timestep" in agent_stats:
                mean[:, :2] = torch.as_tensor(
                    agent_stats["xy_mean_by_timestep"], dtype=torch.float32
                )
            if "xy_std_by_timestep" in agent_stats:
                std[:, :2] = torch.as_tensor(
                    agent_stats["xy_std_by_timestep"], dtype=torch.float32
                )
            return mean, std

        ego_mean, ego_std = expand_agent_stats(data["ego"])
        neighbor_mean, neighbor_std = expand_agent_stats(data["neighbor"])
        mean = [ego_mean] + [neighbor_mean] * args.predicted_neighbor_num
        std = [ego_std] + [neighbor_std] * args.predicted_neighbor_num
        mean = torch.stack(mean, dim=0)
        std = torch.stack(std, dim=0)
        return cls(mean, std)

    def _select_stats(self, data, mean, std):
        if mean.ndim == data.ndim - 1:
            mean = mean[None]
            std = std[None]
        if data.ndim >= 4 and mean.shape[-2] == data.shape[-2] + 1:
            mean = mean[..., 1:, :]
            std = std[..., 1:, :]
        if data.ndim == 3 and mean.ndim == 3:
            mean = mean[:, 0, :]
            std = std[:, 0, :]
        return mean.to(data.device), std.to(data.device)

    def __call__(self, data):
        mean, std = self._select_stats(data, self.mean, self.std)
        return (data - mean) / std

    def inverse(self, data):
        mean, std = self._select_stats(data, self.mean, self.std)
        return data * std + mean

    def normalize_agent_slice(self, data, start: int):
        mean = self.mean[start : start + data.shape[1]].to(data.device)
        std = self.std[start : start + data.shape[1]].to(data.device)
        mean, std = self._select_stats(data, mean, std)
        return (data - mean) / std

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
        if isinstance(args, str):
            path = args
        else:
            path = args.normalization_file_path

        data = openjson(path)
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
            if k not in data:  # Check if key `k` exists in `data`
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = (data[k] - v["mean"].to(data[k].device)) / v["std"].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data

    def inverse(self, data):
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:  # Check if key `k` exists in `data`
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
