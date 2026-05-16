from copy import copy
from importlib import resources
from typing import Any

import torch


def _load_default_normalization() -> dict[str, Any]:
    if hasattr(resources, "files"):
        ref = resources.files("drifting_planner").joinpath("normalization.json")
        with ref.open("r", encoding="utf-8") as f:
            return __import__("json").load(f)
    else:
        with resources.open_text("drifting_planner", "normalization.json") as f:
            return __import__("json").load(f)


class StateNormalizer:
    def __init__(self, mean: list | torch.Tensor, std: list | torch.Tensor) -> None:
        self.mean: torch.Tensor = torch.as_tensor(mean)
        self.std: torch.Tensor = torch.as_tensor(std)

    @classmethod
    def from_json(cls, args: Any) -> "StateNormalizer":
        data = _load_default_normalization()
        mean = [[data["ego"]["mean"]]] + [[data["neighbor"]["mean"]]] * args.predicted_neighbor_num
        std = [[data["ego"]["std"]]] + [[data["neighbor"]["std"]]] * args.predicted_neighbor_num
        return cls(mean, std)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.std.to(data.device) + self.mean.to(data.device)

    def to_dict(self) -> dict[str, list]:
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist(),
        }


class TrajectoryNormalizer(StateNormalizer):
    @classmethod
    def from_json(cls, args: Any) -> "TrajectoryNormalizer":
        data = _load_default_normalization()
        ego_key = "ego_delta" if "ego_delta" in data else "ego"
        neighbor_key = "neighbor_delta" if "neighbor_delta" in data else "neighbor"
        mean = [[data[ego_key]["mean"]]] + [
            [data[neighbor_key]["mean"]]
        ] * args.predicted_neighbor_num
        std = [[data[ego_key]["std"]]] + [[data[neighbor_key]["std"]]] * args.predicted_neighbor_num
        return cls(mean, std)

    @staticmethod
    def _current_xy_for(data: torch.Tensor, current_states_raw: torch.Tensor) -> torch.Tensor:
        if data.dim() == 4:
            return current_states_raw[:, :, None, :2]
        if data.dim() == 5:
            return current_states_raw[:, None, :, None, :2]
        raise ValueError("Expected trajectory shape [B, P, T, 4] or [B, M, P, T, 4].")

    def normalize_future(
        self, future: torch.Tensor, current_states_raw: torch.Tensor
    ) -> torch.Tensor:
        delta = future.clone()
        delta[..., :2] = delta[..., :2] - self._current_xy_for(delta, current_states_raw)
        return self(delta)

    def inverse_future(
        self, normalized_delta: torch.Tensor, current_states_raw: torch.Tensor
    ) -> torch.Tensor:
        future = self.inverse(normalized_delta)
        future[..., :2] = future[..., :2] + self._current_xy_for(future, current_states_raw)
        return future


class ObservationNormalizer:
    def __init__(self, normalization_dict: dict[str, dict[str, torch.Tensor]]) -> None:
        self._normalization_dict: dict[str, dict[str, torch.Tensor]] = normalization_dict

    @classmethod
    def from_json(cls, args: Any) -> "ObservationNormalizer":
        data = _load_default_normalization()
        ndt: dict[str, dict[str, torch.Tensor]] = {}
        for k, v in data.items():
            if k not in ["ego", "neighbor", "ego_delta", "neighbor_delta"]:
                ndt[k] = {
                    "mean": torch.tensor(v["mean"], dtype=torch.float32),
                    "std": torch.tensor(v["std"], dtype=torch.float32),
                }
        return cls(ndt)

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = (data[k] - v["mean"].to(data[k].device)) / v["std"].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data

    def inverse(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = data[k] * v["std"].to(data[k].device) + v["mean"].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data

    def to_dict(self) -> dict[str, dict[str, list]]:
        return {
            k: {kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()}
            for k, v in self._normalization_dict.items()
        }
