import json

import torch

from drifting_planner.utils.normalizer import (
    ObservationNormalizer,
    StateNormalizer,
    TrajectoryNormalizer,
)


class Config:
    def __init__(self, args_file):
        with open(args_file, "r") as f:
            args_dict = json.load(f)

        for key, value in args_dict.items():
            setattr(self, key, value)
        self.state_normalizer = StateNormalizer(
            self.state_normalizer["mean"], self.state_normalizer["std"]
        )
        trajectory_normalizer = getattr(self, "trajectory_normalizer", None)
        if trajectory_normalizer is None:
            trajectory_normalizer = self.state_normalizer.to_dict()
        self.trajectory_normalizer = TrajectoryNormalizer(
            trajectory_normalizer["mean"], trajectory_normalizer["std"]
        )
        self.observation_normalizer = ObservationNormalizer(
            {
                k: {"mean": torch.as_tensor(v["mean"]), "std": torch.as_tensor(v["std"])}
                for k, v in self.observation_normalizer.items()
            }
        )
