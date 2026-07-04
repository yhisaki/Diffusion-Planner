import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson


def add_current_state_to_ego_agent_future(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add the current state of the ego vehicle to the ego_future array.

    Args:
        data: A dictionary containing the data.

    Returns:
        The updated data dictionary with the current state added to the ego_future array.
    """
    future = data["ego_agent_future"]
    state = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # x, y, yaw
    future = np.concatenate([state[None, :], future[:-1, :]], axis=0)
    data["ego_agent_future"] = future
    return data


class DiffusionPlannerData(Dataset):
    def __init__(self, data_list, data_augmentation=None):
        data = openjson(data_list)
        # Accept both legacy list format and sampling.py dict format {"seed": ..., "files": [...]}
        self.data_list = data["files"] if isinstance(data, dict) else data
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle=True)
        data = dict(data)  # npz to dict
        data = add_current_state_to_ego_agent_future(data)
        if self.data_augmentation is not None:
            data = self.data_augmentation.augment(data)
        return data
