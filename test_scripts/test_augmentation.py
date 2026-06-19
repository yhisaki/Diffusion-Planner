import argparse
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.visualize_input import visualize_inputs

parser = argparse.ArgumentParser()
parser.add_argument("target_npz", type=Path)
parser.add_argument("save_dir", type=Path)
parser.add_argument("--augment_min_speed", type=float, default=1.0)
parser.add_argument("--augment_time_interval", type=float, default=0.1)
parser.add_argument("--augment_min_linearization_speed", type=float, default=0.5)
parser.add_argument("--augment_exact_position_gain", type=float, default=2.0)
parser.add_argument("--augment_exact_velocity_gain", type=float, default=3.0)
parser.add_argument("--augment_lateral_offset_std", type=float, default=1.0)
parser.add_argument("--augment_yaw_half_range", type=float, default=0.05)
parser.add_argument("--augment_default_wheel_base", type=float, default=3.0)
parser.add_argument("--augment_speed_scale_half_range", type=float, default=0.05)
args = parser.parse_args()

target_npz = args.target_npz

save_dir = args.save_dir
save_dir.mkdir(parents=True, exist_ok=True)

loaded = np.load(target_npz)
data = {}
for key, value in loaded.items():
    if key == "token":
        continue
    data[key] = np.array(value, copy=True)

aug = StatePerturbation(
    augment_prob=1.0,
    min_speed=args.augment_min_speed,
    time_interval=args.augment_time_interval,
    min_linearization_speed=args.augment_min_linearization_speed,
    exact_position_gain=args.augment_exact_position_gain,
    exact_velocity_gain=args.augment_exact_velocity_gain,
    lateral_offset_std=args.augment_lateral_offset_std,
    yaw_half_range=args.augment_yaw_half_range,
    default_wheel_base=args.augment_default_wheel_base,
    speed_scale_half_range=args.augment_speed_scale_half_range,
)


def heading_to_cos_sin_np(x: np.ndarray) -> np.ndarray:
    if x.shape[-1] != 3:
        return x
    return np.concatenate([x[..., :2], np.cos(x[..., 2:3]), np.sin(x[..., 2:3])], axis=-1)


def as_visualization_batch(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    visualized = {key: np.expand_dims(value, axis=0) for key, value in sample.items()}
    for key in ("goal_pose", "ego_agent_past"):
        if key in visualized:
            visualized[key] = heading_to_cos_sin_np(visualized[key])
    return visualized

# Save original data visualization with augmentation range rectangle
original_save_path = save_dir / "original.png"
fig, ax = plt.subplots(figsize=(10, 10))

# Visualize inputs on the ax
view_range = 30
visualize_inputs(as_visualization_batch(deepcopy(data)), save_path=None, ax=ax, view_ranges=[view_range])

# Get augmentation ranges from the aug object (approximate +/- 3 sigma for normal)
cfg = aug.config
x_min, y_min = 0.0, -3.0 * cfg.lateral_offset_std
x_max, y_max = 0.0, 3.0 * cfg.lateral_offset_std

# Draw the augmentation range rectangle
rect = patches.Rectangle(
    (x_min, y_min),
    max(x_max - x_min, 0.05),
    y_max - y_min,
    linewidth=2,
    edgecolor="red",
    facecolor="none",
    linestyle="--",
    label="Augmentation Range",
)
ax.add_patch(rect)
ax.legend()

plt.tight_layout()
plt.savefig(original_save_path, dpi=100)
plt.close()

trial_num = 10
elapsed_times = []
for i in range(trial_num):
    t0 = time.perf_counter()
    aug_data = aug(deepcopy(data))
    elapsed_times.append(time.perf_counter() - t0)

    # Save augmented data to npz file
    data_dict = {key: value for key, value in aug_data.items() if isinstance(value, np.ndarray)}

    # Save to npz file
    output_path = save_dir / f"augmented_{i:08d}.npz"
    np.savez(output_path, **data_dict)

    # Use deepcopy to avoid side effects from visualize_inputs
    visualize_inputs(
        as_visualization_batch(deepcopy(aug_data)),
        save_dir / f"augmented_{i:08d}.png",
        view_ranges=[view_range],
    )

print(f"Augmented data saved: {trial_num} files to {save_dir}")
print(
    f"Augmentation time: mean={sum(elapsed_times) / len(elapsed_times) * 1000:.1f}ms  min={min(elapsed_times) * 1000:.1f}ms  max={max(elapsed_times) * 1000:.1f}ms"
)
