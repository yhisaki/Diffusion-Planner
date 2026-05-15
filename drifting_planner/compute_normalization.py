import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).parent / "drifting_planner" / "normalization.json"


def collect_npz_files(path):
    path = Path(path).expanduser()
    if path.is_dir():
        return sorted(path.glob("*.npz"))
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return [Path(p).expanduser() for p in json.load(f)]
    if path.suffix == ".npz":
        return [path]
    raise ValueError(f"Unsupported data path: {path}")


def heading_to_cos_sin(x):
    return np.concatenate([x[..., :2], np.cos(x[..., 2:3]), np.sin(x[..., 2:3])], axis=-1)


class RunningStats:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.count = None
        self.sum = None
        self.sumsq = None

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        values = values.reshape(-1, values.shape[-1])
        count = values.shape[0]
        value_sum = values.sum(axis=0)
        value_sumsq = np.square(values).sum(axis=0)
        if self.count is None:
            self.count = count
            self.sum = value_sum
            self.sumsq = value_sumsq
        else:
            self.count += count
            self.sum += value_sum
            self.sumsq += value_sumsq

    def finalize(self, min_std):
        if self.count is None or self.count == 0:
            return [0.0] * self.feature_dim, [1.0] * self.feature_dim
        mean = self.sum / self.count
        var = self.sumsq / self.count - np.square(mean)
        std = np.sqrt(np.maximum(var, 0.0))
        std = np.maximum(std, min_std)
        return mean.tolist(), std.tolist()


def valid_rows(x):
    return np.any(np.abs(x) > 1e-8, axis=-1)


def update_sequence(stats, key, data):
    mask = valid_rows(data)
    stats[key].update(data[mask])


def update_speed_limit(stats, key, speed_limit, has_speed_limit):
    mask = has_speed_limit.astype(bool).reshape(-1)
    values = speed_limit.reshape(-1, speed_limit.shape[-1])
    stats[key].update(values[mask])


def compute_normalization(npz_files, min_std):
    stats = {
        "ego": RunningStats(4),
        "neighbor": RunningStats(4),
        "ego_agent_past": RunningStats(4),
        "ego_current_state": RunningStats(10),
        "neighbor_agents_past": RunningStats(11),
        "static_objects": RunningStats(10),
        "lanes": RunningStats(33),
        "lanes_speed_limit": RunningStats(1),
        "route_lanes": RunningStats(33),
        "route_lanes_speed_limit": RunningStats(1),
        "polygons": RunningStats(3),
        "line_strings": RunningStats(4),
        "goal_pose": RunningStats(4),
    }

    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)

        ego_past = heading_to_cos_sin(data["ego_agent_past"])
        ego_future = heading_to_cos_sin(data["ego_agent_future"])
        neighbor_future_raw = data["neighbor_agents_future"]
        neighbor_future_mask = valid_rows(neighbor_future_raw)
        neighbor_future = heading_to_cos_sin(neighbor_future_raw)
        goal_pose = heading_to_cos_sin(data["goal_pose"])

        update_sequence(stats, "ego_agent_past", ego_past)
        update_sequence(stats, "ego", ego_future)
        stats["ego_current_state"].update(data["ego_current_state"][None, :])
        update_sequence(stats, "neighbor", neighbor_future[neighbor_future_mask])
        update_sequence(stats, "neighbor_agents_past", data["neighbor_agents_past"])
        update_sequence(stats, "static_objects", data["static_objects"])
        update_sequence(stats, "lanes", data["lanes"])
        update_speed_limit(
            stats,
            "lanes_speed_limit",
            data["lanes_speed_limit"],
            data["lanes_has_speed_limit"],
        )
        update_sequence(stats, "route_lanes", data["route_lanes"])
        update_speed_limit(
            stats,
            "route_lanes_speed_limit",
            data["route_lanes_speed_limit"],
            data["route_lanes_has_speed_limit"],
        )
        update_sequence(stats, "polygons", data["polygons"])
        update_sequence(stats, "line_strings", data["line_strings"])
        stats["goal_pose"].update(goal_pose[None, :])

    normalization = {}
    for key, value in stats.items():
        mean, std = value.finalize(min_std)
        normalization[key] = {"mean": mean, "std": std}

    return normalization


def main():
    parser = argparse.ArgumentParser(description="Compute DriftingPlanner normalization stats")
    parser.add_argument("--data", required=True, help="Training data directory, .json list, or .npz")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output normalization.json path")
    parser.add_argument("--min-std", type=float, default=1e-3)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    npz_files = collect_npz_files(args.data)
    if args.max_files is not None:
        npz_files = npz_files[: args.max_files]
    if not npz_files:
        raise ValueError(f"No .npz files found under {args.data}")

    normalization = compute_normalization(npz_files, args.min_std)

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(normalization, f, indent=2)
        f.write("\n")

    print(f"Wrote normalization stats for {len(npz_files)} files to {output}")


if __name__ == "__main__":
    main()
