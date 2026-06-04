import argparse
import json
import random
from pathlib import Path

import numpy as np
from diffusion_planner.dimensions import OUTPUT_T


def load_data_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["files"] if isinstance(data, dict) else data


def sample_data_list(files: list[str], max_data_size: int | None, seed: int) -> list[str]:
    if max_data_size is None:
        return files
    if max_data_size < 1:
        raise ValueError("--max_data_size must be at least 1.")
    if len(files) <= max_data_size:
        return files
    rng = random.Random(seed)
    return rng.sample(files, max_data_size)


def running_update(count, mean, m2, values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return count, mean, m2
    if mean is None:
        mean = np.zeros(values.shape[1:], dtype=np.float64)
        m2 = np.zeros(values.shape[1:], dtype=np.float64)
    for value in values:
        count += 1
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)
    return count, mean, m2


def valid_rows(values):
    return np.sum(values != 0, axis=-1) != 0


def running_update_by_timestep(count, mean, m2, values, valid_mask):
    values = np.asarray(values, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if mean is None:
        mean = np.zeros(values.shape[1:], dtype=np.float64)
        m2 = np.zeros(values.shape[1:], dtype=np.float64)
        count = np.zeros(values.shape[1], dtype=np.int64)
    for timestep in range(values.shape[1]):
        for value in values[valid_mask[:, timestep], timestep]:
            count[timestep] += 1
            delta = value - mean[timestep]
            mean[timestep] += delta / count[timestep]
            m2[timestep] += delta * (value - mean[timestep])
    return count, mean, m2


def finalize(count, mean, m2):
    if count < 2:
        raise ValueError("At least two samples are required to compute normalization stats.")
    var = m2 / count
    return mean.astype(np.float32), np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)


def finalize_by_timestep(count, mean, m2):
    if np.any(count < 2):
        bad = np.where(count < 2)[0].tolist()
        raise ValueError(
            "At least two samples are required for every timestep to compute "
            f"normalization stats. Insufficient timesteps: {bad}"
        )
    var = m2 / count[:, None]
    return mean.astype(np.float32), np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)


def heading_to_xy_cos_sin(values):
    if values.shape[-1] >= 4:
        return values[..., :4].astype(np.float32)
    heading = values[..., 2]
    return np.stack(
        [
            values[..., 0],
            values[..., 1],
            np.cos(heading),
            np.sin(heading),
        ],
        axis=-1,
    ).astype(np.float32)


def transform_future_to_agent_frame(future, current):
    out = future.copy()
    rel = future[..., :2] - current[:, None, :2]
    cur_cos = current[:, None, 2]
    cur_sin = current[:, None, 3]

    out[..., 0] = rel[..., 0] * cur_cos + rel[..., 1] * cur_sin
    out[..., 1] = -rel[..., 0] * cur_sin + rel[..., 1] * cur_cos

    fut_cos = future[..., 2]
    fut_sin = future[..., 3]
    out[..., 2] = fut_cos * cur_cos + fut_sin * cur_sin
    out[..., 3] = fut_sin * cur_cos - fut_cos * cur_sin
    return out


def compute_stats(files: list[str], future_len: int) -> dict:
    ego_count, ego_mean, ego_m2 = None, None, None
    neighbor_count, neighbor_mean, neighbor_m2 = None, None, None
    ego_past_count, ego_past_mean, ego_past_m2 = None, None, None
    neighbor_past_count, neighbor_past_mean, neighbor_past_m2 = None, None, None

    for file_path in files:
        data = np.load(file_path, allow_pickle=True)

        ego_past_raw = data["ego_agent_past"].astype(np.float32)
        ego_past = heading_to_xy_cos_sin(ego_past_raw)
        ego_past_valid = valid_rows(ego_past_raw)
        ego_past_valid[-1] = True
        ego_past[~ego_past_valid] = 0.0
        ego_past_count, ego_past_mean, ego_past_m2 = running_update_by_timestep(
            ego_past_count, ego_past_mean, ego_past_m2, ego_past[None, :, :], ego_past_valid[None, :]
        )

        ego_current = data["ego_current_state"][:4].astype(np.float32)
        ego_future_raw = data["ego_agent_future"][:future_len].astype(np.float32)
        ego_future = heading_to_xy_cos_sin(ego_future_raw)
        ego_agent = np.concatenate([ego_current[None, :], ego_future], axis=0)
        ego_valid = np.concatenate([[True], valid_rows(ego_future_raw)], axis=0)
        ego_agent[~ego_valid] = 0.0
        ego_count, ego_mean, ego_m2 = running_update_by_timestep(
            ego_count, ego_mean, ego_m2, ego_agent[None, :, :], ego_valid[None, :]
        )

        neighbor_past = data["neighbor_agents_past"].astype(np.float32)
        neighbor_past_valid = valid_rows(neighbor_past)
        neighbor_past[~neighbor_past_valid] = 0.0
        if np.any(neighbor_past_valid):
            neighbor_past_count, neighbor_past_mean, neighbor_past_m2 = running_update_by_timestep(
                neighbor_past_count,
                neighbor_past_mean,
                neighbor_past_m2,
                neighbor_past,
                neighbor_past_valid,
            )

        neighbor_future = data["neighbor_agents_future"][:, :future_len].astype(np.float32)
        current = neighbor_past[:, -1, :4]
        valid_current = valid_rows(current)
        valid_future = valid_rows(neighbor_future[..., :3])
        valid_neighbor = valid_current & np.any(valid_future, axis=-1)
        if np.any(valid_neighbor):
            cur = current[valid_neighbor]
            fut = heading_to_xy_cos_sin(neighbor_future[valid_neighbor])
            fut = transform_future_to_agent_frame(fut, cur)
            valid = valid_future[valid_neighbor]

            current_canonical = np.zeros((cur.shape[0], 1, 4), dtype=np.float32)
            current_canonical[..., 2] = 1.0
            neighbor_agent = np.concatenate([current_canonical, fut], axis=1)
            neighbor_valid = np.concatenate(
                [np.ones((cur.shape[0], 1), dtype=bool), valid], axis=1
            )
            neighbor_agent[~neighbor_valid] = 0.0
            neighbor_count, neighbor_mean, neighbor_m2 = running_update_by_timestep(
                neighbor_count, neighbor_mean, neighbor_m2, neighbor_agent, neighbor_valid
            )

    ego_mean, ego_std = finalize_by_timestep(ego_count, ego_mean, ego_m2)
    neighbor_mean, neighbor_std = finalize_by_timestep(neighbor_count, neighbor_mean, neighbor_m2)
    ego_past_mean, ego_past_std = finalize_by_timestep(
        ego_past_count, ego_past_mean, ego_past_m2
    )
    neighbor_past_mean, neighbor_past_std = finalize_by_timestep(
        neighbor_past_count, neighbor_past_mean, neighbor_past_m2
    )

    return {
        "ego": {
            "mean": ego_mean.tolist(),
            "std": ego_std.tolist(),
        },
        "neighbor": {
            "mean": neighbor_mean.tolist(),
            "std": neighbor_std.tolist(),
        },
        "ego_agent_past": {
            "mean": ego_past_mean.tolist(),
            "std": ego_past_std.tolist(),
        },
        "neighbor_agents_past": {
            "mean": neighbor_past_mean.tolist(),
            "std": neighbor_past_std.tolist(),
        },
    }


def merge_with_base(base_path: Path | None, stats: dict) -> dict:
    if base_path is None:
        return stats
    with base_path.open("r", encoding="utf-8") as f:
        base = json.load(f)
    base.update(stats)
    return base


def round_sig_digits(value, digits: int = 4):
    if isinstance(value, dict):
        return {k: round_sig_digits(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [round_sig_digits(v, digits) for v in value]
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    return value


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_list", type=Path, required=True)
    parser.add_argument("--data_list", type=Path, default="/mnt/nvme/dataset/basic_dataset/path_list_train.json")
    parser.add_argument("--output", type=Path, default=Path("normalization.json"))
    parser.add_argument("--base", type=Path, default=None)
    parser.add_argument("--future_len", type=int, default=OUTPUT_T)
    parser.add_argument(
        "--max_data_size",
        type=int,
        default=10**3,
        help="Maximum number of data files to use. If smaller than data_list size, sample without replacement.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for sampling.")
    args = parser.parse_args()

    files = load_data_list(args.data_list)
    sampled_files = sample_data_list(files, args.max_data_size, args.seed)
    if len(sampled_files) != len(files):
        print(
            f"Sampled {len(sampled_files)} files without replacement "
            f"from {len(files)} files (seed={args.seed})."
        )
    else:
        print(f"Using {len(sampled_files)} files.")

    stats = compute_stats(sampled_files, args.future_len)
    output = round_sig_digits(merge_with_base(args.base, stats), 4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
