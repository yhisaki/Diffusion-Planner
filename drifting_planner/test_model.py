import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from drifting_planner.dimensions import (
    INPUT_T,
    MAX_NUM_AGENTS,
    MAX_NUM_NEIGHBORS,
    NUM_LINE_STRINGS,
    NUM_POLYGONS,
    NUM_SEGMENTS_IN_LANE,
    NUM_SEGMENTS_IN_ROUTE,
    OUTPUT_T,
    POINTS_PER_LANELET,
    POINTS_PER_LINE_STRING,
    POINTS_PER_POLYGON,
    POSE_DIM,
    SEGMENT_POINT_DIM,
)
from drifting_planner.model.drifting_planner import DriftingPlanner
from drifting_planner.utils.config import Config
from drifting_planner.utils.visualize_input import visualize_inputs


def heading_to_cos_sin(x):
    return torch.cat(
        [
            x[..., :2],
            x[..., 2:3].cos(),
            x[..., 2:3].sin(),
        ],
        dim=-1,
    )


def load_model_and_config(ckpt_path, device):
    ckpt_dir = Path(ckpt_path).parent
    args_file = ckpt_dir / "args.json"
    if not args_file.exists():
        print(f"ERROR: args.json not found in {ckpt_dir}")
        sys.exit(1)

    config = Config(args_file)
    config.device = device  # type: ignore

    model = DriftingPlanner(config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
        print("Using EMA weights")
        state_dict = ckpt["ema_state_dict"]

    stripped = {k.replace("module.", ""): v for k, v in state_dict.items()}
    incompatible = model.load_state_dict(stripped, strict=False)
    if incompatible.missing_keys:
        print("WARNING: checkpoint is missing model keys:")
        for key in incompatible.missing_keys:
            print(f"  missing: {key}")
    if incompatible.unexpected_keys:
        print("WARNING: checkpoint has unexpected model keys:")
        for key in incompatible.unexpected_keys:
            print(f"  unexpected: {key}")
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {ckpt_path}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    return model, config


EXPECTED_SHAPES = {
    "ego_agent_past": (INPUT_T + 1, 3),
    "ego_agent_future": (OUTPUT_T, 3),
    "ego_current_state": (10,),
    "neighbor_agents_past": (MAX_NUM_NEIGHBORS, INPUT_T + 1, 11),
    "neighbor_agents_future": (MAX_NUM_NEIGHBORS, OUTPUT_T, 3),
    "static_objects": (5, 10),
    "lanes": (NUM_SEGMENTS_IN_LANE, POINTS_PER_LANELET, SEGMENT_POINT_DIM),
    "lanes_speed_limit": (NUM_SEGMENTS_IN_LANE, 1),
    "lanes_has_speed_limit": (NUM_SEGMENTS_IN_LANE, 1),
    "route_lanes": (NUM_SEGMENTS_IN_ROUTE, POINTS_PER_LANELET, SEGMENT_POINT_DIM),
    "route_lanes_speed_limit": (NUM_SEGMENTS_IN_ROUTE, 1),
    "route_lanes_has_speed_limit": (NUM_SEGMENTS_IN_ROUTE, 1),
    "polygons": (NUM_POLYGONS, POINTS_PER_POLYGON, 2),
    "line_strings": (NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 2),
    "goal_pose": (3,),
    "ego_shape": (3,),
    "turn_indicators": (INPUT_T,),
}


def ensure_batch_dim(key, arr):
    expected_shape = EXPECTED_SHAPES.get(key)
    if expected_shape is not None:
        if arr.shape == expected_shape:
            return arr[None, ...]
        if arr.ndim == len(expected_shape) + 1:
            return arr

    if arr.ndim >= 1:
        return arr[None, ...]
    return arr.reshape(1)


def load_npz(npz_path, device):
    data = np.load(npz_path)
    inputs = {}
    for key in data.files:
        arr = ensure_batch_dim(key, data[key])
        inputs[key] = torch.tensor(arr, dtype=torch.float32, device=device)

    return inputs


def prepare_inputs(inputs, config):
    inputs = {key: value.clone() for key, value in inputs.items()}
    B = inputs["ego_current_state"].shape[0]

    if "ego_agent_past" in inputs:
        inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
    if "goal_pose" in inputs:
        inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

    inputs["sampled_trajectories"] = torch.zeros(
        B, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, dtype=torch.float32, device=config.device
    )

    inputs = config.observation_normalizer(inputs)
    return inputs


def run_inference(model, inputs, config):
    with torch.no_grad():
        encoder_outputs, decoder_outputs = model(inputs)
    return encoder_outputs, decoder_outputs


def extract_predictions(decoder_outputs, inputs, config):
    prediction = decoder_outputs["prediction"][0].cpu().numpy()
    turn_indicator_logit = decoder_outputs["turn_indicator_logit"][0].cpu().numpy()
    turn_indicator_pred = turn_indicator_logit.argmax()

    inputs_np = {
        k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    if "goal_pose" in inputs_np and inputs_np["goal_pose"].shape[-1] == 3:
        goal_pose = inputs_np["goal_pose"]
        inputs_np["goal_pose"] = np.concatenate(
            [
                goal_pose[..., :2],
                np.cos(goal_pose[..., 2:3]),
                np.sin(goal_pose[..., 2:3]),
            ],
            axis=-1,
        )
    inputs_np["turn_indicator_pred"] = turn_indicator_pred
    inputs_np["prediction"] = prediction

    return inputs_np, prediction, turn_indicator_pred


def get_active_neighbor_indices(inputs_np, limit=None):
    neighbors = inputs_np["neighbor_agents_past"][0]
    current = neighbors[:, -1, :4]
    active = np.flatnonzero(np.any(np.abs(current) > 1e-6, axis=1)) + 1
    if limit is not None:
        active = active[:limit]
    return active.tolist()


def visualize_results(inputs_np, prediction, output_dir, npz_name, view_ranges=None, show=False):
    if view_ranges is None:
        view_ranges = [60]

    output_dir.mkdir(parents=True, exist_ok=True)

    fig_all, axes_all = visualize_inputs(
        inputs_np,
        save_path=None,
        view_ranges=view_ranges,
    )

    ax = axes_all[0] if len(view_ranges) == 1 else axes_all[0]  # type: ignore
    agent_indices = [0] + get_active_neighbor_indices(inputs_np, limit=20)
    for agent_idx in agent_indices:
        traj = prediction[agent_idx]
        valid = (traj[:, 0] != 0) | (traj[:, 1] != 0)
        if not valid.any():
            continue
        valid_traj = traj[valid]
        color = "cyan" if agent_idx == 0 else "magenta"
        label = "Ego Prediction" if agent_idx == 0 else f"Neighbor {agent_idx - 1} Prediction"
        ax.plot(
            valid_traj[:, 0],
            valid_traj[:, 1],
            color=color,
            linewidth=2,
            linestyle="-",
            label=label,
            alpha=0.8,
        )
        if len(valid_traj) > 0:
            ax.scatter(
                valid_traj[-1, 0],
                valid_traj[-1, 1],
                color=color,
                s=50,
                marker="*",
                zorder=5,
            )

    ax.legend(loc="upper left", fontsize=7)  # type: ignore

    if not show:
        save_path = output_dir / f"{npz_name}_result.png"
        fig_all.tight_layout()
        fig_all.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig_all)
        print(f"Visualization saved to {save_path}")
    else:
        fig_all.tight_layout()
        plt.show()


def print_summary(inputs_np, prediction, turn_indicator_pred):
    _TURN_INDICATOR_LABELS = {
        0: "Straight",
        1: "Straight",
        2: "Left",
        3: "Right",
        4: "Keep",
    }

    print("\n" + "=" * 50)
    print("INFERENCE SUMMARY")
    print("=" * 50)

    ego_state = inputs_np["ego_current_state"][0]
    print(f"Ego position: ({ego_state[0]:.2f}, {ego_state[1]:.2f})")
    print(f"Ego velocity: ({ego_state[4]:.2f}, {ego_state[5]:.2f}) m/s")

    print(f"\nPredicted turn: {_TURN_INDICATOR_LABELS.get(turn_indicator_pred, 'Unknown')}")

    print(f"\nPrediction shape: {prediction.shape}")
    print(f"  Ego final position: ({prediction[0, -1, 0]:.2f}, {prediction[0, -1, 1]:.2f})")

    active_neighbor_indices = get_active_neighbor_indices(inputs_np)
    predicted_neighbor_count = sum(
        (prediction[i, :, 0] != 0).any() or (prediction[i, :, 1] != 0).any()
        for i in active_neighbor_indices
    )
    print(f"  Active neighbors in scene: {len(active_neighbor_indices)}")
    print(f"  Active neighbors predicted: {predicted_neighbor_count}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Test and visualize DriftingPlanner model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--npz", type=str, required=True, help="Path to input data (.npz)")
    parser.add_argument("--output-dir", type=str, default="./test_output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--view-ranges", type=int, nargs="+", default=[60], help="View ranges in meters"
    )
    parser.add_argument("--show", action="store_true", help="Show plot with plt.show()")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, config = load_model_and_config(args.ckpt, device)
    raw_inputs = load_npz(args.npz, device)
    model_inputs = prepare_inputs(raw_inputs, config)

    encoder_outputs, decoder_outputs = run_inference(model, model_inputs, config)

    npz_name = Path(args.npz).stem
    output_dir = Path(args.output_dir)
    inputs_np, prediction, turn_indicator_pred = extract_predictions(
        decoder_outputs, raw_inputs, config
    )

    print_summary(inputs_np, prediction, turn_indicator_pred)
    visualize_results(inputs_np, prediction, output_dir, npz_name, args.view_ranges, args.show)


if __name__ == "__main__":
    main()
