"""Training Data Visualizer — interactive NPZ viewer with Plotly and Gradio.

Launch
------
    training_data_visualizer /path/to/npz/directory
"""

from __future__ import annotations

import argparse
import functools
import random
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from diffusion_planner.utils.data_augmentation import StatePerturbation

from training_data_visualizer.loader import discover_npz_files, load_npz
from training_data_visualizer.visualization import (
    get_traffic_light_summary,
    plot_tcos,
    plot_tdisplacement,
    plot_trajectory,
    plot_tsin,
    plot_tx,
    plot_ty,
)


def go_empty() -> go.Figure:
    """Return an empty Plotly figure (placeholder)."""
    return go.Figure()


class TrainingDataViewer:
    """Holds state for the Gradio interface: list of NPZ paths + current index."""

    def __init__(self, npz_paths: list[Path]) -> None:
        self.npz_paths = list(npz_paths)
        self.current_index: int = 0
        self.augmentor = StatePerturbation(augment_prob=1.0)

    def load_current(
        self,
        time_step: int = 0,
        show_augmented: bool = False,
        augmentation_seed: int = 0,
    ) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, str, str, str, int]:
        """Load data for current index and return UI outputs."""
        if not self.npz_paths:
            return (
                go_empty(),
                go_empty(),
                go_empty(),
                go_empty(),
                go_empty(),
                go_empty(),
                "No NPZ files found",
                "",
                "",
                0,
            )

        idx = max(0, min(self.current_index, len(self.npz_paths) - 1))
        self.current_index = idx

        data = load_npz(self.npz_paths[idx])
        original_ego_state = data["ego_current_state"].reshape(-1).copy()
        if show_augmented:
            data = self._augment_for_visualization(data, augmentation_seed)

        traj_fig = plot_trajectory(data, time_step=time_step if time_step > 0 else None)
        tx_fig = plot_tx(data)
        ty_fig = plot_ty(data)
        tcos_fig = plot_tcos(data)
        tsin_fig = plot_tsin(data)
        tdisplacement_fig = plot_tdisplacement(data)
        tl_summary = get_traffic_light_summary(data)

        info = f"Sample {idx + 1} / {len(self.npz_paths)} — {self.npz_paths[idx].name}"
        if show_augmented and "augmentation_perturbation" in data:
            dx, dy, dyaw = data["augmentation_perturbation"].reshape(-1)[:3]
            info += f" | augmented seed={augmentation_seed} dx={dx:.2f} dy={dy:.2f} dyaw={dyaw:.2f}"

        ego_state = data["ego_current_state"].reshape(-1)
        if show_augmented:
            ego_state_str = (
                "Before augmentation:\n"
                + self._format_ego_state(original_ego_state)
                + "\n\nAfter augmentation:\n"
                + self._format_ego_state(ego_state)
            )
        else:
            ego_state_str = self._format_ego_state(ego_state)

        return traj_fig, tx_fig, ty_fig, tcos_fig, tsin_fig, tdisplacement_fig, info, ego_state_str, tl_summary, idx

    @staticmethod
    def _format_ego_state(state: np.ndarray) -> str:
        labels = ["x", "y", "cos", "sin", "vx", "vy", "ax", "ay", "steering_angle", "unused_9"]
        return "\n".join(f"{labels[i]}: {state[i]:.4f}" for i in range(len(state)))

    def _augment_for_visualization(
        self, data: dict[str, np.ndarray], augmentation_seed: int
    ) -> dict[str, np.ndarray]:
        rng_state = np.random.get_state()
        np.random.seed(int(augmentation_seed) % (2**32 - 1))
        try:
            return self.augmentor.augment_with_aux(data)
        finally:
            np.random.set_state(rng_state)

    def navigate(self, delta: int, *args: Any) -> tuple:
        self.current_index = max(0, min(len(self.npz_paths) - 1, self.current_index + delta))
        return self.load_current(*args)

    def jump(self, idx: int, *args: Any) -> tuple:
        self.current_index = max(0, min(len(self.npz_paths) - 1, int(idx)))
        return self.load_current(*args)

    def shuffle(self, *args: Any) -> tuple:
        random.shuffle(self.npz_paths)
        self.current_index = 0
        return self.load_current(*args)

    def resample_augmentation(self, time_step: int = 0) -> tuple:
        augmentation_seed = random.randint(1, 2**31 - 1)
        return (True, augmentation_seed, *self.load_current(time_step, True, augmentation_seed))

    def download_npz(self) -> str:
        if not self.npz_paths:
            return ""
        idx = max(0, min(self.current_index, len(self.npz_paths) - 1))
        return str(self.npz_paths[idx])


def _build_navigation_column(n_total: int) -> tuple[gr.Slider, list[tuple[int, gr.Button]], gr.Button, gr.Button]:
    """Build the navigation column widgets."""
    sample_slider = gr.Slider(
        0,
        max(1, n_total - 1),
        value=0,
        step=1,
        label=f"Sample (0–{n_total - 1})" if n_total > 0 else "Sample (no data)",
    )

    with gr.Row():
        btn_m30 = gr.Button("<<<  30", size="sm")
        btn_m10 = gr.Button("<<  10", size="sm")
        btn_m1 = gr.Button("<  1", size="sm")
        btn_p1 = gr.Button("1  >", size="sm")
        btn_p10 = gr.Button("10  >>", size="sm")
        btn_p30 = gr.Button("30  >>>", size="sm")

    nav_buttons = [
        (-30, btn_m30),
        (-10, btn_m10),
        (-1, btn_m1),
        (1, btn_p1),
        (10, btn_p10),
        (30, btn_p30),
    ]

    with gr.Row():
        btn_shuffle = gr.Button("Shuffle", size="sm")
        btn_reload = gr.Button("Reload", size="sm")

    return sample_slider, nav_buttons, btn_shuffle, btn_reload


def _build_display_controls() -> tuple[gr.Slider, gr.Checkbox, gr.Button, gr.State]:
    """Build the display control widgets."""
    time_step_sl = gr.Slider(
        0,
        80,
        value=0,
        step=1,
        label="Time Step Marker (0=hidden, 1-80=future)",
    )
    show_augmented_cb = gr.Checkbox(value=False, label="Show Augmented")
    btn_resample_aug = gr.Button("Resample Augmentation", size="sm")
    augmentation_seed_state = gr.State(value=random.randint(1, 2**31 - 1))
    return time_step_sl, show_augmented_cb, btn_resample_aug, augmentation_seed_state


def _bind_events(
    viewer: TrainingDataViewer,
    inputs: list,
    outputs: list,
    sample_slider: gr.Slider,
    nav_buttons: list[tuple[int, gr.Button]],
    btn_shuffle: gr.Button,
    btn_reload: gr.Button,
    btn_resample_aug: gr.Button,
    btn_download: gr.DownloadButton,
    time_step_sl: gr.Slider,
    show_augmented_cb: gr.Checkbox,
    augmentation_seed_state: gr.State,
) -> None:
    """Wire up all Gradio event handlers."""
    for delta, btn in nav_buttons:
        btn.click(functools.partial(viewer.navigate, delta), inputs=inputs, outputs=outputs)

    btn_shuffle.click(viewer.shuffle, inputs=inputs, outputs=outputs)
    btn_reload.click(viewer.load_current, inputs=inputs, outputs=outputs)
    btn_resample_aug.click(
        viewer.resample_augmentation,
        inputs=[time_step_sl],
        outputs=[show_augmented_cb, augmentation_seed_state] + outputs,
    )
    btn_download.click(viewer.download_npz, inputs=[], outputs=btn_download)
    sample_slider.release(viewer.jump, inputs=[sample_slider] + inputs, outputs=outputs)
    time_step_sl.release(viewer.load_current, inputs=inputs, outputs=outputs)
    show_augmented_cb.change(viewer.load_current, inputs=inputs, outputs=outputs)


def build_interface(viewer: TrainingDataViewer) -> gr.Blocks:
    """Build and return the Gradio interface."""
    n_total = len(viewer.npz_paths)

    with gr.Blocks(title="Training Data Visualizer") as demo:
        gr.Markdown("# Training Data Visualizer")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Navigation")
                sample_slider, nav_buttons, btn_shuffle, btn_reload = _build_navigation_column(n_total)
                btn_download = gr.DownloadButton("Download this NPZ")

                gr.Markdown("### Display")
                time_step_sl, show_augmented_cb, btn_resample_aug, augmentation_seed_state = _build_display_controls()

                gr.Markdown("### Data Info")
                info_text = gr.Textbox(label="", interactive=False, lines=1)
                ego_state_text = gr.Textbox(label="ego_current_state", interactive=False, lines=10)
                tl_info_text = gr.Textbox(label="Traffic Light Info", interactive=False, lines=8)

            with gr.Column(scale=2):
                traj_plot = gr.Plot(label="Trajectory View")
                tx_plot = gr.Plot(label="t-x")
                ty_plot = gr.Plot(label="t-y")
                tcos_plot = gr.Plot(label="t-cos")
                tsin_plot = gr.Plot(label="t-sin")
                tdisplacement_plot = gr.Plot(label="t-displacement")

        inputs = [time_step_sl, show_augmented_cb, augmentation_seed_state]
        outputs = [traj_plot, tx_plot, ty_plot, tcos_plot, tsin_plot, tdisplacement_plot, info_text, ego_state_text, tl_info_text, sample_slider]

        _bind_events(
            viewer,
            inputs,
            outputs,
            sample_slider,
            nav_buttons,
            btn_shuffle,
            btn_reload,
            btn_resample_aug,
            btn_download,
            time_step_sl,
            show_augmented_cb,
            augmentation_seed_state,
        )

        demo.load(viewer.load_current, inputs=inputs, outputs=outputs)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize training data NPZ files")
    parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        default=".",
        help="Directory containing NPZ files (default: current directory)",
    )
    parser.add_argument("--port", type=int, default=7861, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    args = parser.parse_args()

    npz_paths = discover_npz_files(args.directory)
    if not npz_paths:
        print(f"No .npz files found in {args.directory}")
        return

    print(f"Found {len(npz_paths)} NPZ files in {args.directory}")
    viewer = TrainingDataViewer(npz_paths)
    demo = build_interface(viewer)
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
