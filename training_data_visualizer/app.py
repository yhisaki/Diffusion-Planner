"""Training Data Visualizer — interactive NPZ viewer with Plotly and Gradio.

Launch
------
    training_data_visualizer /path/to/npz/directory
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from training_data_visualizer.loader import discover_npz_files, load_npz
from training_data_visualizer.visualization import (
    plot_curvature,
    plot_trajectory,
    plot_velocity,
)


class TrainingDataViewer:
    """Holds state for the Gradio interface: list of NPZ paths + current index."""

    def __init__(self, npz_paths: list[Path]) -> None:
        self.npz_paths = list(npz_paths)
        self.current_index: int = 0

    def load_current(
        self,
        time_step: int = 0,
    ) -> tuple[object, object, object, str, int]:
        """Load data for current index and return (traj_fig, vel_fig, lat_fig, info, index)."""
        if not self.npz_paths:
            return go_empty(), go_empty(), go_empty(), "No NPZ files found", 0

        idx = max(0, min(self.current_index, len(self.npz_paths) - 1))
        self.current_index = idx

        data = load_npz(self.npz_paths[idx])
        traj_fig = plot_trajectory(data, time_step=time_step if time_step > 0 else None)
        vel_fig = plot_velocity(data)
        curv_fig = plot_curvature(data)
        info = f"Sample {idx + 1} / {len(self.npz_paths)} — {self.npz_paths[idx].name}"
        return traj_fig, vel_fig, curv_fig, info, idx

    def navigate(self, delta: int, *args) -> tuple:
        self.current_index = max(0, min(len(self.npz_paths) - 1, self.current_index + delta))
        return self.load_current(*args)

    def jump(self, idx: int, *args) -> tuple:
        self.current_index = max(0, min(len(self.npz_paths) - 1, int(idx)))
        return self.load_current(*args)

    def shuffle(self, *args) -> tuple:
        random.shuffle(self.npz_paths)
        self.current_index = 0
        return self.load_current(*args)

    def download_npz(self) -> str:
        if not self.npz_paths:
            return ""
        idx = max(0, min(self.current_index, len(self.npz_paths) - 1))
        return str(self.npz_paths[idx])


def go_empty() -> "go.Figure":
    """Return an empty Plotly figure (placeholder)."""
    import plotly.graph_objects as go
    return go.Figure()


def build_interface(viewer: TrainingDataViewer) -> gr.Blocks:
    n_total = len(viewer.npz_paths)

    with gr.Blocks(title="Training Data Visualizer") as demo:
        gr.Markdown("# Training Data Visualizer")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Navigation")
                sample_slider = gr.Slider(
                    0, max(0, n_total - 1), value=0, step=1,
                    label=f"Sample (0–{n_total - 1})" if n_total > 0 else "Sample (no data)",
                )
                with gr.Row():
                    btn_m30 = gr.Button("<<<  30", size="sm")
                    btn_m10 = gr.Button("<<  10", size="sm")
                    btn_m1 = gr.Button("<  1", size="sm")
                    btn_p1 = gr.Button("1  >", size="sm")
                    btn_p10 = gr.Button("10  >>", size="sm")
                    btn_p30 = gr.Button("30  >>>", size="sm")
                with gr.Row():
                    btn_shuffle = gr.Button("Shuffle", size="sm")
                    btn_reload = gr.Button("Reload", size="sm")
                btn_download = gr.DownloadButton("Download this NPZ")

                gr.Markdown("### Display")
                time_step_sl = gr.Slider(0, 79, value=0, step=1, label="Time Step Marker (0=hidden)")

                gr.Markdown("### Data Info")
                info_text = gr.Textbox(label="", interactive=False, lines=1)

            with gr.Column(scale=2):
                traj_plot = gr.Plot(label="Trajectory View")
                with gr.Accordion("Speed & Curvature Plots", open=False):
                    with gr.Row():
                        vel_plot = gr.Plot(label="Speed")
                        curv_plot = gr.Plot(label="Curvature")

        _gen_inputs = [time_step_sl]
        _outputs = [traj_plot, vel_plot, curv_plot, info_text, sample_slider]

        import functools

        for delta, btn in [(-30, btn_m30), (-10, btn_m10), (-1, btn_m1),
                           (1, btn_p1), (10, btn_p10), (30, btn_p30)]:
            btn.click(functools.partial(viewer.navigate, delta), inputs=_gen_inputs, outputs=_outputs)

        btn_shuffle.click(viewer.shuffle, inputs=_gen_inputs, outputs=_outputs)
        btn_reload.click(viewer.load_current, inputs=_gen_inputs, outputs=_outputs)
        btn_download.click(viewer.download_npz, inputs=[], outputs=btn_download)
        sample_slider.change(viewer.jump, inputs=[sample_slider] + _gen_inputs, outputs=_outputs)

        for slider in [time_step_sl]:
            slider.release(viewer.load_current, inputs=_gen_inputs, outputs=_outputs)

        demo.load(viewer.load_current, inputs=_gen_inputs, outputs=_outputs)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize training data NPZ files")
    parser.add_argument("directory", type=str, nargs="?", default=".",
                        help="Directory containing NPZ files (default: current directory)")
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