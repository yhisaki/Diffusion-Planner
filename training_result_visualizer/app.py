"""Training Result Visualizer.

Launch:
    training_result_visualizer
"""

from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go

from training_data_visualizer.loader import load_npz
from training_result_visualizer.inference import Predictor
from training_result_visualizer.loader import load_path_list
from training_result_visualizer.visualization import (
    plot_prediction_components,
    plot_prediction_vs_gt,
)


class TrainingResultViewer:
    """Holds model, path list, and navigation state for the Gradio UI."""

    def __init__(
        self, model_path: str = "", path_list_path: str = "", device: str = "auto"
    ) -> None:
        self.model_path = model_path
        self.path_list_path = path_list_path
        self.device_name = device
        self.predictor: Predictor | None = None
        self.npz_paths: list[Path] = []
        self.current_index = 0
        if model_path and path_list_path:
            self.configure(model_path, path_list_path, device=device)

    def configure(
        self,
        model_path_text: str,
        path_list_text: str,
        model_file: tempfile._TemporaryFileWrapper | None = None,
        path_list_file: tempfile._TemporaryFileWrapper | None = None,
        device: str = "auto",
    ) -> tuple[object, object, object, str, int, int]:
        model_path = _component_path(model_file) or model_path_text
        path_list_path = _component_path(path_list_file) or path_list_text
        if not model_path:
            raise gr.Error("モデル path を指定してください。")
        if not path_list_path:
            raise gr.Error("path_list.json の path を指定してください。")

        self.model_path = str(Path(model_path).expanduser().resolve())
        self.path_list_path = str(Path(path_list_path).expanduser().resolve())
        self.device_name = device
        self.predictor = Predictor(self.model_path, device)
        self.npz_paths = load_path_list(self.path_list_path)
        if not self.npz_paths:
            raise gr.Error("path_list.json に有効な NPZ path がありません。")
        self.current_index = 0
        traj_fig, fig_x, fig_y, info, idx = self.load_current()
        return traj_fig, fig_x, fig_y, info, idx, max(0, len(self.npz_paths) - 1)

    def load_current(
        self, time_step: int = 0, view_range: int = 60
    ) -> tuple[object, object, object, str, int]:
        empty = go.Figure()
        if not self.npz_paths:
            return empty, empty, empty, "No path list loaded", 0
        if self.predictor is None:
            return empty, empty, empty, "No model loaded", 0

        idx = max(0, min(self.current_index, len(self.npz_paths) - 1))
        self.current_index = idx
        npz_path = self.npz_paths[idx]
        data = load_npz(npz_path)
        prediction = self.predictor.predict(npz_path)
        marker_step = time_step if time_step > 0 else None
        traj_fig = plot_prediction_vs_gt(
            data, prediction, view_range=view_range, time_step=marker_step
        )
        fig_x, fig_y = plot_prediction_components(data, prediction)
        info = (
            f"Sample {idx + 1} / {len(self.npz_paths)}\n"
            f"NPZ: {npz_path}\n"
            f"Model: {self.model_path}\n"
            f"Device: {self.device_name}"
        )
        return traj_fig, fig_x, fig_y, info, idx

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


def _component_path(file_obj: object | None) -> str:
    if file_obj is None:
        return ""
    if isinstance(file_obj, str):
        return file_obj
    return str(getattr(file_obj, "name", "") or "")


def build_interface(viewer: TrainingResultViewer) -> gr.Blocks:
    with gr.Blocks(title="Training Result Visualizer") as demo:
        gr.Markdown("# Training Result Visualizer")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Inputs")
                model_file = gr.File(label="Model checkpoint", file_count="single")
                model_path = gr.Textbox(label="Model path", value=viewer.model_path)
                path_list_file = gr.File(label="path_list.json", file_count="single")
                path_list_path = gr.Textbox(
                    label="path_list.json path", value=viewer.path_list_path
                )
                device = gr.Dropdown(
                    ["auto", "cuda", "cpu"], value=viewer.device_name, label="Device"
                )
                btn_load = gr.Button("Load Model and Path List", variant="primary")

                gr.Markdown("### Navigation")
                sample_slider = gr.Slider(0, 1, value=0, step=1, label="Sample")
                with gr.Row():
                    btn_m10 = gr.Button("<< 10", size="sm")
                    btn_m1 = gr.Button("< 1", size="sm")
                    btn_p1 = gr.Button("1 >", size="sm")
                    btn_p10 = gr.Button("10 >>", size="sm")
                with gr.Row():
                    btn_shuffle = gr.Button("Shuffle", size="sm")
                    btn_reload = gr.Button("Reload", size="sm")

                gr.Markdown("### Display")
                time_step = gr.Slider(
                    0, 79, value=0, step=1, label="GT Time Step Marker (0=hidden)"
                )
                view_range = gr.Slider(20, 200, value=60, step=5, label="View Range [m]")
                info_text = gr.Textbox(label="Info", interactive=False, lines=5)

            with gr.Column(scale=2):
                traj_plot = gr.Plot(label="Prediction vs GT")
                with gr.Row():
                    plot_x = gr.Plot(label="Prediction x")
                    plot_y = gr.Plot(label="Prediction y")

        reload_inputs = [time_step, view_range]
        outputs = [traj_plot, plot_x, plot_y, info_text, sample_slider]

        def _configure(*args):
            traj_fig, fig_x, fig_y, info, idx, max_idx = viewer.configure(*args)
            return traj_fig, fig_x, fig_y, info, gr.update(value=idx, maximum=max(1, max_idx))

        btn_load.click(
            _configure,
            inputs=[model_path, path_list_path, model_file, path_list_file, device],
            outputs=outputs,
        )

        import functools

        for delta, btn in [(-10, btn_m10), (-1, btn_m1), (1, btn_p1), (10, btn_p10)]:
            btn.click(
                functools.partial(viewer.navigate, delta), inputs=reload_inputs, outputs=outputs
            )

        btn_shuffle.click(viewer.shuffle, inputs=reload_inputs, outputs=outputs)
        btn_reload.click(viewer.load_current, inputs=reload_inputs, outputs=outputs)
        sample_slider.change(viewer.jump, inputs=[sample_slider] + reload_inputs, outputs=outputs)
        time_step.release(viewer.load_current, inputs=reload_inputs, outputs=outputs)
        view_range.release(viewer.load_current, inputs=reload_inputs, outputs=outputs)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize model prediction vs GT from a path_list.json"
    )
    parser.add_argument("--model-path", default="", help="Initial model checkpoint path")
    parser.add_argument("--path-list", default="", help="Initial path_list.json path")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device"
    )
    parser.add_argument("--port", type=int, default=7862, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    args = parser.parse_args()

    viewer = TrainingResultViewer(args.model_path, args.path_list, args.device)
    demo = build_interface(viewer)
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
