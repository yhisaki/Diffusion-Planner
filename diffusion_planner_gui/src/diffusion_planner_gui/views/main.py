"""Main area rendering for diffusion_planner_gui."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import streamlit as st

from diffusion_planner_gui.loader import load_npz
from diffusion_planner_gui.predictor_cache import get_predictor
from diffusion_planner_gui.visualization import (
    plot_prediction_vs_gt,
    plot_tcos,
    plot_trajectory,
    plot_tsin,
    plot_turn_indicator,
    plot_tv,
    plot_tx,
    plot_ty,
)


def _format_ego_state(state: np.ndarray) -> str:
    labels = ["x", "y", "cos", "sin", "vx", "vy", "ax", "ay", "steering_angle", "yaw_rate"]
    return "\n".join(f"{labels[i]}: {state[i]:.4f}" for i in range(len(state)))


def _current_data(npz_path) -> tuple[dict[str, np.ndarray], bool]:
    augmented = st.session_state.get("augmented_data")
    if augmented is not None and st.session_state.get("augmented_path") == str(npz_path):
        return augmented, True
    return load_npz(npz_path), False


def _npz_bytes(data: dict[str, np.ndarray]) -> bytes:
    buffer = BytesIO()
    np.savez(buffer, **data)
    buffer.seek(0)
    return buffer.getvalue()


def _predict_current_sample(
    npz_path, data: dict[str, np.ndarray], is_augmented: bool, model_loaded: bool
) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
    """Run the loaded model on the current sample, if any model is loaded."""
    if not model_loaded:
        return None, None, None
    predictor = get_predictor(st.session_state.model_path, "auto")
    if predictor is None:
        return None, None, None
    return predictor.predict_with_velocity(
        npz_path=None if is_augmented else npz_path,
        data=data if is_augmented else None,
        noise_scale=st.session_state.noise_scale,
        noise_seed=st.session_state.noise_seed,
    )


def _render_trajectory_and_timeseries(
    data: dict[str, np.ndarray],
    prediction: np.ndarray | None,
    ego_velocity_prediction: np.ndarray | None,
    turn_indicator_prediction: int | None = None,
) -> None:
    view_range = 40
    gt_interval = st.session_state.footprint_interval if st.session_state.show_gt_footprint else 0
    pred_interval = (
        st.session_state.footprint_interval if st.session_state.show_pred_footprint else 0
    )

    if prediction is not None:
        traj_fig = plot_prediction_vs_gt(
            data,
            prediction,
            view_range=view_range,
            footprint_interval=gt_interval,
            pred_footprint_interval=pred_interval,
        )
    else:
        traj_fig = plot_trajectory(data, view_range=view_range, footprint_interval=gt_interval)
    st.plotly_chart(traj_fig, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_tx(data, prediction), width="stretch")
    with col2:
        st.plotly_chart(plot_ty(data, prediction), width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_tcos(data, prediction), width="stretch")
    with col4:
        st.plotly_chart(plot_tsin(data, prediction), width="stretch")

    st.plotly_chart(plot_tv(data, prediction, ego_velocity_prediction), width="stretch")

    st.plotly_chart(
        plot_turn_indicator(data, turn_indicator_prediction), width="stretch"
    )


def _render_sample_info(
    npz_path,
    data: dict[str, np.ndarray],
    is_augmented: bool,
    idx: int,
    n_total: int,
    model_loaded: bool,
) -> None:
    info = f"Sample {idx + 1} / {n_total} — {npz_path.name}"
    if model_loaded:
        info += (
            f"\nModel: {st.session_state.model_path}\n"
            f"Noise scale: {st.session_state.noise_scale:.2f}\n"
            f"Noise seed: {st.session_state.noise_seed}"
        )
    st.info(info)

    ego_state = data["ego_current_state"].reshape(-1)
    st.text_area(
        "ego_current_state",
        _format_ego_state(ego_state),
        height=200,
        disabled=True,
        label_visibility="collapsed",
    )

    with open(str(npz_path), "rb") as f:
        download_data = _npz_bytes(data) if is_augmented else f.read()
        download_name = f"{npz_path.stem}_augmented.npz" if is_augmented else npz_path.name
        st.download_button(
            "Download this NPZ",
            download_data,
            file_name=download_name,
            mime="application/octet-stream",
        )


def render_main() -> None:
    npz_paths = st.session_state.npz_paths
    idx = st.session_state.current_index
    n_total = len(npz_paths)
    npz_path = npz_paths[idx]
    model_loaded = st.session_state.model_loaded

    data, is_augmented = _current_data(npz_path)
    prediction, ego_velocity_prediction, turn_indicator_prediction = _predict_current_sample(
        npz_path, data, is_augmented, model_loaded
    )

    _render_trajectory_and_timeseries(
        data, prediction, ego_velocity_prediction, turn_indicator_prediction
    )

    with st.sidebar:
        _render_sample_info(npz_path, data, is_augmented, idx, n_total, model_loaded)
