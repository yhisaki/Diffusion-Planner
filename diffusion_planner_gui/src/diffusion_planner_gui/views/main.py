"""Main area rendering for diffusion_planner_gui."""

from __future__ import annotations

import numpy as np
import streamlit as st

from diffusion_planner_gui.loader import load_npz
from diffusion_planner_gui.views.sidebar import _get_predictor
from diffusion_planner_gui.visualization import (
    plot_prediction_vs_gt,
    plot_tcos,
    plot_trajectory,
    plot_tsin,
    plot_tv,
    plot_tx,
    plot_ty,
)


def _format_ego_state(state: np.ndarray) -> str:
    labels = ["x", "y", "cos", "sin", "vx", "vy", "ax", "ay", "steering_angle", "yaw_rate"]
    return "\n".join(f"{labels[i]}: {state[i]:.4f}" for i in range(len(state)))


def render_main() -> None:
    npz_paths = st.session_state.npz_paths
    idx = st.session_state.current_index
    n_total = len(npz_paths)
    npz_path = npz_paths[idx]

    data = load_npz(npz_path)

    model_loaded = st.session_state.model_loaded
    view_range = 40
    gt_interval = st.session_state.footprint_interval if st.session_state.show_gt_footprint else 0
    pred_interval = (
        st.session_state.footprint_interval if st.session_state.show_pred_footprint else 0
    )

    prediction = None
    if model_loaded:
        predictor = _get_predictor(st.session_state.model_path, "auto")
        if predictor is not None:
            prediction = predictor.predict(
                npz_path=npz_path,
                noise_scale=st.session_state.noise_scale,
                noise_seed=st.session_state.noise_seed,
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

    st.plotly_chart(plot_tv(data, prediction), width="stretch")

    with st.sidebar:
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
            st.download_button(
                "Download this NPZ",
                f.read(),
                file_name=npz_path.name,
                mime="application/octet-stream",
            )
