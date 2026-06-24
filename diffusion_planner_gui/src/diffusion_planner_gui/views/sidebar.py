"""Sidebar widgets for diffusion_planner_gui."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from diffusion_planner.utils.data_augmentation import StatePerturbation

from diffusion_planner_gui.inference import Predictor
from diffusion_planner_gui.loader import load_npz, resolve_data_path


@st.cache_resource
def _get_predictor(model_path: str, device: str) -> Predictor | None:
    if not model_path:
        return None
    try:
        return Predictor(model_path, device)
    except Exception as e:
        st.session_state.last_model_error = str(e)
        return None


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Inputs")

        st.text_input(
            "Data path", placeholder="directory / path_list.json / single.npz", key="data_path"
        )
        st.text_input("Model path (optional)", placeholder="model checkpoint", key="model_path")

        if st.session_state.get("load_error"):
            st.error(st.session_state.load_error)
            st.session_state.load_error = None

        st.button("Load", on_click=_on_load, type="primary")

        if not st.session_state.data_loaded:
            return

        _render_navigation()
        _render_augmentation()
        _render_display()
        if st.session_state.model_loaded:
            _render_noise()
        st.divider()
        st.header("Info")


def _on_load() -> None:
    data_path = st.session_state.data_path
    model_path = st.session_state.model_path

    if not data_path:
        st.session_state.load_error = "Please specify a data path."
        return

    try:
        npz_paths = resolve_data_path(data_path)
    except Exception as e:
        st.session_state.load_error = f"Failed to resolve data path: {e}"
        return

    if not npz_paths:
        st.session_state.load_error = "No NPZ files found."
        return

    model_loaded = False
    if model_path:
        model_path_resolved = str(Path(model_path).expanduser().resolve())
        predictor = _get_predictor(model_path_resolved, "auto")
        if predictor is not None:
            st.session_state.model_path = model_path_resolved
            model_loaded = True
        else:
            err = st.session_state.get("last_model_error", "unknown error")
            st.session_state.load_error = f"Failed to load model: {err}"

    st.session_state.npz_paths = npz_paths
    st.session_state.current_index = 0
    st.session_state.data_loaded = True
    st.session_state.model_loaded = model_loaded
    st.session_state.noise_seed = 0
    _clear_augmentation()


def _render_navigation() -> None:
    st.divider()
    st.header("Navigation")

    st.slider(
        "Sample",
        0,
        max(1, len(st.session_state.npz_paths) - 1),
        key="current_index",
        on_change=_clear_augmentation,
    )

    def _nav_callback(delta: int) -> None:
        n = len(st.session_state.npz_paths)
        st.session_state.current_index = max(0, min(n - 1, st.session_state.current_index + delta))
        _clear_augmentation()

    cols = st.columns(10)
    labels_deltas = [
        ("< 100", -100),
        ("< 50", -50),
        ("< 30", -30),
        ("< 10", -10),
        ("< 1", -1),
        ("1 >", 1),
        ("10 >", 10),
        ("30 >", 30),
        ("50 >", 50),
        ("100 >", 100),
    ]
    for col, (label, delta) in zip(cols, labels_deltas):
        with col:
            st.button(label, on_click=_nav_callback, args=(delta,), key=f"nav_{delta}")


def _render_augmentation() -> None:
    st.divider()
    st.header("Data Augmentation")

    if st.session_state.get("augmentation_error"):
        st.error(st.session_state.augmentation_error)
        st.session_state.augmentation_error = None

    st.checkbox(
        "State Perturbation",
        key="state_perturbation_enabled",
        on_change=_on_state_perturbation_toggle,
    )
    if st.session_state.get("augmented_data") is not None:
        st.caption("Showing augmented data for the current sample.")


def _on_state_perturbation_toggle() -> None:
    if st.session_state.get("state_perturbation_enabled"):
        _on_augment_current_sample()
    else:
        _clear_augmentation()


def _on_augment_current_sample() -> None:
    try:
        npz_path = st.session_state.npz_paths[st.session_state.current_index]
        data = load_npz(npz_path)
        augmented = StatePerturbation(augment_prob=1.0).augment_with_aux(data)
    except Exception as e:
        st.session_state.augmentation_error = f"Failed to augment current sample: {e}"
        _clear_augmentation()
        return

    if augmented is data:
        st.session_state.augmentation_error = (
            "Data augmentation was skipped by the augmenter conditions."
        )
        _clear_augmentation()
        return

    st.session_state.augmented_data = augmented
    st.session_state.augmented_path = str(npz_path)


def _clear_augmentation() -> None:
    st.session_state.augmented_data = None
    st.session_state.augmented_path = ""
    st.session_state.state_perturbation_enabled = False


def _render_display() -> None:
    st.divider()
    st.header("Display")

    st.checkbox("Show GT Footprint", key="show_gt_footprint")
    if st.session_state.get("model_loaded"):
        st.checkbox("Show Prediction Footprint", key="show_pred_footprint")
    st.selectbox("Footprint Step Interval", [1, 2, 5, 10], key="footprint_interval")


def _render_noise() -> None:
    st.divider()
    st.header("Diffusion Input Noise")

    st.slider("Noise Scale", 0.0, 1.0, step=0.01, key="noise_scale")

    def _resample_noise() -> None:
        st.session_state.noise_seed += 1

    st.button("Resample Noise", on_click=_resample_noise)
