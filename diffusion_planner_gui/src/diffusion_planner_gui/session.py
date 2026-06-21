"""Session state initialization for diffusion_planner_gui."""

from __future__ import annotations

import os

import streamlit as st


def init_session() -> None:
    defaults: dict[str, object] = {
        "data_path": os.environ.get("DP_GUI_DATA", ""),
        "model_path": os.environ.get("DP_GUI_MODEL", ""),
        "data_loaded": False,
        "model_loaded": False,
        "npz_paths": [],
        "current_index": 0,
        "noise_seed": 0,
        "noise_scale": 0.0,
        "show_gt_footprint": False,
        "show_pred_footprint": False,
        "footprint_interval": 1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
