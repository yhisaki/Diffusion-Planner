"""Cached predictor construction shared across views."""

from __future__ import annotations

import streamlit as st

from diffusion_planner_gui.inference import Predictor


@st.cache_resource
def get_predictor(model_path: str, device: str = "auto") -> Predictor | None:
    if not model_path:
        return None
    try:
        return Predictor(model_path, device)
    except Exception as e:
        st.session_state.last_model_error = str(e)
        return None
