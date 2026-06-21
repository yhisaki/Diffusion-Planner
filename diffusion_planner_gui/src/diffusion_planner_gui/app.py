"""Diffusion Planner GUI — unified training data and prediction visualization.

Launch
------
    diffusion_planner_gui
"""

from __future__ import annotations

import streamlit as st

from diffusion_planner_gui.session import init_session
from diffusion_planner_gui.views.main import render_main
from diffusion_planner_gui.views.sidebar import render_sidebar

st.set_page_config(page_title="Diffusion Planner GUI", layout="wide")
st.title("Diffusion Planner GUI")

init_session()

render_sidebar()
if st.session_state.data_loaded:
    render_main()
