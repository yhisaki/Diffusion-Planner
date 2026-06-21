"""CLI entry point for diffusion_planner_gui."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diffusion Planner GUI — visualize training data and model predictions"
    )
    parser.add_argument(
        "--data", default="", help="Data path (directory, path_list.json, or single .npz)"
    )
    parser.add_argument("--model", default="", help="Model checkpoint path (optional)")
    parser.add_argument("--port", type=int, default=8503, help="Streamlit server port")
    args, _ = parser.parse_known_args()

    os.environ["DP_GUI_DATA"] = args.data
    os.environ["DP_GUI_MODEL"] = args.model

    pkg_dir = Path(__file__).resolve().parent
    sys.argv = ["streamlit", "run", str(pkg_dir / "app.py"), "--server.port", str(args.port)]

    from streamlit.web import cli as stcli

    stcli.main()


if __name__ == "__main__":
    main()
