"""Per-coordinate time-series plots for ego vehicle state.

Each function takes an NPZ data dict and returns a Plotly Figure showing the
evolution of a single coordinate over time.

Past values are drawn in orange; future GT values in black.
When a *prediction* is provided, GT future is shown lighter and a blue dashed
prediction trace is overlaid.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ._timeseries import (
    future_heading_cos_sin,
    past_heading_cos_sin,
    past_positions,
)

_PRED_COLOR = "#1F77B4"


def _extract_ego_prediction(prediction: np.ndarray | None) -> np.ndarray | None:
    """Extract ego-only trajectory from a multi-agent prediction array.

    Args:
        prediction: Array of shape ``(T, D)`` or ``(P, T, D)``.

    Returns:
        ``(T, D)`` array for the ego agent, or ``None``.
    """
    if prediction is None:
        return None
    pred = np.asarray(prediction)
    if pred.ndim == 2:
        return pred
    if pred.ndim == 3:
        return pred[0]
    return None


def _add_pred_trace(fig: go.Figure, t: np.ndarray, y: np.ndarray, name: str) -> None:
    """Add a dashed prediction trace to *fig*."""
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            mode="lines+markers",
            name=name,
            line=dict(color=_PRED_COLOR, width=2, dash="dash"),
            marker=dict(size=3),
        )
    )


# ---------------------------------------------------------------------------
# t-x
# ---------------------------------------------------------------------------


def plot_tx(data: dict[str, np.ndarray], prediction: np.ndarray | None = None) -> go.Figure:
    """Plot ego x-coordinate over time (past and future).

    Args:
        data: NPZ data dict.
        prediction: Optional model output of shape ``(T, 4)`` or ``(P, T, 4)``.
            When given, GT future is shown lighter and prediction is overlaid.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    has_data = False

    pr = past_positions(data)
    if pr is not None:
        t, positions = pr
        fig.add_trace(
            go.Scatter(
                x=t,
                y=positions[:, 0],
                mode="lines+markers",
                name="Past X",
                line=dict(color="orange", width=2),
                marker=dict(size=3),
            )
        )
        has_data = True

    ego_pred = _extract_ego_prediction(prediction)

    if "ego_agent_future" in data:
        ego_state = data["ego_current_state"].reshape(-1)
        future = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
        positions = np.vstack([[ego_state[0], ego_state[1]], future[:, :2]])
        t = np.arange(len(positions))
        gt_opacity = 0.35 if ego_pred is not None else 1.0
        fig.add_trace(
            go.Scatter(
                x=t,
                y=positions[:, 0],
                mode="lines+markers",
                name="GT X",
                line=dict(color="black", width=2),
                marker=dict(size=3),
                opacity=gt_opacity,
            )
        )
        has_data = True

    if ego_pred is not None:
        _add_pred_trace(fig, np.arange(len(ego_pred)), ego_pred[:, 0], "Pred X")
        has_data = True

    title = "X"
    if not has_data:
        fig.update_layout(template="plotly_white", title="X", height=400)
        return fig

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Time step",
        yaxis_title="X [m]",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig


# ---------------------------------------------------------------------------
# t-y
# ---------------------------------------------------------------------------


def plot_ty(data: dict[str, np.ndarray], prediction: np.ndarray | None = None) -> go.Figure:
    """Plot ego y-coordinate over time (past and future).

    Args:
        data: NPZ data dict.
        prediction: Optional model output of shape ``(T, 4)`` or ``(P, T, 4)``.
            When given, GT future is shown lighter and prediction is overlaid.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    has_data = False

    pr = past_positions(data)
    if pr is not None:
        t, positions = pr
        fig.add_trace(
            go.Scatter(
                x=t,
                y=positions[:, 1],
                mode="lines+markers",
                name="Past Y",
                line=dict(color="orange", width=2),
                marker=dict(size=3),
            )
        )
        has_data = True

    ego_pred = _extract_ego_prediction(prediction)

    if "ego_agent_future" in data:
        ego_state = data["ego_current_state"].reshape(-1)
        future = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
        positions = np.vstack([[ego_state[0], ego_state[1]], future[:, :2]])
        t = np.arange(len(positions))
        gt_opacity = 0.35 if ego_pred is not None else 1.0
        fig.add_trace(
            go.Scatter(
                x=t,
                y=positions[:, 1],
                mode="lines+markers",
                name="GT Y",
                line=dict(color="black", width=2),
                marker=dict(size=3),
                opacity=gt_opacity,
            )
        )
        has_data = True

    if ego_pred is not None:
        _add_pred_trace(fig, np.arange(len(ego_pred)), ego_pred[:, 1], "Pred Y")
        has_data = True

    title = "Y"
    if not has_data:
        fig.update_layout(template="plotly_white", title="Y", height=400)
        return fig

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Time step",
        yaxis_title="Y [m]",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig


# ---------------------------------------------------------------------------
# t-cos
# ---------------------------------------------------------------------------


def plot_tcos(data: dict[str, np.ndarray], prediction: np.ndarray | None = None) -> go.Figure:
    """Plot ego ``cos(heading)`` over time (past and future).

    Args:
        data: NPZ data dict.
        prediction: Optional model output of shape ``(T, 4)`` or ``(P, T, 4)``.

    Returns:
        Plotly Figure with past cos (orange) and future cos (black) traces.
    """
    fig = go.Figure()
    has_data = False

    pr = past_heading_cos_sin(data)
    if pr is not None:
        t, cos_vals, _ = pr
        fig.add_trace(
            go.Scatter(
                x=t,
                y=cos_vals,
                mode="lines+markers",
                name="Past cos",
                line=dict(color="orange", width=2),
                marker=dict(size=3),
            )
        )
        has_data = True

    ego_pred = _extract_ego_prediction(prediction)

    fr = future_heading_cos_sin(data)
    if fr is not None:
        t, cos_vals, _ = fr
        gt_opacity = 0.35 if ego_pred is not None else 1.0
        fig.add_trace(
            go.Scatter(
                x=t,
                y=cos_vals,
                mode="lines+markers",
                name="GT cos",
                line=dict(color="black", width=2),
                marker=dict(size=3),
                opacity=gt_opacity,
            )
        )
        has_data = True

    if ego_pred is not None:
        _add_pred_trace(fig, np.arange(len(ego_pred)), ego_pred[:, 2], "Pred cos")
        has_data = True

    title = "cos"
    if not has_data:
        fig.update_layout(template="plotly_white", title="cos", height=400)
        return fig

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Time step",
        yaxis_title="cos(heading)",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig


# ---------------------------------------------------------------------------
# t-sin
# ---------------------------------------------------------------------------


def plot_tsin(data: dict[str, np.ndarray], prediction: np.ndarray | None = None) -> go.Figure:
    """Plot ego ``sin(heading)`` over time (past and future).

    Args:
        data: NPZ data dict.
        prediction: Optional model output of shape ``(T, 4)`` or ``(P, T, 4)``.

    Returns:
        Plotly Figure with past sin (orange) and future sin (black) traces.
    """
    fig = go.Figure()
    has_data = False

    pr = past_heading_cos_sin(data)
    if pr is not None:
        t, _, sin_vals = pr
        fig.add_trace(
            go.Scatter(
                x=t,
                y=sin_vals,
                mode="lines+markers",
                name="Past sin",
                line=dict(color="orange", width=2),
                marker=dict(size=3),
            )
        )
        has_data = True

    ego_pred = _extract_ego_prediction(prediction)

    fr = future_heading_cos_sin(data)
    if fr is not None:
        t, _, sin_vals = fr
        gt_opacity = 0.35 if ego_pred is not None else 1.0
        fig.add_trace(
            go.Scatter(
                x=t,
                y=sin_vals,
                mode="lines+markers",
                name="GT sin",
                line=dict(color="black", width=2),
                marker=dict(size=3),
                opacity=gt_opacity,
            )
        )
        has_data = True

    if ego_pred is not None:
        _add_pred_trace(fig, np.arange(len(ego_pred)), ego_pred[:, 3], "Pred sin")
        has_data = True

    title = "sin"
    if not has_data:
        fig.update_layout(template="plotly_white", title="sin", height=400)
        return fig

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Time step",
        yaxis_title="sin(heading)",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig


# ---------------------------------------------------------------------------
