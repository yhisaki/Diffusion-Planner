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
    future_positions,
    future_speeds,
    past_heading_cos_sin,
    past_positions,
    past_speeds,
)

_PRED_COLOR = "#1F77B4"

Series = tuple[np.ndarray, np.ndarray]


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


def _plot_timeseries(
    past: Series | None,
    future: Series | None,
    pred: Series | None,
    *,
    title: str,
    yaxis_title: str,
    past_name: str,
    future_name: str,
    pred_name: str,
    dim_future: bool | None = None,
) -> go.Figure:
    """Build a standard past/future/prediction time-series Plotly figure.

    Args:
        past: Optional ``(t, values)`` for the past trace (drawn orange).
        future: Optional ``(t, values)`` for the GT future trace (drawn black,
            dimmed to 0.35 opacity when *dim_future* — defaulting to whether
            *pred* is given — is true).
        pred: Optional ``(t, values)`` for the prediction trace (drawn as a
            blue dashed line).
        title: Figure title.
        yaxis_title: Y-axis label.
        past_name: Legend name for the past trace.
        future_name: Legend name for the GT future trace.
        pred_name: Legend name for the prediction trace.
        dim_future: Overrides whether the GT future trace is dimmed. Defaults
            to ``pred is not None``.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    has_data = False

    if past is not None:
        t, values = past
        fig.add_trace(
            go.Scatter(
                x=t,
                y=values,
                mode="lines+markers",
                name=past_name,
                line=dict(color="orange", width=2),
                marker=dict(size=3),
            )
        )
        has_data = True

    if future is not None:
        t, values = future
        if dim_future is None:
            dim_future = pred is not None
        gt_opacity = 0.35 if dim_future else 1.0
        fig.add_trace(
            go.Scatter(
                x=t,
                y=values,
                mode="lines+markers",
                name=future_name,
                line=dict(color="black", width=2),
                marker=dict(size=3),
                opacity=gt_opacity,
            )
        )
        has_data = True

    if pred is not None:
        t, values = pred
        fig.add_trace(
            go.Scatter(
                x=t,
                y=values,
                mode="lines+markers",
                name=pred_name,
                line=dict(color=_PRED_COLOR, width=2, dash="dash"),
                marker=dict(size=3),
            )
        )
        has_data = True

    if not has_data:
        fig.update_layout(template="plotly_white", title=title, height=400)
        return fig

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Time step",
        yaxis_title=yaxis_title,
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig


# ---------------------------------------------------------------------------
# t-x / t-y (position)
# ---------------------------------------------------------------------------


def _plot_position_axis(
    data: dict[str, np.ndarray],
    prediction: np.ndarray | None,
    *,
    axis_index: int,
    axis_label: str,
) -> go.Figure:
    past = past_positions(data)
    past_series = (past[0], past[1][:, axis_index]) if past is not None else None

    future = future_positions(data)
    future_series = (future[0], future[1][:, axis_index]) if future is not None else None

    ego_pred = _extract_ego_prediction(prediction)
    pred_series = (
        (np.arange(len(ego_pred)), ego_pred[:, axis_index]) if ego_pred is not None else None
    )

    return _plot_timeseries(
        past_series,
        future_series,
        pred_series,
        title=axis_label,
        yaxis_title=f"{axis_label} [m]",
        past_name=f"Past {axis_label}",
        future_name=f"GT {axis_label}",
        pred_name=f"Pred {axis_label}",
    )


def plot_tx(data: dict[str, np.ndarray], prediction: np.ndarray | None = None) -> go.Figure:
    """Plot ego x-coordinate over time (past and future).

    Args:
        data: NPZ data dict.
        prediction: Optional model output of shape ``(T, 4)`` or ``(P, T, 4)``.
            When given, GT future is shown lighter and prediction is overlaid.

    Returns:
        Plotly Figure.
    """
    return _plot_position_axis(data, prediction, axis_index=0, axis_label="X")


def plot_ty(data: dict[str, np.ndarray], prediction: np.ndarray | None = None) -> go.Figure:
    """Plot ego y-coordinate over time (past and future).

    Args:
        data: NPZ data dict.
        prediction: Optional model output of shape ``(T, 4)`` or ``(P, T, 4)``.
            When given, GT future is shown lighter and prediction is overlaid.

    Returns:
        Plotly Figure.
    """
    return _plot_position_axis(data, prediction, axis_index=1, axis_label="Y")


# ---------------------------------------------------------------------------
# t-cos / t-sin (heading)
# ---------------------------------------------------------------------------


def _plot_heading_component(
    data: dict[str, np.ndarray],
    prediction: np.ndarray | None,
    *,
    component_index: int,
    component_label: str,
) -> go.Figure:
    past = past_heading_cos_sin(data)
    past_series = (past[0], past[component_index]) if past is not None else None

    future = future_heading_cos_sin(data)
    future_series = (future[0], future[component_index]) if future is not None else None

    ego_pred = _extract_ego_prediction(prediction)
    pred_series = (
        (np.arange(len(ego_pred)), ego_pred[:, component_index + 1])
        if ego_pred is not None
        else None
    )

    return _plot_timeseries(
        past_series,
        future_series,
        pred_series,
        title=component_label,
        yaxis_title=f"{component_label}(heading)",
        past_name=f"Past {component_label}",
        future_name=f"GT {component_label}",
        pred_name=f"Pred {component_label}",
    )


def plot_tcos(data: dict[str, np.ndarray], prediction: np.ndarray | None = None) -> go.Figure:
    """Plot ego ``cos(heading)`` over time (past and future).

    Args:
        data: NPZ data dict.
        prediction: Optional model output of shape ``(T, 4)`` or ``(P, T, 4)``.

    Returns:
        Plotly Figure with past cos (orange) and future cos (black) traces.
    """
    return _plot_heading_component(data, prediction, component_index=1, component_label="cos")


def plot_tsin(data: dict[str, np.ndarray], prediction: np.ndarray | None = None) -> go.Figure:
    """Plot ego ``sin(heading)`` over time (past and future).

    Args:
        data: NPZ data dict.
        prediction: Optional model output of shape ``(T, 4)`` or ``(P, T, 4)``.

    Returns:
        Plotly Figure with past sin (orange) and future sin (black) traces.
    """
    return _plot_heading_component(data, prediction, component_index=2, component_label="sin")


# ---------------------------------------------------------------------------
# t-v (velocity)
# ---------------------------------------------------------------------------


def plot_tv(
    data: dict[str, np.ndarray],
    prediction: np.ndarray | None = None,
    ego_velocity_prediction: np.ndarray | None = None,
) -> go.Figure:
    """Plot ego speed over time (past and future).

    Uses ``ego_velocity_past`` / ``ego_velocity_future`` if available;
    falls back to displacement-based speed otherwise.

    Args:
        data: NPZ data dict.
        prediction: Optional model output.
        ego_velocity_prediction: Optional ego future velocity prediction, shape ``(T,)``
            or ``(T, 1)``.

    Returns:
        Plotly Figure with past speed (orange) and future speed (black) traces.
    """
    past = past_speeds(data)

    future = None
    if "ego_velocity_future" in data or "ego_agent_future" in data:
        future = future_speeds(data)

    pred_speed = None
    if ego_velocity_prediction is not None:
        pred_speed = np.asarray(ego_velocity_prediction).reshape(-1)
    has_pred_speed = pred_speed is not None and pred_speed.size > 0
    pred = (np.arange(1, pred_speed.size + 1), pred_speed) if has_pred_speed else None

    ego_pred = _extract_ego_prediction(prediction)
    dim_future = ego_pred is not None or has_pred_speed

    return _plot_timeseries(
        past,
        future,
        pred,
        title="V",
        yaxis_title="speed [m/s]",
        past_name="Past speed",
        future_name="GT speed",
        pred_name="Pred speed",
        dim_future=dim_future,
    )
