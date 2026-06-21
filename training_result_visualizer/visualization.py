"""Plotly overlays for model prediction vs GT."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from training_data_visualizer.visualization import plot_trajectory


def plot_prediction_vs_gt(
    data: dict[str, np.ndarray],
    prediction: np.ndarray | None,
    view_range: int = 60,
    time_step: int | None = None,
) -> go.Figure:
    """Create a scene plot with GT future and model prediction overlaid."""
    fig = plot_trajectory(data, view_range=view_range, time_step=time_step)

    if prediction is not None and len(prediction) > 0:
        pred = _as_agent_prediction(prediction)
        ego_pred = pred[0]
        ego_valid = _valid_xy_mask(ego_pred)
        if np.any(ego_valid):
            fig.add_trace(
                go.Scatter(
                    x=ego_pred[ego_valid, 0],
                    y=ego_pred[ego_valid, 1],
                    mode="lines+markers",
                    line=dict(color="#00A6D6", width=3),
                    marker=dict(size=4, color="#00A6D6"),
                    name="Ego Prediction",
                    showlegend=True,
                )
            )

        neighbor_valid_mask = _neighbor_validity_mask(data, len(pred) - 1)

        for neighbor_index, neighbor_pred in enumerate(pred[1:], start=1):
            if (
                neighbor_index - 1 < len(neighbor_valid_mask)
                and not neighbor_valid_mask[neighbor_index - 1]
            ):
                continue
            neighbor_valid = _valid_xy_mask(neighbor_pred)
            if not np.any(neighbor_valid):
                continue
            fig.add_trace(
                go.Scatter(
                    x=neighbor_pred[neighbor_valid, 0],
                    y=neighbor_pred[neighbor_valid, 1],
                    mode="lines",
                    line=dict(color="#00897B", width=2),
                    opacity=0.45,
                    name="Neighbor Prediction"
                    if neighbor_index == 1
                    else f"Neighbor Prediction {neighbor_index}",
                    showlegend=neighbor_index == 1,
                )
            )

    fig.update_layout(title="Prediction vs GT")
    return fig


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, height=280)
    return fig


def plot_prediction_components(
    data: dict[str, np.ndarray],
    prediction: np.ndarray | None,
) -> tuple[go.Figure, go.Figure]:
    """Plot per-step x and y of GT and prediction over time as two separate figures."""
    if prediction is None or "ego_agent_future" not in data:
        return _empty_figure("Prediction x"), _empty_figure("Prediction y")

    gt = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
    pred = _as_agent_prediction(prediction)[0]
    n = min(len(gt), len(pred))
    if n == 0:
        return _empty_figure("Prediction x"), _empty_figure("Prediction y")

    t = np.arange(n)

    fig_x = go.Figure()
    fig_x.add_trace(
        go.Scatter(x=t, y=gt[:n, 0], mode="lines", line=dict(color="#1F77B4", width=2), name="GT")
    )
    fig_x.add_trace(
        go.Scatter(
            x=t,
            y=pred[:n, 0],
            mode="lines",
            line=dict(color="#1F77B4", width=2, dash="dash"),
            name="Pred",
        )
    )
    fig_x.update_layout(
        title="Prediction x",
        xaxis_title="Time step",
        yaxis_title="x [m]",
        margin=dict(l=40, r=20, t=40, b=40),
        height=280,
    )

    fig_y = go.Figure()
    fig_y.add_trace(
        go.Scatter(x=t, y=gt[:n, 1], mode="lines", line=dict(color="#FF7F0E", width=2), name="GT")
    )
    fig_y.add_trace(
        go.Scatter(
            x=t,
            y=pred[:n, 1],
            mode="lines",
            line=dict(color="#FF7F0E", width=2, dash="dash"),
            name="Pred",
        )
    )
    fig_y.update_layout(
        title="Prediction y",
        xaxis_title="Time step",
        yaxis_title="y [m]",
        margin=dict(l=40, r=20, t=40, b=40),
        height=280,
    )

    return fig_x, fig_y


def _as_agent_prediction(prediction: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction)
    if pred.ndim == 2:
        pred = pred[None, ...]
    if pred.ndim != 3:
        raise ValueError(f"prediction must have shape [T, D] or [P, T, D], got {pred.shape}")
    return pred


def _valid_xy_mask(trajectory: np.ndarray) -> np.ndarray:
    return ~((trajectory[:, 0] == 0) & (trajectory[:, 1] == 0))


def _neighbor_validity_mask(
    data: dict[str, np.ndarray], num_predicted_neighbors: int
) -> np.ndarray:
    if "neighbor_agents_past" not in data or num_predicted_neighbors <= 0:
        return np.ones(num_predicted_neighbors, dtype=bool)

    neighbors = data["neighbor_agents_past"]
    neighbors = neighbors.reshape(neighbors.shape[0], neighbors.shape[1], -1)
    last_t = neighbors.shape[1] - 1

    mask = np.zeros(num_predicted_neighbors, dtype=bool)
    for i in range(min(num_predicted_neighbors, neighbors.shape[0])):
        mask[i] = np.sum(np.abs(neighbors[i, last_t, :4])) >= 1e-6
    return mask
