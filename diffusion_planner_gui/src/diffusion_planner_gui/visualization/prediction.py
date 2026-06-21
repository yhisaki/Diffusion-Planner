"""Prediction overlay plots."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ._scene import draw_prediction_footprints, valid_xy_mask
from .trajectory import plot_trajectory


def _as_agent_prediction(prediction: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction)
    if pred.ndim == 2:
        pred = pred[None, ...]
    if pred.ndim != 3:
        raise ValueError(f"prediction must have shape [T, D] or [P, T, D], got {pred.shape}")
    return pred


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


def plot_prediction_vs_gt(
    data: dict[str, np.ndarray],
    prediction: np.ndarray | None,
    view_range: int = 60,
    footprint_interval: int = 0,
    pred_footprint_interval: int = 0,
) -> go.Figure:
    fig = plot_trajectory(data, view_range=view_range, footprint_interval=footprint_interval)
    has_pred = prediction is not None and len(prediction) > 0

    if has_pred:
        pred = _as_agent_prediction(prediction)
        ego_pred = pred[0]
        ego_valid = valid_xy_mask(ego_pred)
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
            neighbor_valid = valid_xy_mask(neighbor_pred)
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
                    legendgroup="Neighbor Prediction",
                    showlegend=neighbor_index == 1,
                )
            )

        for trace in draw_prediction_footprints(
            prediction,
            pred_footprint_interval,
            shape=data.get("ego_shape"),
        ):
            fig.add_trace(trace)

    title = "Prediction vs GT" if has_pred else "Trajectory View"
    fig.update_layout(template="plotly_white", uirevision="1", title=title)
    return fig
