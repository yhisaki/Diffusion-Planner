"""Scene trajectory plot — bird's-eye view of lanes, agents, and ego.

The main entry point is :func:`plot_trajectory`, which builds a square-ratio
Plotly Figure centred on the ego vehicle with all scene elements drawn.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ._scene import (
    draw_ego_footprints,
    draw_ego_vehicle,
    draw_goal_pose,
    draw_lanes,
    draw_neighbor_agents,
    draw_polygons_and_lines,
    draw_route,
    draw_static_objects,
)


def plot_trajectory(
    data: dict[str, np.ndarray],
    view_range: int = 60,
    footprint_interval: int = 0,
) -> go.Figure:
    """Build the primary scene plot (bird's-eye view of lanes, agents, and ego).

    The figure is centred on the ego vehicle's current position and shows:

    * Lane boundaries (coloured by traffic-light state)
    * Route lanes (dashed olive)
    * Static objects (filled rectangles)
    * Polygons and line strings
    * Neighbor agents (past trails, bounding boxes, future)
    * Ego vehicle (bounding box, past, future GT)
    * Goal pose
    * Optional GT footprints at regular step intervals

    Args:
        data: NPZ data dict (see parent package docstring for required keys).
        view_range: Half-extent of the axis limits in metres.  The x- and
            y-axes both span ``[centre - view_range, centre + view_range]``.
        footprint_interval: If > 0, draw ego vehicle outline footprints along
            the future GT every *footprint_interval* steps.  Default 0 (off).

    Returns:
        A Plotly ``Figure`` with a square aspect ratio and legend on the
        top left.
    """
    fig = go.Figure()

    for trace in draw_lanes(data):
        fig.add_trace(trace)
    for trace in draw_route(data):
        fig.add_trace(trace)
    for trace in draw_static_objects(data):
        fig.add_trace(trace)
    for trace in draw_polygons_and_lines(data):
        fig.add_trace(trace)
    for trace in draw_neighbor_agents(data):
        fig.add_trace(trace)
    for trace in draw_ego_vehicle(data):
        fig.add_trace(trace)
    for trace in draw_goal_pose(data):
        fig.add_trace(trace)

    for trace in draw_ego_footprints(data, footprint_interval):
        fig.add_trace(trace)

    ego_state = data["ego_current_state"].reshape(-1)
    center_x = float(ego_state[0])
    center_y = float(ego_state[1])

    fig.update_xaxes(range=[center_x - view_range, center_x + view_range])
    fig.update_yaxes(
        range=[center_y - view_range, center_y + view_range], scaleanchor="x", scaleratio=1
    )
    title = "Trajectory View"

    fig.update_layout(
        template="plotly_white",
        uirevision="1",
        title=title,
        xaxis_title="X [m]",
        yaxis_title="Y [m]",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=9)),
        margin=dict(l=40, r=20, t=40, b=40),
        height=700,
    )
    return fig
