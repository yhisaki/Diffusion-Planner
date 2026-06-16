"""Plotly-based visualization utilities for training data NPZ files.

All drawing is done with Plotly graph objects — no matplotlib dependency.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

_PLOTLY_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3FZ",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def sample_color(index: int) -> str:
    return _PLOTLY_COLORS[index % len(_PLOTLY_COLORS)]


# ---------------------------------------------------------------------------
# Dimension constants (mirroring diffusion_planner.dimensions)
# ---------------------------------------------------------------------------

TRAFFIC_LIGHT = 8
TRAFFIC_LIGHT_GREEN = 8
TRAFFIC_LIGHT_YELLOW = 9
TRAFFIC_LIGHT_RED = 10
TRAFFIC_LIGHT_WHITE = 11
TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT = 12
LINE_TYPE_LEFT_START = 13
LINE_TYPE_NUM = 10
SEGMENT_POINT_DIM = LINE_TYPE_LEFT_START + LINE_TYPE_NUM + LINE_TYPE_NUM


def _traffic_light_color(tl: np.ndarray) -> str:
    """Return a colour string for a 5-element traffic-light one-hot vector."""
    if tl[0] == 1:
        return "green"
    if tl[1] == 1:
        return "yellow"
    if tl[2] == 1:
        return "red"
    if tl[3] == 1:
        return "purple"
    if tl[4] == 1:
        return "black"
    return "purple"


# ---------------------------------------------------------------------------
# Scene drawing helpers (all return lists of Plotly traces)
# ---------------------------------------------------------------------------


def _valid_xy_mask(points: np.ndarray) -> np.ndarray:
    return ~((points[..., 0] == 0) & (points[..., 1] == 0))


def _ego_marker_pose(data: dict[str, np.ndarray], marker_step: int) -> tuple[float, float, float, str] | None:
    if marker_step <= 0:
        return None

    cursor = marker_step - 1
    if "ego_agent_future" not in data:
        return None

    future = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
    if cursor >= len(future):
        return None

    state = future[cursor]
    heading = float(state[2]) if future.shape[1] >= 3 else 0.0
    return float(state[0]), float(state[1]), heading, f"future[{cursor}]"


def _draw_ego_marker(data: dict[str, np.ndarray], marker_step: int) -> list[go.Scatter]:
    pose = _ego_marker_pose(data, marker_step)
    if pose is None:
        return []

    x, y, heading, label = pose
    ego_shape = data["ego_shape"].reshape(-1)
    wheelbase = float(ego_shape[0])
    car_length = float(ego_shape[1])
    car_width = float(ego_shape[2])
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    cx = x + (wheelbase / 2) * cos_h
    cy = y + (wheelbase / 2) * sin_h
    bx, by = [], []
    for dx_frac, dy_frac in [
        (-0.5, -0.5),
        (0.5, -0.5),
        (0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, -0.5),
    ]:
        dx = car_length * dx_frac
        dy = car_width * dy_frac
        bx.append(dx * cos_h - dy * sin_h + cx)
        by.append(dx * sin_h + dy * cos_h + cy)

    return [
        go.Scatter(
            x=bx,
            y=by,
            mode="lines",
            line=dict(color="darkviolet", width=2),
            fill="toself",
            fillcolor="rgba(148,0,211,0.25)",
            name=f"marker {label}",
            showlegend=True,
        ),
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            marker=dict(size=8, color="darkviolet", symbol="x"),
            showlegend=False,
        ),
    ]


def _draw_ego_vehicle(data: dict[str, np.ndarray]) -> list[go.Scatter]:
    """Draw ego vehicle, past and future trajectories."""
    traces: list[go.Scatter] = []
    ego_state = data["ego_current_state"].reshape(-1)
    ego_x, ego_y = float(ego_state[0]), float(ego_state[1])
    ego_shape = data["ego_shape"].reshape(-1)
    wheelbase = float(ego_shape[0])
    car_length = float(ego_shape[1])
    car_width = float(ego_shape[2])
    ego_cos, ego_sin = float(ego_state[2]), float(ego_state[3])

    cx = ego_x + (wheelbase / 2) * ego_cos
    cy = ego_y + (wheelbase / 2) * ego_sin

    corners_x = []
    corners_y = []
    for dx_frac, dy_frac in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)]:
        dx = car_length * dx_frac
        dy = car_width * dy_frac
        corners_x.append(dx * ego_cos - dy * ego_sin + cx)
        corners_y.append(dx * ego_sin + dy * ego_cos + cy)
    traces.append(
        go.Scatter(
            x=corners_x,
            y=corners_y,
            mode="lines",
            line=dict(color="red", width=2),
            fill="toself",
            fillcolor="rgba(255,0,0,0.3)",
            name="Ego Vehicle",
            showlegend=True,
        )
    )

    if "ego_agent_past" in data:
        past = data["ego_agent_past"].reshape(-1, data["ego_agent_past"].shape[-1])
        traces.append(
            go.Scatter(
                x=past[:, 0],
                y=past[:, 1],
                mode="lines+markers",
                line=dict(color="orange", width=2, dash="dash"),
                marker=dict(size=3),
                name="Ego Past",
                showlegend=True,
            )
        )

    if "original_ego_agent_past_in_augmented_frame" in data:
        original_past = data["original_ego_agent_past_in_augmented_frame"].reshape(
            -1, data["original_ego_agent_past_in_augmented_frame"].shape[-1]
        )
        traces.append(
            go.Scatter(
                x=original_past[:, 0],
                y=original_past[:, 1],
                mode="lines",
                line=dict(color="rgba(80,80,80,0.8)", width=1, dash="dot"),
                name="Original Past in Aug Frame",
                showlegend=True,
            )
        )

    if "ego_agent_future" in data:
        future = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
        t_vals = np.linspace(0, 1, len(future))
        colors_str = [f"rgb({int(255 * t)},{0},{int(255 * (1 - t))})" for t in t_vals]
        traces.append(
            go.Scatter(
                x=future[:, 0],
                y=future[:, 1],
                mode="lines+markers",
                line=dict(color="purple", width=2),
                marker=dict(size=4, color=colors_str),
                name="Ego Future (GT)",
                showlegend=True,
            )
        )

    if "original_ego_agent_future_in_augmented_frame" in data:
        original_future = data["original_ego_agent_future_in_augmented_frame"].reshape(
            -1, data["original_ego_agent_future_in_augmented_frame"].shape[-1]
        )
        traces.append(
            go.Scatter(
                x=original_future[:, 0],
                y=original_future[:, 1],
                mode="lines+markers",
                line=dict(color="black", width=1, dash="dot"),
                marker=dict(size=3, color="black", symbol="circle-open"),
                name="Original Future in Aug Frame",
                showlegend=True,
            )
        )

    return traces


def _draw_neighbor_agents(data: dict[str, np.ndarray]) -> list[go.Scatter]:
    """Draw neighbor agents — past trajectories, bounding boxes, future trajectories."""
    traces: list[go.Scatter] = []
    if "neighbor_agents_past" not in data:
        return traces

    neighbors = data["neighbor_agents_past"]  # (N, T, 11)
    neighbors = neighbors.reshape(neighbors.shape[0], neighbors.shape[1], -1)
    last_t = neighbors.shape[1] - 1

    for i in range(min(neighbors.shape[0], 32)):
        neighbor = neighbors[i, last_t]
        if np.sum(np.abs(neighbor[:4])) < 1e-6:
            continue
        n_x, n_y = float(neighbor[0]), float(neighbor[1])
        n_cos, n_sin = float(neighbor[2]), float(neighbor[3])
        vel_x, vel_y = float(neighbor[4]), float(neighbor[5])
        len_y_dim, len_x_dim = float(neighbor[6]), float(neighbor[7])

        vehicle_type = int(np.argmax(neighbor[8:11])) if neighbor.shape[0] > 10 else 0
        color = ["blue", "green", "purple"][vehicle_type] if vehicle_type < 3 else "blue"

        past_pts = neighbors[i, :, :2]
        past_mask = _valid_xy_mask(past_pts)
        if np.any(past_mask):
            traces.append(
                go.Scatter(
                    x=past_pts[past_mask, 0],
                    y=past_pts[past_mask, 1],
                    mode="lines",
                    line=dict(color=color, width=1, dash="dash"),
                    opacity=0.6,
                    showlegend=False,
                )
            )

        heading_n = np.arctan2(n_sin, n_cos)
        cos_h, sin_h = np.cos(heading_n), np.sin(heading_n)
        bx, by = [], []
        for dx_frac, dy_frac in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)]:
            dx = len_x_dim * dx_frac
            dy = len_y_dim * dy_frac
            bx.append(dx * cos_h - dy * sin_h + n_x)
            by.append(dx * sin_h + dy * cos_h + n_y)
        traces.append(
            go.Scatter(
                x=bx,
                y=by,
                mode="lines",
                line=dict(color=color, width=1),
                fill="toself",
                fillcolor=f"rgba({100},{150},{255},0.15)",
                showlegend=False,
            )
        )

        if "neighbor_agents_future" in data:
            n_future = data["neighbor_agents_future"].reshape(
                data["neighbor_agents_future"].shape[0],
                -1,
                data["neighbor_agents_future"].shape[-1],
            )
            if i < n_future.shape[0]:
                nf = n_future[i]
                future_mask = _valid_xy_mask(nf)
                if np.any(future_mask):
                    traces.append(
                        go.Scatter(
                            x=nf[future_mask, 0],
                            y=nf[future_mask, 1],
                            mode="lines+markers",
                            line=dict(color=color, width=1),
                            marker=dict(size=2, color=color),
                            opacity=0.4,
                            showlegend=False,
                        )
                    )

    return traces


def _draw_lanes(data: dict[str, np.ndarray]) -> list[go.Scatter]:
    """Draw lane boundaries (left + right) from lanes data."""
    traces: list[go.Scatter] = []
    if "lanes" not in data:
        return traces

    lanes = data["lanes"]  # (140, 20, 33)
    if lanes.ndim == 3:
        lanes = lanes.reshape(lanes.shape[0], lanes.shape[1], -1)

    for i in range(lanes.shape[0]):
        tl = lanes[i, 0, TRAFFIC_LIGHT : TRAFFIC_LIGHT + 5]
        color = _traffic_light_color(tl)

        lx = lanes[i, :, 0] + lanes[i, :, 4]
        ly = lanes[i, :, 1] + lanes[i, :, 5]
        rx = lanes[i, :, 0] + lanes[i, :, 6]
        ry = lanes[i, :, 1] + lanes[i, :, 7]
        lane_mask = np.any(np.abs(lanes[i, :, :8]) > 1e-6, axis=-1)
        if not np.any(lane_mask):
            continue

        traces.append(
            go.Scatter(
                x=lx[lane_mask],
                y=ly[lane_mask],
                mode="lines",
                line=dict(color=color, width=1),
                opacity=0.25,
                showlegend=False,
            )
        )
        traces.append(
            go.Scatter(
                x=rx[lane_mask],
                y=ry[lane_mask],
                mode="lines",
                line=dict(color=color, width=1),
                opacity=0.25,
                showlegend=False,
            )
        )

    return traces


def _draw_route(data: dict[str, np.ndarray]) -> list[go.Scatter]:
    """Draw route lanes as dashed olive lines."""
    traces: list[go.Scatter] = []
    if "route_lanes" not in data:
        return traces

    route = data["route_lanes"]
    if route.ndim == 3:
        route = route.reshape(route.shape[0], route.shape[1], -1)

    for i in range(route.shape[0]):
        route_mask = _valid_xy_mask(route[i])
        if np.any(route_mask):
            traces.append(
                go.Scatter(
                    x=route[i, route_mask, 0],
                    y=route[i, route_mask, 1],
                    mode="lines",
                    line=dict(color="olive", width=2, dash="dash"),
                    opacity=0.5,
                    showlegend=False,
                )
            )

    return traces


def _draw_static_objects(data: dict[str, np.ndarray]) -> list[go.Scatter]:
    """Draw static objects as filled rectangles."""
    traces: list[go.Scatter] = []
    if "static_objects" not in data:
        return traces

    statics = data["static_objects"]  # (5, 10)
    if statics.ndim > 2:
        statics = statics.reshape(statics.shape[0], -1)

    colors_map = ["orange", "gray", "yellow", "brown"]
    for i in range(statics.shape[0]):
        obj = statics[i]
        if np.sum(np.abs(obj[:4])) < 1e-6:
            continue
        obj_x, obj_y = float(obj[0]), float(obj[1])
        obj_heading = np.arctan2(float(obj[3]), float(obj[2]))
        obj_width = float(obj[4]) if obj.shape[0] > 4 else 1.0
        obj_length = float(obj[5]) if obj.shape[0] > 5 else 1.0

        obj_type = int(np.argmax(obj[-4:])) if obj.shape[0] >= 10 else 0
        obj_color = colors_map[obj_type % len(colors_map)]

        cos_h, sin_h = np.cos(obj_heading), np.sin(obj_heading)
        bx, by = [], []
        for dx_frac, dy_frac in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)]:
            dx = obj_length * dx_frac
            dy = obj_width * dy_frac
            bx.append(dx * cos_h - dy * sin_h + obj_x)
            by.append(dx * sin_h + dy * cos_h + obj_y)
        traces.append(
            go.Scatter(
                x=bx,
                y=by,
                mode="lines",
                line=dict(color=obj_color, width=1),
                fill="toself",
                fillcolor=obj_color,
                opacity=0.4,
                showlegend=False,
            )
        )

    return traces


def _draw_goal_pose(data: dict[str, np.ndarray]) -> list[go.Scatter]:
    """Draw goal pose as an arrow approximated by a line."""
    traces: list[go.Scatter] = []
    if "goal_pose" not in data:
        return traces

    goal = data["goal_pose"].reshape(-1)
    if np.sum(np.abs(goal[:2])) < 1e-6:
        return traces
    if len(goal) >= 4:
        gx, gy = float(goal[0]), float(goal[1])
        gcos, gsin = float(goal[2]), float(goal[3])
    elif len(goal) >= 3:
        gx, gy = float(goal[0]), float(goal[1])
        heading = float(goal[2])
        gcos, gsin = np.cos(heading), np.sin(heading)
    else:
        return traces

    arrow_dx = 2.0 * gcos
    arrow_dy = 2.0 * gsin
    traces.append(
        go.Scatter(
            x=[gx, gx + arrow_dx],
            y=[gy, gy + arrow_dy],
            mode="lines+markers",
            line=dict(color="blue", width=3),
            marker=dict(size=8, symbol="arrow", angle=np.degrees(np.arctan2(gsin, gcos))),
            name="Goal Pose",
            showlegend=True,
        )
    )

    return traces


def _draw_polygons_and_lines(data: dict[str, np.ndarray]) -> list[go.Scatter]:
    """Draw polygons (filled) and line strings (road borders vs stop lines)."""
    traces: list[go.Scatter] = []

    if "polygons" in data:
        polygons = data["polygons"]
        if polygons.ndim == 3:
            for i in range(polygons.shape[0]):
                polygon = polygons[i]
                if np.sum(np.abs(polygon[:, :2])) < 1e-6:
                    continue
                traces.append(
                    go.Scatter(
                        x=polygon[:, 0],
                        y=polygon[:, 1],
                        mode="lines",
                        line=dict(color="gray", width=1),
                        fill="toself",
                        fillcolor="rgba(128,128,128,0.4)",
                        showlegend=False,
                    )
                )

    if "line_strings" in data:
        ls = data["line_strings"]
        if ls.ndim == 3:
            for i in range(ls.shape[0]):
                line = ls[i]
                if np.sum(np.abs(line)) < 1e-6:
                    continue
                is_road_border = line.shape[-1] > 3 and np.any(line[:, 3] > 0.5)
                color = "red" if is_road_border else "orange"
                traces.append(
                    go.Scatter(
                        x=line[:, 0],
                        y=line[:, 1],
                        mode="lines",
                        line=dict(color=color, width=1),
                        opacity=0.7,
                        showlegend=False,
                    )
                )

    return traces


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------


def plot_trajectory(
    data: dict[str, np.ndarray],
    view_range: int = 60,
    time_step: int | None = None,
) -> go.Figure:
    """Create the main trajectory visualization as a Plotly Figure.

    Args:
        data: Dict of numpy arrays loaded from an NPZ file.
        view_range: Half-range for axis limits in metres.
        time_step: Optional unified ego timeline marker. 0 hides it; positive
            values walk through past, current, then future.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    for trace in _draw_lanes(data):
        fig.add_trace(trace)
    for trace in _draw_route(data):
        fig.add_trace(trace)
    for trace in _draw_static_objects(data):
        fig.add_trace(trace)
    for trace in _draw_polygons_and_lines(data):
        fig.add_trace(trace)
    for trace in _draw_neighbor_agents(data):
        fig.add_trace(trace)
    for trace in _draw_ego_vehicle(data):
        fig.add_trace(trace)
    for trace in _draw_goal_pose(data):
        fig.add_trace(trace)

    if time_step is not None:
        for trace in _draw_ego_marker(data, time_step):
            fig.add_trace(trace)

    ego_state = data["ego_current_state"].reshape(-1)
    center_x = float(ego_state[0])
    center_y = float(ego_state[1])

    fig.update_xaxes(range=[center_x - view_range, center_x + view_range])
    fig.update_yaxes(
        range=[center_y - view_range, center_y + view_range], scaleanchor="x", scaleratio=1
    )
    title = "Trajectory View"
    if "augmentation_perturbation" in data:
        dx, dy, dyaw = data["augmentation_perturbation"].reshape(-1)[:3]
        title = f"Augmented Trajectory View (dx={dx:.2f}m, dy={dy:.2f}m, dyaw={dyaw:.2f}rad)"

    fig.update_layout(
        title=title,
        xaxis_title="X [m]",
        yaxis_title="Y [m]",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=9)),
        margin=dict(l=40, r=20, t=40, b=40),
        height=700,
    )

    return fig


def _past_positions(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    if "ego_agent_past" not in data:
        return None
    past = data["ego_agent_past"].reshape(-1, data["ego_agent_past"].shape[-1])
    ego_state = data["ego_current_state"].reshape(-1)
    positions = np.vstack([past[:, :2], [[ego_state[0], ego_state[1]]]])
    n = len(positions)
    t = np.arange(-n + 1, 1)
    return t, positions


def _past_heading_cos_sin(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if "ego_agent_past" not in data:
        return None
    past = data["ego_agent_past"].reshape(-1, data["ego_agent_past"].shape[-1])
    ego_state = data["ego_current_state"].reshape(-1)
    if past.shape[1] >= 4:
        cos_vals = past[:, 2]
        sin_vals = past[:, 3]
    elif past.shape[1] == 3:
        heading = past[:, 2]
        cos_vals = np.cos(heading)
        sin_vals = np.sin(heading)
    else:
        return None
    cos_vals = np.append(cos_vals, ego_state[2])
    sin_vals = np.append(sin_vals, ego_state[3])
    n = len(cos_vals)
    t = np.arange(-n + 1, 1)
    return t, cos_vals, sin_vals


def _future_heading_cos_sin(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if "ego_agent_future" not in data:
        return None
    ego_state = data["ego_current_state"].reshape(-1)
    future = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
    if future.shape[1] >= 4:
        cos_vals = np.hstack([ego_state[2], future[:, 2]])
        sin_vals = np.hstack([ego_state[3], future[:, 3]])
    elif future.shape[1] >= 3:
        heading = np.hstack([np.arctan2(ego_state[3], ego_state[2]), future[:, 2]])
        cos_vals = np.cos(heading)
        sin_vals = np.sin(heading)
    else:
        return None
    t = np.arange(len(cos_vals))
    return t, cos_vals, sin_vals


def plot_tx(data: dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    has_data = False

    past_result = _past_positions(data)
    if past_result is not None:
        t, positions = past_result
        fig.add_trace(go.Scatter(
            x=t, y=positions[:, 0], mode="lines+markers", name="Past X",
            line=dict(color="orange", width=2), marker=dict(size=3),
        ))
        has_data = True

    if "ego_agent_future" in data:
        ego_state = data["ego_current_state"].reshape(-1)
        future = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
        positions = np.vstack([[ego_state[0], ego_state[1]], future[:, :2]])
        t = np.arange(len(positions))
        fig.add_trace(go.Scatter(
            x=t, y=positions[:, 0], mode="lines+markers", name="Future X",
            line=dict(color="black", width=2), marker=dict(size=3),
        ))
        has_data = True

    if not has_data:
        fig.update_layout(title="t-x (no data)", height=400)
        return fig

    fig.update_layout(
        title="t-x",
        xaxis_title="Time step",
        yaxis_title="X [m]",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig


def plot_ty(data: dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    has_data = False

    past_result = _past_positions(data)
    if past_result is not None:
        t, positions = past_result
        fig.add_trace(go.Scatter(
            x=t, y=positions[:, 1], mode="lines+markers", name="Past Y",
            line=dict(color="orange", width=2), marker=dict(size=3),
        ))
        has_data = True

    if "ego_agent_future" in data:
        ego_state = data["ego_current_state"].reshape(-1)
        future = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
        positions = np.vstack([[ego_state[0], ego_state[1]], future[:, :2]])
        t = np.arange(len(positions))
        fig.add_trace(go.Scatter(
            x=t, y=positions[:, 1], mode="lines+markers", name="Future Y",
            line=dict(color="black", width=2), marker=dict(size=3),
        ))
        has_data = True

    if not has_data:
        fig.update_layout(title="t-y (no data)", height=400)
        return fig

    fig.update_layout(
        title="t-y",
        xaxis_title="Time step",
        yaxis_title="Y [m]",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig


def plot_tcos(data: dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    has_data = False

    past_result = _past_heading_cos_sin(data)
    if past_result is not None:
        t, cos_vals, _ = past_result
        fig.add_trace(go.Scatter(
            x=t, y=cos_vals, mode="lines+markers", name="Past cos",
            line=dict(color="orange", width=2), marker=dict(size=3),
        ))
        has_data = True

    future_result = _future_heading_cos_sin(data)
    if future_result is not None:
        t, cos_vals, _ = future_result
        fig.add_trace(go.Scatter(
            x=t, y=cos_vals, mode="lines+markers", name="Future cos",
            line=dict(color="black", width=2), marker=dict(size=3),
        ))
        has_data = True

    if not has_data:
        fig.update_layout(title="t-cos (no data)", height=400)
        return fig

    fig.update_layout(
        title="t-cos",
        xaxis_title="Time step",
        yaxis_title="cos(heading)",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig


def plot_tsin(data: dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    has_data = False

    past_result = _past_heading_cos_sin(data)
    if past_result is not None:
        t, _, sin_vals = past_result
        fig.add_trace(go.Scatter(
            x=t, y=sin_vals, mode="lines+markers", name="Past sin",
            line=dict(color="orange", width=2), marker=dict(size=3),
        ))
        has_data = True

    future_result = _future_heading_cos_sin(data)
    if future_result is not None:
        t, _, sin_vals = future_result
        fig.add_trace(go.Scatter(
            x=t, y=sin_vals, mode="lines+markers", name="Future sin",
            line=dict(color="black", width=2), marker=dict(size=3),
        ))
        has_data = True

    if not has_data:
        fig.update_layout(title="t-sin (no data)", height=400)
        return fig

    fig.update_layout(
        title="t-sin",
        xaxis_title="Time step",
        yaxis_title="sin(heading)",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig


def plot_tdisplacement(data: dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    has_data = False

    if "ego_agent_past" in data:
        past = data["ego_agent_past"].reshape(-1, data["ego_agent_past"].shape[-1])
        if past.shape[0] >= 2:
            displacements = np.linalg.norm(np.diff(past[:, :2].astype(np.float64), axis=0), axis=1)
            speed = displacements / 0.1
            t = np.arange(-len(speed), 0)
            fig.add_trace(go.Scatter(
                x=t, y=speed, mode="lines+markers", name="Past speed",
                line=dict(color="orange", width=2), marker=dict(size=3),
            ))
            has_data = True

    if not has_data:
        fig.update_layout(title="t-displacement (no data)", height=400)
        return fig

    fig.update_layout(
        title="t-displacement",
        xaxis_title="Time step",
        yaxis_title="speed [m/s]",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig
