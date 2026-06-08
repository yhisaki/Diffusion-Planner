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
        valid = ~((past[:, 0] == 0) & (past[:, 1] == 0))
        if np.any(valid):
            traces.append(
                go.Scatter(
                    x=past[valid, 0],
                    y=past[valid, 1],
                    mode="lines+markers",
                    line=dict(color="orange", width=2, dash="dash"),
                    marker=dict(size=3),
                    name="Ego Past",
                    showlegend=True,
                )
            )

    if "ego_agent_future" in data:
        future = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
        valid = ~((future[:, 0] == 0) & (future[:, 1] == 0))
        if np.any(valid):
            vf = future[valid]
            t_vals = np.linspace(0, 1, len(vf))
            colors_str = [f"rgb({int(255 * t)},{0},{int(255 * (1 - t))})" for t in t_vals]
            traces.append(
                go.Scatter(
                    x=vf[:, 0],
                    y=vf[:, 1],
                    mode="lines+markers",
                    line=dict(color="purple", width=2),
                    marker=dict(size=4, color=colors_str),
                    name="Ego Future (GT)",
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
        valid_past = ~((past_pts[:, 0] == 0) & (past_pts[:, 1] == 0))
        if np.any(valid_past):
            traces.append(
                go.Scatter(
                    x=past_pts[valid_past, 0],
                    y=past_pts[valid_past, 1],
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
                valid_nf = ~((nf[:, 0] == 0) & (nf[:, 1] == 0))
                if np.any(valid_nf):
                    traces.append(
                        go.Scatter(
                            x=nf[valid_nf, 0],
                            y=nf[valid_nf, 1],
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

        traces.append(
            go.Scatter(
                x=lx,
                y=ly,
                mode="lines",
                line=dict(color=color, width=1),
                opacity=0.25,
                showlegend=False,
            )
        )
        traces.append(
            go.Scatter(
                x=rx,
                y=ry,
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
        valid = ~((route[i, :, 0] == 0) & (route[i, :, 1] == 0))
        if np.any(valid):
            traces.append(
                go.Scatter(
                    x=route[i, valid, 0],
                    y=route[i, valid, 1],
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
        time_step: Optional time step index to mark on ego future trajectory.

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

    if time_step is not None and "ego_agent_future" in data:
        ego_state = data["ego_current_state"].reshape(-1)
        ego_shape = data["ego_shape"].reshape(-1)
        wheelbase = float(ego_shape[0])
        car_length = float(ego_shape[1])
        car_width = float(ego_shape[2])
        future = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
        if 0 <= time_step < len(future):
            fx, fy = float(future[time_step, 0]), float(future[time_step, 1])
            if future.shape[1] >= 3:
                heading = float(future[time_step, 2])
                fcos, fsin = np.cos(heading), np.sin(heading)
            else:
                fcos, fsin = float(ego_state[2]), float(ego_state[3])
            cx = fx + (wheelbase / 2) * fcos
            cy = fy + (wheelbase / 2) * fsin
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
                bx.append(dx * fcos - dy * fsin + cx)
                by.append(dx * fsin + dy * fcos + cy)
            fig.add_trace(
                go.Scatter(
                    x=bx,
                    y=by,
                    mode="lines",
                    line=dict(color="darkviolet", width=2),
                    fill="toself",
                    fillcolor="rgba(148,0,211,0.25)",
                    name=f"t={time_step}",
                    showlegend=True,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[fx],
                    y=[fy],
                    mode="markers",
                    marker=dict(size=8, color="darkviolet", symbol="x"),
                    showlegend=False,
                )
            )

    ego_state = data["ego_current_state"].reshape(-1)
    center_x = float(ego_state[0])
    center_y = float(ego_state[1])

    fig.update_xaxes(range=[center_x - view_range, center_x + view_range])
    fig.update_yaxes(
        range=[center_y - view_range, center_y + view_range], scaleanchor="x", scaleratio=1
    )
    fig.update_layout(
        title="Trajectory View",
        xaxis_title="X [m]",
        yaxis_title="Y [m]",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=9)),
        margin=dict(l=40, r=20, t=40, b=40),
        height=700,
    )

    return fig


def _compute_speeds(
    positions: np.ndarray, dt: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-step speed in km/h and time-step indices from (N, 2) positions.

    Returns:
        (t, speed_kmh) where t starts at 0 and speed_kmh has length N-1.
    """
    diffs = np.diff(positions, axis=0)
    speed_ms = np.sqrt((diffs**2).sum(axis=1)) / dt
    speed_kmh = speed_ms * 3.6
    t = np.arange(len(speed_kmh))
    return t, speed_kmh


def _compute_curvature(
    headings: np.ndarray, positions: np.ndarray, dt: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-step curvature (1/m) and corresponding time indices.

    Args:
        headings: (N,) unwrapped headings in radians.
        positions: (N, 2) xy positions.
        dt: Time step in seconds.

    Returns:
        (t, curvature) where curvature has length N-1.
    """
    diffs = np.diff(positions, axis=0)
    arc_lengths = np.sqrt((diffs**2).sum(axis=1))
    heading_diffs = np.diff(np.unwrap(headings))
    curvature = np.zeros(len(heading_diffs))
    valid_mask = arc_lengths > 1e-6
    curvature[valid_mask] = heading_diffs[valid_mask] / arc_lengths[valid_mask]
    t = np.arange(len(curvature))
    return t, curvature


def _past_speeds(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute speed from past trajectory. Returns (t, speed_kmh) with t < 0."""
    if "ego_agent_past" not in data:
        return None
    past = data["ego_agent_past"].reshape(-1, data["ego_agent_past"].shape[-1])
    valid = ~((past[:, 0] == 0) & (past[:, 1] == 0))
    if np.sum(valid) < 2:
        return None
    vp = past[valid]
    # Temporal order: oldest past -> newest past -> current
    positions = np.vstack([vp[:, :2], [[data["ego_current_state"].reshape(-1)[0], data["ego_current_state"].reshape(-1)[1]]]])
    _, speed_kmh = _compute_speeds(positions)
    n = len(speed_kmh)
    t = np.arange(-n, 0)
    return t, speed_kmh


def _past_curvature(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute curvature from past trajectory. Returns (t, curvature) with t < 0."""
    if "ego_agent_past" not in data:
        return None
    past = data["ego_agent_past"].reshape(-1, data["ego_agent_past"].shape[-1])
    valid = ~((past[:, 0] == 0) & (past[:, 1] == 0))
    if np.sum(valid) < 2:
        return None
    vp = past[valid]
    ego_state = data["ego_current_state"].reshape(-1)
    cos_init = ego_state[2]
    sin_init = ego_state[3]

    # Temporal order: oldest past -> newest past -> current
    headings = np.concatenate([vp[:, 2], [np.arctan2(sin_init, cos_init)]])
    positions = np.vstack([vp[:, :2], [[ego_state[0], ego_state[1]]]])
    _, curvature = _compute_curvature(headings, positions)
    n = len(curvature)
    t = np.arange(-n, 0)
    return t, curvature


def plot_velocity(data: dict[str, np.ndarray]) -> go.Figure:
    """Speed plot for the ego vehicle (past + future)."""
    fig = go.Figure()
    has_data = False

    # Past speed
    past_result = _past_speeds(data)
    if past_result is not None:
        t, speed_kmh = past_result
        fig.add_trace(go.Scatter(
            x=t, y=speed_kmh, mode="lines", name="Past Speed",
            line=dict(color="orange", width=2),
        ))
        has_data = True

    # Future (GT) speed
    if "ego_agent_future" in data:
        ego_state = data["ego_current_state"].reshape(-1)
        future = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
        valid = ~((future[:, 0] == 0) & (future[:, 1] == 0))
        if np.any(valid):
            positions = np.vstack([ego_state[:2], future[valid, :2]])
            t, speed_kmh = _compute_speeds(positions)
            fig.add_trace(go.Scatter(
                x=t, y=speed_kmh, mode="lines", name="Future Speed",
                line=dict(color="black", width=2, dash="dash"),
            ))
            has_data = True

    if not has_data:
        fig.update_layout(title="Speed (no data)", height=400)
        return fig

    fig.update_layout(
        title="Speed (km/h)",
        xaxis_title="Time step",
        yaxis_title="Speed (km/h)",
        yaxis=dict(range=[0, 80]),
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig


def plot_curvature(data: dict[str, np.ndarray]) -> go.Figure:
    """Path curvature plot for the ego vehicle (past + future)."""
    fig = go.Figure()
    has_data = False

    # Past curvature
    past_result = _past_curvature(data)
    if past_result is not None:
        t, curvature = past_result
        fig.add_trace(go.Scatter(
            x=t, y=curvature, mode="lines", name="Past Curvature",
            line=dict(color="orange", width=2),
        ))
        has_data = True

    # Future (GT) curvature
    if "ego_agent_future" in data:
        ego_state = data["ego_current_state"].reshape(-1)
        future = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
        valid = ~((future[:, 0] == 0) & (future[:, 1] == 0))
        if np.any(valid):
            vf = future[valid]
            cos_init = ego_state[2]
            sin_init = ego_state[3]
            if vf.shape[1] >= 3:
                headings = np.concatenate([[np.arctan2(sin_init, cos_init)], vf[:, 2]])
            else:
                headings = np.concatenate([
                    [np.arctan2(sin_init, cos_init)],
                    np.arctan2(vf[:, 1] - np.roll(vf[:, 1], 1), vf[:, 0] - np.roll(vf[:, 0], 1)),
                ])
                headings[0] = np.arctan2(sin_init, cos_init)
            positions = np.vstack([ego_state[:2], vf[:, :2]])
            t, curvature = _compute_curvature(headings, positions)
            fig.add_trace(go.Scatter(
                x=t, y=curvature, mode="lines", name="Future Curvature",
                line=dict(color="black", width=2, dash="dash"),
            ))
            has_data = True

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)

    if not has_data:
        fig.update_layout(title="Curvature (no data)", height=400)
        return fig

    fig.update_layout(
        title="Path Curvature (1/m)",
        xaxis_title="Time step",
        yaxis_title="Curvature (1/m)",
        yaxis=dict(range=[-0.2, 0.2]),
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig
