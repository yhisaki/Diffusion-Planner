"""Scene drawing helpers — lane boundaries, agents, static objects, etc.

Each function accepts an NPZ data dict and returns a list of Plotly Scatter
traces.  None of these functions create a full Figure — they only produce
traces that can be added to an existing Figure via ``fig.add_trace()``.

All functions are reusable: pass a data dict matching the expected key schema
(see module-level docstring of the parent package) and the traces will be
generated independently of the container Figure or layout.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ._constants import TRAFFIC_LIGHT
from ._traffic_light import traffic_light_color, traffic_light_label, traffic_light_legend_order


def valid_xy_mask(points: np.ndarray) -> np.ndarray:
    """Return a boolean mask of points whose (x, y) are not both zero.

    Args:
        points: Array with (x, y) in the last dimension's first two positions.

    Returns:
        Boolean mask with same leading dimensions as *points*, selecting
        entries where at least one of x or y is non-zero.
    """
    return ~((points[..., 0] == 0) & (points[..., 1] == 0))


# ---------------------------------------------------------------------------
# Ego vehicle
# ---------------------------------------------------------------------------


def _make_vehicle_box(
    x: float,
    y: float,
    heading: float,
    wheelbase: float,
    car_length: float,
    car_width: float,
) -> tuple[list[float], list[float]]:
    """Compute the 5-corner outline of a vehicle bounding box."""
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
    return bx, by


def draw_ego_footprints(
    data: dict[str, np.ndarray],
    interval: int = 0,
) -> list[go.Scatter]:
    """Draw ego vehicle outline at every *interval* steps along the future trajectory.

    Args:
        data: NPZ data dict.
        interval: Step interval.  0 or negative hides all footprints.

    Returns:
        List of Plotly Scatter traces (all with ``showlegend=False``), or empty.
    """
    if interval <= 0 or "ego_agent_future" not in data:
        return []
    future = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
    ego_shape = data["ego_shape"].reshape(-1)
    wheelbase = float(ego_shape[0])
    car_length = float(ego_shape[1])
    car_width = float(ego_shape[2])

    traces: list[go.Scatter] = []
    for i in range(0, len(future), interval):
        state = future[i]
        heading = float(state[2]) if future.shape[1] >= 3 else 0.0
        bx, by = _make_vehicle_box(
            float(state[0]),
            float(state[1]),
            heading,
            wheelbase,
            car_length,
            car_width,
        )
        traces.append(
            go.Scatter(
                x=bx,
                y=by,
                mode="lines",
                line=dict(color="darkviolet", width=2),
                fill="toself",
                fillcolor="rgba(148,0,211,0.15)",
                opacity=0.6,
                legendgroup="GT Footprint",
                showlegend=(i == 0),
                name="GT Footprint",
            )
        )
    return traces


def draw_prediction_footprints(
    prediction: np.ndarray | None,
    interval: int = 0,
    shape: np.ndarray | None = None,
) -> list[go.Scatter]:
    """Draw prediction vehicle outlines at every *interval* steps.

    Args:
        prediction: Model output of shape ``(T, 4)`` or ``(P, T, 4)``.
        interval: Step interval.  0 or negative hides all footprints.
        shape: ``(3,)`` array [wheelbase, length, width].  Falls back to
            reasonable defaults if ``None``.

    Returns:
        List of Plotly Scatter traces (all with ``showlegend=False``), or empty.
    """
    if interval <= 0 or prediction is None:
        return []
    pred = np.asarray(prediction)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2 or pred.shape[1] < 4:
        return []

    if shape is not None:
        s = shape.reshape(-1)
    else:
        s = np.array([3.0, 4.5, 2.0])
    wheelbase = float(s[0])
    car_length = float(s[1])
    car_width = float(s[2])

    traces: list[go.Scatter] = []
    for i in range(0, len(pred), interval):
        state = pred[i]
        heading = float(np.arctan2(state[3], state[2]))
        bx, by = _make_vehicle_box(
            float(state[0]),
            float(state[1]),
            heading,
            wheelbase,
            car_length,
            car_width,
        )
        traces.append(
            go.Scatter(
                x=bx,
                y=by,
                mode="lines",
                line=dict(color="#00A6D6", width=2),
                fill="toself",
                fillcolor="rgba(0,166,214,0.1)",
                opacity=0.6,
                legendgroup="Pred Footprint",
                showlegend=(i == 0),
                name="Pred Footprint",
            )
        )
    return traces


def draw_ego_vehicle(data: dict[str, np.ndarray]) -> list[go.Scatter]:
    """Draw the ego vehicle bounding-box and its past/future trajectories.

    Args:
        data: NPZ data dict (requires ``ego_current_state``, ``ego_shape``).

    Returns:
        List of Plotly Scatter traces (ego box, past, GT future).
    """
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

    corners_x, corners_y = [], []
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

    if "ego_agent_future" in data:
        future = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
        t_vals = np.linspace(0, 1, len(future))
        colors = [f"rgb({int(255 * t)},{0},{int(255 * (1 - t))})" for t in t_vals]
        traces.append(
            go.Scatter(
                x=future[:, 0],
                y=future[:, 1],
                mode="lines+markers",
                line=dict(color="purple", width=2),
                marker=dict(size=4, color=colors),
                name="Ego Future (GT)",
                showlegend=True,
            )
        )

    return traces


# ---------------------------------------------------------------------------
# Neighbor agents
# ---------------------------------------------------------------------------


def draw_neighbor_agents(data: dict[str, np.ndarray]) -> list[go.Scatter]:
    """Draw neighbor agent past trajectories, bounding boxes, and future.

    Each agent is colour-coded by vehicle type (blue / green / purple).
    Up to 32 agents are drawn.

    Args:
        data: NPZ data dict (requires ``neighbor_agents_past``).

    Returns:
        List of Plotly Scatter traces (all with ``showlegend=False``).
    """
    traces: list[go.Scatter] = []
    if "neighbor_agents_past" not in data:
        return traces

    neighbors = data["neighbor_agents_past"]
    neighbors = neighbors.reshape(neighbors.shape[0], neighbors.shape[1], -1)
    last_t = neighbors.shape[1] - 1

    for i in range(min(neighbors.shape[0], 32)):
        neighbor = neighbors[i, last_t]
        if np.sum(np.abs(neighbor[:4])) < 1e-6:
            continue
        n_x, n_y = float(neighbor[0]), float(neighbor[1])
        n_cos, n_sin = float(neighbor[2]), float(neighbor[3])
        len_y_dim, len_x_dim = float(neighbor[6]), float(neighbor[7])

        vehicle_type = int(np.argmax(neighbor[8:11])) if neighbor.shape[0] > 10 else 0
        color = ["blue", "green", "purple"][vehicle_type] if vehicle_type < 3 else "blue"

        past_pts = neighbors[i, :, :2]
        past_mask = valid_xy_mask(past_pts)
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
                f_mask = valid_xy_mask(nf)
                if np.any(f_mask):
                    traces.append(
                        go.Scatter(
                            x=nf[f_mask, 0],
                            y=nf[f_mask, 1],
                            mode="lines+markers",
                            line=dict(color=color, width=1),
                            marker=dict(size=2, color=color),
                            opacity=0.4,
                            showlegend=False,
                        )
                    )

    return traces


# ---------------------------------------------------------------------------
# Lanes, route, static objects, polygons, line strings, goal pose
# ---------------------------------------------------------------------------


def draw_lanes(data: dict[str, np.ndarray]) -> list[go.Scatter]:
    """Draw lane boundaries coloured by traffic-light state.

    Each lane is drawn as two lines (left and right boundary).  Dummy legend
    traces are appended for each distinct traffic-light state present.

    Args:
        data: NPZ data dict (requires ``"lanes"`` key).

    Returns:
        List of Plotly Scatter traces.
    """
    traces: list[go.Scatter] = []
    if "lanes" not in data:
        return traces

    lanes = data["lanes"]
    if lanes.ndim == 3:
        lanes = lanes.reshape(lanes.shape[0], lanes.shape[1], -1)

    _seen: set[str] = set()

    for i in range(lanes.shape[0]):
        tl = lanes[i, 0, TRAFFIC_LIGHT : TRAFFIC_LIGHT + 5]
        color = traffic_light_color(tl)
        state_label = traffic_light_label(tl)
        _seen.add(state_label)

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
                line=dict(color=color, width=2),
                opacity=0.55,
                showlegend=False,
            )
        )
        traces.append(
            go.Scatter(
                x=rx[lane_mask],
                y=ry[lane_mask],
                mode="lines",
                line=dict(color=color, width=2),
                opacity=0.55,
                showlegend=False,
            )
        )

    for state_label, color in traffic_light_legend_order():
        if state_label in _seen:
            traces.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"TL: {state_label}",
                    showlegend=True,
                )
            )

    return traces


def draw_route(data: dict[str, np.ndarray]) -> list[go.Scatter]:
    """Draw route lanes as dashed olive lines.

    Args:
        data: NPZ data dict (requires ``"route_lanes"`` key).

    Returns:
        List of Plotly Scatter traces (all with ``showlegend=False``).
    """
    traces: list[go.Scatter] = []
    if "route_lanes" not in data:
        return traces
    route = data["route_lanes"]
    if route.ndim == 3:
        route = route.reshape(route.shape[0], route.shape[1], -1)
    for i in range(route.shape[0]):
        mask = valid_xy_mask(route[i])
        if np.any(mask):
            traces.append(
                go.Scatter(
                    x=route[i, mask, 0],
                    y=route[i, mask, 1],
                    mode="lines",
                    line=dict(color="olive", width=2, dash="dash"),
                    opacity=0.5,
                    showlegend=False,
                )
            )
    return traces


def draw_static_objects(data: dict[str, np.ndarray]) -> list[go.Scatter]:
    """Draw static objects as filled rectangles colour-coded by type.

    Args:
        data: NPZ data dict (requires ``"static_objects"`` key).

    Returns:
        List of Plotly Scatter traces (all with ``showlegend=False``).
    """
    traces: list[go.Scatter] = []
    if "static_objects" not in data:
        return traces
    statics = data["static_objects"]
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


def draw_goal_pose(data: dict[str, np.ndarray]) -> list[go.Scatter]:
    """Draw the goal pose as a short arrow.

    The goal is read from ``data["goal_pose"]``.  Accepts both ``[x, y, cos, sin]``
    and ``[x, y, heading]`` formats.

    Args:
        data: NPZ data dict (requires ``"goal_pose"`` key).

    Returns:
        List containing a single Plotly Scatter trace, or empty list.
    """
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


def draw_polygons_and_lines(data: dict[str, np.ndarray]) -> list[go.Scatter]:
    """Draw filled polygons and line strings (road borders / stop lines).

    Args:
        data: NPZ data dict (uses ``"polygons"`` and ``"line_strings"`` keys).

    Returns:
        List of Plotly Scatter traces (all with ``showlegend=False``).
    """
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
