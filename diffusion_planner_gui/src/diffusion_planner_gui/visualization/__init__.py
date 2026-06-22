"""Visualization package for Diffusion Planner training data and predictions.

Expected NPZ data format
------------------------
* ``ego_current_state``  — ``(10,)``: [x, y, cos, sin, vx, vy, ax, ay, steering_angle, yaw_rate]
* ``ego_shape``          — ``(3,)``: [wheelbase, length, width]
* ``ego_agent_past``     — ``(T_past, D)``: past trajectory
* ``ego_agent_future``   — ``(T_future, D)``: ground-truth future trajectory
* ``lanes``              — ``(N, 20, SEG)``: lane boundaries + traffic-light states
* ``neighbor_agents_past``   — ``(N, T, D)``: neighbor past states
* ``neighbor_agents_future`` — ``(N, T, D)``: neighbor future positions
* ``route_lanes``, ``static_objects``, ``goal_pose``, ``polygons``, ``line_strings``

Prediction format
-----------------
``np.ndarray`` of shape ``(T, 4)`` or ``(P, T, 4)``. Last dim: ``[x, y, cos, sin]``.
"""

from .prediction import plot_prediction_vs_gt
from .timeseries import plot_tcos, plot_tsin, plot_tv, plot_tx, plot_ty
from .trajectory import plot_trajectory
