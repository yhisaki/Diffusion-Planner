from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _valid_xy(x: np.ndarray) -> np.ndarray:
    return np.any(np.abs(x[..., :2]) > 1e-6, axis=-1)


@dataclass
class AugumentedEgoPosition:
    x: float
    y: float
    yaw: float


@dataclass
class StatePerturbationConfig:
    path_augment_prob: float
    velocity_augment_prob: float
    stop_lon_range: float
    lat_range: float
    yaw_range: float
    velocity_scale_range: float
    hermite_connect_distance_m: float


class StatePerturbation:
    """Scene-level ego-centric data augmentation.

    The raw dataset is already expressed in the current ego frame. This augmenter
    samples a small virtual current ego pose in that frame, then rewrites every
    geometric field into the new virtual ego frame.
    """

    def __init__(
        self,
        path_augment_prob: float = 0.5,
        velocity_augment_prob: float = 0.5,
        stop_lon_range: float = 4.0,
        lat_range: float = 1.0,
        yaw_range: float = 0.2,
        velocity_scale_range: float = 0.6,
        hermite_connect_distance_m: float = 10.0,
    ) -> None:
        self.config = StatePerturbationConfig(
            path_augment_prob=path_augment_prob,
            velocity_augment_prob=velocity_augment_prob,
            stop_lon_range=stop_lon_range,
            lat_range=lat_range,
            yaw_range=yaw_range,
            velocity_scale_range=velocity_scale_range,
            hermite_connect_distance_m=hermite_connect_distance_m,
        )

    def augment(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        augmented = {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            for key, value in data.items()
        }

        if np.random.random() <= self.config.path_augment_prob:
            is_vehicle_stopping = data.get("ego_current_state")[4] < 1e-3
            augumented_ego_pos = self._get_augment_ego_position(is_vehicle_stopping)
            self._add_original_gt_in_augmented_frame(
                augmented, augumented_ego_pos, source_data=data
            )
            self._transform_scene_to_new_ego_frame(augmented, augumented_ego_pos)
            self._connect_ego_future_with_hermite(augmented)
            self._reset_ego_current_state(augmented, augumented_ego_pos)
        else:
            self._add_default_aux_keys(augmented)

        if np.random.random() <= self.config.velocity_augment_prob:
            self._augument_velocity_past(augmented)
        return augmented

    def _add_original_gt_in_augmented_frame(
        self,
        data: dict[str, np.ndarray],
        perturbation: AugumentedEgoPosition,
        source_data: dict[str, np.ndarray] | None = None,
    ) -> None:
        source_data = data if source_data is None else source_data
        origin = np.array([perturbation.x, perturbation.y], dtype=np.float32)
        for source_key, aux_key in (
            ("ego_agent_future", "original_ego_agent_future_in_augmented_frame"),
        ):
            source = source_data.get(source_key)
            if source is None:
                continue
            temp = {aux_key: np.array(source, copy=True)}
            self._transform_xy_heading_angle(
                temp, aux_key, origin, perturbation.yaw, use_mask=False
            )
            data[aux_key] = temp[aux_key]

        data["augmentation_perturbation"] = np.array(
            [perturbation.x, perturbation.y, perturbation.yaw], dtype=np.float32
        )

    @staticmethod
    def _add_default_aux_keys(data: dict[str, np.ndarray]) -> None:
        ego_future = data.get("ego_agent_future")
        if ego_future is not None:
            data["original_ego_agent_future_in_augmented_frame"] = np.array(ego_future, copy=True)
        data["augmentation_perturbation"] = np.zeros(3, dtype=np.float32)

    def _get_augment_ego_position(self, is_vehicle_stopping: bool) -> AugumentedEgoPosition:
        cfg = self.config
        if is_vehicle_stopping:
            x = float(np.random.uniform(-cfg.stop_lon_range, cfg.stop_lon_range))
        else:
            x = 0.0
        y = float(np.random.uniform(-cfg.lat_range, cfg.lat_range))
        theta = float(np.random.uniform(-cfg.yaw_range, cfg.yaw_range))
        return AugumentedEgoPosition(float(x), float(y), float(theta))

    def _augument_velocity_past(self, data: dict[str, np.ndarray]) -> None:
        ego_velocity_past = data.get("ego_velocity_past")
        past_length = ego_velocity_past.shape[0]
        veloctiy_scale = np.random.uniform(
            1.0 - self.config.velocity_scale_range,
            1.0 + self.config.velocity_scale_range,
            (past_length,),
        )
        ego_velocity_past *= veloctiy_scale[:, np.newaxis]

    def _connect_ego_future_with_hermite(self, data: dict[str, np.ndarray]) -> None:
        future = data.get("ego_agent_future")
        if future is None or future.shape[-1] < 3:
            return

        flat = future.reshape(-1, future.shape[-1])
        if flat.shape[0] == 0:
            return

        distances = self._future_sample_distances(flat)
        anchor_idx = self._hermite_anchor_index(distances)
        if anchor_idx is None:
            return
        if anchor_idx == 0:
            return

        anchor_xy = flat[anchor_idx, :2].astype(np.float64)
        anchor_distance = float(distances[anchor_idx])
        if anchor_distance <= 1e-6:
            return

        start_xy = np.zeros(2, dtype=np.float64)
        start_heading = 0.0
        end_heading = float(flat[anchor_idx, 2])
        start_tangent = anchor_distance * np.array(
            [np.cos(start_heading), np.sin(start_heading)], dtype=np.float64
        )
        end_tangent = anchor_distance * np.array(
            [np.cos(end_heading), np.sin(end_heading)], dtype=np.float64
        )

        sample_u = np.linspace(0.0, 1.0, anchor_idx + 1, dtype=np.float64)[:-1]

        for i, u in enumerate(sample_u):
            xy, tangent = self._cubic_hermite(start_xy, anchor_xy, start_tangent, end_tangent, u)
            flat[i, 0] = xy[0]
            flat[i, 1] = xy[1]
            if np.linalg.norm(tangent) > 1e-6:
                flat[i, 2] = np.arctan2(tangent[1], tangent[0])

    @staticmethod
    def _future_sample_distances(future: np.ndarray) -> np.ndarray:
        distances = np.zeros(future.shape[0], dtype=np.float64)
        if future.shape[0] == 0:
            return distances

        xy = future[:, :2].astype(np.float64)
        if future.shape[0] == 1:
            distances[0] = float(np.linalg.norm(xy[0]))
            return distances

        segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        valid_segment_lengths = segment_lengths[segment_lengths > 1e-6]
        if valid_segment_lengths.size == 0:
            distances[0] = float(np.linalg.norm(xy[0]))
            return distances

        interval = float(np.median(valid_segment_lengths))
        return interval * (np.arange(future.shape[0], dtype=np.float64) + 1.0)

    def _hermite_anchor_index(self, distances: np.ndarray) -> int | None:
        valid = np.flatnonzero(distances > 1e-6)
        if valid.size == 0:
            return None

        connect_distance = max(0.0, float(self.config.hermite_connect_distance_m))
        if connect_distance <= 1e-6:
            return None

        candidates = np.flatnonzero(distances >= connect_distance)
        if candidates.size > 0:
            return int(candidates[0])
        return int(valid[-1])

    @staticmethod
    def _cubic_hermite(
        p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray, u: float
    ) -> tuple[np.ndarray, np.ndarray]:
        u2 = u * u
        u3 = u2 * u

        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2
        xy = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

        dh00 = 6.0 * u2 - 6.0 * u
        dh10 = 3.0 * u2 - 4.0 * u + 1.0
        dh01 = -6.0 * u2 + 6.0 * u
        dh11 = 3.0 * u2 - 2.0 * u
        tangent = dh00 * p0 + dh10 * m0 + dh01 * p1 + dh11 * m1
        return xy, tangent

    def _transform_scene_to_new_ego_frame(
        self, data: dict[str, np.ndarray], perturbation: AugumentedEgoPosition
    ) -> None:
        origin = np.array([perturbation.x, perturbation.y], dtype=np.float32)
        yaw = perturbation.yaw

        self._transform_xy_heading_angle(data, "ego_agent_future", origin, yaw, use_mask=False)
        self._transform_xy_heading_angle(data, "neighbor_agents_future", origin, yaw, use_mask=True)
        self._transform_xy_heading_angle(data, "goal_pose", origin, yaw, use_mask=True)

        self._transform_xy_heading_cossin(
            data, "neighbor_agents_past", origin, yaw, rotate_velocity=True
        )
        self._transform_xy_heading_cossin(
            data, "static_objects", origin, yaw, rotate_velocity=False
        )

        self._transform_lane_like(data, "lanes", origin, yaw)
        self._transform_lane_like(data, "route_lanes", origin, yaw)
        self._transform_points_only(data, "polygons", origin, yaw)
        self._transform_points_only(data, "line_strings", origin, yaw)

    def _reset_ego_current_state(
        self, data: dict[str, np.ndarray], perturbation: AugumentedEgoPosition
    ) -> None:
        state = data["ego_current_state"]
        state[0] = 0.0
        state[1] = 0.0
        state[2] = 1.0
        state[3] = 0.0

    def _transform_xy_heading_angle(
        self,
        data: dict[str, np.ndarray],
        key: str,
        origin: np.ndarray,
        yaw: float,
        use_mask: bool,
    ) -> None:
        value = data.get(key)
        if value is None or value.shape[-1] < 3:
            return
        if use_mask:
            mask = _valid_xy(value)
            value[..., :2][mask] = self._transform_points(value[..., :2][mask], origin, yaw)
            value[..., 2][mask] = _wrap_angle(value[..., 2][mask] - yaw)
        else:
            value[..., :2] = self._transform_points(value[..., :2], origin, yaw)
            value[..., 2] = _wrap_angle(value[..., 2] - yaw)

    def _transform_xy_heading_cossin(
        self,
        data: dict[str, np.ndarray],
        key: str,
        origin: np.ndarray,
        yaw: float,
        rotate_velocity: bool,
    ) -> None:
        value = data.get(key)
        if value is None or value.shape[-1] < 4:
            return
        mask = _valid_xy(value)
        value[..., :2][mask] = self._transform_points(value[..., :2][mask], origin, yaw)

        heading = np.arctan2(value[..., 3][mask], value[..., 2][mask]) - yaw
        value[..., 2][mask] = np.cos(heading)
        value[..., 3][mask] = np.sin(heading)

        if rotate_velocity and value.shape[-1] >= 6:
            value[..., 4:6][mask] = self._rotate_vectors(value[..., 4:6][mask], yaw)

    def _transform_lane_like(
        self, data: dict[str, np.ndarray], key: str, origin: np.ndarray, yaw: float
    ) -> None:
        value = data.get(key)
        if value is None or value.shape[-1] < 2:
            return

        mask = _valid_xy(value)
        value[..., :2][mask] = self._transform_points(value[..., :2][mask], origin, yaw)

        for start in (2, 4, 6):
            end = start + 2
            if value.shape[-1] >= end:
                value[..., start:end][mask] = self._rotate_vectors(value[..., start:end][mask], yaw)

    def _transform_points_only(
        self, data: dict[str, np.ndarray], key: str, origin: np.ndarray, yaw: float
    ) -> None:
        value = data.get(key)
        if value is None or value.shape[-1] < 2:
            return
        mask = _valid_xy(value)
        value[..., :2][mask] = self._transform_points(value[..., :2][mask], origin, yaw)

    @staticmethod
    def _transform_points(points: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
        return StatePerturbation._rotate_vectors(points - origin.astype(points.dtype), yaw)

    @staticmethod
    def _rotate_vectors(vectors: np.ndarray, yaw: float) -> np.ndarray:
        c = np.cos(yaw)
        s = np.sin(yaw)
        x = vectors[..., 0].copy()
        y = vectors[..., 1].copy()
        out = np.empty_like(vectors)
        out[..., 0] = c * x + s * y
        out[..., 1] = -s * x + c * y
        return out
