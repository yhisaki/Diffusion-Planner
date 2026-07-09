from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_DISABLE


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
    lat_range: float
    yaw_range: float
    velocity_scale_range: float
    turn_indicator_onset_prob: float


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
        lat_range: float = 1.0,
        yaw_range: float = 0.2,
        velocity_scale_range: float = 0.6,
        turn_indicator_onset_prob: float = 0.0,
    ) -> None:
        self.config = StatePerturbationConfig(
            path_augment_prob=path_augment_prob,
            velocity_augment_prob=velocity_augment_prob,
            lat_range=lat_range,
            yaw_range=yaw_range,
            velocity_scale_range=velocity_scale_range,
            turn_indicator_onset_prob=turn_indicator_onset_prob,
        )

    def augment(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        augmented = {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            for key, value in data.items()
        }

        if np.random.random() <= self.config.path_augment_prob:
            augumented_ego_pos = self._get_augment_ego_position()
            self._add_original_gt_in_augmented_frame(
                augmented, augumented_ego_pos, source_data=data
            )
            self._transform_scene_to_new_ego_frame(augmented, augumented_ego_pos)
            self._reset_ego_current_state(augmented, augumented_ego_pos)
        else:
            self._add_default_aux_keys(augmented)

        if np.random.random() <= self.config.velocity_augment_prob:
            self._augument_velocity_past(augmented)

        if np.random.random() <= self.config.turn_indicator_onset_prob:
            self._augment_turn_indicator_onset(augmented)
        return augmented

    @staticmethod
    def _augment_turn_indicator_onset(data: dict[str, np.ndarray]) -> None:
        """Turn a steady turn-signal sample into a signal *onset* sample.

        Overwrites the whole turn indicator history with straight (DISABLE)
        *except the current step* (index -1). The encoder, which reads index -2,
        therefore sees a straight signal, while ``make_turn_indicator_gt``
        (comparing indices -2 and -1) now sees a change and yields the current
        turn class instead of KEEP. This teaches the planner to emit a turn
        signal from the trajectory even when the recent signal was straight
        (e.g. left turn + currently left -> input straight, GT left, not keep).
        """
        turn_indicators = data.get("turn_indicators")
        if turn_indicators is None:
            return
        turn_indicators[:-1] = TURN_INDICATOR_OUTPUT_DISABLE

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

    def _get_augment_ego_position(self) -> AugumentedEgoPosition:
        cfg = self.config
        y = float(np.random.uniform(-cfg.lat_range, cfg.lat_range))
        theta = float(np.random.uniform(-cfg.yaw_range, cfg.yaw_range))
        return AugumentedEgoPosition(0.0, y, float(theta))

    def _augument_velocity_past(self, data: dict[str, np.ndarray]) -> None:
        ego_velocity_past = data.get("ego_velocity_past")
        past_length = ego_velocity_past.shape[0]
        veloctiy_scale = np.random.uniform(
            1.0 - self.config.velocity_scale_range,
            1.0 + self.config.velocity_scale_range,
            (past_length,),
        )
        ego_velocity_past *= veloctiy_scale[:, np.newaxis]

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
