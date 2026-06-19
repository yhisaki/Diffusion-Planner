from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _valid_xy(x: np.ndarray) -> np.ndarray:
    return np.any(np.abs(x[..., :2]) > 1e-6, axis=-1)


@dataclass
class EgoPerturbation:
    x: float
    y: float
    yaw: float
    speed: float
    speed_scale: float = 1.0


@dataclass
class StatePerturbationConfig:
    augment_prob: float
    min_speed: float
    min_length: float
    time_interval: float
    max_steering_angle: float
    min_linearization_speed: float
    exact_position_gain: float
    exact_velocity_gain: float
    lateral_offset_std: float
    yaw_half_range: float
    default_wheel_base: float
    speed_scale_half_range: float


class StatePerturbation:
    """Scene-level ego-centric data augmentation.

    The raw dataset is already expressed in the current ego frame. This augmenter
    samples a small virtual current ego pose in that frame, bends only the ego
    history/future so they meet and leave that virtual pose smoothly, then
    rewrites every geometric field into the new virtual ego frame.
    """

    def __init__(
        self,
        augment_prob: float = 0.5,
        min_speed: float = 1.0,
        min_length: float = 15.0,
        time_interval: float = 0.1,
        max_steering_rate: float = 0.5,
        min_linearization_speed: float = 1.0,
        exact_position_gain: float = 1.0,
        exact_velocity_gain: float = 3.0,
        lateral_offset_std: float = 1.5,
        yaw_half_range: float = 0.2,
        default_wheel_base: float = 3.0,
        speed_scale_half_range: float = 0.2,
    ) -> None:
        self.config = StatePerturbationConfig(
            augment_prob=augment_prob,
            min_speed=min_speed,
            min_length=min_length,
            time_interval=time_interval,
            max_steering_angle=max_steering_rate,
            min_linearization_speed=min_linearization_speed,
            exact_position_gain=exact_position_gain,
            exact_velocity_gain=exact_velocity_gain,
            lateral_offset_std=lateral_offset_std,
            yaw_half_range=yaw_half_range,
            default_wheel_base=default_wheel_base,
            speed_scale_half_range=speed_scale_half_range,
        )

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self.augment(data)

    def augment(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self._augment(data, include_aux=False)

    def augment_with_aux(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self._augment(data, include_aux=True)

    def _augment(
        self, data: dict[str, np.ndarray], include_aux: bool = False
    ) -> dict[str, np.ndarray]:
        if np.random.random() >= self.config.augment_prob:
            return data
        if "ego_current_state" not in data:
            return data
        if abs(float(data["ego_current_state"][4])) < self.config.min_speed:
            return data

        future_length = self._future_trajectory_length(data)
        std_scale = min(1.0, future_length / self.config.min_length)
        cfg = self.config
        cfg.lateral_offset_std *= std_scale
        cfg.yaw_half_range *= std_scale

        augmented = {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            for key, value in data.items()
        }

        perturbation = self._augment_ego_current(data["ego_current_state"])
        augmented["ego_current_state"][4] = perturbation.speed
        self._scale_ego_past(augmented, perturbation.speed_scale)
        if include_aux:
            self._add_original_gt_in_augmented_frame(augmented, perturbation, source_data=data)
        self._rollout_ego_future_with_dynamics(augmented, perturbation)
        self._transform_scene_to_new_ego_frame(augmented, perturbation)
        self._reset_ego_current_state(augmented, perturbation)
        return augmented

    def _add_original_gt_in_augmented_frame(
        self,
        data: dict[str, np.ndarray],
        perturbation: EgoPerturbation,
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

    def _augment_ego_current(self, current_state: np.ndarray) -> EgoPerturbation:
        cfg = self.config
        x = 0.0
        # yaw and speed_scale use uniform distributions over [-half_range, half_range].
        # lateral_offset uses a normal distribution N(0, lateral_offset_std^2).
        y = float(np.random.normal(0.0, cfg.lateral_offset_std))
        theta = float(
            np.random.uniform(
                -cfg.yaw_half_range,
                cfg.yaw_half_range,
            )
        )
        speed_scale = max(
            0.0,
            float(
                np.random.uniform(
                    1.0 - cfg.speed_scale_half_range,
                    1.0 + cfg.speed_scale_half_range,
                )
            ),
        )
        current_speed = max(0.0, float(np.linalg.norm(current_state[4:6])))
        speed = current_speed * speed_scale
        return EgoPerturbation(float(x), float(y), float(theta), float(speed), float(speed_scale))

    def _scale_ego_past(self, data: dict[str, np.ndarray], speed_scale: float) -> None:
        past = data.get("ego_agent_past")
        if past is None or past.shape[-1] < 2:
            return
        past = past.reshape(-1, past.shape[-1])
        past[:, :2] *= speed_scale

    @staticmethod
    def _integrate_bicycle_velocity_steering_step(
        x: float,
        y: float,
        theta: float,
        velocity: float,
        steering: float,
        wheel_base: float,
        dt: float,
    ) -> tuple[float, float, float]:
        heading_delta = velocity / max(wheel_base, 1e-3) * np.tan(steering) * dt
        next_theta = float(_wrap_angle(theta + heading_delta))
        next_x = x + velocity * np.cos(next_theta) * dt
        next_y = y + velocity * np.sin(next_theta) * dt
        return float(next_x), float(next_y), next_theta

    @staticmethod
    def _integrate_bicycle_accel_steering_step(
        x: float,
        y: float,
        theta: float,
        velocity: float,
        acceleration: float,
        steering: float,
        wheel_base: float,
        dt: float,
    ) -> tuple[float, float, float, float]:
        next_velocity = max(0.0, velocity + acceleration * dt)
        next_x, next_y, next_theta = StatePerturbation._integrate_bicycle_velocity_steering_step(
            x, y, theta, next_velocity, steering, wheel_base, dt
        )
        return next_x, next_y, next_theta, float(next_velocity)

    def _rollout_ego_future_with_dynamics(
        self, data: dict[str, np.ndarray], perturbation: EgoPerturbation
    ) -> None:
        future = data.get("ego_agent_future")
        if future is None or future.shape[-1] < 3:
            return

        original = np.array(future, copy=True)
        states = self._dynamics_converging_future(
            original=original,
            current_state=data["ego_current_state"],
            wheel_base=self._ego_wheel_base(data),
            perturbation=perturbation,
        )
        future[...] = states.astype(future.dtype, copy=False)

    def _dynamics_converging_future(
        self,
        original: np.ndarray,
        current_state: np.ndarray,
        wheel_base: float,
        perturbation: EgoPerturbation,
    ) -> np.ndarray:
        cfg = self.config
        dt = cfg.time_interval
        n = original.shape[0]
        out = np.zeros_like(original)

        ref_position, ref_velocity, ref_acceleration = self._reference_states(
            original, current_state
        )
        x = perturbation.x
        y = perturbation.y
        theta = perturbation.yaw
        v = perturbation.speed

        for t in range(n):
            target_xy = ref_position[t]
            current_xy = np.array([x, y], dtype=np.float64)
            current_velocity = v * np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)

            position_error = current_xy - target_xy
            velocity_error = current_velocity - ref_velocity[t]
            virtual_input = (
                ref_acceleration[t]
                - cfg.exact_position_gain * position_error
                - cfg.exact_velocity_gain * velocity_error
            )

            cos_h = float(np.cos(theta))
            sin_h = float(np.sin(theta))
            ux = float(virtual_input[0])
            uy = float(virtual_input[1])
            linearization_speed = max(abs(v), cfg.min_linearization_speed)

            acceleration = cos_h * ux + sin_h * uy
            lateral_command = -sin_h * ux + cos_h * uy
            steering = np.arctan2(
                wheel_base * lateral_command, linearization_speed * linearization_speed
            )
            steering = np.clip(
                steering,
                -cfg.max_steering_angle,
                cfg.max_steering_angle,
            )
            x, y, theta, v = self._integrate_bicycle_accel_steering_step(
                x, y, theta, v, acceleration, steering, wheel_base, dt
            )

            out[t, 0] = x
            out[t, 1] = y
            out[t, 2] = theta

        return out

    def _reference_states(
        self, original: np.ndarray, current_state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        dt = cfg.time_interval
        current_xy = np.asarray(current_state[:2], dtype=np.float64)
        positions = np.vstack([current_xy[None], original[:, :2].astype(np.float64)])

        velocity = np.gradient(positions, dt, axis=0, edge_order=1)
        acceleration = np.gradient(velocity, dt, axis=0, edge_order=1)

        return positions[:-1], velocity[:-1], acceleration[:-1]

    def _ego_wheel_base(self, data: dict[str, np.ndarray]) -> float:
        ego_shape = data.get("ego_shape")
        if ego_shape is None or ego_shape.size == 0:
            return self.config.default_wheel_base
        return max(0.1, float(np.asarray(ego_shape).reshape(-1)[0]))

    @staticmethod
    def _future_trajectory_length(data: dict[str, np.ndarray]) -> float:
        if "ego_agent_future" not in data:
            return 0.0
        future = data["ego_agent_future"].reshape(-1, data["ego_agent_future"].shape[-1])
        if future.shape[0] < 2:
            return 0.0
        diffs = np.diff(future[:, :2].astype(np.float64), axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def _transform_scene_to_new_ego_frame(
        self, data: dict[str, np.ndarray], perturbation: EgoPerturbation
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
        self, data: dict[str, np.ndarray], perturbation: EgoPerturbation
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
