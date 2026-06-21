import numpy as np
from diffusion_planner.utils.data_augmentation import EgoPerturbation, StatePerturbation


def _make_data() -> dict[str, np.ndarray]:
    data = {
        "ego_current_state": np.zeros(10, dtype=np.float32),
        "ego_shape": np.array([3.2, 4.8, 1.9], dtype=np.float32),
        "ego_agent_future": np.zeros((8, 3), dtype=np.float32),
        "neighbor_agents_future": np.zeros((2, 8, 3), dtype=np.float32),
        "goal_pose": np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "neighbor_agents_past": np.zeros((2, 4, 11), dtype=np.float32),
        "static_objects": np.zeros((2, 10), dtype=np.float32),
        "lanes": np.zeros((2, 4, 8), dtype=np.float32),
        "route_lanes": np.zeros((2, 4, 8), dtype=np.float32),
        "polygons": np.zeros((2, 4, 3), dtype=np.float32),
        "line_strings": np.zeros((2, 4, 4), dtype=np.float32),
    }
    data["ego_current_state"][2] = 1.0
    data["ego_current_state"][4] = 5.0
    data["ego_agent_future"][:, 0] = np.arange(1, 9, dtype=np.float32)
    data["lanes"][0, :, 0] = np.arange(1, 5, dtype=np.float32)
    data["lanes"][0, :, 2] = 1.0
    data["route_lanes"][0, :, 0] = np.arange(1, 5, dtype=np.float32)
    data["route_lanes"][0, :, 2] = 1.0
    return data


def test_augment_prob_zero_returns_original_object():
    data = _make_data()
    aug = StatePerturbation(augment_prob=0.0)

    result = aug(data)

    assert result is data


def test_slow_ego_returns_original_object():
    data = _make_data()
    data["ego_current_state"][4] = 0.5
    aug = StatePerturbation(augment_prob=1.0, min_speed=1.0)

    result = aug(data)

    assert result is data


def test_augmentation_copies_input_and_resets_current_ego_frame():
    data = _make_data()
    original_future = data["ego_agent_future"].copy()
    aug = StatePerturbation(
        augment_prob=1.0,
        lateral_offset_std=0.0,
        yaw_half_range=0.0,
    )

    result = aug(data)

    assert result is not data
    np.testing.assert_array_equal(data["ego_agent_future"], original_future)
    np.testing.assert_allclose(result["ego_current_state"][:4], [0.0, 0.0, 1.0, 0.0])
    assert not np.allclose(result["ego_agent_future"], original_future)


def test_augment_with_aux_records_original_gt_in_augmented_frame():
    data = _make_data()
    aug = StatePerturbation(
        augment_prob=1.0,
        lateral_offset_std=0.0,
        yaw_half_range=0.0,
    )

    result = aug.augment_with_aux(data)

    assert "original_ego_agent_future_in_augmented_frame" in result
    assert "augmentation_perturbation" in result
    np.testing.assert_allclose(result["augmentation_perturbation"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        result["original_ego_agent_future_in_augmented_frame"][:, :2],
        data["ego_agent_future"][:, :2],
    )


def test_configurable_perturbation_half_range_is_used():
    current_state = np.zeros(10, dtype=np.float32)
    current_state[4] = 5.0
    aug = StatePerturbation(
        lateral_offset_std=0.0,
        yaw_half_range=0.0,
        speed_scale_half_range=0.0,
    )

    perturbation = aug._augment_ego_current(current_state)

    assert perturbation == EgoPerturbation(x=0.0, y=0.0, yaw=0.0, speed=5.0)


def test_speed_perturbation_changes_ego_current_state():
    data = _make_data()
    aug = StatePerturbation(
        augment_prob=1.0,
        lateral_offset_std=0.0,
        yaw_half_range=0.0,
        speed_scale_half_range=0.1,
    )

    np.random.seed(42)
    result = aug(data)

    original_speed = float(np.linalg.norm(_make_data()["ego_current_state"][4:6]))
    result_speed = float(np.linalg.norm(result["ego_current_state"][4:6]))
    assert not np.isclose(result_speed, original_speed)


def test_speed_perturbation_affects_augmented_future():
    data = _make_data()
    data["ego_agent_future"][:, 0] = np.arange(1, 9, dtype=np.float32)
    data["ego_agent_future"][:, 1] = 0.0
    data["ego_agent_future"][:, 2] = 0.0

    aug_no_speed = StatePerturbation(
        augment_prob=1.0,
        lateral_offset_std=0.0,
        yaw_half_range=0.0,
        speed_scale_half_range=0.0,
    )
    aug_with_speed = StatePerturbation(
        augment_prob=1.0,
        lateral_offset_std=0.0,
        yaw_half_range=0.0,
        speed_scale_half_range=0.1,
    )

    result_no_speed = aug_no_speed(data)
    data2 = _make_data()
    data2["ego_agent_future"][:, 0] = np.arange(1, 9, dtype=np.float32)
    data2["ego_agent_future"][:, 1] = 0.0
    data2["ego_agent_future"][:, 2] = 0.0
    result_with_speed = aug_with_speed(data2)

    assert not np.allclose(
        result_no_speed["ego_agent_future"], result_with_speed["ego_agent_future"]
    )


def test_default_wheel_base_is_configurable_when_ego_shape_missing():
    aug = StatePerturbation(default_wheel_base=2.7)

    assert aug._ego_wheel_base({}) == 2.7
