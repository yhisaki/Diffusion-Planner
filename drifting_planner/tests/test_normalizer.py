import torch

from drifting_planner.utils.normalizer import TrajectoryNormalizer


def test_trajectory_normalizer_round_trip_with_multiple_positive_samples():
    mean = torch.tensor(
        [
            [[1.0, 2.0, 0.5, -0.5]],
            [[0.25, -0.75, 0.1, 0.2]],
        ]
    )
    std = torch.tensor(
        [
            [[2.0, 4.0, 0.5, 0.25]],
            [[8.0, 16.0, 2.0, 4.0]],
        ]
    )
    normalizer = TrajectoryNormalizer(mean, std)

    current = torch.tensor(
        [
            [[10.0, 20.0, 1.0, 0.0], [100.0, 200.0, 0.0, 1.0]],
            [[-5.0, 2.0, 1.0, 0.0], [7.0, -11.0, 0.0, 1.0]],
        ]
    )
    future = torch.randn(2, 3, 2, 4, 4)
    future[..., :2] += current[:, None, :, None, :2]

    normalized = normalizer.normalize_future(future, current)
    restored = normalizer.inverse_future(normalized, current)

    assert torch.allclose(restored, future, atol=1e-6)


def test_trajectory_normalizer_uses_delta_xy_not_absolute_xy():
    mean = torch.zeros(1, 1, 4)
    std = torch.ones(1, 1, 4)
    normalizer = TrajectoryNormalizer(mean, std)

    current = torch.tensor([[[100.0, -50.0, 1.0, 0.0]]])
    future = torch.tensor([[[[103.0, -48.0, 0.0, 1.0]]]])

    normalized = normalizer.normalize_future(future, current)

    assert torch.allclose(normalized[..., :2], torch.tensor([[[[3.0, 2.0]]]]))
