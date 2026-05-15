import pytest
import torch
from drifting_planner.drifting_loss import (
    compute_drifting_field,
    compute_drifting_loss,
    compute_scene_drifting_loss,
)


def test_drifting_field_is_zero_for_identical_distributions_without_self_mask():
    torch.manual_seed(0)
    x = torch.randn(4, 8)

    field = compute_drifting_field(
        x,
        x,
        x,
        temperatures=[0.2],
        ignore_self_negatives=False,
    )

    assert torch.allclose(field, torch.zeros_like(field), atol=1e-6)


def test_drifting_loss_rejects_single_negative_with_self_mask():
    x = torch.randn(1, 8)

    with pytest.raises(ValueError, match="At least two negative samples"):
        compute_drifting_loss(
            x,
            x,
            x,
            temperatures=[0.2],
            ignore_self_negatives=True,
        )


def test_drifting_loss_gradients_only_flow_through_prediction():
    torch.manual_seed(1)
    x = torch.randn(3, 8, requires_grad=True)
    y_pos = torch.randn(4, 8, requires_grad=True)

    loss, drift_norm = compute_drifting_loss(
        x,
        y_pos,
        x,
        temperatures=[0.2],
        ignore_self_negatives=True,
    )
    loss.backward()

    assert loss.detach().item() == pytest.approx(drift_norm.detach().item())
    assert x.grad is not None
    assert torch.any(x.grad != 0)
    assert y_pos.grad is None


def test_scene_drifting_loss_uses_generated_samples_as_negatives():
    torch.manual_seed(2)
    x = torch.randn(2, 4, 6, requires_grad=True)
    y_pos = torch.randn(2, 2, 6)

    loss, drift_norm = compute_scene_drifting_loss(x, y_pos, temperatures=[0.2, 0.5])
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(drift_norm)
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_scene_drifting_loss_rejects_too_few_generated_samples():
    x = torch.randn(2, 1, 6)
    y_pos = torch.randn(2, 1, 6)

    with pytest.raises(ValueError, match="At least two generated samples"):
        compute_scene_drifting_loss(x, y_pos, temperatures=[0.2])


def test_high_dimensional_scene_loss_does_not_underflow_to_zero():
    torch.manual_seed(3)
    x = torch.zeros(2, 2, 1024, requires_grad=True)
    y_pos = torch.randn(2, 1, 1024)

    loss, drift_norm = compute_scene_drifting_loss(
        x,
        y_pos,
        temperatures=[0.5, 1.0, 2.0],
    )
    loss.backward()

    assert loss.detach().item() > 0
    assert drift_norm.detach().item() > 0
    assert x.grad is not None
    assert torch.any(x.grad != 0)

