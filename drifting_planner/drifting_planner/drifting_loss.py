import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_drifting_field_single_temperature(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    tau: float,
    ignore_self_negatives: bool = False,
    normalize_distance_by_dim: bool = True,
) -> torch.Tensor:
    """Compute the Algorithm A1 drifting field for one temperature.

    The paper normalizes the concatenated positive/negative logits across both
    the sample and query axes, then reweights positive and negative terms
    jointly. This preserves the anti-symmetric field construction better than
    normalizing attraction and repulsion independently.

    Args:
        x: [N, D] query points (generated samples)
        y_pos: [N_pos, D] positive samples from data distribution
        y_neg: [N_neg, D] negative samples from generated distribution
        tau: temperature scalar
        ignore_self_negatives: whether to mask the diagonal when y_neg is x

    Returns:
        field: [N, D] drifting field
    """
    dist_pos = torch.cdist(x, y_pos)
    dist_neg = torch.cdist(x, y_neg)
    if normalize_distance_by_dim:
        distance_scale = math.sqrt(x.shape[-1])
        dist_pos = dist_pos / distance_scale
        dist_neg = dist_neg / distance_scale

    if ignore_self_negatives and x.shape[0] == y_neg.shape[0] and x.shape[0] > 1:
        eye = torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
        dist_neg = dist_neg.masked_fill(eye, 1e6)

    logits = torch.cat([-dist_pos / tau, -dist_neg / tau], dim=1)
    a_row = F.softmax(logits, dim=1)
    a_col = F.softmax(logits, dim=0)
    weights = torch.sqrt(a_row * a_col)

    n_pos = y_pos.shape[0]
    w_pos, w_neg = torch.split(weights, [n_pos, y_neg.shape[0]], dim=1)
    pos_mass = w_pos.sum(dim=1, keepdim=True)
    neg_mass = w_neg.sum(dim=1, keepdim=True)
    w_pos = w_pos * neg_mass
    w_neg = w_neg * pos_mass

    drift_pos = w_pos @ y_pos
    drift_neg = w_neg @ y_neg
    return drift_pos - drift_neg


def compute_drifting_field(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperatures: List[float],
    ignore_self_negatives: bool = False,
    normalize_distance_by_dim: bool = True,
) -> torch.Tensor:
    """Compute anti-symmetric drifting field V_{p,q}(x).

    Supports multiple temperatures for multi-scale kernel computation.

    Args:
        x: [N, D] generated samples (query points)
        y_pos: [M, D] positive samples from data distribution
        y_neg: [K, D] negative samples from generated distribution
        temperatures: list of temperature values for multi-scale kernels
        ignore_self_negatives: whether to mask self negatives

    Returns:
        field: [N, D] combined drifting field
    """
    total_field: Optional[torch.Tensor] = None

    for tau in temperatures:
        field = compute_drifting_field_single_temperature(
            x,
            y_pos,
            y_neg,
            tau,
            ignore_self_negatives=ignore_self_negatives,
            normalize_distance_by_dim=normalize_distance_by_dim,
        )
        if total_field is None:
            total_field = field
        else:
            total_field = total_field + field

    assert total_field is not None
    return total_field / len(temperatures)


def _flatten_features(x: torch.Tensor) -> torch.Tensor:
    """Flatten tensor to [batch, features]."""
    return x.reshape(x.shape[0], -1)


def compute_drifting_loss(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperatures: List[float],
    ignore_self_negatives: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute drifting loss per Algorithm 1 in the paper.

    L = E[||x - stopgrad(x + V(x))||^2]

    Args:
        x: [N, D] generated samples f(e)
        y_pos: [M, D] positive samples from data distribution
        y_neg: [K, D] negative samples from generated distribution
        temperatures: list of temperature values
        ignore_self_negatives: whether to mask the diagonal when y_neg is x

    Returns:
        loss: scalar drifting loss
        drift_norm: scalar ||V||^2 for monitoring
    """
    x_feat = x
    y_pos_feat = y_pos
    y_neg_feat = y_neg

    if x_feat.dim() > 2:
        x_feat = _flatten_features(x_feat)
        y_pos_feat = _flatten_features(y_pos_feat)
        y_neg_feat = _flatten_features(y_neg_feat)

    if ignore_self_negatives and y_neg_feat.shape[0] < 2:
        raise ValueError("At least two negative samples are required when self negatives are ignored.")

    V = compute_drifting_field(
        x_feat,
        y_pos_feat,
        y_neg_feat,
        temperatures,
        ignore_self_negatives=ignore_self_negatives,
    )

    drift_norm = (V ** 2).sum(dim=-1).mean()

    x_drifted = x_feat + V

    loss = ((x_feat - x_drifted.detach()) ** 2).sum(dim=-1).mean()

    return loss, drift_norm


def compute_scene_drifting_loss(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    temperatures: List[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute drifting loss per conditional scene.

    Args:
        x: [B, K, D] generated samples for each conditioning scene
        y_pos: [B, M, D] positive data samples for each conditioning scene
        temperatures: list of temperature values

    Returns:
        loss: scalar drifting loss
        drift_norm: scalar drift norm
    """
    if x.dim() != 3 or y_pos.dim() != 3:
        raise ValueError("Expected x [B, K, D] and y_pos [B, M, D].")
    if x.shape[0] != y_pos.shape[0] or x.shape[-1] != y_pos.shape[-1]:
        raise ValueError("Generated and positive samples must share batch and feature dimensions.")
    if x.shape[1] < 2:
        raise ValueError("At least two generated samples are required for drifting negatives.")
    if y_pos.shape[1] < 1:
        raise ValueError("At least one positive sample is required.")

    losses = []
    drift_norms = []
    for x_i, y_pos_i in zip(x, y_pos):
        loss_i, drift_norm_i = compute_drifting_loss(
            x_i,
            y_pos_i,
            x_i,
            temperatures,
            ignore_self_negatives=True,
        )
        losses.append(loss_i)
        drift_norms.append(drift_norm_i)

    return torch.stack(losses).mean(), torch.stack(drift_norms).mean()
