from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_kernel(x, y, tau):
    """Compute exponential kernel k(x, y) = exp(-||x - y|| / tau).

    Args:
        x: [..., D] query points
        y: [..., D] key points (broadcastable with x)
        tau: temperature scalar

    Returns:
        kernel: [...] similarity values
    """
    dist = torch.linalg.norm(x - y, dim=-1)
    return torch.exp(-dist / tau)


def compute_drifting_field_single_temperature(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    tau: float,
    ignore_self_negatives: bool = False,
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
    feature_extractor: Optional[Callable] = None,
    ignore_self_negatives: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute drifting loss per Algorithm 1 in the paper.

    L = E[||x - stopgrad(x + V(x))||^2]

    Args:
        x: [N, D] generated samples f(e)
        y_pos: [M, D] positive samples from data distribution
        y_neg: [K, D] negative samples from generated distribution
        temperatures: list of temperature values
        feature_extractor: optional feature extractor phi for feature-space loss
        ignore_self_negatives: whether to mask the diagonal when y_neg is x

    Returns:
        loss: scalar drifting loss
        drift_norm: scalar ||V||^2 for monitoring
    """
    if feature_extractor is not None:
        x_feat = feature_extractor(x)
        y_pos_feat = feature_extractor(y_pos)
        y_neg_feat = feature_extractor(y_neg)
    else:
        x_feat = x
        y_pos_feat = y_pos
        y_neg_feat = y_neg

    if x_feat.dim() > 2:
        x_feat = _flatten_features(x_feat)
        y_pos_feat = _flatten_features(y_pos_feat)
        y_neg_feat = _flatten_features(y_neg_feat)

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


def compute_grouped_drifting_loss(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    valid_mask: torch.Tensor,
    temperatures: List[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute drifting loss independently for each batch item.

    Args:
        x: [B, P, D] generated samples
        y_pos: [B, P, D] positive samples
        y_neg: [B, P, D] negative samples
        valid_mask: [B, P] valid query/sample mask
        temperatures: list of temperature values

    Returns:
        loss: scalar drifting loss
        drift_norm: scalar drift norm
    """
    losses = []
    drift_norms = []

    for x_i, y_pos_i, y_neg_i, valid_i in zip(x, y_pos, y_neg, valid_mask):
        if not torch.any(valid_i):
            continue

        x_valid = x_i[valid_i]
        y_pos_valid = y_pos_i[valid_i]
        y_neg_valid = y_neg_i[valid_i]

        loss_i, drift_norm_i = compute_drifting_loss(
            x_valid,
            y_pos_valid,
            y_neg_valid,
            temperatures,
            ignore_self_negatives=True,
        )
        losses.append(loss_i)
        drift_norms.append(drift_norm_i)

    if not losses:
        zero = x.sum() * 0.0
        return zero, zero

    return torch.stack(losses).mean(), torch.stack(drift_norms).mean()


def compute_masked_drifting_loss(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    valid_mask: torch.Tensor,
    temperatures: List[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute drifting loss over all valid samples in a mini-batch.

    Args:
        x: [..., D] generated samples
        y_pos: [..., D] positive samples
        y_neg: [..., D] negative samples
        valid_mask: [...] valid sample mask
        temperatures: list of temperature values

    Returns:
        loss: scalar drifting loss
        drift_norm: scalar drift norm
    """
    x_flat = x.reshape(-1, x.shape[-1])
    y_pos_flat = y_pos.reshape(-1, y_pos.shape[-1])
    y_neg_flat = y_neg.reshape(-1, y_neg.shape[-1])
    valid_flat = valid_mask.reshape(-1)

    if not torch.any(valid_flat):
        zero = x.sum() * 0.0
        return zero, zero

    return compute_drifting_loss(
        x_flat[valid_flat],
        y_pos_flat[valid_flat],
        y_neg_flat[valid_flat],
        temperatures,
        ignore_self_negatives=True,
    )


def compute_drifting_loss_multi_scale(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperatures: List[float],
    feature_extractor: Optional[Callable] = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute drifting loss with multi-scale features (Eq. 7 in paper).

    For feature extractors that return features at multiple scales/locations,
    compute drifting loss at each scale and sum.

    Args:
        x: [N, ...] generated samples
        y_pos: [M, ...] positive samples
        y_neg: [K, ...] negative samples
        temperatures: list of temperature values
        feature_extractor: feature extractor that returns list of feature tensors

    Returns:
        loss: scalar total drifting loss
        drift_norms: dict of drift norms per scale
    """
    x_feats: List[torch.Tensor]
    y_pos_feats: List[torch.Tensor]
    y_neg_feats: List[torch.Tensor]

    if feature_extractor is not None:
        x_feats_raw = feature_extractor(x)
        y_pos_feats_raw = feature_extractor(y_pos)
        y_neg_feats_raw = feature_extractor(y_neg)

        if isinstance(x_feats_raw, (list, tuple)):
            x_feats = list(x_feats_raw)
            y_pos_feats = list(y_pos_feats_raw)
            y_neg_feats = list(y_neg_feats_raw)
        else:
            x_feats = [x_feats_raw]
            y_pos_feats = [y_pos_feats_raw]
            y_neg_feats = [y_neg_feats_raw]
    else:
        x_feats = [_flatten_features(x)]
        y_pos_feats = [_flatten_features(y_pos)]
        y_neg_feats = [_flatten_features(y_neg)]

    total_loss: torch.Tensor = torch.tensor(0.0, device=x.device)
    drift_norms = {}

    for j, (x_f, y_pos_f, y_neg_f) in enumerate(zip(x_feats, y_pos_feats, y_neg_feats)):
        x_f = _flatten_features(x_f)
        y_pos_f = _flatten_features(y_pos_f)
        y_neg_f = _flatten_features(y_neg_f)

        V = compute_drifting_field(
            x_f,
            y_pos_f,
            y_neg_f,
            temperatures,
            ignore_self_negatives=True,
        )

        drift_norm = (V ** 2).sum(dim=-1).mean()
        drift_norms[f"drift_norm_scale_{j}"] = drift_norm

        x_drifted = x_f + V
        scale_loss = ((x_f - x_drifted.detach()) ** 2).sum(dim=-1).mean()
        total_loss = total_loss + scale_loss

    return total_loss, drift_norms
