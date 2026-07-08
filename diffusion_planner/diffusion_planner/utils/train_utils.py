import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

StateDict = dict[str, torch.Tensor]
Checkpoint = dict[str, Any]
PathLike = str | Path


def openjson(path: PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        dict = json.load(f)
    return dict


def set_seed(CUR_SEED: int) -> None:
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_params_for_weight_decay(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split trainable parameters into decay / no-decay groups.

    Weight decay is applied only to matrix-like weights (e.g. ``nn.Linear`` /
    ``nn.Conv`` weights). It is *not* applied to:

    * biases and other 1D tensors,
    * normalization parameters (LayerNorm/BatchNorm weight & bias — 1D),
    * ``nn.Embedding`` weights,
    * bare ``nn.Parameter`` tensors such as positional embeddings, queries and
      tokens (their leaf name does not end with ``"weight"``).

    Returns:
        A tuple ``(decay_params, no_decay_params)``.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    seen: set[int] = set()

    for module in model.modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))

            if (
                param.ndim <= 1
                or isinstance(module, nn.Embedding)
                or not param_name.endswith("weight")
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    return decay_params, no_decay_params


@torch.no_grad()
def compute_params_norm(
    params: list[nn.Parameter] | nn.Parameter,
    norm_type: float = 2.0,
) -> float:
    """Compute the global norm of a collection of parameter *values*.

    This matches the aggregation used by ``clip_grad_norm_``: the per-parameter
    norms are stacked and reduced with the same ``norm_type``, so for
    ``norm_type=2`` the result equals ``sqrt(sum(p**2))`` over all elements.

    Args:
        params: Parameters (or a single parameter) whose values are measured.
        norm_type: Order of the norm.

    Returns:
        The global norm as a python float (0.0 if there are no parameters).
    """
    if isinstance(params, nn.Parameter):
        params = [params]
    params = [p for p in params if p is not None]
    if not params:
        return 0.0

    device = params[0].device
    per_param = torch.stack([p.detach().norm(norm_type).to(device) for p in params])
    return torch.norm(per_param, norm_type).item()


def compute_weight_decay_norms(model: nn.Module, norm_type: float = 2.0) -> dict[str, float]:
    """Compute parameter-value norms split by weight-decay group.

    Useful for monitoring the effect of weight decay: the ``"decay"`` norm is
    expected to shrink relative to a run without weight decay, while the
    ``"no_decay"`` group is left untouched by weight decay.

    Args:
        model: Model to measure.
        norm_type: Order of the norm.

    Returns:
        Dict with keys ``"decay"``, ``"no_decay"`` and ``"total"``.
    """
    decay_params, no_decay_params = split_params_for_weight_decay(model)
    return {
        "decay": compute_params_norm(decay_params, norm_type),
        "no_decay": compute_params_norm(no_decay_params, norm_type),
        "total": compute_params_norm(decay_params + no_decay_params, norm_type),
    }


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    total_steps: int,
    warmup_steps: int = 0,
    lr_min_ratio: float = 0.0,
    weight_decay: float = 0.01,
) -> tuple[Optimizer, LambdaLR]:
    """Build an AdamW optimizer with a linear-warmup + cosine-decay LR schedule.

    The scheduler is expected to be stepped once per optimizer update (i.e. once
    per batch), so ``total_steps`` must be the total number of training steps
    (``train_epochs * len(train_loader)``).

    Weight decay is applied only to appropriate parameters
    (see :func:`split_params_for_weight_decay`).

    Args:
        model: Model whose ``requires_grad`` parameters are optimized.
        learning_rate: Base (peak) learning rate reached at the end of warmup.
        total_steps: Total number of optimizer steps over the whole run.
        warmup_steps: Number of steps to linearly warm up from 0 to the base LR.
        lr_min_ratio: Final LR as a fraction of the base LR at the end of decay.
        weight_decay: Weight decay applied to the decay parameter group.

    Returns:
        A tuple ``(optimizer, scheduler)``.
    """
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= lr_min_ratio <= 1.0:
        raise ValueError("lr_min_ratio must be in [0, 1]")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")

    decay_params, no_decay_params = split_params_for_weight_decay(model)
    if not decay_params and not no_decay_params:
        raise RuntimeError("No trainable parameters found")

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    optimizer = optim.AdamW(param_groups, lr=learning_rate)

    def lr_lambda(step: int) -> float:
        # Linear warmup.
        if step < warmup_steps:
            return step / warmup_steps
        # Cosine decay from 1.0 down to lr_min_ratio over the remaining steps.
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 0:
            return 1.0
        progress = (step - warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler


def get_epoch_mean_loss(epoch_loss: list[dict[str, Any]]) -> dict[str, float]:
    epoch_mean_loss: dict[str, list[float]] = {}
    for current_loss in epoch_loss:
        for key, value in current_loss.items():
            scalar = value if isinstance(value, (int, float)) else value.item()
            if key in epoch_mean_loss:
                epoch_mean_loss[key].append(float(scalar))
            else:
                epoch_mean_loss[key] = [float(scalar)]

    result: dict[str, float] = {}
    for key, values in epoch_mean_loss.items():
        result[key] = float(np.mean(np.array(values)))

    return result


def get_model(model: nn.Module) -> nn.Module:
    while isinstance(model, DDP) or hasattr(model, "_orig_mod"):
        if isinstance(model, DDP):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
    return model


def normalize_state_dict_keys(state_dict: StateDict) -> StateDict:
    normalized: StateDict = {}
    for k, v in state_dict.items():
        while k.startswith(("module.", "_orig_mod.")):
            if k.startswith("module."):
                k = k.removeprefix("module.")
            elif k.startswith("_orig_mod."):
                k = k.removeprefix("_orig_mod.")
        k = k.replace("._orig_mod.", ".")
        normalized[k] = v
    return normalized


def get_model_state_dict(model: nn.Module) -> StateDict:
    return normalize_state_dict_keys(get_model(model).state_dict())


def get_checkpoint_state_dict(ckpt: Checkpoint | StateDict) -> StateDict:
    if "model" in ckpt:
        return ckpt["model"]
    if "ema_state_dict" in ckpt:
        return ckpt["ema_state_dict"]
    return ckpt


def resume_model(
    path: PathLike,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    ema: Any,
    device: torch.device | str,
) -> (
    tuple[nn.Module, Optimizer, int, str | None, Any]
    | tuple[nn.Module, Optimizer, Any, int, str | None, Any]
):
    """
    load ckpt from path
    """
    ckpt = torch.load(path, map_location=device)

    # load model
    model.load_state_dict(normalize_state_dict_keys(get_checkpoint_state_dict(ckpt)))
    print("Model load done")

    # load optimizer
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("Optimizer load done")
    except:
        print("no pretrained optimizer found")

    if scheduler is not None:
        # load schedule
        try:
            scheduler.load_state_dict(ckpt["schedule"])
            print("Schedule load done")
        except:
            print("no schedule found,")

    # load step
    try:
        init_epoch = ckpt["epoch"]
        print("Step load done")
    except:
        init_epoch = 0

    # Load wandb id
    try:
        wandb_id = ckpt["wandb_id"]
        print("wandb id load done")
    except:
        wandb_id = None

    try:
        ema.ema.load_state_dict(ckpt["ema_state_dict"])
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        print("ema load done")
    except:
        print("no ema shadow found")

    if scheduler is not None:
        return model, optimizer, scheduler, init_epoch, wandb_id, ema
    return model, optimizer, init_epoch, wandb_id, ema


def resume_encoder_model(path: PathLike, model: nn.Module, device: torch.device | str) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    state_dict = normalize_state_dict_keys(get_checkpoint_state_dict(ckpt))

    cleaned: StateDict = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            cleaned[k.replace("encoder.", "", 1)] = v

    if not cleaned:
        raise RuntimeError(f"No encoder weights found in checkpoint: {path}")

    model.encoder.load_state_dict(cleaned)
    print(f"Encoder loaded from {path} ({len(cleaned)} keys)")

    return model
