import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

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
    scheduler: Any,
    ema: Any,
    device: torch.device | str,
) -> tuple[nn.Module, Optimizer, Any, int, str | None, Any]:
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

    return model, optimizer, scheduler, init_epoch, wandb_id, ema


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
