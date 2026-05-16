import json
import random
from typing import Any

import numpy as np
import torch


def openjson(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d


def set_seed(CUR_SEED: int) -> None:
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_epoch_mean_loss(
    epoch_loss: list[dict[str, float | torch.Tensor]],
) -> dict[str, float]:
    epoch_mean_loss: dict[str, list[float]] = {}
    for current_loss in epoch_loss:
        for key, value in current_loss.items():
            if key in epoch_mean_loss:
                epoch_mean_loss[key].append(
                    value if isinstance(value, (int, float)) else value.item()
                )
            else:
                epoch_mean_loss[key] = [value if isinstance(value, (int, float)) else value.item()]

    result: dict[str, float] = {}
    for key, values in epoch_mean_loss.items():
        result[key] = float(np.mean(np.array(values)))

    return result


def resume_model(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    ema: Any,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, Any, int, str | None, Any]:
    """
    load ckpt from path
    """
    ckpt = torch.load(path, map_location=device)

    # load model
    try:
        model.load_state_dict(ckpt["model"])
    except:
        model.load_state_dict(ckpt)
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
    init_epoch: int
    try:
        init_epoch = ckpt["epoch"]
        print("Step load done")
    except:
        init_epoch = 0

    # Load wandb id
    wandb_id: str | None
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
