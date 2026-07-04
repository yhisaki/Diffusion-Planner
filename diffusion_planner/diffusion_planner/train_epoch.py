import os

import torch
from torch import nn
from tqdm import tqdm

from diffusion_planner.model.module.decoder import compute_training_loss
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.train_utils import get_epoch_mean_loss


def heading_to_cos_sin(x):
    """
    Convert heading angle to cosine and sine.
    Args:
        x: [B, T, 3] where last dimension is (x, y, heading)
    Output:
        x: [B, T, 4] where last dimension is (x, y, cos(heading), sin(heading))
    """
    return torch.cat(
        [
            x[..., :2],
            x[..., 2:3].cos(),
            x[..., 2:3].sin(),
        ],
        dim=-1,
    )


def train_epoch(
    data_loader,
    model,
    optimizer,
    args,
    ema,
    aug: StatePerturbation = None,
    scheduler=None,
    epoch=0,
    save_path=None,
    wandb_id=None,
    save_step_interval=100,
):
    epoch_loss = []
    recent_loss = []

    model.train()

    if args.ddp:
        torch.cuda.synchronize()

    is_main = ddp.get_rank() == 0

    iterator = data_loader
    if is_main:
        iterator = tqdm(data_loader, desc="Training", unit="batch")

    for step, inputs in enumerate(iterator):
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
        inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

        ego_future = inputs["ego_agent_future"]
        neighbors_future = inputs["neighbor_agents_future"]
        # Normalize to ego-centric
        if aug is not None:
            inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

        # heading to cos sin
        ego_future = heading_to_cos_sin(ego_future)

        mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
        neighbors_future = heading_to_cos_sin(neighbors_future)
        neighbors_future[mask] = 0.0
        inputs = args.observation_normalizer(inputs)

        # call the model
        optimizer.zero_grad()

        loss = compute_training_loss(model, inputs, (ego_future, neighbors_future, mask), args)

        loss["loss"] = (
            args.alpha_neighbor_loss * loss["neighbor_prediction_loss"]
            + args.alpha_planning_loss * loss["ego_planning_loss"]
            + loss["turn_indicator_loss"]
            + args.coeff_road_border_loss * loss["road_border_loss"]
            + args.coeff_neighbor_collision_loss * loss["neighbor_collision_loss"]
        )

        # loss backward
        loss["loss"].backward()

        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        ema.update(model)

        if args.ddp:
            torch.cuda.synchronize()
        epoch_loss.append(loss)
        recent_loss.append(loss)

        # Periodic (per-step) loss print and checkpoint save on the main process
        if is_main and save_step_interval > 0 and (step + 1) % save_step_interval == 0:
            recent_mean_loss = get_epoch_mean_loss(recent_loss)
            recent_loss = []
            print(
                f"[epoch {epoch + 1} | step {step + 1}] "
                f"loss={recent_mean_loss['loss']:.4f} "
                f"turn_indicator_accuracy={recent_mean_loss['turn_indicator_accuracy']:.4f}"
            )

            if save_path is not None:
                model_dict = {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "ema_state_dict": ema.ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "schedule": scheduler.state_dict() if scheduler is not None else None,
                    "loss": recent_mean_loss["loss"],
                    "wandb_id": wandb_id,
                }
                torch.save(model_dict, os.path.join(save_path, "latest.pth"))

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"{epoch_mean_loss['loss']=:.4f}")
        print(f"{epoch_mean_loss['turn_indicator_accuracy']=:.4f}")

    return epoch_mean_loss, epoch_mean_loss["loss"]
