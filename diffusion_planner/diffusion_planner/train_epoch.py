import time

import torch
from torch import nn

from diffusion_planner.loss import compute_training_loss
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.train_utils import get_epoch_mean_loss, get_model, get_model_state_dict


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
    aug: StatePerturbation | None = None,
    log_interval: int = 100,
    epoch: int = 0,
    save_path: str | None = None,
    scheduler=None,
):
    print(f"Epoch {epoch + 1} training started.")
    epoch_loss = []

    model.train()

    scaler = torch.amp.GradScaler("cuda", enabled=getattr(args, "use_amp", False))

    if args.ddp:
        torch.cuda.synchronize()

    log_start_time = time.perf_counter()

    for batch_idx, inputs in enumerate(data_loader):
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
        inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

        ego_future = inputs["ego_agent_future"]
        neighbors_future = inputs["neighbor_agents_future"]

        # heading to cos sin
        ego_future = heading_to_cos_sin(ego_future)

        mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
        neighbors_future = heading_to_cos_sin(neighbors_future)
        neighbors_future[mask] = 0.0
        inputs = args.observation_normalizer(inputs)

        # call the model
        optimizer.zero_grad()

        use_amp = getattr(args, "use_amp", False)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = compute_training_loss(model, inputs, (ego_future, neighbors_future, mask), args)

            loss["loss"] = (
                args.alpha_neighbor_loss * loss["neighbor_prediction_loss"]
                + args.alpha_planning_loss * loss["ego_planning_loss"]
                + loss["turn_indicator_loss"]
            )

        # loss backward
        scaler.scale(loss["loss"]).backward()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5)

        scaler.step(optimizer)
        scaler.update()

        ema.update(get_model(model))

        if args.ddp:
            torch.cuda.synchronize()
        epoch_loss.append(loss)

        if ddp.get_rank() == 0 and (batch_idx + 1) % log_interval == 0:
            elapsed_sec = time.perf_counter() - log_start_time
            log_start_time = time.perf_counter()
            recent_losses = epoch_loss[-log_interval:]
            avg_loss = sum(l["loss"].item() for l in recent_losses) / len(recent_losses)
            avg_ego_planning = sum(l["ego_planning_loss"].item() for l in recent_losses) / len(
                recent_losses
            )
            avg_neighbor_pred = sum(
                l["neighbor_prediction_loss"].item() for l in recent_losses
            ) / len(recent_losses)
            avg_turn_indicator = sum(l["turn_indicator_loss"].item() for l in recent_losses) / len(
                recent_losses
            )
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Batch {batch_idx + 1}/{len(data_loader)} | "
                f"Loss: {avg_loss:.4f} | "
                f"Ego: {avg_ego_planning:.4f} | "
                f"Neighbor: {avg_neighbor_pred:.4f} | "
                f"Turn: {avg_turn_indicator:.4f} | "
                f"LR: {lr:.6f} | "
                f"{log_interval} batches: {elapsed_sec:.2f}s"
            )

            if save_path:
                model_dict = {
                    "epoch": epoch + 1,
                    "model": get_model_state_dict(model),
                    "ema_state_dict": ema.ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "schedule": scheduler.state_dict() if scheduler is not None else None,
                    "loss": loss["loss"].item(),
                    "wandb_id": getattr(args, "wandb_id", None),
                }
                torch.save(model_dict, f"{save_path}/latest.pth")

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"{epoch_mean_loss['loss']=:.4f}")
        print(f"{epoch_mean_loss['turn_indicator_accuracy']=:.4f}")

    return epoch_mean_loss, epoch_mean_loss["loss"]
