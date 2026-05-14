import argparse
import json
import os

import pandas as pd
import torch
import wandb
from drifting_planner.dimensions import *
from drifting_planner.loss import loss_func, make_turn_indicator_gt
from drifting_planner.model.drifting_planner import DriftingPlanner
from drifting_planner.train_epoch import train_epoch
from drifting_planner.utils.data_augmentation import StatePerturbation
from drifting_planner.utils.dataset import DiffusionPlannerData
from drifting_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from drifting_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from drifting_planner.utils.train_utils import resume_model, set_seed
from timm.utils.model_ema import ModelEma
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def heading_to_cos_sin(x):
    return torch.cat(
        [
            x[..., :2],
            x[..., 2:3].cos(),
            x[..., 2:3].sin(),
        ],
        dim=-1,
    )


@torch.no_grad()
def validate_model(model, val_loader, args):
    device = args.device
    model.eval()
    total_loss_ego = 0.0
    total_loss_neighbor = 0.0
    total_samples_ego = 0
    total_samples_neighbor = 0

    turn_indicator_correct = 0.0
    turn_indicator_total = 0
    turn_indicator_change_correct = 0.0
    turn_indicator_change_total = 0

    for inputs in val_loader:
        inputs = {key: value.to(device) for key, value in inputs.items()}
        B = inputs["ego_current_state"].shape[0]

        turn_indicator_seq = inputs["turn_indicators"]

        inputs["sampled_trajectories"] = torch.zeros(
            B, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, dtype=torch.float32, device=device
        )

        inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
        inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

        ego_future = inputs["ego_agent_future"]
        ego_future = heading_to_cos_sin(ego_future)
        neighbors_future = inputs["neighbor_agents_future"]
        neighbor_future_mask = (
            torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
        )
        neighbors_future = heading_to_cos_sin(neighbors_future)
        neighbors_future[neighbor_future_mask] = 0.0

        B, Pn, T, _ = neighbors_future.shape
        ego_current, neighbors_current = (
            inputs["ego_current_state"][:, :4],
            inputs["neighbor_agents_past"][:, :Pn, -1, :4],
        )
        inputs = args.observation_normalizer(inputs)

        _, outputs = model(inputs)

        P = 1 + Pn
        prediction = outputs["prediction"][:, :P, :, :]

        neighbor_current_mask = (
            torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        )
        neighbor_mask = torch.concat(
            (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1
        )

        gt_future = torch.cat(
            [ego_future[:, None, :, :], neighbors_future[..., :]], dim=1
        )
        current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)

        all_gt = torch.cat(
            [current_states[:, :, None, :], gt_future], dim=2
        )
        all_gt[:, 1:][neighbor_mask] = 0.0

        prediction = outputs["prediction"]
        turn_indicator_logit = outputs["turn_indicator_logit"]
        turn_indicator = turn_indicator_logit.argmax(dim=-1)
        turn_indicator_gt = make_turn_indicator_gt(turn_indicator_seq)
        correct = (turn_indicator == turn_indicator_gt).long()
        turn_indicator_correct += correct.sum().item()
        turn_indicator_total += correct.numel()
        change_mask = turn_indicator_seq[:, -1] != turn_indicator_seq[:, -2]
        change_count = change_mask.sum().item()
        if change_count > 0:
            turn_indicator_change_correct += correct[change_mask].sum().item()
            turn_indicator_change_total += change_count

        neighbors_future_valid = ~neighbor_future_mask
        all_gt = all_gt[:, :, 1:, :]
        loss_tensor = (prediction - all_gt) ** 2
        loss_ego = loss_tensor[:, 0, :]
        loss_nei = loss_tensor[:, 1:, :]
        loss_nei = loss_nei[neighbors_future_valid]
        total_loss_ego += loss_ego.mean().item() * B
        total_samples_ego += B
        if loss_nei.shape[0] > 0:
            nei_B = loss_nei.shape[0]
            total_loss_neighbor += loss_nei.mean().item() * nei_B
            total_samples_neighbor += nei_B

    avg_loss_ego = total_loss_ego / max(total_samples_ego, 1)
    avg_loss_neighbor = total_loss_neighbor / max(total_samples_neighbor, 1)

    return {
        "avg_loss_ego": avg_loss_ego,
        "avg_loss_neighbor": avg_loss_neighbor,
        "turn_indicator_accuracy": turn_indicator_correct / max(turn_indicator_total, 1),
        "turn_indicator_change_accuracy": turn_indicator_change_correct / max(turn_indicator_change_total, 1),
        "turn_indicator_change_total": turn_indicator_change_total,
    }


def get_args():
    parser = argparse.ArgumentParser(description="Drifting Planner Training")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--train_set_list", type=str, required=True)
    parser.add_argument("--valid_set_list", type=str, required=True)

    parser.add_argument("--future_len", type=int, default=OUTPUT_T)
    parser.add_argument("--time_len", type=int, default=INPUT_T + 1)
    parser.add_argument("--ego_prediction_horizon", type=int, default=OUTPUT_T)

    parser.add_argument("--agent_state_dim", type=int, default=11)
    parser.add_argument("--agent_num", type=int, default=320)

    parser.add_argument("--static_objects_state_dim", type=int, default=10)
    parser.add_argument("--static_objects_num", type=int, default=5)

    parser.add_argument("--lane_num", type=int, default=NUM_SEGMENTS_IN_LANE)
    parser.add_argument("--lane_len", type=int, default=POINTS_PER_LANELET)

    parser.add_argument("--route_num", type=int, default=NUM_SEGMENTS_IN_ROUTE)
    parser.add_argument("--route_len", type=int, default=POINTS_PER_LANELET)

    parser.add_argument("--polygon_num", type=int, default=NUM_POLYGONS)
    parser.add_argument("--polygon_len", type=int, default=POINTS_PER_POLYGON)

    parser.add_argument("--line_string_num", type=int, default=NUM_LINE_STRINGS)
    parser.add_argument("--line_string_len", type=int, default=POINTS_PER_LINE_STRING)

    parser.add_argument("--use_data_augment", default=True, type=boolean)
    parser.add_argument("--augment_prob", type=float, default=0.5)
    parser.add_argument("--normalization_file_path", default="normalization.json", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--save_utd", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warm_up_epoch", type=int, default=5)
    parser.add_argument("--encoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--decoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--use_ego_history", type=boolean, default=True)
    parser.add_argument("--ego_history_dropout_rate", type=float, default=0.6)
    parser.add_argument("--use_turn_indicators", type=boolean, default=True)

    parser.add_argument("--coeff_position_lat_loss", type=float, default=1.0)
    parser.add_argument("--coeff_position_lon_loss", type=float, default=1.0)
    parser.add_argument("--coeff_heading_l2_loss", type=float, default=1.0)
    parser.add_argument("--coeff_velocity", type=float, default=1.0)

    parser.add_argument("--coeff_road_border_loss", type=float, default=0.0)
    parser.add_argument("--road_border_margin", type=float, default=0.25)
    parser.add_argument("--road_border_n_interp", type=int, default=2)

    parser.add_argument("--coeff_neighbor_collision_loss", type=float, default=0.0)
    parser.add_argument("--neighbor_collision_margin", type=float, default=2.0)

    parser.add_argument("--alpha_planning_loss", type=float, default=1.0)
    parser.add_argument("--alpha_neighbor_loss", type=float, default=0.1)

    parser.add_argument("--use_velocity_representation", type=boolean, default=False)

    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--use_ema", default=True, type=boolean)

    parser.add_argument("--encoder_mixer_depth", type=int, default=6)
    parser.add_argument("--encoder_fusion_depth", type=int, default=6)
    parser.add_argument("--decoder_depth", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--predicted_neighbor_num", type=int, default=320)

    parser.add_argument(
        "--drifting_temperatures",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.2],
    )
    parser.add_argument(
        "--drifting_loss_weight",
        type=float,
        default=1.0,
    )

    parser.add_argument("--resume_model_path", type=str, default=None)

    parser.add_argument("--use_wandb", default=False, type=boolean)
    parser.add_argument("--notes", default="", type=str)

    parser.add_argument("--ddp", default=True, type=boolean)
    parser.add_argument("--port", default="22323", type=str)

    args = parser.parse_args()

    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)

    return args


def model_training(args):
    from drifting_planner.utils import ddp

    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
    print(f"{global_rank=}, {rank=}")

    if global_rank == 0:
        print("------------- {} -------------".format(args.exp_name))
        print("Batch size: {}".format(args.batch_size))
        print("Learning rate: {}".format(args.learning_rate))
        print("Use device: {}".format(args.device))

        if args.resume_model_path is not None:
            save_path = os.path.dirname(args.resume_model_path)
        else:
            save_path = args.save_dir
            os.makedirs(save_path, exist_ok=True)

        args_dict = vars(args)
        args_dict = {
            k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict()
            for k, v in args_dict.items()
        }
        args_dict["model_type"] = "drifting"

        with open(os.path.join(save_path, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=4)
    else:
        save_path = None

    set_seed(args.seed + global_rank)

    train_epochs = args.train_epochs
    batch_size = args.batch_size
    save_utd = args.save_utd

    aug = (
        StatePerturbation(augment_prob=args.augment_prob, device=args.device)
        if args.use_data_augment
        else None
    )

    train_set = DiffusionPlannerData(args.train_set_list)
    valid_set = DiffusionPlannerData(args.valid_set_list)

    train_sampler = DistributedSampler(
        train_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=True
    )
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=batch_size // ddp.get_world_size(),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if global_rank == 0:
        valid_loader = DataLoader(
            valid_set,
            batch_size=batch_size // 2,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False,
        )
    else:
        valid_loader = None

    if global_rank == 0:
        print("Dataset Prepared: {} train data\n".format(len(train_set)))

    if args.ddp:
        torch.distributed.barrier()

    drifting_planner = DriftingPlanner(args)
    drifting_planner = drifting_planner.to(rank if args.device == "cuda" else args.device)

    if args.ddp:
        drifting_planner = DDP(drifting_planner, device_ids=[rank], find_unused_parameters=True)

    if args.use_ema:
        model_ema = ModelEma(
            drifting_planner,
            decay=0.999,
            device=args.device,
        )
    else:
        model_ema = None

    if global_rank == 0:
        print(
            "Model Params: {}".format(
                sum(p.numel() for p in ddp.get_model(drifting_planner, args.ddp).parameters())
            )
        )

    params = [
        {
            "params": ddp.get_model(drifting_planner, args.ddp).parameters(),
            "lr": args.learning_rate,
        }
    ]

    optimizer = optim.AdamW(params)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, train_epochs, args.warm_up_epoch)

    if args.resume_model_path is not None:
        print(f"Model loaded from {args.resume_model_path}")
        drifting_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(
            args.resume_model_path, drifting_planner, optimizer, scheduler, model_ema, args.device
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate
        print(f"Learning rate reset to {args.learning_rate}")
    else:
        init_epoch = 0
        wandb_id = None

    if global_rank == 0:
        os.environ["WANDB_MODE"] = "online" if args.use_wandb else "offline"
        wandb.init(
            project="Drifting-Planner",
            name=args.exp_name,
            notes=args.notes,
            resume="allow",
            id=wandb_id,
            dir=f"{save_path}",
        )
        wandb.config.update(args)

    if args.ddp:
        torch.distributed.barrier()

    data_list = []
    best_loss = float("inf")

    if global_rank == 0:
        valid_dict = validate_model(drifting_planner, valid_loader, args)
        valid_loss_ego = valid_dict["avg_loss_ego"]
        valid_loss_neighbor = valid_dict["avg_loss_neighbor"]
        turn_indicator_accuracy = valid_dict["turn_indicator_accuracy"]
        turn_indicator_change_accuracy = valid_dict["turn_indicator_change_accuracy"]
        turn_indicator_change_total = valid_dict["turn_indicator_change_total"]
        print(
            f"{valid_loss_ego=:.3f}\n"
            f"{valid_loss_neighbor=:.3f}\n"
            f"{turn_indicator_accuracy=:.3f}\n"
            f"{turn_indicator_change_accuracy=:.3f}\n"
            f"{turn_indicator_change_total=:.3f}"
        )

    for epoch in range(init_epoch, train_epochs):
        if args.ddp:
            torch.distributed.barrier()

        final_epoch_count = 10
        if epoch >= train_epochs - final_epoch_count:
            base_lr = args.learning_rate
            if epoch >= train_epochs - final_epoch_count // 2:
                adjusted_lr = base_lr * 0.01
            else:
                adjusted_lr = base_lr * 0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = adjusted_lr
            if global_rank == 0:
                print(f"Final phase: Epoch {epoch + 1}, LR adjusted to {adjusted_lr}")

        train_loss, train_total_loss = train_epoch(
            train_loader, drifting_planner, optimizer, args, model_ema, aug
        )

        if global_rank == 0:
            valid_dict = validate_model(drifting_planner, valid_loader, args)
            valid_loss_ego = valid_dict["avg_loss_ego"]
            valid_loss_neighbor = valid_dict["avg_loss_neighbor"]
            turn_indicator_accuracy = valid_dict["turn_indicator_accuracy"]
            turn_indicator_change_accuracy = valid_dict["turn_indicator_change_accuracy"]
            turn_indicator_change_total = valid_dict["turn_indicator_change_total"]
            print(
                f"Epoch {epoch + 1}/{train_epochs}\n"
                f"{valid_loss_ego=:.3f}\n"
                f"{valid_loss_neighbor=:.3f}\n"
                f"{turn_indicator_accuracy=:.3f}\n"
                f"{turn_indicator_change_accuracy=:.3f}\n"
                f"{turn_indicator_change_total=:.3f}"
            )

            lr_dict = {"lr": optimizer.param_groups[0]["lr"]}
            wandb.log(
                {
                    **{f"train_loss/{k}": v for k, v in train_loss.items()},
                    **{f"lr/{k}": v for k, v in lr_dict.items()},
                    "valid_loss/ego": valid_loss_ego,
                    "valid_loss/neighbors": valid_loss_neighbor,
                    "valid_loss/turn_indicator_accuracy": turn_indicator_accuracy,
                    "valid_loss/turn_indicator_change_accuracy": turn_indicator_change_accuracy,
                },
                step=epoch + 1,
            )

            curr_data = {
                "epoch": epoch + 1,
                "train_loss": train_total_loss,
                "valid_loss_ego": valid_loss_ego,
                "valid_loss_neighbor": valid_loss_neighbor,
            }
            data_list.append(curr_data)
            df = pd.DataFrame(data_list)
            df.to_csv(os.path.join(save_path, "train_log.tsv"), index=False, sep="\t")

            model_dict = {
                "epoch": epoch + 1,
                "model": drifting_planner.state_dict(),
                "ema_state_dict": (
                    model_ema.ema.state_dict() if model_ema is not None else None
                ),
                "optimizer": optimizer.state_dict(),
                "schedule": scheduler.state_dict(),
                "loss": valid_loss_ego,
                "wandb_id": wandb_id,
            }
            torch.save(model_dict, f"{save_path}/latest.pth")

            if (epoch + 1 - init_epoch) % save_utd == 0:
                curr_dir = os.path.join(save_path, f"epoch{epoch + 1:04d}")
                os.makedirs(curr_dir, exist_ok=True)
                torch.save(model_dict, f"{curr_dir}/best_model.pth")
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as f:
                    json.dump(curr_data, f, indent=4)
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=4)

            if valid_loss_ego < best_loss:
                curr_dir = os.path.join(save_path, "best_model")
                os.makedirs(curr_dir, exist_ok=True)
                torch.save(model_dict, f"{curr_dir}/best_model.pth")
                best_loss = valid_loss_ego
                curr_data["best_loss"] = best_loss
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as f:
                    json.dump(curr_data, f, indent=4)
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=4)

        scheduler.step()
        train_sampler.set_epoch(epoch + 1)


if __name__ == "__main__":
    args = get_args()
    model_training(args)
