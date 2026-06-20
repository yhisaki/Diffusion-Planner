import argparse
import json
import os

import torch
import wandb
from diffusion_planner.dimensions import *
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_epoch import train_epoch
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.train_utils import (
    get_model,
    get_model_state_dict,
    resume_encoder_model,
    resume_model,
    set_seed,
)
from timm.utils import ModelEma
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


def get_args():
    # Arguments
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, help="save path for model ckpt", required=True)

    # Data
    parser.add_argument("--train_set_list", type=str, required=True)

    parser.add_argument("--future_len", type=int, default=OUTPUT_T)
    parser.add_argument("--time_len", type=int, default=INPUT_T + 1)

    parser.add_argument("--agent_state_dim", type=int, help="past state dim for agents", default=11)
    parser.add_argument("--agent_num", type=int, default=MAX_NUM_NEIGHBORS)

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

    # DataLoader parameters
    parser.add_argument("--augment_prob", type=float, help="augmentation probability", default=0.5)
    parser.add_argument("--normalization_file_path", default="normalization.json", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin-mem", action="store_true", help="Pin CPU memory in DataLoader")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Training
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--encoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--decoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument(
        "--velocity_dropout_ratio",
        type=float,
        default=0.5,
        help="probability of dropping the ego velocity token during training",
    )
    parser.add_argument("--use_turn_indicators", type=boolean, default=True)

    parser.add_argument("--coeff_pos_ego", type=float, default=1.0)
    parser.add_argument("--coeff_pos_neighbor", type=float, default=1.0)
    parser.add_argument("--coeff_heading_ego", type=float, default=1.0)
    parser.add_argument("--coeff_heading_neighbor", type=float, default=1.0)

    parser.add_argument("--alpha_planning_loss", type=float, default=1.0)
    parser.add_argument("--alpha_neighbor_loss", type=float, default=0.1)

    parser.add_argument("--device", type=str, help="run on which device", default="cuda")

    parser.add_argument("--use_ema", default=True, type=boolean)

    # Model
    parser.add_argument("--encoder_mixer_depth", type=int, default=3)
    parser.add_argument("--encoder_fusion_depth", type=int, default=3)
    parser.add_argument("--decoder_depth", type=int, help="number of decoding layers", default=3)
    parser.add_argument("--num_heads", type=int, help="number of multi-head", default=8)
    parser.add_argument("--hidden_dim", type=int, help="hidden dimension", default=256)
    parser.add_argument("--predicted_neighbor_num", type=int, default=MAX_NUM_NEIGHBORS)

    parser.add_argument(
        "--resume_model_path",
        type=str,
        help="path to resume a full training checkpoint",
        default=None,
    )
    parser.add_argument(
        "--resume_encoder_model_path",
        type=str,
        help="path to load only encoder weights from a checkpoint",
        default=None,
    )
    parser.add_argument(
        "--freeze_encoder",
        type=boolean,
        default=False,
        help="freeze encoder parameters during training",
    )
    parser.add_argument(
        "--compile_model",
        type=boolean,
        default=False,
        help="compile model with torch.compile() for faster forward pass",
    )
    parser.add_argument(
        "--use_amp",
        type=boolean,
        default=False,
        help="use automatic mixed precision for faster forward/backward pass",
    )

    parser.add_argument("--use_wandb", default=False, type=boolean)
    parser.add_argument("--notes", default="", type=str)

    # distributed training parameters
    parser.add_argument(
        "--find_unused_parameters",
        type=boolean,
        default=True,
        help="passed to DDP find_unused_parameters",
    )
    parser.add_argument("--ddp", default=True, type=boolean, help="use ddp or not")
    parser.add_argument("--port", default="22323", type=str, help="port")

    args = parser.parse_args()

    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)

    if args.resume_model_path is not None and args.resume_encoder_model_path is not None:
        raise ValueError("resume_model_path and resume_encoder_model_path cannot both be set")

    return args


def mean_ego_loss(loss_dict):
    result = {}
    for key, val in loss_dict.items():
        if key.startswith("ego_"):
            result[f"valid_loss/{key}"] = val.mean().item()
    return result


def model_training(args):
    # init ddp
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
    print(f"{global_rank=}, {rank=}")

    if global_rank == 0:
        # Logging
        print("------------- {} -------------".format(args.exp_name))
        print("Batch size: {}".format(args.batch_size))
        print("Learning rate: {}".format(args.learning_rate))
        print("Use device: {}".format(args.device))

        if args.resume_model_path is not None:
            save_path = os.path.dirname(args.resume_model_path)
        else:
            save_path = args.save_dir
            os.makedirs(save_path, exist_ok=True)

        # Save args
        args_dict = vars(args)
        args_dict = {
            k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict()
            for k, v in args_dict.items()
        }
        args_dict["major_version"] = 4

        with open(os.path.join(save_path, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=4)

    else:
        save_path = None

    # set seed
    set_seed(args.seed + global_rank)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size

    aug = StatePerturbation(augment_prob=args.augment_prob)

    # prepare dataset
    train_set = DiffusionPlannerData(args.train_set_list, data_augmentation=aug)

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
        print("Dataset Prepared: {} train data\n".format(len(train_set)))

    if args.ddp:
        torch.distributed.barrier()

    init_epoch = 0
    wandb_id = None

    # set up model and training state before wrapping with DDP
    diffusion_planner = Diffusion_Planner(args)
    diffusion_planner = diffusion_planner.to(rank if args.device == "cuda" else args.device)

    if args.resume_encoder_model_path is not None:
        diffusion_planner = resume_encoder_model(
            args.resume_encoder_model_path,
            diffusion_planner,
            args.device,
        )
        print("Encoder loaded.")

        if args.freeze_encoder:
            for parameter in diffusion_planner.encoder.parameters():
                parameter.requires_grad_(False)
            print("Encoder frozen. Training decoder and other non-encoder parameters.")

    trainable_parameters = [p for p in diffusion_planner.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found")

    optimizer = optim.AdamW([
        {
            "params": trainable_parameters,
            "lr": args.learning_rate,
        }
    ])

    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )

    if args.use_ema:
        model_ema = ModelEma(
            diffusion_planner,
            decay=0.999,
            device=args.device,
        )

    if args.resume_model_path is not None:
        print(f"Model loaded from {args.resume_model_path}")
        diffusion_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(
            args.resume_model_path,
            diffusion_planner,
            optimizer,
            scheduler,
            model_ema,
            args.device,
        )

        if args.freeze_encoder:
            for parameter in diffusion_planner.encoder.parameters():
                parameter.requires_grad_(False)
            print("Encoder frozen. Training decoder and other non-encoder parameters.")

    if args.compile_model:
        torch.set_float32_matmul_precision("high")
        print("Compiling model with torch.compile()...")
        diffusion_planner = torch.compile(diffusion_planner)
        print("Model compiled")

    if args.ddp:
        diffusion_planner = DDP(
            diffusion_planner, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
        )

    if global_rank == 0:
        print(
            "Model Params: {}".format(
                sum(p.numel() for p in get_model(diffusion_planner).parameters())
            )
        )

    # logger
    if global_rank == 0:
        wandb_init_kwargs = {}
        if not args.use_wandb:
            wandb_init_kwargs["mode"] = "offline"
        wandb.init(
            project="Diffusion-Planner",
            name=args.exp_name,
            notes=args.notes,
            resume="allow",
            id=wandb_id,
            dir=f"{save_path}",
            **wandb_init_kwargs,
        )
        wandb.config.update(args)

    if args.ddp:
        torch.distributed.barrier()

    # begin training
    for epoch in range(init_epoch, train_epochs):
        # Synchronize all processes before training
        if args.ddp:
            torch.distributed.barrier()

        # training step
        train_loss, train_total_loss = train_epoch(
            train_loader,
            diffusion_planner,
            optimizer,
            args,
            model_ema,
            aug,
            epoch=epoch,
            save_path=save_path,
            scheduler=scheduler,
        )

        if global_rank == 0:
            wandb.log(
                {
                    **{f"train/{key}": value for key, value in train_loss.items()},
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch + 1,
            )

            model_dict = {
                "epoch": epoch + 1,
                "model": get_model_state_dict(diffusion_planner),
                "ema_state_dict": model_ema.ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "schedule": scheduler.state_dict(),
                "wandb_id": wandb_id,
            }
            torch.save(model_dict, f"{save_path}/model_epoch_{epoch + 1}.pth")

        train_sampler.set_epoch(epoch + 1)


if __name__ == "__main__":
    args = get_args()
    torch.set_printoptions(threshold=1000)

    # Run
    model_training(args)
