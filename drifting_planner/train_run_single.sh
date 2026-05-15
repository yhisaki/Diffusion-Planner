#!/bin/bash
set -ux
exp_name=DriftingPlanner
TRAIN_SET_LIST=/home/hisaki/diffusion-planner-data/data/
VALID_SET_LIST=/home/hisaki/diffusion-planner-data/data/

TRAIN_SET_LIST=$(readlink -f $TRAIN_SET_LIST)
VALID_SET_LIST=$(readlink -f $VALID_SET_LIST)

cd $(dirname $0)

export CUDA_VISIBLE_DEVICES=0

SAVE_DIR="./drifting_training_result"
TIME=$(date +%Y%m%d-%H%M%S)
SAVE_PATH="${SAVE_DIR}/${TIME}_${exp_name}"

mkdir -p ${SAVE_PATH}

git show -s > ${SAVE_PATH}/git_show.txt 2>/dev/null || true
git diff > ${SAVE_PATH}/git_diff.txt 2>/dev/null || true

python3 compute_normalization.py \
--data ${TRAIN_SET_LIST} \
--output drifting_planner/normalization.json

python3 train_drifting.py \
--exp_name ${exp_name} \
--train_set_list $TRAIN_SET_LIST \
--valid_set_list $VALID_SET_LIST \
--use_wandb False \
--save_dir ${SAVE_PATH} \
--train_epochs 100 \
--batch_size 10 \
--save_utd 10 \
--learning_rate 3e-4 \
--warm_up_epoch 5 \
--drifting_loss_weight 1.0 \
--drifting_temperatures 0.02 0.05 0.2 \
--ddp False \
--num_workers 4 \
2>&1 | tee ${SAVE_PATH}/train_log.txt
