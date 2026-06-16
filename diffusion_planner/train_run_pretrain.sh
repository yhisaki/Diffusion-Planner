#!/bin/bash
set -ux
exp_name=${1}
TRAIN_SET_LIST=${2:-/mnt/nvme/dataset/basic_dataset/path_list_train.json}
DEBUG=${3:-False}

# to convert full paths
TRAIN_SET_LIST=$(readlink -f $TRAIN_SET_LIST)

cd $(dirname $0)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

export DIST_INIT_FILE=/tmp/tmp_dist_init_$$
rm -f ${DIST_INIT_FILE}

SAVE_DIR="./training_result"
TIME=$(date +%Y%m%d-%H%M%S)
SAVE_PATH="${SAVE_DIR}/${TIME}_${exp_name}"

mkdir -p ${SAVE_PATH}

# Save git info
git show -s > ${SAVE_PATH}/git_show.txt
git diff > ${SAVE_PATH}/git_diff.txt

# pretraining
LAUNCHER=(python3)
if [ "${DEBUG}" = "True" ]; then
    LAUNCHER=(python3 -m debugpy --listen 5678)
fi

"${LAUNCHER[@]}" -m torch.distributed.run \
--nnodes 1 \
--nproc-per-node $NUM_GPUS \
--standalone train_predictor.py \
--resume_model_path "/home/hisaki/Diffusion-Planner/diffusion_planner/training_result/20260616-150116_new_v4/latest.pth" \
--exp_name ${exp_name} \
--train_set_list $TRAIN_SET_LIST \
--use_wandb False \
--save_dir ${SAVE_PATH} \
--train_epochs 80 \
--batch_size 480 \
--find_unused_parameters False \
--compile_model True \
--use_amp True \
2>&1 | tee ${SAVE_PATH}/train_log.txt
