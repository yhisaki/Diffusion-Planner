#!/bin/bash
set -ux
exp_name=${1}
TRAIN_SET_LIST=${2}
VALID_SET_LIST=${3}

TRAIN_SET_LIST=$(readlink -f $TRAIN_SET_LIST)
VALID_SET_LIST=$(readlink -f $VALID_SET_LIST)

cd $(dirname $0)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO

rm -f /tmp/tmp_dist_init

SAVE_DIR="/mnt/nvme/drifting_training_result"
TIME=$(date +%Y%m%d-%H%M%S)
SAVE_PATH="${SAVE_DIR}/${TIME}_${exp_name}"

mkdir -p ${SAVE_PATH}

git show -s > ${SAVE_PATH}/git_show.txt
git diff > ${SAVE_PATH}/git_diff.txt

python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_drifting.py \
--exp_name ${exp_name} \
--train_set_list $TRAIN_SET_LIST \
--valid_set_list $VALID_SET_LIST \
--use_wandb True \
--save_dir ${SAVE_PATH} \
--train_epochs 50 \
--save_utd 10 \
--drifting_loss_weight 1.0 \
--drifting_temperatures 0.02 0.05 0.2 \
2>&1 | tee ${SAVE_PATH}/train_log.txt
