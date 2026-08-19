#!/usr/bin/env bash

set -x
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SOAP_STALE_Q_K=4
export FUNCTION=${1}

DEBUG=${DEBUG:-false}
MODE=${MODE:-single}

export GPUS=${GPUS:-8}
export MASTER_PORT=${MASTER_PORT:-29504}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

if [ $MODE == multi ]; then
    export NCCL_DEBUG=INFO 
    echo "Init NCCL paramers"
fi
# train
echo "zyq: ${2}"
export CONFIG=${2}
EXP_NAME=$(basename "$CONFIG")
EXP_NAME=${EXP_NAME%.py}

EXP_FOLDER=$(dirname "$CONFIG")
EXP_FOLDER=$(basename "$EXP_FOLDER")
WORK_DIR_BASE="${WORK_DIR_BASE:-work_dirs}"
WORK_DIRS="${WORK_DIRS:-${WORK_DIR_BASE}/${EXP_FOLDER}/${EXP_NAME}}"
mkdir -p $WORK_DIRS

if [ -z $CONFIG ]; then
    echo "config is required"
    exit 1
fi


if [ -z $MLP_WORKER_0_PORT ]; then  #local
    MLP_WORKER_0_PORT=${MASTER_PORT}
    MODE=single
fi
# --resume-from ${WORK_DIRS}/latest.pth \
if [ $MODE == multi ]; then
    echo "Multi GPUS training start"
    python -m torch.distributed.launch --master_port ${MLP_WORKER_0_PORT} --nnodes ${MLP_WORKER_NUM} --node_rank ${MLP_ROLE_INDEX} --master_addr ${MLP_WORKER_0_HOST} --nproc_per_node ${GPUS} --use_env \
    ${FUNCTION} ${CONFIG} --work-dir=${WORK_DIRS} \

    --gpus ${GPUS} --autoscale-lr --no-validate \
    --resume-from ${WORK_DIRS}/latest.pth \
    --max-iters "${MAX_ITERS:-30}" --launcher="pytorch" 2>&1 | tee ${WORK_DIRS}/train.log
elif [ $MODE == local ]; then
    python ${FUNCTION} ${CONFIG} --work-dir=${WORK_DIRS} \
    --gpus ${GPUS} --autoscale-lr \
    --launcher='none' 2>&1 | tee ${WORK_DIRS}/train.log
else
    echo "Single GPUS training start"
    python -m torch.distributed.launch --master_port ${MLP_WORKER_0_PORT} --nproc_per_node ${GPUS} --use_env \
    ${FUNCTION} ${CONFIG} --work-dir=${WORK_DIRS} \
    --gpus ${GPUS} --autoscale-lr \
    --max-iters "${MAX_ITERS:-30}" --launcher="pytorch" 2>&1 | tee ${WORK_DIRS}/train.log
fi
