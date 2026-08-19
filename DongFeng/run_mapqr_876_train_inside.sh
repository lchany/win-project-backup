#!/usr/bin/env bash
# 在 mapqr-leicheng 容器内直接运行:
#   bash /mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang/run_mapqr_876_train_inside.sh

set -euo pipefail

REPO="/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang"
GCONTRACT="/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/gpu_contract_alignment_f922c38_8npu_20260814T172611"
HARNESS="${GCONTRACT}/test_harness/ddp_train_30.sh"
TRAIN="${GCONTRACT}/tools/train_spetr_gpu_seed0_runtime.py"
CONFIG="${GCONTRACT}/config/aligned_gpu_contract_npu_runtime.py"

PORT="${PORT:-30050}"
MAX_ITERS="${MAX_ITERS:-876}"
STAMP="$(date +%Y%m%dT%H%M%S)"
OUT="${OUT:-/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/mapqr_876_${STAMP}}"

cd "$REPO"
mkdir -p "${OUT}/work"

export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SOAP_STALE_Q_K=4
export MASTER_PORT="$PORT"
export GPUS=8
export MODE=single
export MAX_ITERS="$MAX_ITERS"
export WORK_DIRS="${OUT}/work"

unset TASK_QUEUE_ENABLE COMBINED_ENABLE CPU_AFFINITY_CONF || true
unset ASCEND_LAUNCH_BLOCKING PROFILING_MODE PROFILE PROFILE_DIR || true
unset HCCL_DETERMINISTIC DETERMINISTIC PYTHONHASHSEED || true
unset DBG_NPU RESET_DEBUG SAMPLER_DEBUG || true

export PYTHONPATH="${GCONTRACT}/tools:${REPO}/mmdetection3d-0.17.1:${REPO}:${PYTHONPATH:-}"

echo "OUT=$OUT"
echo "HEAD=$(git rev-parse --short HEAD)"
echo "MAX_ITERS=$MAX_ITERS SOAP_STALE_Q_K=$SOAP_STALE_Q_K"

bash "$HARNESS" "$TRAIN" "$CONFIG" 2>&1 | tee "${OUT}/work/train.log"

echo "done checkpoint=${OUT}/work/iter_${MAX_ITERS}.pth"
