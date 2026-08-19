#!/usr/bin/env bash
# MapQR 876-step 正式训练（fa95a2a 合同）
# 宿主机执行: bash run_mapqr_876_train.sh

set -euo pipefail

REPO="/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang"
GCONTRACT="/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/gpu_contract_alignment_f922c38_8npu_20260814T172611"
CONTAINER="mapqr-leicheng"
PORT="${PORT:-30050}"
MAX_ITERS="${MAX_ITERS:-876}"
STAMP="$(date +%Y%m%dT%H%M%S)"
OUT="${OUT:-/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/mapqr_876_${STAMP}}"

mkdir -p "${OUT}/work" "${OUT}/logs"

docker exec \
  -e ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e SOAP_STALE_Q_K=4 \
  -e MASTER_PORT="${PORT}" \
  -e GPUS=8 \
  -e MODE=single \
  -e MAX_ITERS="${MAX_ITERS}" \
  -e WORK_DIRS="${OUT}/work" \
  -e REPO="${REPO}" \
  -e GCONTRACT="${GCONTRACT}" \
  "${CONTAINER}" bash -lc '
set -euo pipefail
cd "$REPO"

unset TASK_QUEUE_ENABLE COMBINED_ENABLE CPU_AFFINITY_CONF || true
unset ASCEND_LAUNCH_BLOCKING PROFILING_MODE PROFILE PROFILE_DIR || true
unset HCCL_DETERMINISTIC DETERMINISTIC PYTHONHASHSEED || true
unset DBG_NPU RESET_DEBUG SAMPLER_DEBUG || true

export PYTHONPATH="${GCONTRACT}/tools:${REPO}/mmdetection3d-0.17.1:${REPO}:${PYTHONPATH:-}"

echo "HEAD=$(git rev-parse --short HEAD) MAX_ITERS=${MAX_ITERS} SOAP_STALE_Q_K=${SOAP_STALE_Q_K}"

bash "${GCONTRACT}/test_harness/ddp_train_30.sh" \
  "${GCONTRACT}/tools/train_spetr_gpu_seed0_runtime.py" \
  "${GCONTRACT}/config/aligned_gpu_contract_npu_runtime.py"
' 2>&1 | tee "${OUT}/logs/train.log"

echo "done OUT=${OUT}"
