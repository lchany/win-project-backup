#!/bin/bash
# STEP-275: 800-step training with STEP-271 mx QR fix on back 8 NPUs (no env bypass).
set -euo pipefail
REPO=${REPO:-/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang}
CONFIG=${CONFIG:-projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py}
HARNESS=${HARNESS:-tools/ddp_train.sh}
TRAIN=${TRAIN:-tools/train_spetr.py}
STAMP=${STAMP:-$(date +%Y%m%dT%H%M%S)}
OUT=${OUT:-/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step275_mx_qr_bypass_800step_${STAMP}}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}
export TORCH_DEVICE_BACKEND_AUTOLOAD=${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}
export MASTER_PORT=${MASTER_PORT:-30091}
export GPUS=${GPUS:-8}
export MODE=${MODE:-single}
export MAX_ITERS=${MAX_ITERS:-800}
export WORK_DIRS=${WORK_DIRS:-${OUT}/work}
export LOG_DIR=${LOG_DIR:-${OUT}}

mkdir -p "$WORK_DIRS" "$LOG_DIR/logs"
cd "$REPO"

unset SOAP_DIST_QR SOAP_QR_SHAPE_LOG SOAP_QR_DUMP_DIR || true
unset HCCL_DETERMINISTIC DETERMINISTIC PYTHONHASHSEED || true
unset MX_QR_VALIDATION_BYPASS || true

export PYTHONPATH="${PWD}/mmdetection3d-0.17.1:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

{
  echo "STEP275_MX_QR_BYPASS_800STEP_START $(date -Iseconds)"
  echo "HEAD=$(git rev-parse --short HEAD)"
  echo "OUT=$OUT"
  echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
  echo "MAX_ITERS=$MAX_ITERS"
  echo "SOAP_DIST_QR=${SOAP_DIST_QR:-unset}"
  python -c "import torch, torch_npu; print('torch_npu', torch_npu.__version__, 'npu_count', torch.npu.device_count())"
  bash "$HARNESS" "$TRAIN" "$CONFIG"
} >"$LOG_DIR/logs/launcher.log" 2>&1
rc=$?
echo "$rc" >"$LOG_DIR/launcher_rc.txt"
echo "STEP275_MX_QR_BYPASS_800STEP_END rc=$rc $(date -Iseconds)" >>"$LOG_DIR/logs/launcher.log"
exit $rc
