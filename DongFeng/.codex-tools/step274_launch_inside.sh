#!/bin/bash
# STEP-274: 30-step full test with STEP-271 mx QR bypass on back 8 NPUs only.
set -euo pipefail
REPO=${REPO:-/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang}
CONFIG=${CONFIG:-projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py}
HARNESS=${HARNESS:-tools/ddp_train.sh}
TRAIN=${TRAIN:-tools/train_spetr.py}
STAMP=${STAMP:-$(date +%Y%m%dT%H%M%S)}
OUT=${OUT:-/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step274_mx_qr_bypass_30step_${STAMP}}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}
export TORCH_DEVICE_BACKEND_AUTOLOAD=${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}
export MX_QR_VALIDATION_BYPASS=${MX_QR_VALIDATION_BYPASS:-1}
export MASTER_PORT=${MASTER_PORT:-30090}
export GPUS=${GPUS:-8}
export MODE=${MODE:-single}
export MAX_ITERS=${MAX_ITERS:-30}
export WORK_DIRS=${WORK_DIRS:-${OUT}/work}
export LOG_DIR=${LOG_DIR:-${OUT}}

mkdir -p "$WORK_DIRS" "$LOG_DIR/logs"
cd "$REPO"

unset SOAP_DIST_QR SOAP_QR_SHAPE_LOG SOAP_QR_DUMP_DIR || true
unset HCCL_DETERMINISTIC DETERMINISTIC PYTHONHASHSEED || true

export PYTHONPATH="${PWD}/mmdetection3d-0.17.1:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

{
  echo "STEP274_MX_QR_BYPASS_30STEP_START $(date -Iseconds)"
  echo "HEAD=$(git rev-parse --short HEAD)"
  echo "OUT=$OUT"
  echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
  echo "MX_QR_VALIDATION_BYPASS=$MX_QR_VALIDATION_BYPASS"
  echo "SOAP_DIST_QR=${SOAP_DIST_QR:-unset}"
  python -c "from mmcv import Config; c=Config.fromfile('$CONFIG'); print('one_sided', c.optimizer.get('one_sided_dim_threshold', 'MISSING'))"
  python -c "import torch, torch_npu; print('torch_npu', torch_npu.__version__, 'npu_count', torch.npu.device_count())"
  grep -n "mx_driving_cloud\|MX_QR_VALIDATION\|linalg.qr" projects/mmdet3d_plugin/optimizers/soap.py | head -20 || true
  bash "$HARNESS" "$TRAIN" "$CONFIG"
} >"$LOG_DIR/logs/launcher.log" 2>&1
rc=$?
echo "$rc" >"$LOG_DIR/launcher_rc.txt"
echo "STEP274_MX_QR_BYPASS_30STEP_END rc=$rc $(date -Iseconds)" >>"$LOG_DIR/logs/launcher.log"
exit $rc
