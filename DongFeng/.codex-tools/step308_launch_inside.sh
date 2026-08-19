#!/bin/bash
set -euo pipefail

source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1

REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
CONFIG=projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
EXPECTED_HEAD=27b1d6d3f363619ad2faa244abe8fbc5a97faef6
OUT=${STEP308_OUT:?STEP308_OUT is required}

export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
export GPUS=8
export MODE=single
export MAX_ITERS=30
export MASTER_PORT=30172
export WORK_DIRS="$OUT/work"
export LOG_DIR="$OUT"

unset MX_QR_VALIDATION_BYPASS SOAP_DIST_QR SOAP_QR_SHAPE_LOG SOAP_QR_DUMP_DIR || true
unset HCCL_DETERMINISTIC DETERMINISTIC PYTHONHASHSEED || true
unset ASCEND_LAUNCH_BLOCKING PROFILING_MODE PROFILE PROFILE_DIR HCCL_IF_BASE_PORT || true

mkdir -p "$OUT/logs" "$WORK_DIRS"
cd "$REPO"
export PYTHONPATH="${PWD}/mmdetection3d-0.17.1:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

{
  echo "STEP308_START $(date -Iseconds)"
  echo "HEAD=$(git rev-parse HEAD)"
  echo "BRANCH=$(git branch --show-current)"
  echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
  echo "MAX_ITERS=$MAX_ITERS GPUS=$GPUS MASTER_PORT=$MASTER_PORT"
  if [ "$(git rev-parse HEAD)" != "$EXPECTED_HEAD" ]; then
    echo "HEAD_MISMATCH"
    exit 91
  fi
  for path in tools/train_spetr.py tools/ddp_train.sh "$CONFIG" projects/mmdet3d_plugin/optimizers/soap.py; do
    if ! git diff --quiet HEAD -- "$path"; then
      echo "TRACKED_PATH_DIRTY $path"
      exit 92
    fi
  done
  python - <<'PY'
import inspect
import torch
import torch_npu
import mx_driving_cloud
import mx_driving_cloud.ops.linalg as linalg
import mmcv.runner.dist_utils as dist_utils

source = inspect.getsource(dist_utils._init_dist_pytorch)
assert "torch.npu.set_device" in source, "MMCV init_dist lacks explicit NPU set_device"
wrapper = inspect.getsource(linalg)
assert "QR_SOAP_FIXED_SHAPE" not in wrapper
assert "MX_QR_VALIDATION_BYPASS" not in wrapper
print("TORCH_NPU", torch_npu.__version__, "NPU_COUNT", torch.npu.device_count())
print("MMCV_EXPLICIT_SET_DEVICE", True)
print("MX_QR_MODULE", mx_driving_cloud.linalg.qr.__module__)
PY
  python -c "from mmcv import Config; c=Config.fromfile('$CONFIG'); print('ONE_SIDED', c.optimizer.get('one_sided_dim_threshold', 'MISSING'))"
  grep -n "mx_driving_cloud.linalg.qr" projects/mmdet3d_plugin/optimizers/soap.py
  bash tools/ddp_train.sh tools/train_spetr.py "$CONFIG"
} >"$OUT/logs/launcher.log" 2>&1
rc=$?
echo "$rc" >"$OUT/launcher_rc.txt"
echo "STEP308_END rc=$rc $(date -Iseconds)" >>"$OUT/logs/launcher.log"
exit $rc
