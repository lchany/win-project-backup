#!/bin/bash
set -euo pipefail

source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1

REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
CONFIG=projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
EXPECTED_HEAD=3a1d7633582d079a2f3e3ddba6fa2555c14da77f
OUT=${STEP330_OUT:?STEP330_OUT is required}
SOAP_K=${SOAP_STALE_Q_K:?SOAP_STALE_Q_K is required}
BACK8_DEVICES="8,9,10,11,12,13,14,15"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$BACK8_DEVICES}"
if [ "$ASCEND_RT_VISIBLE_DEVICES" != "$BACK8_DEVICES" ]; then
  echo "BACK8_GUARD_FAIL visible=$ASCEND_RT_VISIBLE_DEVICES expected=$BACK8_DEVICES"
  exit 93
fi
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export GPUS=8
export MODE=single
export MAX_ITERS=${MAX_ITERS:-30}
export MASTER_PORT=${MASTER_PORT:?MASTER_PORT is required}
export WORK_DIRS="$OUT/work"
export LOG_DIR="$OUT"
export SOAP_STALE_Q_K="$SOAP_K"

unset MX_QR_VALIDATION_BYPASS SOAP_DIST_QR SOAP_QR_SHAPE_LOG SOAP_QR_DUMP_DIR SOAP_QR_DUMP_MAX_CALLS SOAP_QR_BACKEND || true
unset HCCL_DETERMINISTIC DETERMINISTIC PYTHONHASHSEED || true
unset ASCEND_LAUNCH_BLOCKING PROFILING_MODE PROFILE PROFILE_DIR HCCL_IF_BASE_PORT || true

mkdir -p "$OUT/logs" "$WORK_DIRS"
cd "$REPO"
export PYTHONPATH="${PWD}/mmdetection3d-0.17.1:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
rm -f "${REPO}/sitecustomize.py"

{
  echo "STEP330_START $(date -Iseconds)"
  echo "HEAD=$(git rev-parse HEAD)"
  echo "SOAP_STALE_Q_K=$SOAP_STALE_Q_K"
  echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
  echo "MAX_ITERS=$MAX_ITERS GPUS=$GPUS MASTER_PORT=$MASTER_PORT"
  if [ "$(git rev-parse HEAD)" != "$EXPECTED_HEAD" ]; then
    echo "HEAD_MISMATCH expected=$EXPECTED_HEAD"
    exit 91
  fi
  for path in tools/train_spetr.py tools/ddp_train.sh "$CONFIG" projects/mmdet3d_plugin/optimizers/soap.py; do
    if ! git diff --quiet HEAD -- "$path"; then
      echo "TRACKED_PATH_DIRTY $path"
      exit 92
    fi
  done
  python -c "from mmcv import Config; c=Config.fromfile('$CONFIG'); print('ONE_SIDED', c.optimizer.get('one_sided_dim_threshold', 'MISSING'))"
  grep -n "mx_driving_cloud.linalg.qr\|torch.linalg.qr" projects/mmdet3d_plugin/optimizers/soap.py | head -5 || true
  tmp=$(mktemp)
  sed "s/^export SOAP_STALE_Q_K=.*/export SOAP_STALE_Q_K=${SOAP_K}/" tools/ddp_train.sh >"$tmp"
  bash "$tmp" tools/train_spetr.py "$CONFIG"
  rm -f "$tmp"
} >"$OUT/logs/launcher.log" 2>&1
rc=$?
echo "$rc" >"$OUT/launcher_rc.txt"
echo "STEP330_END rc=$rc SOAP_STALE_Q_K=$SOAP_K $(date -Iseconds)" >>"$OUT/logs/launcher.log"
exit $rc
