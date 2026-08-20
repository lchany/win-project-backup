#!/bin/bash
set -euo pipefail

source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1

REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
BASE_CONFIG=projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
EXPECTED_HEAD=3a1d7633582d079a2f3e3ddba6fa2555c14da77f
OUT=${STEP331_OUT:?STEP331_OUT is required}
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
export MAX_ITERS=${MAX_ITERS:-16}
export MASTER_PORT=${MASTER_PORT:?MASTER_PORT is required}
export WORK_DIRS="$OUT/work"
export LOG_DIR="$OUT"
export SOAP_STALE_Q_K=${SOAP_STALE_Q_K:-4}
export STEP331_PROFILE_OUTPUT="$OUT/profile_raw"
export STEP331_PROFILE_WORK_DIR="$OUT/work"

unset MX_QR_VALIDATION_BYPASS SOAP_DIST_QR SOAP_QR_SHAPE_LOG SOAP_QR_DUMP_DIR SOAP_QR_BACKEND || true
unset HCCL_DETERMINISTIC DETERMINISTIC PYTHONHASHSEED || true
unset ASCEND_LAUNCH_BLOCKING PROFILING_MODE PROFILE PROFILE_DIR HCCL_IF_BASE_PORT || true

mkdir -p "$OUT/logs" "$OUT/work" "$OUT/profile_raw" "$OUT/tools"
cd "$REPO"
export PYTHONPATH="${OUT}:${PWD}/mmdetection3d-0.17.1:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
rm -f "${REPO}/sitecustomize.py"

cp -a "$OUT/step331_cycle_profiler_hook.py" "$OUT/step331_cycle_profiler_hook.py.bak" 2>/dev/null || true
test -f "$OUT/step331_cycle_profiler_hook.py"
export STEP331_REPO="$REPO"
python3 "$OUT/step331_build_profile_config.py" --repo "$REPO" "$REPO/$BASE_CONFIG" "$OUT/step331_profile_config.py"

{
  echo "STEP331_START $(date -Iseconds)"
  echo "HEAD=$(git rev-parse HEAD)"
  echo "SOAP_STALE_Q_K=$SOAP_STALE_Q_K"
  echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
  echo "MAX_ITERS=$MAX_ITERS GPUS=$GPUS MASTER_PORT=$MASTER_PORT"
  echo "PROFILE wait=8 warmup=1 active=7 => train iters 10-16"
  if [ "$(git rev-parse HEAD)" != "$EXPECTED_HEAD" ]; then
    echo "HEAD_MISMATCH expected=$EXPECTED_HEAD"
    exit 91
  fi
  for path in tools/train_spetr.py tools/ddp_train.sh "$BASE_CONFIG" projects/mmdet3d_plugin/optimizers/soap.py; do
    if ! git diff --quiet HEAD -- "$path"; then
      echo "TRACKED_PATH_DIRTY $path"
      exit 92
    fi
  done
  tmp=$(mktemp)
  sed "s/^export SOAP_STALE_Q_K=.*/export SOAP_STALE_Q_K=${SOAP_STALE_Q_K}/" tools/ddp_train.sh >"$tmp"
  bash "$tmp" tools/train_spetr.py "$OUT/step331_profile_config.py"
  rm -f "$tmp"
  train_rc=$?
  exit "$train_rc"
} >"$OUT/logs/launcher.log" 2>&1
rc=$?
echo "$rc" >"$OUT/launcher_rc.txt"
echo "STEP331_END rc=$rc $(date -Iseconds)" >>"$OUT/logs/launcher.log"
exit $rc
