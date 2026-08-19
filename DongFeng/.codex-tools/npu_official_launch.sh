#!/usr/bin/env bash
# Canonical 8-NPU official training launch. Run INSIDE container mapqr-leicheng.
# Do not docker -e PYTHONPATH. Override only WORK_DIRS / MASTER_PORT / MAX_ITERS / CONFIG.
set -euo pipefail

REPO=${REPO:-/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang}
CONFIG=${CONFIG:-projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py}
HARNESS=${HARNESS:-tools/ddp_train.sh}
TRAIN=${TRAIN:-tools/train_spetr.py}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}
export TORCH_DEVICE_BACKEND_AUTOLOAD=${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}
export MASTER_PORT=${MASTER_PORT:-30050}
export GPUS=${GPUS:-8}
export MODE=${MODE:-single}
export MAX_ITERS=${MAX_ITERS:-30}

STAMP=$(date +%Y%m%dT%H%M%S)
WORK_DIRS=${WORK_DIRS:-/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/official_run_${STAMP}/work}
export WORK_DIRS
LOG_DIR=${LOG_DIR:-$(dirname "$WORK_DIRS")}
mkdir -p "$WORK_DIRS" "$LOG_DIR/logs"

cd "$REPO"

unset TASK_QUEUE_ENABLE COMBINED_ENABLE CPU_AFFINITY_CONF || true
unset ASCEND_LAUNCH_BLOCKING PROFILING_MODE PROFILE PROFILE_DIR || true
unset HCCL_DETERMINISTIC DETERMINISTIC PYTHONHASHSEED || true

# Prepend vendor mmdet3d + repo; keep container CANN/tbe paths.
export PYTHONPATH="${PWD}/mmdetection3d-0.17.1:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

{
  echo "NPU_OFFICIAL_LAUNCH_START $(date -Iseconds)"
  echo "HEAD=$(git rev-parse --short HEAD)"
  echo "REPO=$PWD"
  echo "CONFIG=$CONFIG"
  echo "WORK_DIRS=$WORK_DIRS"
  echo "MASTER_PORT=$MASTER_PORT MAX_ITERS=$MAX_ITERS GPUS=$GPUS"
  echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
  echo "SOAP_DIST_QR=${SOAP_DIST_QR:-unset} SOAP_STALE_Q_K=${SOAP_STALE_Q_K:-unset}"
  python -c "from mmcv import Config; c=Config.fromfile('$CONFIG'); print('onesided', c.optimizer.get('one_sided_dim_threshold', 'MISSING'))"
  python -c "import torch, torch_npu; print('torch_npu', torch_npu.__version__, 'count', torch.npu.device_count())"
  bash "$HARNESS" "$TRAIN" "$CONFIG"
} >"$LOG_DIR/logs/launcher.log" 2>&1
rc=$?
echo "$rc" >"$LOG_DIR/launcher_rc.txt"
echo "NPU_OFFICIAL_LAUNCH_END rc=$rc $(date -Iseconds)" >>"$LOG_DIR/logs/launcher.log"
exit $rc
