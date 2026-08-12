#!/usr/bin/env bash
set -euo pipefail

# Background training launcher for physical NPU card 10.
# Run this script inside the training container.

REPO_ROOT=/mnt/sfs_turbo/workdir/wfc1/l2.9-df-for-yuexiang
CONFIG=projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
DATE_TAG=$(date +%Y%m%d_%H%M)

# Kill stale train_spetr processes (WARNING: also kills card11 if running).
# Remove or comment out the next line if you want to launch card10 and card11 simultaneously.
pkill -9 -f train_spetr || true

cd "$REPO_ROOT/mmdetection3d-0.17.1"
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
cd "$REPO_ROOT"

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_COMPILE_DISABLE=1
export TORCH_DYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export TORCH_USE_INDUCTOR=0
export ASCEND_RT_VISIBLE_DEVICES=10
export MASTER_PORT=29617
export WORK_DIR_BASE=/mnt/sfs_turbo/workdir/wfc1/work_dirs_card10_${DATE_TAG}
mkdir -p "$WORK_DIR_BASE" /mnt/sfs_turbo/workdir/wfc1/diagnostics

nohup ./run_train.sh "$CONFIG" > /mnt/sfs_turbo/workdir/wfc1/diagnostics/train_card10_${DATE_TAG}.log 2>&1 &

echo "card10 training started in background"
echo "PID: $!"
echo "work_dir: $WORK_DIR_BASE"
echo "launcher log: /mnt/sfs_turbo/workdir/wfc1/diagnostics/train_card10_${DATE_TAG}.log"
echo "training log: $WORK_DIR_BASE/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune/train.log"
