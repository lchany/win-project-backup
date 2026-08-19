#!/bin/bash
set -euo pipefail
source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PYTHONDONTWRITEBYTECODE=1
export STEP270_OUT=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step270_device_pin
mkdir -p "$STEP270_OUT"
python -u /mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step270_device_pin/step270_device_pin.py
