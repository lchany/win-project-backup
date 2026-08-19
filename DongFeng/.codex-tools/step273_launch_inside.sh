#!/bin/bash
set -euo pipefail
source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MX_QR_VALIDATION_BYPASS=1
export PYTHONDONTWRITEBYTECODE=1
DIR=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step273_front8_bypass_verify
export STEP273_OUT="$DIR"
mkdir -p "$DIR"
python -u "$DIR/step273_front8_bypass_verify.py"
