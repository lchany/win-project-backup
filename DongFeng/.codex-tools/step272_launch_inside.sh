#!/bin/bash
set -euo pipefail
source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONDONTWRITEBYTECODE=1
unset MX_QR_VALIDATION_BYPASS || true
DIR=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step272_front8_eye192
export STEP272_OUT="$DIR"
mkdir -p "$DIR"
python -u "$DIR/step272_front8_eye192.py"
