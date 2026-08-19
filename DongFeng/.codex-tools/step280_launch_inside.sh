#!/bin/bash
set -euo pipefail
source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PYTHONDONTWRITEBYTECODE=1
export STEP280_OUT=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step280_qr_cpu_vs_mx
mkdir -p "$STEP280_OUT"
cd "$STEP280_OUT"
python -u "$STEP280_OUT/step280_qr_cpu_vs_mx_scan.py"
echo "$?" > "$STEP280_OUT/launcher_rc.txt"
