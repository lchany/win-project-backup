#!/bin/bash
set -euo pipefail
source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PYTHONDONTWRITEBYTECODE=1
export MX_QR_VALIDATION_BYPASS=1
DIR=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step271_qr_patch
bash "$DIR/step271_apply_in_container.sh"
OUT=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step271_qr_bypass_verify
export STEP271_OUT="$OUT"
mkdir -p "$OUT"
python -u /mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step271_qr_bypass_verify/step271_verify_qr_bypass.py
