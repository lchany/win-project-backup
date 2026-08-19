#!/bin/bash
set -euo pipefail
source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PYTHONDONTWRITEBYTECODE=1
unset MX_QR_VALIDATION_BYPASS SOAP_DIST_QR SOAP_QR_SHAPE_LOG SOAP_QR_DUMP_DIR || true
OUT=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step285_bad8_official_qr
export STEP285_OUT="$OUT"
mkdir -p "$OUT"
cd "$OUT"
python -u "$OUT/step285_bad8_official_qr.py"
echo "$?" > "$OUT/launcher_rc.txt"
