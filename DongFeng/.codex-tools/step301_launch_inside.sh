#!/bin/bash
set -u
source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PYTHONDONTWRITEBYTECODE=1
unset MX_QR_VALIDATION_BYPASS SOAP_DIST_QR SOAP_QR_SHAPE_LOG SOAP_QR_DUMP_DIR || true
python -u "$STEP301_RUN_DIR/step301_bad_multivisible_setdevice_qr7.py"
rc=$?
echo "$rc" > "$STEP301_RUN_DIR/launcher_rc.txt"
exit "$rc"
