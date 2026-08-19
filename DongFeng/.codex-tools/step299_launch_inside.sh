#!/bin/bash
set -u
source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
unset MX_QR_VALIDATION_BYPASS SOAP_DIST_QR SOAP_QR_SHAPE_LOG SOAP_QR_DUMP_DIR || true
python -u "$STEP299_RUN_DIR/step299_bad_single_visible_qr7.py"
rc=$?
echo "$rc" > "$STEP299_RUN_DIR/launcher_rc.txt"
exit "$rc"
