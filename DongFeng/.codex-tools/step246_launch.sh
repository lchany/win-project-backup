#!/usr/bin/env bash
# STEP-246: 63861df CPU FP64 bilateral SOAP + one_sided=None, 30-step loss gate
set -euo pipefail

REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
STAMP=$(date +%Y%m%dT%H%M%S)
OUT=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step246_cpu_fp64_no_onesided_30step_${STAMP}
COMPAT=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step238_soap_preqr_30step_20260818T000235/overlay/soap_63861df_compat.py
SOAP=${REPO}/projects/mmdet3d_plugin/optimizers/soap.py
LAUNCH=${REPO}/diagnostics/tools/npu_official_launch.sh

mkdir -p "$OUT/overlay" "$OUT/work" "$OUT/logs"
echo "$OUT" > /mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step246_latest_dir.txt

cd "$REPO"
cp "$SOAP" "$OUT/overlay/soap_HEAD_step244.py"
sha256sum "$SOAP" > "$OUT/overlay/soap_HEAD.sha256"

cp "$COMPAT" "$SOAP"
python3 -m py_compile "$SOAP"

python3 - <<'PY'
from mmcv import Config
c = Config.fromfile("projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py")
print("onesided", c.optimizer.get("one_sided_dim_threshold", "MISSING"))
PY

t=$(python3 -c "from pathlib import Path; t=Path('$SOAP').read_text(encoding='utf-8', errors='replace'); print('eigh', t.count('eigh'), 'cpu', t.count('.cpu()'), 'float64', t.count('float64'), 'wave', t.count('_parallel_qr_waves'))")
echo "soap_audit $t" | tee "$OUT/overlay/soap_audit.txt"

echo "STEP246_OUT=$OUT"
