#!/bin/bash
set -euo pipefail
PKG=/home/ma-user/anaconda3/envs/PyTorch-2.7.1/lib/python3.11/site-packages/mx_driving_cloud
OUT=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step269_192_root/kernel_src
mkdir -p "$OUT"
cp -f "$PKG/ops/linalg.py" "$OUT/linalg.py"
cp -f "$PKG/packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.py" "$OUT/qr_v2.py"
cp -f "$PKG/packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.cpp" "$OUT/qr_v2.cpp"
cp -f "$PKG/packages/vendors/customize/op_api/include/aclnn_qr_v2.h" "$OUT/aclnn_qr_v2.h"
cp -f "$PKG/packages/vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend910b/qr_v2.json" "$OUT/qr_v2_910b.json"
cp -f "$PKG/packages/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/qr_v2/QrV2_566c2e1c0e6c8c92152ad84416d77006.json" "$OUT/QrV2_kernel.json"
echo "=== LINALG ==="
cat "$OUT/linalg.py"
echo "=== QR_V2_PY ==="
cat "$OUT/qr_v2.py"
echo "=== HEADER ==="
cat "$OUT/aclnn_qr_v2.h"
echo "=== JSON ==="
cat "$OUT/qr_v2_910b.json"
echo "=== KERNEL JSON ==="
cat "$OUT/QrV2_kernel.json"
echo "=== CPP HITS ==="
grep -n -E "workspace|Workspace|TILE|tile|64|AICPU|aicpu|pad|Pad|remainder|last|MTE|ubSize|UB_|blockDim|geQrt|GEQRT|TSQRT|LARFB" "$OUT/qr_v2.cpp" | head -n 200
echo "=== DUMP EXISTS ==="
ls -l /mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step260_qr_tensor_dump_30step_20260818T194457/qr_tensors/rank0_step10_ind0_192x192_BAD.pt
