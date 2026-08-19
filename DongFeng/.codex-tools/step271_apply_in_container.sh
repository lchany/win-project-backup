#!/bin/bash
set -euo pipefail
PKG=/home/ma-user/anaconda3/envs/PyTorch-2.7.1/lib/python3.11/site-packages/mx_driving_cloud
DIR=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step271_qr_patch
LINALG="$PKG/ops/linalg.py"
BACKUP="$DIR/linalg.py.bak"
PATCHED="$DIR/linalg_step271.py"
mkdir -p "$DIR"
if [ ! -f "$BACKUP" ]; then
  cp -a "$LINALG" "$BACKUP"
  echo "backup -> $BACKUP"
fi
cp -f "$PATCHED" "$LINALG"
echo "applied STEP-271 linalg.py"
