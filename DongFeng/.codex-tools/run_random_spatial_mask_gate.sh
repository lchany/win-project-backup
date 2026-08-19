#!/usr/bin/env bash
set -euo pipefail

: "${OUT_DIR:?OUT_DIR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"

export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  "${HARNESS_PATH:?HARNESS_PATH is required}" \
  --output-dir "${OUT_DIR}" \
  --warmup 5 \
  --repeats 15
