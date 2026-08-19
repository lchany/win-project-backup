#!/usr/bin/env bash
set -euo pipefail
: "${STEP215_ROOT:?}"
: "${OUTPUT_DIR:?}"
: "${MASTER_PORT:?}"
root=$(readlink -f "$STEP215_ROOT")
output=$(readlink -m "$OUTPUT_DIR")
[[ "$output" == "$root"/runs/step215_qr_24shape_* && ! -e "$output" ]]
harness="$root/harness/step215_qr_24shape_gate.py"
controller="$root/harness/step214_j_ready_controller.py"
summarizer="$root/harness/step215_qr_24shape_summarize.py"
mkdir -p "$output"
exec > >(tee "$output/wrapper.log") 2>&1
echo contract=step215_qr_24_real_shapes_world8_numeric_gate
date -u +start_utc=%FT%TZ
sha256sum "$harness" "$controller" "$summarizer" > "$output/source_sha256.txt"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
export OUTPUT_DIR
setsid python3 -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=8 --master-addr=127.0.0.1 \
  --master-port="$MASTER_PORT" "$harness" --output-dir "$output" \
  > "$output/torchrun.log" 2>&1 &
launcher_pid=$!
cleanup() {
  touch "$output/release_after_npu_smi" 2>/dev/null || true
  kill -TERM -- "-$launcher_pid" 2>/dev/null || true
  wait "$launcher_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
python3 "$controller" --output-dir "$output" --launcher-pid "$launcher_pid" --timeout-seconds 900
wait "$launcher_pid"
trap - EXIT INT TERM
[[ "$(find "$output/done" -name 'rank*.json' | wc -l)" -eq 8 ]]
python3 "$summarizer" --output-dir "$output"
find "$output" -type f -print0 | sort -z | xargs -0 sha256sum > "$output/artifacts_sha256.txt"
date -u +end_utc=%FT%TZ
echo decision=COMPLETE_24SHAPE_NUMERICAL_GATE
