#!/usr/bin/env bash
set -euo pipefail
: "${STEP214_ROOT:?}"
: "${OUTPUT_DIR:?}"
: "${MASTER_PORT:?}"
root=$(readlink -f "$STEP214_ROOT")
output=$(readlink -m "$OUTPUT_DIR")
[[ "$output" == "$root"/runs/step215_a_qr_* && ! -e "$output" ]]
harness="$root/harness/step215_a_qr_exact_primitive_gate.py"
controller="$root/harness/step214_j_ready_controller.py"
mkdir -p "$output"
exec > >(tee "$output/wrapper.log") 2>&1
echo contract=step215_a_qr2560_raw_q_world8
date -u +start_utc=%FT%TZ
sha256sum "$harness" "$controller" > "$output/source_sha256.txt"
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
python3 "$controller" --output-dir "$output" --launcher-pid "$launcher_pid" --timeout-seconds 105
wait "$launcher_pid"
trap - EXIT INT TERM
[[ "$(find "$output/done" -name 'rank*.json' | wc -l)" -eq 8 ]]
find "$output" -type f -print0 | sort -z | xargs -0 sha256sum > "$output/artifacts_sha256.txt"
date -u +end_utc=%FT%TZ
echo decision=COMPLETE_STRICT_RAW_Q_GATE
