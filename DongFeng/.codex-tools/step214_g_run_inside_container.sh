#!/usr/bin/env bash
set -euo pipefail

: "${STEP214_ROOT:?STEP214_ROOT is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"

expected_suffix='/diagnostics/step214_triton_ascend_3.2.0rc4'
resolved_root=$(readlink -f -- "$STEP214_ROOT")
resolved_output=$(readlink -m -- "$OUTPUT_DIR")
[[ "$resolved_root" == *"$expected_suffix" ]]
[[ "$resolved_output" == "$resolved_root"/runs/step214_g_* ]]
[[ ! -L "$STEP214_ROOT" ]]
[[ ! -e "$OUTPUT_DIR" ]]
[[ -x "$resolved_root/venv/bin/python" ]]
harness="$resolved_root/harness/step214_g_triton_vector_add_gate.py"
controller="$resolved_root/harness/step214_g_ready_controller.py"
[[ -f "$harness" && ! -L "$harness" ]]
[[ -f "$controller" && ! -L "$controller" ]]
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]]
(( MASTER_PORT >= 1024 && MASTER_PORT <= 65535 ))

mkdir -p -- "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/wrapper.log") 2>&1
printf 'contract=step214_g_triton_ascend_vector_add_world8_back8\n'
printf 'start_utc=%s\n' "$(date -u +%FT%TZ)"
sha256sum "$harness" "$controller" "$resolved_root/harness/step214_g_run_inside_container.sh" > "$OUTPUT_DIR/source_sha256.txt"

global_snapshot() {
  python3 - <<'PY'
import importlib.metadata as metadata
import importlib.util
import json
import pathlib
import triton

try:
    ascend_dist = metadata.version("triton-ascend")
except metadata.PackageNotFoundError:
    ascend_dist = "MISSING"
payload = {
    "triton_dist": metadata.version("triton"),
    "triton_ascend_dist": ascend_dist,
    "triton_module": str(pathlib.Path(triton.__file__).resolve()),
    "ascend_backend_spec": bool(importlib.util.find_spec("triton.backends.ascend")),
}
print(json.dumps(payload, sort_keys=True))
PY
}

global_snapshot > "$OUTPUT_DIR/global_before.json"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
export OUTPUT_DIR

setsid "$resolved_root/venv/bin/python" -m torch.distributed.run \
  --nnodes=1 \
  --nproc-per-node=8 \
  --master-addr=127.0.0.1 \
  --master-port="$MASTER_PORT" \
  "$harness" \
  --output-dir "$OUTPUT_DIR" \
  > "$OUTPUT_DIR/torchrun.log" 2>&1 &
launcher_pid=$!
launcher_pgid=$launcher_pid
cleanup() {
  touch "$OUTPUT_DIR/release_after_npu_smi" 2>/dev/null || true
  if kill -0 "$launcher_pid" 2>/dev/null; then
    kill -TERM -- "-$launcher_pgid" 2>/dev/null || true
    wait "$launcher_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$resolved_root/venv/bin/python" "$controller" \
  --output-dir "$OUTPUT_DIR" \
  --launcher-pid "$launcher_pid" \
  --timeout-seconds 105
wait "$launcher_pid"
trap - EXIT INT TERM
done_count=$(find "$OUTPUT_DIR/done" -maxdepth 1 -type f -name 'rank*.json' | wc -l)
[[ "$done_count" -eq 8 ]]
[[ ! -s "$OUTPUT_DIR/torchrun.log" ]] || ! grep -Eqi 'traceback|fatal|error|oom|failed' "$OUTPUT_DIR/torchrun.log"
global_snapshot > "$OUTPUT_DIR/global_after.json"
cmp "$OUTPUT_DIR/global_before.json" "$OUTPUT_DIR/global_after.json"
find "$OUTPUT_DIR" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUTPUT_DIR/artifacts_sha256.txt"
printf 'end_utc=%s\n' "$(date -u +%FT%TZ)"
printf 'decision=PASS_WORLD8_BACK8_EXACT_GLOBAL_UNCHANGED\n'
