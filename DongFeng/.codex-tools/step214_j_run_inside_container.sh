#!/usr/bin/env bash
set -euo pipefail
: "${STEP214_ROOT:?}"; : "${OUTPUT_DIR:?}"; : "${MASTER_PORT:?}"
root=$(readlink -f -- "$STEP214_ROOT"); out=$(readlink -m -- "$OUTPUT_DIR")
[[ "$root" == *'/diagnostics/step214_triton_ascend_3.2.0rc4' && "$out" == "$root"/runs/step214_j_aggregate_* && ! -e "$OUTPUT_DIR" ]]
h="$root/harness/step214_j_triton_atomic_aggregate_gate.py"; c="$root/harness/step214_j_ready_controller.py"
[[ -x "$root/venv/bin/python" && -f "$h" && -f "$c" ]]
mkdir -p "$out"; exec > >(tee -a "$out/wrapper.log") 2>&1
echo contract=step214_j_register_aggregate_world8_back8; date -u +start_utc=%FT%TZ
sha256sum "$h" "$c" "$root/harness/step214_j_run_inside_container.sh" > "$out/source_sha256.txt"
snapshot(){ python3 - <<'PY'
import importlib.metadata as m,importlib.util,json,pathlib,triton
try:a=m.version('triton-ascend')
except m.PackageNotFoundError:a='MISSING'
print(json.dumps({'triton':m.version('triton'),'triton_ascend':a,'module':str(pathlib.Path(triton.__file__).resolve()),'ascend':bool(importlib.util.find_spec('triton.backends.ascend'))},sort_keys=True))
PY
}
snapshot > "$out/global_before.json"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 OUTPUT_DIR
setsid "$root/venv/bin/python" -m torch.distributed.run --nnodes=1 --nproc-per-node=8 --master-addr=127.0.0.1 --master-port="$MASTER_PORT" "$h" --output-dir "$out" > "$out/torchrun.log" 2>&1 &
pid=$!; cleanup(){ touch "$out/release_after_npu_smi" 2>/dev/null||true; kill -TERM -- "-$pid" 2>/dev/null||true; wait "$pid" 2>/dev/null||true; }; trap cleanup EXIT INT TERM
"$root/venv/bin/python" "$c" --output-dir "$out" --launcher-pid "$pid" --timeout-seconds 105
wait "$pid"; trap - EXIT INT TERM
[[ "$(find "$out/done" -type f -name 'rank*.json' | wc -l)" -eq 8 ]]
snapshot > "$out/global_after.json"; cmp "$out/global_before.json" "$out/global_after.json"
find "$out" -type f -print0|sort -z|xargs -0 sha256sum > "$out/artifacts_sha256.txt"
date -u +end_utc=%FT%TZ; echo decision=PASS_REGISTER_AGGREGATE_MECHANISM
