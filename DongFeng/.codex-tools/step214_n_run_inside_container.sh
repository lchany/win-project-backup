#!/usr/bin/env bash
set -euo pipefail
: "${STEP214_ROOT:?}";: "${OUTPUT_DIR:?}";: "${MASTER_PORT:?}";r=$(readlink -f "$STEP214_ROOT");o=$(readlink -m "$OUTPUT_DIR")
[[ "$o" == "$r"/runs/step214_n_forward_* && ! -e "$o" ]];h="$r/harness/step214_n_msda_forward_persistent_gate.py";c="$r/harness/step214_j_ready_controller.py";mkdir -p "$o";exec > >(tee "$o/wrapper.log") 2>&1
echo contract=step214_n_msda_forward_b1_persistent64_world8;date -u +start_utc=%FT%TZ;sha256sum "$h" "$r/harness/step214_k_msda_forward_gate.py" "$c" > "$o/source_sha256.txt"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 OUTPUT_DIR;unset TRITON_ALL_BLOCKS_PARALLEL||true
setsid "$r/venv/bin/python" -m torch.distributed.run --nnodes=1 --nproc-per-node=8 --master-addr=127.0.0.1 --master-port="$MASTER_PORT" "$h" --output-dir "$o" > "$o/torchrun.log" 2>&1 &p=$!
cleanup() { touch "$o/release_after_npu_smi" 2>/dev/null||true;kill -TERM -- "-$p" 2>/dev/null||true;wait "$p" 2>/dev/null||true;};trap cleanup EXIT INT TERM
"$r/venv/bin/python" "$c" --output-dir "$o" --launcher-pid "$p" --timeout-seconds 105;wait "$p";trap - EXIT INT TERM
[[ "$(find "$o/done" -name 'rank*.json'|wc -l)" -eq 8 ]];find "$o" -type f -print0|sort -z|xargs -0 sha256sum > "$o/artifacts_sha256.txt";date -u +end_utc=%FT%TZ;echo decision=PASS_MSDA_FORWARD_B1_PERSISTENT64
