#!/usr/bin/env bash
set -euo pipefail
: "${STEP215_ROOT:?}"
: "${BUSINESS_REPO:?}"
: "${CONFIG_PATH:?}"
: "${CHECKPOINT_PATH:?}"
: "${ADAPTER_PATH:?}"
: "${OUTPUT_DIR:?}"
: "${MASTER_PORT:?}"
: "${EXPECTED_SOAP_SHA256:?}"
basis_relaxed_diagnostic=${BASIS_RELAXED_DIAGNOSTIC:-0}
q_orthogonality_limit=${Q_ORTHOGONALITY_LIMIT:-1e-5}
if [ "$basis_relaxed_diagnostic" != 0 ]; then
  if [ "$basis_relaxed_diagnostic" != 1 ]; then
    echo "BASIS_RELAXED_DIAGNOSTIC must be 0 or 1" >&2
    exit 64
  fi
fi
if [ "$q_orthogonality_limit" != 1e-5 ]; then
  if [ "$q_orthogonality_limit" != 2e-5 ]; then
    echo "Q_ORTHOGONALITY_LIMIT must be 1e-5 or calibrated 2e-5" >&2
    exit 64
  fi
fi
root=$(readlink -f "$STEP215_ROOT")
repo=$(readlink -f "$BUSINESS_REPO")
config=$(readlink -f "$CONFIG_PATH")
checkpoint=$(readlink -f "$CHECKPOINT_PATH")
adapter=$(readlink -f "$ADAPTER_PATH")
output=$(readlink -m "$OUTPUT_DIR")
contract_root=$(dirname "$(dirname "$config")")
[[ -d "$repo/.git" && -f "$config" && -f "$checkpoint" && -f "$adapter" && ! -e "$output" ]]
[[ -d "$contract_root/tools" && -d "$repo/mmdetection3d-0.17.1" ]]
[[ "$adapter" != "$repo"/* && "$output" != "$repo"/* ]]
harness="$root/harness/step215_e_soap_two_cycle_gate.py"
controller="$root/harness/step214_j_ready_controller.py"
mkdir -p "$output"
exec > >(tee "$output/wrapper.log") 2>&1
echo contract=step215_e_real_soap_three_track_two_cycle_resume_world8
date -u +start_utc=%FT%TZ
sha256sum "$harness" "$controller" "$adapter" > "$output/source_sha256.txt"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$contract_root/tools:$repo/mmdetection3d-0.17.1:$repo:${PYTHONPATH:-}"
unset TASK_QUEUE_ENABLE
cd "$repo"
basis_args=()
if [ "$basis_relaxed_diagnostic" = 1 ]; then
  basis_args+=(--basis-relaxed-diagnostic)
fi
setsid python3 -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=8 --master-addr=127.0.0.1 \
  --master-port="$MASTER_PORT" "$harness" \
  --repo "$repo" --config "$config" --checkpoint "$checkpoint" \
  --adapter "$adapter" --output-dir "$output" \
  --expected-soap-sha256 "$EXPECTED_SOAP_SHA256" \
  --q-orthogonality-limit "$q_orthogonality_limit" \
  "${basis_args[@]}" \
  > "$output/torchrun.log" 2>&1 &
launcher_pid=$!
cleanup() {
  touch "$output/release_after_npu_smi" 2>/dev/null || true
  kill -TERM -- "-$launcher_pid" 2>/dev/null || true
  wait "$launcher_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
python3 "$controller" --output-dir "$output" --launcher-pid "$launcher_pid" --timeout-seconds 180
wait "$launcher_pid"
trap - EXIT INT TERM
[[ "$(find "$output/done" -name 'rank*.json' | wc -l)" -eq 8 ]]
python3 - "$output" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])/'world_summary.json'
x=json.loads(p.read_text())
assert x['status']=='PASS' and x['all_rank_pass'] and x['rank_count']==8
PY
find "$output" -type f -print0 | sort -z | xargs -0 sha256sum > "$output/artifacts_sha256.txt"
date -u +end_utc=%FT%TZ
if [ "$basis_relaxed_diagnostic" = 1 ]; then
  echo decision=PASS_STEP215_E_BASIS_RELAXED_DIAGNOSTIC_ONLY
else
  echo decision=PASS_STEP215_E_STRICT_RAW_Q_TWO_CYCLE_RESUME_GATE
fi
