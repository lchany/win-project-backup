#!/usr/bin/env bash
set -euo pipefail
: "${STEP216_TOOL_ROOT:?}" "${BUSINESS_REPO:?}" "${CONFIG_PATH:?}" "${CHECKPOINT_PATH:?}"
: "${ADAPTER_PATH:?}" "${SUMMARIZER:?}" "${OUTPUT_DIR:?}" "${MASTER_PORT:?}"
timeout_seconds=${STEP216_TIMEOUT_SECONDS:-1200}; [[ "$timeout_seconds" == 1200 ]]
tool_root=$(readlink -f "$STEP216_TOOL_ROOT"); repo=$(readlink -f "$BUSINESS_REPO")
config=$(readlink -f "$CONFIG_PATH"); checkpoint=$(readlink -f "$CHECKPOINT_PATH")
adapter=$(readlink -f "$ADAPTER_PATH"); output=$(readlink -m "$OUTPUT_DIR")
summarizer=$(readlink -f "$SUMMARIZER")
contract_root=$(dirname "$(dirname "$config")"); harness_root="$tool_root/harness"
contract="$harness_root/step216_a_source_contract.json"
gate="$harness_root/step216_a_brockett_qr_gate.py"; policy="$harness_root/step216_a_brockett_policy.py"
controller="$harness_root/step216_a_world8_controller.py"
[[ -d "$repo/.git" && -f "$config" && -f "$checkpoint" && -f "$adapter" && -f "$summarizer" && -f "$contract" && ! -e "$output" ]]
[[ -d "$contract_root/tools" && -d "$repo/mmdetection3d-0.17.1" ]]
[[ "$tool_root" != "$repo" && "$tool_root" != "$repo"/* ]]
[[ "$adapter" == "$harness_root"/* && "$adapter" != "$repo"/* ]]
[[ "$summarizer" == "$harness_root"/* && "$summarizer" != "$repo"/* ]]
[[ "$output" == "$tool_root"/runs/step216_a_brockett_* && "$output" != "$repo"/* ]]
mkdir -p "$output"
exec > >(tee "$output/wrapper.log") 2>&1
echo contract=step216_a_real_checkpoint_brockett_cubic_core_world8_local_screen_v3
date -u +start_utc=%FT%TZ
python3 "$policy" --source-contract "$contract" --source-root "$harness_root"
sha256sum "$contract" "$gate" "$policy" "$controller" "$summarizer" "$adapter" > "$output/source_sha256.txt"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$harness_root:$contract_root/tools:$repo/mmdetection3d-0.17.1:$repo:${PYTHONPATH:-}"
unset TASK_QUEUE_ENABLE
cd "$repo"
launcher_pid=; launcher_pgid=
terminate_launcher_group() {
  if [[ -n "$launcher_pgid" ]]; then
    kill -TERM -- "-$launcher_pgid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do kill -0 "$launcher_pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$launcher_pgid" 2>/dev/null || true
    wait "$launcher_pid" 2>/dev/null || true
  fi
}
trap terminate_launcher_group EXIT INT TERM
setsid python3 -m torch.distributed.run --nnodes=1 --nproc-per-node=8 \
  --master-addr=127.0.0.1 --master-port="$MASTER_PORT" "$gate" \
  --repo "$repo" --config "$config" --checkpoint "$checkpoint" --adapter "$adapter" \
  --source-contract "$contract" --output-dir "$output" > "$output/torchrun.log" 2>&1 &
launcher_pid=$!; launcher_pgid=$launcher_pid
printf '%s\n' "$launcher_pid" > "$output/launcher.pid"
printf '%s\n' "$launcher_pgid" > "$output/launcher.pgid"
wait "$launcher_pid"; launcher_pid=; launcher_pgid=
trap - EXIT INT TERM
[[ "$(find "$output/done" -name 'rank*.json' | wc -l)" -eq 8 ]]
date -u +end_utc=%FT%TZ
