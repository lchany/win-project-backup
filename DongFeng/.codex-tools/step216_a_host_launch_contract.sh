#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 7 ]; then
  echo "usage: $0 TOOL_ROOT REPO CONFIG CHECKPOINT ADAPTER OUTPUT PORT" >&2; exit 64
fi
tool_root=$(readlink -f "$1"); repo=$(readlink -f "$2")
config=$(readlink -f "$3"); checkpoint=$(readlink -f "$4"); adapter=$(readlink -f "$5")
output=$(readlink -m "$6"); port=$7
container=mapqr-leicheng; timeout_seconds=1200; harness_root="$tool_root/harness"
controller="$harness_root/step216_a_world8_controller.py"
summarizer="$harness_root/step216_a_brockett_qr_summarize.py"
contract="$harness_root/step216_a_source_contract.json"
[[ "$tool_root" != "$repo" && "$tool_root" != "$repo"/* ]]
[[ "$adapter" == "$harness_root"/* && "$adapter" != "$repo"/* ]]
[[ "$output" == "$tool_root"/runs/step216_a_brockett_* && "$output" != "$repo"/* && ! -e "$output" ]]
mapfile -t names < <(docker ps --filter "name=^/${container}$" --format '{{.Names}}')
[[ "${#names[@]}" -eq 1 && "${names[0]}" == "$container" ]]
active=$(docker exec "$container" bash -lc "pgrep -af 'torchrun|torch.distributed.run|train_spetr.py|tools/train.py|msprof' | grep -v -E 'pgrep|grep' | wc -l")
[[ "$active" -eq 0 ]]
python3 "$controller" --output-dir "$tool_root" --port "$port" --preflight-only

host_job_pid=; host_job_pgid=; controller_pid=; controller_pgid=; cleanup_done=0
terminate_container_launcher_group() {
  [[ -f "$output/launcher.pgid" ]] || return 0
  docker exec -i "$container" bash -s -- "$output/launcher.pgid" <<'SH'
set -u
pgid_file=$1
[[ -f "$pgid_file" && ! -L "$pgid_file" ]]
IFS= read -r pgid < "$pgid_file"
[[ "$pgid" =~ ^[1-9][0-9]*$ && "$pgid" -gt 1 ]]
kill -TERM -- "-$pgid" 2>/dev/null || true
for _ in 1 2 3 4 5; do kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; done
kill -KILL -- "-$pgid" 2>/dev/null || true
SH
}
terminate_exact_group() {
  local pid=$1 pgid=$2
  [[ -n "$pid" && -n "$pgid" ]] || return 0
  kill -TERM -- "-$pgid" 2>/dev/null || true
  for _ in 1 2 3 4 5; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$pgid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
cleanup_all_groups() {
  [[ "$cleanup_done" -eq 0 ]] || return 0
  cleanup_done=1; set +e
  touch "$output/release_after_npu_smi" 2>/dev/null || true
  terminate_container_launcher_group
  terminate_exact_group "$controller_pid" "$controller_pgid"
  terminate_exact_group "$host_job_pid" "$host_job_pgid"
  set -e
}
trap cleanup_all_groups EXIT INT TERM

setsid timeout --signal=TERM --kill-after=5 "$timeout_seconds" docker exec \
  -e STEP216_TOOL_ROOT="$tool_root" -e BUSINESS_REPO="$repo" -e CONFIG_PATH="$config" \
  -e CHECKPOINT_PATH="$checkpoint" -e ADAPTER_PATH="$adapter" -e OUTPUT_DIR="$output" \
  -e SUMMARIZER="$summarizer" \
  -e MASTER_PORT="$port" -e STEP216_TIMEOUT_SECONDS="$timeout_seconds" \
  "$container" bash "$harness_root/step216_a_run_inside_container.sh" &
host_job_pid=$!; host_job_pgid=$host_job_pid
printf '%s\n' "$host_job_pid" > "$tool_root/$(basename "$output").host_job.pid"
printf '%s\n' "$host_job_pgid" > "$tool_root/$(basename "$output").host_job.pgid"
for _ in $(seq 1 120); do
  [[ -d "$output" ]] && break
  kill -0 "$host_job_pid" 2>/dev/null || break
  sleep 0.25
done
if [[ ! -d "$output" ]]; then
  cleanup_all_groups; status=71
else
  setsid timeout --signal=TERM --kill-after=5 "$timeout_seconds" python3 "$controller" \
    --output-dir "$output" --port "$port" --launcher-pid "$host_job_pid" \
    --timeout-seconds "$timeout_seconds" > "$output/host_controller.log" 2>&1 &
  controller_pid=$!; controller_pgid=$controller_pid
  printf '%s\n' "$controller_pid" > "$output/host_controller.pid"
  printf '%s\n' "$controller_pgid" > "$output/host_controller.pgid"
  set +e; wait "$controller_pid"; controller_status=$?; set -e
  controller_pid=; controller_pgid=
  if [[ "$controller_status" -ne 0 ]]; then
    cleanup_all_groups; status=$controller_status
  else
    set +e; wait "$host_job_pid"; runner_status=$?; set -e
    host_job_pid=; host_job_pgid=
    if [[ "$runner_status" -ne 0 ]]; then cleanup_all_groups; fi
    status=$runner_status
  fi
fi
[[ -d "$output" ]] || mkdir -p "$output"
set +e
python3 "$controller" --output-dir "$output" --port "$port" --postflight-only
postflight_status=$?
if [[ "$status" -eq 0 && "$postflight_status" -eq 0 ]]; then
  python3 "$summarizer" --output-dir "$output" --source-contract "$contract"
  summary_status=$?
  [[ "$summary_status" -eq 0 ]] || status=$summary_status
fi
set -e
[[ "$postflight_status" -eq 0 ]] || status=70
find "$output" -type f ! -name artifacts_sha256.txt ! -name artifacts_sha256.txt.tmp \
  -print0 | sort -z | xargs -0 sha256sum > "$output/artifacts_sha256.txt.tmp"
mv "$output/artifacts_sha256.txt.tmp" "$output/artifacts_sha256.txt"
host_job_pid=; host_job_pgid=; controller_pid=; controller_pgid=; trap - EXIT INT TERM
exit "$status"
