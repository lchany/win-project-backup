#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 3 ]]; then echo "usage: $0 STEP214_ROOT OUTPUT_DIR MASTER_PORT" >&2; exit 2; fi
step214_root=$1; output_dir=$2; master_port=$3; container='mapqr-leicheng'
mapfile -t exact_names < <(docker ps --filter "name=^/${container}$" --format '{{.Names}}')
[[ "${#exact_names[@]}" -eq 1 && "${exact_names[0]}" == "$container" ]]
[[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]]
[[ ! -L "$step214_root" ]]
[[ "$(readlink -f -- "$step214_root")" == *'/diagnostics/step214_triton_ascend_3.2.0rc4' ]]
[[ "$(readlink -m -- "$output_dir")" == "$(readlink -f -- "$step214_root")"/runs/step214_h_atomic_* ]]
[[ ! -e "$output_dir" ]]
active=$(docker exec "$container" bash -lc "pgrep -af 'torchrun|torch.distributed.run|train_spetr.py|tools/train.py|msprof' | grep -v -E 'pgrep|grep' | wc -l")
[[ "$active" -eq 0 ]]
set +e
timeout --signal=TERM --kill-after=5 120 docker exec -e STEP214_ROOT="$step214_root" \
  -e OUTPUT_DIR="$output_dir" -e MASTER_PORT="$master_port" "$container" \
  bash "$step214_root/harness/step214_h_run_inside_container.sh"
rc=$?
set -e
if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
  docker exec "$container" bash -c "touch '$output_dir/release_after_npu_smi' 2>/dev/null || true; pkill -TERM -f '$output_dir' 2>/dev/null || true"
fi
exit "$rc"
