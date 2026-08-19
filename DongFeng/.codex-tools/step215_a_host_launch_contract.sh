#!/usr/bin/env bash
set -euo pipefail
[[ "$#" -eq 3 ]]
root=$1
output=$2
port=$3
container=mapqr-leicheng
mapfile -t names < <(docker ps --filter "name=^/${container}$" --format '{{.Names}}')
[[ "${#names[@]}" -eq 1 && "${names[0]}" == "$container" && ! -e "$output" ]]
active=$(docker exec "$container" bash -lc \
  "pgrep -af 'torchrun|torch.distributed.run|train_spetr.py|tools/train.py|msprof' | grep -v -E 'pgrep|grep' | wc -l")
[[ "$active" -eq 0 ]]
set +e
timeout --signal=TERM --kill-after=5 120 docker exec \
  -e STEP214_ROOT="$root" -e OUTPUT_DIR="$output" -e MASTER_PORT="$port" \
  "$container" bash "$root/harness/step215_a_run_inside_container.sh"
status=$?
set -e
if [[ "$status" -eq 124 || "$status" -eq 137 ]]; then
  docker exec "$container" bash -c \
    "touch '$output/release_after_npu_smi'; pkill -TERM -f '$output' || true"
fi
exit "$status"
