#!/usr/bin/env bash
set -euo pipefail
[[ "$#" -eq 3 ]]; root=$1; out=$2; port=$3; c=mapqr-leicheng
mapfile -t n < <(docker ps --filter "name=^/${c}$" --format '{{.Names}}'); [[ "${#n[@]}" -eq 1 && "${n[0]}" == "$c" ]]
[[ "$(readlink -m -- "$out")" == "$(readlink -f -- "$root")"/runs/step214_j_aggregate_* && ! -e "$out" ]]
active=$(docker exec "$c" bash -lc "pgrep -af 'torchrun|torch.distributed.run|train_spetr.py|tools/train.py|msprof'|grep -v -E 'pgrep|grep'|wc -l"); [[ "$active" -eq 0 ]]
set +e; timeout --signal=TERM --kill-after=5 120 docker exec -e STEP214_ROOT="$root" -e OUTPUT_DIR="$out" -e MASTER_PORT="$port" "$c" bash "$root/harness/step214_j_run_inside_container.sh"; rc=$?; set -e
if [[ "$rc" -eq 124 || "$rc" -eq 137 ]];then docker exec "$c" bash -c "touch '$out/release_after_npu_smi' 2>/dev/null||true;pkill -TERM -f '$out' 2>/dev/null||true";fi
exit "$rc"
