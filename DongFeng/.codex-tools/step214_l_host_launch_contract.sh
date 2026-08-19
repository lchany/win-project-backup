#!/usr/bin/env bash
set -euo pipefail
[[ "$#" -eq 3 ]];r=$1;o=$2;p=$3;c=mapqr-leicheng;mapfile -t n < <(docker ps --filter "name=^/${c}$" --format '{{.Names}}');[[ "${#n[@]}" -eq 1 && "${n[0]}" == "$c" && ! -e "$o" ]]
a=$(docker exec "$c" bash -lc "pgrep -af 'torchrun|torch.distributed.run|train_spetr.py|tools/train.py|msprof'|grep -v -E 'pgrep|grep'|wc -l");[[ "$a" -eq 0 ]]
set +e;timeout --signal=TERM --kill-after=5 120 docker exec -e STEP214_ROOT="$r" -e OUTPUT_DIR="$o" -e MASTER_PORT="$p" "$c" bash "$r/harness/step214_l_run_inside_container.sh";x=$?;set -e
if [[ "$x" -eq 124 || "$x" -eq 137 ]];then docker exec "$c" bash -c "touch '$o/release_after_npu_smi';pkill -TERM -f '$o'||true";fi;exit "$x"
