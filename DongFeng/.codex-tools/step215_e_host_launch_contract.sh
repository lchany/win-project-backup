#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 8 ]; then
  if [ "$#" -ne 9 ] && [ "$#" -ne 10 ]; then
    echo "usage: $0 ROOT REPO CONFIG CHECKPOINT ADAPTER OUTPUT PORT SOAP_SHA [--basis-relaxed-diagnostic [--q-orthogonality-limit=2e-5]]" >&2
    exit 64
  fi
  if [ "$9" != --basis-relaxed-diagnostic ]; then
    echo "usage: $0 ROOT REPO CONFIG CHECKPOINT ADAPTER OUTPUT PORT SOAP_SHA [--basis-relaxed-diagnostic [--q-orthogonality-limit=2e-5]]" >&2
    exit 64
  fi
fi
root=$1
repo=$2
config=$3
checkpoint=$4
adapter=$5
output=$6
port=$7
soap_sha=$8
basis_relaxed_diagnostic=0
if [ "$#" -eq 9 ]; then
  basis_relaxed_diagnostic=1
fi
q_orthogonality_limit=1e-5
if [ "$#" -eq 10 ]; then
  if [ "${10}" != --q-orthogonality-limit=2e-5 ]; then
    echo "only the calibrated --q-orthogonality-limit=2e-5 is accepted" >&2
    exit 64
  fi
  basis_relaxed_diagnostic=1
  q_orthogonality_limit=2e-5
fi
container=mapqr-leicheng
mapfile -t names < <(docker ps --filter "name=^/${container}$" --format '{{.Names}}')
[[ "${#names[@]}" -eq 1 && "${names[0]}" == "$container" && ! -e "$output" ]]
active=$(docker exec "$container" bash -lc \
  "pgrep -af 'torchrun|torch.distributed.run|train_spetr.py|tools/train.py|msprof' | grep -v -E 'pgrep|grep' | wc -l")
[[ "$active" -eq 0 ]]
set +e
timeout --signal=TERM --kill-after=5 1800 docker exec \
  -e STEP215_ROOT="$root" -e BUSINESS_REPO="$repo" -e CONFIG_PATH="$config" \
  -e CHECKPOINT_PATH="$checkpoint" -e ADAPTER_PATH="$adapter" -e OUTPUT_DIR="$output" \
  -e MASTER_PORT="$port" -e EXPECTED_SOAP_SHA256="$soap_sha" \
  -e BASIS_RELAXED_DIAGNOSTIC="$basis_relaxed_diagnostic" \
  -e Q_ORTHOGONALITY_LIMIT="$q_orthogonality_limit" \
  "$container" bash "$root/harness/step215_e_run_inside_container.sh"
status=$?
set -e
if [[ "$status" -eq 124 || "$status" -eq 137 ]]; then
  docker exec "$container" bash -c \
    "touch '$output/release_after_npu_smi'; pkill -TERM -f '$output' || true"
fi
exit "$status"
