#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 REPO CONTRACT_DIR OUTPUT_DIR PORT LABEL" >&2
  exit 2
fi

repo=$1
contract_dir=$2
output_dir=$3
master_port=$4
label=$5
entry="$contract_dir/tools/train_spetr_gpu_seed0_runtime.py"
config="$contract_dir/config/aligned_gpu_contract_npu_runtime.py"
launcher="$contract_dir/test_harness/ddp_train_30.sh"

test -d "$repo/.git" || test -f "$repo/.git"
test -f "$entry"
test -f "$config"
test -f "$launcher"
test ! -e "$output_dir"
mkdir -p "$output_dir/work"

exec > >(tee -a "$output_dir/wrapper.log") 2>&1
echo "label=$label"
echo "start_utc=$(date -u +%FT%TZ)"
echo "repo_head=$(git -C "$repo" rev-parse HEAD)"
git -C "$repo" status --porcelain > "$output_dir/git_status_before.txt"
sha256sum "$entry" "$config" "$launcher" > "$output_dir/contract_sha256.txt"

export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
base_py=${PYTHONPATH:-}
export PYTHONPATH="$contract_dir/tools:$repo/mmdetection3d-0.17.1:$repo:$base_py"
export GPUS=8
export MODE=single
export MASTER_PORT="$master_port"
export MLP_WORKER_0_PORT="$master_port"
export MAX_ITERS=30
export WORK_DIRS="$output_dir/work"
export REPO_DIR="$repo"

cd "$repo"
bash "$launcher" "$entry" "$config"
launcher_rc=$?
echo "$launcher_rc" > "$output_dir/launcher_rc.txt"
echo "end_utc=$(date -u +%FT%TZ)"
find "$output_dir" -maxdepth 3 -type f -print0 | sort -z | xargs -0 sha256sum > "$output_dir/files_sha256_before_manifest.txt"
exit "$launcher_rc"
