#!/usr/bin/env bash
set -euo pipefail

repo=$1
diag=$2
port=$3
run_dir="$diag/profile_once"
raw_dir="$run_dir/raw"
work_dir="$run_dir/work"

mkdir -p "$raw_dir" "$work_dir"
if find "$raw_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to reuse non-empty raw profile directory" >&2
    exit 41
fi
if find "$work_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to reuse non-empty profile work directory" >&2
    exit 42
fi

echo "PROFILE_CONTRACT=container_mapqr-leicheng_world8_back_devices8-15_seed0_deterministicFalse_wait22_warmup1_active4_steps23-26_stack_shapes"

set +e
docker exec \
    -e MASTER_PORT="$port" \
    -e GPUS=8 \
    -e MODE=single \
    -e MAX_ITERS=28 \
    -e WORK_DIRS="$work_dir" \
    -e GPU_CONTRACT_PROFILE_WORK_DIR="$work_dir" \
    -e GPU_CONTRACT_PROFILE_OUTPUT="$raw_dir" \
    -e REPO_DIR="$repo" \
    -e ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    mapqr-leicheng bash -lc "
        set -euo pipefail
        repo='$repo'
        diag='$diag'
        base_py=\"\$PYTHONPATH\"
        export PYTHONPATH=\"\$diag/tools:\$repo/mmdetection3d-0.17.1:\$repo:\$base_py\"
        unset HCCL_DETERMINISTIC DETERMINISTIC PYTHONHASHSEED
        cd \"\$repo\"
        bash \"\$diag/test_harness/ddp_train_30.sh\" \
            \"\$diag/tools/train_spetr_gpu_seed0_runtime.py\" \
            \"\$diag/config/gpu_contract_profile_once.py\"
    "
rc=$?
set -e
echo "GPU_CONTRACT_PROFILE_TRAIN_EXIT=$rc"
exit "$rc"
