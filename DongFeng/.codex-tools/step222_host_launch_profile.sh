#!/usr/bin/env bash
# STEP-222: one-shot Level0 low-overhead ordinary-step profile under SOAP_STALE_Q_K=4.
set -euo pipefail

REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
TOOL=/mnt/sfs_turbo/workdir/wfc1_leicheng/step222_toolroot
DIAG_ROOT=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics
GCONTRACT=$DIAG_ROOT/gpu_contract_alignment_f922c38_8npu_20260814T172611
HARNESS=$GCONTRACT/test_harness/ddp_train_30.sh
TRAIN=$GCONTRACT/tools/train_spetr_gpu_seed0_runtime.py
BASE_CFG=$GCONTRACT/config/aligned_gpu_contract_npu_runtime.py
EXPECTED_HARNESS_SHA=10ad92c723164d52b32734734c8b466f313200165ec1307cb7199e298bb1e0fc
EXPECTED_CFG_SHA=02aca0c7a33e05e972e268aecc1932bbf10f611fa523ec4696b94d58cd7f56a5
EXPECTED_HEAD=2846401b8e9742e2b8ac51c4fc99331b5874530a
CONTAINER=mapqr-leicheng

STAMP=${STAMP:-$(date +%Y%m%dT%H%M%S)}
OUT=${OUT:-$DIAG_ROOT/step222_p1_level0_ordinary_k4_8npu_$STAMP}
PORT=${PORT:-30010}
MAX_ITERS=${MAX_ITERS:-26}

sha_of() { sha256sum "$1" | awk '{print $1}'; }

require_idle() {
  local busy
  busy=$(npu-smi info 2>/dev/null | grep -c 'python' || true)
  if [[ "$busy" != "0" ]]; then
    echo "FAIL: NPU still has python processes" >&2
    npu-smi info 2>/dev/null | sed -n '/Process id/,$p' | head -n 40 >&2 || true
    exit 2
  fi
  if ss -ltn | awk '{print $4}' | grep -q ":${PORT}$"; then
    echo "FAIL: port $PORT busy" >&2
    exit 2
  fi
}

mkdir -p "$OUT/raw" "$OUT/work" "$OUT/logs" "$TOOL/tools"
[[ "$(sha_of "$HARNESS")" == "$EXPECTED_HARNESS_SHA" ]] || { echo "FAIL harness sha"; exit 2; }
[[ "$(sha_of "$BASE_CFG")" == "$EXPECTED_CFG_SHA" ]] || { echo "FAIL config sha"; exit 2; }
docker inspect -f '{{.State.Running}}' "$CONTAINER" | grep -qx true \
  || { echo "FAIL container not running"; exit 2; }
head_now=$(cd "$REPO" && git rev-parse HEAD)
[[ "$head_now" == "$EXPECTED_HEAD" ]] || { echo "FAIL HEAD $head_now"; exit 2; }
echo "$head_now" | tee "$OUT/repo_head.txt"

# Build overlay config into OUT (never touch business repo config).
python3 "$TOOL/tools/step222_build_profile_config.py" "$BASE_CFG" "$OUT/step222_profile_config.py"
cp -a "$TOOL/tools/step222_low_overhead_profiler_hook.py" "$OUT/step222_low_overhead_profiler_hook.py"

# Also stage hook next to config import path via PYTHONPATH=$OUT:$GCONTRACT/tools:...
require_idle

cat > "$OUT/launch_inside.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SOAP_STALE_Q_K=4
export MASTER_PORT=$PORT
export GPUS=8
export MODE=single
export MAX_ITERS=$MAX_ITERS
export WORK_DIRS=$OUT/work
export STEP222_PROFILE_WORK_DIR=$OUT/work
export STEP222_PROFILE_OUTPUT=$OUT/raw
unset TASK_QUEUE_ENABLE || true
unset HCCL_DETERMINISTIC DETERMINISTIC PYTHONHASHSEED || true
base_py="\${PYTHONPATH:-}"
export PYTHONPATH="$OUT:$GCONTRACT/tools:$REPO/mmdetection3d-0.17.1:$REPO:\$base_py"
echo "STEP222_PROFILE_CONTRACT=k4_level0_no_stack_no_shapes_wait22_warmup1_active2_steps23-24"
bash "$HARNESS" "$TRAIN" "$OUT/step222_profile_config.py"
EOF
chmod +x "$OUT/launch_inside.sh"

set +e
docker exec \
  -e ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e SOAP_STALE_Q_K=4 \
  -e MASTER_PORT="$PORT" \
  -e GPUS=8 \
  -e MODE=single \
  -e MAX_ITERS="$MAX_ITERS" \
  -e WORK_DIRS="$OUT/work" \
  -e STEP222_PROFILE_WORK_DIR="$OUT/work" \
  -e STEP222_PROFILE_OUTPUT="$OUT/raw" \
  -e REPO_DIR="$REPO" \
  "$CONTAINER" bash "$OUT/launch_inside.sh" \
  >"$OUT/logs/train.stdout" 2>"$OUT/logs/train.stderr"
rc=$?
set -e
echo "STEP222_PROFILE_TRAIN_EXIT=$rc" | tee "$OUT/logs/exit.txt"
exit "$rc"
