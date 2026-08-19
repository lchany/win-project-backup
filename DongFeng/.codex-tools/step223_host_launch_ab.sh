#!/usr/bin/env bash
# STEP-223: 30-step A/B of DataContainer pin under SOAP_STALE_Q_K=4.
set -euo pipefail

REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
TOOL=/mnt/sfs_turbo/workdir/wfc1_leicheng/step223_toolroot
DIAG_ROOT=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics
GCONTRACT=$DIAG_ROOT/gpu_contract_alignment_f922c38_8npu_20260814T172611
DC_REL=mmcv/parallel/data_container.py
DC_SRC=$REPO/$DC_REL
HARNESS=$GCONTRACT/test_harness/ddp_train_30.sh
TRAIN=$GCONTRACT/tools/train_spetr_gpu_seed0_runtime.py
CONFIG=$GCONTRACT/config/aligned_gpu_contract_npu_runtime.py
EXPECTED_HARNESS_SHA=10ad92c723164d52b32734734c8b466f313200165ec1307cb7199e298bb1e0fc
EXPECTED_CFG_SHA=02aca0c7a33e05e972e268aecc1932bbf10f611fa523ec4696b94d58cd7f56a5
EXPECTED_HEAD=2846401b8e9742e2b8ac51c4fc99331b5874530a
EXPECTED_DC_SHA=a4e4fad55023bf0f384946de54bca8c02a4546fbd4b6abb22984560d9dc39cce
CONTAINER=mapqr-leicheng

STAMP=${STAMP:-$(date +%Y%m%dT%H%M%S)}
OUT=${OUT:-$DIAG_ROOT/step223_pin_staleq_k4_30step_8npu_$STAMP}
PORT_BASE=${PORT_BASE:-30020}
MAX_ITERS=${MAX_ITERS:-30}
LABEL=${1:?usage: $0 <baseline|candidate|restore|both>}

sha_of() { sha256sum "$1" | awk '{print $1}'; }

require_idle() {
  local busy
  busy=$(npu-smi info 2>/dev/null | grep -c 'python' || true)
  if [[ "$busy" != "0" ]]; then
    echo "FAIL: NPU still has python processes" >&2
    exit 2
  fi
  if ss -ltn | awk '{print $4}' | grep -q ":${PORT}$"; then
    echo "FAIL: port $PORT busy" >&2
    exit 2
  fi
}

install_pin() {
  mkdir -p "$OUT/backup"
  local cur
  cur=$(sha_of "$DC_SRC")
  if [[ "$cur" == "$EXPECTED_DC_SHA" ]]; then
    cp -a "$DC_SRC" "$OUT/backup/data_container.py.orig"
    python3 "$TOOL/tools/step223_apply_pin.py" "$DC_SRC"
    echo "installed_pin sha=$(sha_of "$DC_SRC")" | tee "$OUT/dc_install.txt"
  elif [[ -f "$OUT/backup/data_container.py.orig" ]] && [[ "$(sha_of "$OUT/backup/data_container.py.orig")" == "$EXPECTED_DC_SHA" ]]; then
    echo "pin already installed; original backup present"
  else
    echo "FAIL: data_container.py SHA $cur unexpected" >&2
    exit 2
  fi
}

restore_dc() {
  if [[ -f "$OUT/backup/data_container.py.orig" ]]; then
    cp -a "$OUT/backup/data_container.py.orig" "$DC_SRC"
    echo "restored_dc sha=$(sha_of "$DC_SRC")" | tee -a "$OUT/dc_install.txt"
  fi
  local cur
  cur=$(sha_of "$DC_SRC")
  if [[ "$cur" != "$EXPECTED_DC_SHA" ]]; then
    echo "FAIL: data_container.py not restored to expected SHA" >&2
    exit 2
  fi
}

preflight() {
  mkdir -p "$OUT"
  [[ "$(sha_of "$HARNESS")" == "$EXPECTED_HARNESS_SHA" ]] || { echo "FAIL harness sha"; exit 2; }
  [[ "$(sha_of "$CONFIG")" == "$EXPECTED_CFG_SHA" ]] || { echo "FAIL config sha"; exit 2; }
  [[ "$(sha_of "$DC_SRC")" == "$EXPECTED_DC_SHA" ]] || { echo "FAIL dc sha before run"; exit 2; }
  docker inspect -f '{{.State.Running}}' "$CONTAINER" | grep -qx true \
    || { echo "FAIL container not running"; exit 2; }
  head_now=$(cd "$REPO" && git rev-parse HEAD)
  [[ "$head_now" == "$EXPECTED_HEAD" ]] || { echo "FAIL HEAD $head_now"; exit 2; }
  echo "$head_now" | tee "$OUT/repo_head.txt"
  echo "preflight_ok" | tee "$OUT/preflight.txt"
}

run_one() {
  local name=$1
  local port=$2
  local use_pin=$3
  local run=$OUT/${name}_run
  mkdir -p "$run/work" "$run/logs"
  PORT=$port
  require_idle
  if [[ "$use_pin" == "1" ]]; then
    install_pin
  else
    restore_dc || true
    # baseline must be clean original
    [[ "$(sha_of "$DC_SRC")" == "$EXPECTED_DC_SHA" ]] || { echo "FAIL baseline dc dirty"; exit 2; }
  fi

  cat > "$run/launch_inside.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SOAP_STALE_Q_K=4
export MASTER_PORT=$port
export GPUS=8
export MODE=single
export MAX_ITERS=$MAX_ITERS
export WORK_DIRS=$run/work
unset TASK_QUEUE_ENABLE || true
unset HCCL_DETERMINISTIC DETERMINISTIC PYTHONHASHSEED || true
base_py="\${PYTHONPATH:-}"
export PYTHONPATH="$GCONTRACT/tools:$REPO/mmdetection3d-0.17.1:$REPO:\$base_py"
echo "STEP223_CONTRACT=k4_pin_${name}_dcsha=\$(sha256sum $DC_SRC | awk '{print \$1}')"
python3 - <<'PY'
from mmcv.parallel.data_container import DataContainer
import torch
print("DC_FILE", __import__("mmcv.parallel.data_container", fromlist=["x"]).__file__)
print("HAS_PIN", hasattr(DataContainer, "pin_memory"))
t = torch.randn(8, 8)
dc = DataContainer(t)
if hasattr(dc, "pin_memory"):
    dc.pin_memory()
    print("PINNED", dc.data.is_pinned())
else:
    print("PINNED", False)
PY
bash "$HARNESS" "$TRAIN" "$CONFIG"
EOF
  chmod +x "$run/launch_inside.sh"

  set +e
  docker exec \
    -e ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
    -e SOAP_STALE_Q_K=4 \
    -e MASTER_PORT="$port" \
    -e GPUS=8 \
    -e MODE=single \
    -e MAX_ITERS="$MAX_ITERS" \
    -e WORK_DIRS="$run/work" \
    -e REPO_DIR="$REPO" \
    "$CONTAINER" bash "$run/launch_inside.sh" \
    >"$run/logs/train.stdout" 2>"$run/logs/train.stderr"
  rc=$?
  set -e
  echo "STEP223_${name}_EXIT=$rc" | tee "$run/logs/exit.txt"
  return "$rc"
}

summarize() {
  python3 "$TOOL/tools/step223_summarize.py" "$OUT" || true
}

case "$LABEL" in
  preflight) preflight ;;
  baseline)
    preflight
    run_one baseline $((PORT_BASE)) 0
    ;;
  candidate)
    preflight
    run_one candidate $((PORT_BASE+1)) 1
    restore_dc
    ;;
  restore)
    restore_dc
    ;;
  both)
    preflight
    run_one baseline $((PORT_BASE)) 0
    run_one candidate $((PORT_BASE+1)) 1
    restore_dc
    summarize
    ;;
  *)
    echo "bad label" >&2
    exit 2
    ;;
esac
