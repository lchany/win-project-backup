#!/usr/bin/env bash
# STEP-221 Stage C host launcher: single-variable 30-step A/B for stale-Q.
# Runs inside the host; the training itself is docker-exec'd into mapqr-leicheng.
set -euo pipefail

REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
TOOL=/mnt/sfs_turbo/workdir/wfc1_leicheng/step221_toolroot
DIAG_ROOT=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics
GCONTRACT=$DIAG_ROOT/gpu_contract_alignment_f922c38_8npu_20260814T172611
SOAP_REL=projects/mmdet3d_plugin/optimizers/soap.py
SOAP_SRC=$REPO/$SOAP_REL
PATCHED=$TOOL/harness/soap_stale_q.py
HARNESS=$GCONTRACT/test_harness/ddp_train_30.sh
TRAIN=$GCONTRACT/tools/train_spetr_gpu_seed0_runtime.py
CONFIG=$GCONTRACT/config/aligned_gpu_contract_npu_runtime.py
EXPECTED_SOAP_SHA=0e49429dbca9d9a2546c29f54e79639265f7468703ba4b36fa3b3796861a1077
EXPECTED_HARNESS_SHA=10ad92c723164d52b32734734c8b466f313200165ec1307cb7199e298bb1e0fc
EXPECTED_CFG_SHA=02aca0c7a33e05e972e268aecc1932bbf10f611fa523ec4696b94d58cd7f56a5
CONTAINER=mapqr-leicheng

STAMP=${STAMP:-$(date +%Y%m%dT%H%M%S)}
OUT=${OUT:-$DIAG_ROOT/step221_stage_c_stale_q_30step_8npu_$STAMP}
PORT_BASE=${PORT_BASE:-29998}
MAX_ITERS=${MAX_ITERS:-30}
LABEL=${1:?usage: $0 <baseline|candidate|restore|both>}

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

install_patched() {
  mkdir -p "$OUT/backup"
  local cur
  cur=$(sha_of "$SOAP_SRC")
  if [[ "$cur" == "$EXPECTED_SOAP_SHA" ]]; then
    cp -a "$SOAP_SRC" "$OUT/backup/soap.py.orig"
    cp -a "$PATCHED" "$SOAP_SRC"
    echo "installed_patched sha=$(sha_of "$SOAP_SRC")" | tee "$OUT/soap_install.txt"
  elif [[ -f "$OUT/backup/soap.py.orig" ]] && [[ "$(sha_of "$OUT/backup/soap.py.orig")" == "$EXPECTED_SOAP_SHA" ]]; then
    echo "patched already installed; original backup present"
  else
    echo "FAIL: soap.py SHA $cur is neither original nor a known install" >&2
    exit 2
  fi
}

restore_soap() {
  if [[ -f "$OUT/backup/soap.py.orig" ]]; then
    cp -a "$OUT/backup/soap.py.orig" "$SOAP_SRC"
    echo "restored_soap sha=$(sha_of "$SOAP_SRC")" | tee -a "$OUT/soap_install.txt"
  fi
  local cur
  cur=$(sha_of "$SOAP_SRC")
  if [[ "$cur" != "$EXPECTED_SOAP_SHA" ]]; then
    echo "FAIL: soap.py not restored to expected SHA" >&2
    exit 2
  fi
}

preflight() {
  mkdir -p "$OUT"
  [[ "$(sha_of "$HARNESS")" == "$EXPECTED_HARNESS_SHA" ]] || { echo "FAIL harness sha"; exit 2; }
  [[ "$(sha_of "$CONFIG")" == "$EXPECTED_CFG_SHA" ]] || { echo "FAIL config sha"; exit 2; }
  [[ "$(sha_of "$PATCHED")" != "" ]] || { echo "FAIL patched missing"; exit 2; }
  docker inspect -f '{{.State.Running}}' "$CONTAINER" | grep -qx true \
    || { echo "FAIL container not running"; exit 2; }
  (cd "$REPO" && git rev-parse HEAD) | tee "$OUT/repo_head.txt"
  echo "preflight_ok" | tee "$OUT/preflight.txt"
}

run_one() {
  local name=$1
  local k=$2
  local port=$3
  local run=$OUT/${name}_run
  mkdir -p "$run/work"
  PORT=$port
  require_idle
  install_patched

  cat > "$run/launch_inside.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset TASK_QUEUE_ENABLE || true
export SOAP_STALE_Q_K=$k
export MODE=single
export GPUS=8
export MASTER_PORT=$port
export MAX_ITERS=$MAX_ITERS
export WORK_DIRS=$run/work
export PYTHONPATH="$REPO/mmdetection3d-0.17.1:$REPO:\${PYTHONPATH:-}"
echo "SOAP_STALE_Q_K=\$SOAP_STALE_Q_K"
echo "MAX_ITERS=\$MAX_ITERS"
echo "soap_sha=\$(sha256sum $SOAP_SRC | awk '{print \$1}')"
python3 - <<'PY'
import torch, torch_npu
print('torch', torch.__version__, 'torch_npu', torch_npu.__version__, 'npu', torch.npu.is_available(), 'count', torch.npu.device_count())
PY
set +e
bash "$HARNESS" "$TRAIN" "$CONFIG"
rc=\$?
set -e
echo TRAIN_EXIT=\$rc
exit \$rc
EOF
  chmod +x "$run/launch_inside.sh"

  echo "start_$name port=$port k=$k" | tee "$run/start.txt"
  # login shell so CANN env is present; keep allocator export and stale-Q k.
  nohup docker exec "$CONTAINER" bash -lc "bash $run/launch_inside.sh" \
    > "$run/wrapper.log" 2>&1 &
  echo $! > "$run/docker_exec_host_pid.txt"
  echo "launched host_pid=$(cat "$run/docker_exec_host_pid.txt")"
}

wait_one() {
  local name=$1
  local run=$OUT/${name}_run
  local pid
  pid=$(cat "$run/docker_exec_host_pid.txt")
  echo "waiting $name pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    # progress crumb
    if [[ -f "$run/work/train.log" ]]; then
      tail -n 1 "$run/work/train.log" | tr '\r' '\n' | tail -n 1 || true
    fi
  done
  wait "$pid" || true
  echo "finished_$name" | tee "$run/finished.txt"
}

case "$LABEL" in
  preflight) preflight ;;
  baseline)
    preflight
    run_one baseline 0 "$PORT_BASE"
    wait_one baseline
    ;;
  candidate)
    preflight
    run_one candidate 4 "$((PORT_BASE+1))"
    wait_one candidate
    ;;
  restore) restore_soap ;;
  both)
    preflight
    run_one baseline 0 "$PORT_BASE"
    wait_one baseline
    run_one candidate 4 "$((PORT_BASE+1))"
    wait_one candidate
    restore_soap
    ;;
  *) echo "unknown label $LABEL"; exit 2 ;;
esac
