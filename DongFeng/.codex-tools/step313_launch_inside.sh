#!/bin/bash
set -euo pipefail

source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1

REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
CONFIG=projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
EXPECTED_HEAD=27b1d6d3f363619ad2faa244abe8fbc5a97faef6
OUT=${STEP313_OUT:?STEP313_OUT is required}

export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
export GPUS=8
export MODE=single
export MAX_ITERS=${MAX_ITERS:-10}
export MASTER_PORT=30182
export WORK_DIRS="$OUT/work"
export LOG_DIR="$OUT"
export SOAP_QR_SHAPE_LOG=1
export SOAP_QR_DUMP_DIR="$OUT/qr_tensors"

unset MX_QR_VALIDATION_BYPASS SOAP_DIST_QR || true
unset HCCL_DETERMINISTIC DETERMINISTIC PYTHONHASHSEED || true
unset ASCEND_LAUNCH_BLOCKING PROFILING_MODE PROFILE PROFILE_DIR HCCL_IF_BASE_PORT || true

mkdir -p "$OUT/logs" "$WORK_DIRS" "$SOAP_QR_DUMP_DIR"
cd "$REPO"
export PYTHONPATH="${PWD}/mmdetection3d-0.17.1:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

cat > "${REPO}/sitecustomize.py" <<'PY'
import os
import time
import json
try:
    import torch
except Exception:
    torch = None

dump_dir = os.environ.get("SOAP_QR_DUMP_DIR", "") or ""
shape_log = os.environ.get("SOAP_QR_SHAPE_LOG", "") not in ("", "0", "false", "False")
rank = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_PROCID") or 0)
qr_backend = (os.environ.get("SOAP_QR_BACKEND", "mx") or "mx").lower()
_counter = 0
_dumped_count = 0
try:
    DUMP_MAX_CALLS = int(os.environ.get("SOAP_QR_DUMP_MAX_CALLS", "8"))
except Exception:
    DUMP_MAX_CALLS = 8

def _next_counter():
    global _counter
    _counter += 1
    return _counter

def _is_finite(x):
    try:
        return bool(torch.isfinite(x).all().item())
    except Exception:
        return False

def _dump_one(A, where, note):
    if (torch is None) or (not dump_dir):
        return
    try:
        global _dumped_count
        if _dumped_count >= DUMP_MAX_CALLS:
            return
        n = None
        if hasattr(A, "shape") and len(A.shape) == 2 and A.shape[0] == A.shape[1]:
            n = int(A.shape[0])
        c = _next_counter()
        fname = f"rank{rank}_{where}_qr{c}_{n}x{n}_{note}.pt"
        torch.save(A.detach().to("cpu"), os.path.join(dump_dir, fname))
        _dumped_count += 1
        if shape_log:
            rec = {"ts": time.time(), "rank": rank, "where": where, "counter": c, "shape": list(getattr(A, "shape", [])), "note": note}
            with open(os.path.join(dump_dir, f"qr_dump_rank{rank}.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

if torch is not None and getattr(torch, "linalg", None) is not None:
    _orig_torch_qr = getattr(torch.linalg, "qr", None)
    if _orig_torch_qr is not None:
        def _patched_torch_qr(A, *args, **kwargs):
            dump = False
            note = ""
            try:
                if hasattr(A, "shape") and len(A.shape) == 2 and A.shape[0] == A.shape[1]:
                    n = int(A.shape[0])
                    if n in (192, 220, 256):
                        dump = True
                        note = f"shape{n}"
                if not dump and not _is_finite(A):
                    dump = True
                    note = "nonfinite_input"
            except Exception:
                pass
            Q, R = _orig_torch_qr(A, *args, **kwargs)
            try:
                if not dump and ((not _is_finite(Q)) or (not _is_finite(R))):
                    dump = True
                    note = "nonfinite_output"
            except Exception:
                pass
            try:
                if (not dump) and (_dumped_count < DUMP_MAX_CALLS):
                    dump = True
                    note = "early"
            except Exception:
                pass
            if dump:
                _dump_one(A, "torch", note)
            return Q, R
        torch.linalg.qr = _patched_torch_qr

try:
    import mx_driving_cloud
    _orig_mx_qr = getattr(mx_driving_cloud.linalg, "qr", None)
    if _orig_mx_qr is not None:
        def _patched_mx_qr(A, *args, **kwargs):
            dump = False
            note = ""
            try:
                if hasattr(A, "shape") and len(A.shape) == 2 and A.shape[0] == A.shape[1]:
                    n = int(A.shape[0])
                    if n in (192, 220, 256):
                        dump = True
                        note = f"shape{n}"
                if not dump and (torch is not None) and (not _is_finite(A)):
                    dump = True
                    note = "nonfinite_input"
            except Exception:
                pass
            if qr_backend == "torch":
                Q, R = torch.linalg.qr(A)
            else:
                Q, R = _orig_mx_qr(A, *args, **kwargs)
            try:
                if (not dump) and (torch is not None) and ((not _is_finite(Q)) or (not _is_finite(R))):
                    dump = True
                    note = "nonfinite_output"
            except Exception:
                pass
            try:
                if (not dump) and (_dumped_count < DUMP_MAX_CALLS):
                    dump = True
                    note = "early"
            except Exception:
                pass
            if dump:
                _dump_one(A, "mx", note)
            return Q, R
        mx_driving_cloud.linalg.qr = _patched_mx_qr
except Exception:
    pass
PY

{
  echo "STEP313_START $(date -Iseconds)"
  echo "HEAD=$(git rev-parse HEAD)"
  echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
  echo "MAX_ITERS=$MAX_ITERS GPUS=$GPUS MASTER_PORT=$MASTER_PORT"
  echo "SOAP_QR_DUMP_DIR=$SOAP_QR_DUMP_DIR"
  if [ "$(git rev-parse HEAD)" != "$EXPECTED_HEAD" ]; then
    echo "HEAD_MISMATCH"
    exit 91
  fi
  for path in tools/train_spetr.py tools/ddp_train.sh "$CONFIG" projects/mmdet3d_plugin/optimizers/soap.py; do
    if ! git diff --quiet HEAD -- "$path"; then
      echo "TRACKED_PATH_DIRTY $path"
      exit 92
    fi
  done
  bash tools/ddp_train.sh tools/train_spetr.py "$CONFIG"
} >"$OUT/logs/launcher.log" 2>&1
rc=$?
echo "$rc" >"$OUT/launcher_rc.txt"
echo "STEP313_END rc=$rc $(date -Iseconds)" >>"$OUT/logs/launcher.log"
exit $rc
