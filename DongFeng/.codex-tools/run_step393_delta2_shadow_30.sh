#!/usr/bin/env bash
# STEP393 inside-container runner.  The host controller must supply the live
# world8/npu-smi gate and approved exact-PID cleanup before this may be armed.
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: $0 ISOLATED_REPO CONTRACT_DIR SHADOW_ROOT OUTPUT_DIR MASTER_PORT INSTALLED_CUSTOM_OPP" >&2
  exit 2
fi

repo=$1
contract=$2
shadow=$3
output=$4
master_port=$5
installed_custom_opp=$6
tool_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

expected_commit=27b1d6d3f363619ad2faa244abe8fbc5a97faef6
expected_soap_blob=77e412c4bece2ca95fd6dbf95732b89951924874
soap_rel=projects/mmdet3d_plugin/optimizers/soap.py
entry="$contract/tools/train_spetr_gpu_seed0_runtime.py"
training_entry="$tool_root/step393_training_entry.py"
config="$tool_root/step393_canonical_aligned_gpu_contract_npu_runtime.py"
launcher="$contract/test_harness/ddp_train_30.sh"
shadow_package="$shadow/mx_driving_cloud"
shadow_opp="$shadow_package/packages/vendors/customize"

sha_of() { sha256sum -- "$1" | awk '{print $1}'; }
require_sha() {
  local path=$1 expected=$2
  [[ -f "$path" && ! -L "$path" && "$(sha_of "$path")" == "$expected" ]]
}

[[ "$repo" == /* && "$contract" == /* && "$shadow" == /* && "$output" == /* ]]
[[ "$repo" != / && "$output" != / && "$repo" != "$output" ]]
[[ ! -e "$repo/.git" ]]
[[ -f "$repo/$soap_rel" && ! -L "$repo/$soap_rel" ]]
[[ "$(git hash-object -- "$repo/$soap_rel")" == "$expected_soap_blob" ]]
[[ -f "$repo/.step393_archived_commit" ]]
[[ "$(<"$repo/.step393_archived_commit")" == "$expected_commit" ]]

require_sha "$config" 02aca0c7a33e05e972e268aecc1932bbf10f611fa523ec4696b94d58cd7f56a5
require_sha "$launcher" 10ad92c723164d52b32734734c8b466f313200165ec1307cb7199e298bb1e0fc
require_sha "$entry" 8c5b315b1741a1557293db1df1bd6c6699494970bc136c434b5b84af9aad65fa
require_sha "$training_entry" fb0e48cfb9593bc70188b0a3b30ee5265eda205aa1f69fad407c6ba7e8a21f40
require_sha "$tool_root/step340_loss_gate.py" b4e20111333f066183c5474d931b6248129065f4b80cfc9ce7177df5e44d9b7d
require_sha "$tool_root/gpu_loss_800.json" 67b36f3dbb36ff50b2a2bf68062d2e1589e2f55cb94207505fdd504e380a8851
[[ -d "$shadow_package" && ! -L "$shadow_package" ]]
[[ -d "$shadow_opp" && ! -L "$shadow_opp" ]]
[[ -d "$installed_custom_opp" && ! -L "$installed_custom_opp" ]]
[[ -d "$output" && ! -L "$output" ]]
active_base=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang/mmdetection3d-0.17.1/configs/_base_/default_runtime.py
archived_base="$repo/mmdetection3d-0.17.1/configs/_base_/default_runtime.py"
[[ -f "$active_base" && ! -L "$active_base" && -f "$archived_base" && ! -L "$archived_base" ]]
[[ "$(sha_of "$active_base")" == "$(sha_of "$archived_base")" ]]
for child in work evidence ready done failure gate_ack; do
  [[ -d "$output/$child" && ! -L "$output/$child" ]]
done

# AST, rather than grep, proves that the only two MX QR call sites are exactly
# the locked source lines 429 and 529.
python3 - "$repo/$soap_rel" <<'PY' > "$output/evidence/soap_ast_contract.json"
import ast, hashlib, json, sys
from pathlib import Path
p = Path(sys.argv[1])
data = p.read_bytes()
tree = ast.parse(data.decode("utf-8"))
def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr); node = node.value
    if isinstance(node, ast.Name): parts.append(node.id)
    return ".".join(reversed(parts))
lines = sorted(
    node.lineno for node in ast.walk(tree)
    if isinstance(node, ast.Call) and dotted(node.func) == "mx_driving_cloud.linalg.qr"
)
assert lines == [429, 529], lines
print(json.dumps({"sha256": hashlib.sha256(data).hexdigest(), "call_lines": lines}, sort_keys=True))
PY

# Deliberately remove every known QR fallback/capture/profile/dump/debug switch.
# No sitecustomize, wrapper monkey patch, profiler, tensor dump, or per-QR sync is
# introduced by this runner, and no import-time hook file is created.
unset MX_QR_VALIDATION_BYPASS MX_QR_BYPASS QR_SOAP_FIXED QR_SOAP_FIXED_SHAPE || true
unset SOAP_DIST_QR SOAP_QR_BACKEND SOAP_QR_FALLBACK SOAP_QR_SHAPE_LOG || true
unset SOAP_QR_DUMP SOAP_QR_DUMP_DIR SOAP_QR_DUMP_MAX_CALLS || true
unset QR_CAPTURE_BACKEND QR_CAPTURE_DIR QR_CAPTURE_MAX_PER_RANK QR_CAPTURE_RUN_ID || true
unset QR_CAPTURE_SEED QR_CAPTURE_TARGET_FACTOR QR_CAPTURE_TARGET_SHAPE QR_CAPTURE_TARGET_STEP || true
unset PROFILING_MODE PROFILE PROFILE_DIR PROFILE_OUTPUT ASCEND_PROFILER_OUTPUT || true
unset ASCEND_LAUNCH_BLOCKING TORCH_NPU_DEBUG HCCL_DEBUG || true
unset PYTHONPATH ASCEND_CUSTOM_OPP_PATH LD_PRELOAD PYTHONSTARTUP PYTHONHOME || true
unset PYTHONINSPECT PYTHONWARNINGS PYTHONBREAKPOINT || true

export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
export GPUS=8
export MODE=single
export MAX_ITERS=30
export MASTER_PORT="$master_port"
export MLP_WORKER_0_PORT="$master_port"
export WORK_DIRS="$output/work"
export LOG_DIR="$output"
export REPO_DIR="$repo"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export STEP393_ORIGINAL_ENTRY="$entry"
export STEP393_READY_DIR="$output/ready"
export STEP393_DONE_DIR="$output/done"
export STEP393_FAILURE_DIR="$output/failure"
export STEP393_GATE_ACK_DIR="$output/gate_ack"
export STEP393_GATE_FILE="$output/start.gate"
export STEP393_GATE_TOKEN_SHA256="${STEP393_GATE_TOKEN_SHA256:?host gate token SHA is required}"
export STEP393_SHADOW_PACKAGE="$shadow_package"

# Shadow paths are intentionally first.  The installed original remains only
# as the base OPP after the shadow, and is part of the controller closure.
export PYTHONPATH="$shadow:$contract/tools:$repo/mmdetection3d-0.17.1:$repo"
export ASCEND_CUSTOM_OPP_PATH="$shadow_opp:$installed_custom_opp"
for root in "$shadow" "$contract/tools" "$repo/mmdetection3d-0.17.1" "$repo"; do
  [[ ! -e "$root/sitecustomize.py" && ! -e "$root/usercustomize.py" ]]
done

python3 - "$shadow_package" "$shadow_opp" "$config" <<'PY' > "$output/evidence/environment_preflight.json"
import hashlib, importlib.util, json, os, sys, torch, torch_npu
from mmcv import Config
package, opp, config = map(os.path.realpath, sys.argv[1:])
spec = importlib.util.find_spec("mx_driving_cloud")
assert spec and spec.origin
origin = os.path.realpath(spec.origin)
assert origin == package + "/__init__.py", (origin, package)
assert os.environ["PYTHONPATH"].split(":", 1)[0] == os.path.dirname(package)
assert os.environ["ASCEND_CUSTOM_OPP_PATH"].split(":", 1)[0] == opp
assert torch.npu.is_available() and torch.npu.device_count() == 8
resolved = Config.fromfile(config)
assert resolved.filename == config
print(json.dumps({
    "torch": torch.__version__, "torch_npu": torch_npu.__version__,
    "npu_available": True, "device_count": 8,
    "visible": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
    "shadow_module": origin, "shadow_opp_first": True,
    "task_queue_state": "production-preserved",
    "task_queue_present": "TASK_QUEUE_ENABLE" in os.environ,
    "canonical_config_resolved": True,
    "task_queue_value_sha256": hashlib.sha256(os.environ["TASK_QUEUE_ENABLE"].encode()).hexdigest()
    if "TASK_QUEUE_ENABLE" in os.environ else None,
}, sort_keys=True))
PY

# Preserve the canonical launcher output.  Live 8-rank and host npu-smi
# attestation is a mandatory responsibility of the still-disarmed host
# controller; this runner never substitutes CPU/CUDA or a standalone worker.
set +e
set -o noclobber
(cd "$repo" && bash "$launcher" "$training_entry" "$config") > "$output/native_launcher.log" 2>&1
launcher_rc=$?
set +o noclobber
set -e
printf '%s\n' "$launcher_rc" > "$output/launcher_rc.txt"
[[ "$launcher_rc" -eq 0 ]]

native_log="$output/work/train.log"
[[ -f "$native_log" && ! -L "$native_log" ]]
python3 "$tool_root/step340_loss_gate.py" \
  --gpu "$tool_root/gpu_loss_800.json" --gpu-format json \
  --npu "$native_log" --npu-format log \
  --threshold 0.02 --start-iter 1 --end-iter 30 \
  > "$output/loss_gate.json"

python3 - "$native_log" <<'PY' > "$output/timing_window_report.json"
import json, re, statistics, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
rows = [(int(a), float(b)) for a, b in re.findall(r"Iter\s*\[\s*(\d+)\s*/[^\n]*?\btime:\s*([-+0-9.eE]+)", text)]
assert [i for i, _ in rows] == list(range(1, 31)), [i for i, _ in rows]
window = [v for i, v in rows if 15 <= i <= 29 and i != 24]
assert len(window) == 14
print(json.dumps({
    "report_only": True, "iter_start": 15, "iter_end": 29,
    "excluded": [24], "count": len(window),
    "mean_seconds": statistics.fmean(window),
    "median_seconds": statistics.median(window),
}, sort_keys=True))
PY

printf '%s\n' PASS > "$output/step393_complete.txt"
