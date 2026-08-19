#!/usr/bin/env bash
set -euo pipefail

venv="$STEP214_ROOT/venv"
wheel=$(find "$STEP214_ROOT/wheelhouse" -maxdepth 1 -type f -name 'triton_ascend-3.2.0rc4-*.whl' -print -quit)
[ -x "$venv/bin/python" ] || { echo 'venv_missing'; exit 52; }
[ -n "$wheel" ] || { echo 'wheel_missing'; exit 51; }

"$venv/bin/python" -m pip install \
    --no-index \
    --no-deps \
    --no-cache-dir \
    "$wheel"

printf '%s\n' '---global-default-python---'
python3 - <<'PY'
import importlib.metadata as metadata
import triton

print("global_triton_dist=" + metadata.version("triton"))
try:
    print("global_triton_ascend_dist=" + metadata.version("triton-ascend"))
except metadata.PackageNotFoundError:
    print("global_triton_ascend_dist=MISSING")
print("global_triton_module=" + triton.__file__)
PY

printf '%s\n' '---isolated-venv---'
"$venv/bin/python" - "$venv" <<'PY'
import importlib.metadata as metadata
import importlib.util
import pathlib
import sys

venv = pathlib.Path(sys.argv[1]).resolve()
import torch
import torch_npu
import triton
import triton.backends as triton_backends

print("venv_torch_dist=" + metadata.version("torch"))
print("venv_torch_npu_dist=" + metadata.version("torch-npu"))
print("venv_triton_ascend_dist=" + metadata.version("triton-ascend"))
print("venv_triton_module_version=" + str(getattr(triton, "__version__", "UNKNOWN")))
print("venv_torch_module=" + torch.__file__)
print("venv_torch_npu_module=" + torch_npu.__file__)
print("venv_triton_module=" + triton.__file__)
spec = importlib.util.find_spec("triton.backends.ascend")
print("venv_ascend_backend_spec=" + (spec.origin if spec else "MISSING"))
registry = getattr(triton_backends, "backends", None)
if isinstance(registry, dict):
    print("venv_backend_registry=" + ",".join(sorted(registry)))
else:
    print("venv_backend_registry=UNAVAILABLE")

if not pathlib.Path(triton.__file__).resolve().is_relative_to(venv):
    raise SystemExit("isolated triton module did not shadow global triton")
if spec is None:
    raise SystemExit("Ascend backend module missing")
if isinstance(registry, dict) and "ascend" not in registry:
    raise SystemExit("Ascend backend not registered")
PY

printf '%s\n' '---post-install-processes---'
printf 'training_like_count=%s\n' "$(pgrep -af 'torchrun|torch.distributed|train_spetr.py|tools/train.py|msprof' | grep -v -E 'pgrep|grep' | wc -l)"
