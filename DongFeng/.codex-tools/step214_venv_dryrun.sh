#!/usr/bin/env bash
set -euo pipefail

venv="$STEP214_ROOT/venv"
wheel=$(find "$STEP214_ROOT/wheelhouse" -maxdepth 1 -type f -name 'triton_ascend-3.2.0rc4-*.whl' -print -quit)
[ -n "$wheel" ] || { echo 'wheel_missing'; exit 51; }

if [ ! -x "$venv/bin/python" ]; then
    python3 -m venv --system-site-packages "$venv"
fi

"$venv/bin/python" - <<'PY'
import importlib.metadata as metadata
import pathlib
import sys

print("venv_prefix=" + sys.prefix)
print("venv_base_prefix=" + sys.base_prefix)
print("venv_system_site_packages=" + pathlib.Path(sys.prefix, "pyvenv.cfg").read_text().strip().replace("\n", ";"))
for dist in ("torch", "torch-npu", "triton", "triton-ascend"):
    try:
        print(f"venv_before_{dist}={metadata.version(dist)}")
    except metadata.PackageNotFoundError:
        print(f"venv_before_{dist}=MISSING")
PY

report="$STEP214_ROOT/pip_dry_run_report.json"
"$venv/bin/python" -m pip install \
    --dry-run \
    --report "$report" \
    --no-index \
    --no-cache-dir \
    "$wheel"

"$venv/bin/python" - "$report" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
names = [item["metadata"]["name"].lower().replace("_", "-") for item in report.get("install", [])]
print("dry_run_install_count=" + str(len(names)))
for name in names:
    print("dry_run_install=" + name)
forbidden = {"torch", "torch-npu", "triton", "cann"}
if any(name in forbidden for name in names):
    raise SystemExit("forbidden dependency replacement planned")
if names != ["triton-ascend"]:
    raise SystemExit("unexpected dry-run install plan")
PY
