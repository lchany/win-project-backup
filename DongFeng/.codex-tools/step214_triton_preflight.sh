#!/usr/bin/env bash
set -euo pipefail

printf 'container_name=%s\n' "$(hostname)"
printf 'arch=%s\n' "$(uname -m)"
printf 'glibc=%s\n' "$(ldd --version 2>&1 | head -1)"
printf 'python=%s\n' "$(python3 -V 2>&1)"
printf 'python_exe=%s\n' "$(python3 -c 'import sys; print(sys.executable)')"

python3 - <<'PY'
import importlib.metadata as metadata
import importlib.util
import platform
import sys

print("platform_machine=" + platform.machine())
for dist in ("torch", "torch-npu", "triton", "triton-ascend", "pip", "virtualenv"):
    try:
        print(f"dist_{dist}={metadata.version(dist)}")
    except metadata.PackageNotFoundError:
        print(f"dist_{dist}=MISSING")
for module in ("torch", "torch_npu", "triton", "triton.backends"):
    spec = importlib.util.find_spec(module)
    print(f"module_{module}=" + (spec.origin if spec else "MISSING"))
print("venv_module=" + str(importlib.util.find_spec("venv") is not None))
print("prefix=" + sys.prefix)
print("base_prefix=" + sys.base_prefix)
PY

printf '%s\n' '---cann---'
for version_file in \
    /usr/local/Ascend/ascend-toolkit/latest/*/version.info \
    /usr/local/Ascend/ascend-toolkit/latest/version.info \
    /usr/local/Ascend/ascend-toolkit/*/version.info; do
    [ -f "$version_file" ] || continue
    printf 'version_file=%s\n' "$version_file"
    grep -E '^(Version|version|package_version|version_dir)=' "$version_file" 2>/dev/null | head -5 || true
    break
done

printf '%s\n' '---disk---'
df -h "$STEP214_ROOT" | tail -1
df -Pi "$STEP214_ROOT" | tail -1

printf '%s\n' '---processes---'
printf 'training_like_count=%s\n' "$(pgrep -af 'torchrun|torch.distributed|train_spetr.py|tools/train.py|msprof' | grep -v -E 'pgrep|grep' | wc -l)"

printf '%s\n' '---network---'
python3 - <<'PY'
import urllib.request

for url in (
    "https://pypi.org/pypi/triton-ascend/3.2.0rc4/json",
    "https://gitcode.com/Ascend/triton-ascend",
):
    host = url.split("/")[2]
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            print("network_ok", response.status, host)
    except Exception as error:
        print("network_fail", type(error).__name__, host)
PY
