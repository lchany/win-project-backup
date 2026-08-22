#!/usr/bin/env python3
"""Read-only provenance and environment gate for the installed MX QR kernel."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from remote_exec import connect, parse_machine_info, redact


COMMAND = r"""
set -eu

container_count=$(docker ps --format '{{.Names}}' | awk '$0 == "mapqr-leicheng" {n++} END {print n+0}')
echo "container_exact_count=$container_count"
if [ "$container_count" -ne 1 ]; then
  exit 81
fi

docker inspect --format 'container_running={{.State.Running}} container_name={{.Name}}' mapqr-leicheng

docker exec mapqr-leicheng bash --noprofile --norc -lc '
set -eu
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 - <<'"'"'PY'"'"'
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mx_driving_cloud
import torch
import torch_npu


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


site = Path(mx_driving_cloud.__file__).resolve().parent
paths = {
    "linalg": site / "ops/linalg.py",
    "qr_cpp": site / "packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.cpp",
    "qr_py": site / "packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.py",
    "kernel_cfg": site / "packages/vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend910b/qr_v2.json",
}

print("torch_version=" + str(torch.__version__))
print("torch_npu_version=" + str(getattr(torch_npu, "__version__", "unknown")))
print("mx_version=" + str(getattr(mx_driving_cloud, "__version__", "unknown")))
print("npu_available=" + str(torch.npu.is_available()))
print("npu_count=" + str(torch.npu.device_count()))

for name, path in paths.items():
    print(f"{name}_exists={path.is_file()}")
    if path.is_file():
        print(f"{name}_size={path.stat().st_size}")
        print(f"{name}_sha256={digest(path)}")

cpp = paths["qr_cpp"].read_text(encoding="utf-8", errors="replace")
print("cpp_zero_col_expr=" + str("this->blockp - k" in cpp))
print("cpp_unconditional_core0_calc=" + str("if (coreId == 0)" in cpp))
print("cpp_guarded_core0_calc=" + str("coreId == 0 && tilingInfo.useCoreNum" in cpp))
print("cpp_larfb_frees_vlocal=" + str("vTQue.FreeTensor<DTYPE_A>(vLocal)" in cpp))

kernel_root = site / "packages/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/qr_v2"
kernel_files = sorted(p for p in kernel_root.rglob("*") if p.is_file())
print("kernel_file_count=" + str(len(kernel_files)))
for path in kernel_files:
    rel = path.relative_to(site)
    print(f"kernel_file={rel}|size={path.stat().st_size}|sha256={digest(path)}")

config = paths["kernel_cfg"]
if config.is_file():
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
        print("kernel_cfg_json_type=" + type(payload).__name__)
    except Exception as exc:
        print("kernel_cfg_json_error=" + type(exc).__name__)
PY
'
"""


def main() -> int:
    info = parse_machine_info()
    jump = connect(
        str(info["jump_host"]),
        int(info["jump_port"]),
        str(info["jump_user"]),
        str(info["jump_password"]),
    )
    target = None
    try:
        transport = jump.get_transport()
        channel = transport.open_channel(
            "direct-tcpip",
            (str(info["target_host"]), int(info["target_port"])),
            ("127.0.0.1", 0),
        )
        target = connect(
            str(info["target_host"]),
            int(info["target_port"]),
            str(info["target_user"]),
            str(info["target_password"]),
            sock=channel,
        )
        _, stdout, stderr = target.exec_command(COMMAND, timeout=120)
        output = stdout.read().decode("utf-8", errors="replace")
        error = stderr.read().decode("utf-8", errors="replace")
        status = stdout.channel.recv_exit_status()
        print(redact(output + error, info), end="" if (output + error).endswith("\n") else "\n")
        return status
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
