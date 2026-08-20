#!/usr/bin/env python3
"""Apply formal SOAP QR fix: mx_driving_cloud.linalg.qr -> torch.linalg.qr, no compat layer."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
set -euo pipefail
REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
SOAP=projects/mmdet3d_plugin/optimizers/soap.py
cd "$REPO"
test "$(git rev-parse --abbrev-ref HEAD)" = "ascend_npu_optimize"

python3 - <<'PY'
from pathlib import Path
import subprocess

repo = Path("/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang")
rel = "projects/mmdet3d_plugin/optimizers/soap.py"
path = repo / rel
raw = path.read_bytes()
text = raw.decode("utf-8")

mx_calls = text.count("mx_driving_cloud.linalg.qr(")
if mx_calls < 1:
    raise SystemExit(f"expected mx QR calls, found {mx_calls}")
if "SOAP_QR_BACKEND" in text:
    raise SystemExit("refusing: SOAP_QR_BACKEND compat layer present")

text = text.replace("mx_driving_cloud.linalg.qr(", "torch.linalg.qr(")
for needle in ("import mx_driving_cloud\r\n", "import mx_driving_cloud\n"):
    if needle in text:
        text = text.replace(needle, "", 1)
        break
else:
    if "import mx_driving_cloud" in text:
        raise SystemExit("import mx_driving_cloud present but newline pattern unknown")

if "mx_driving_cloud" in text:
    raise SystemExit("mx_driving_cloud still referenced after patch")
if text.count("torch.linalg.qr(") < mx_calls:
    raise SystemExit("torch QR call count mismatch")

path.write_bytes(text.encode("utf-8"))
print("mx_replaced", mx_calls)
print("torch_qr", text.count("torch.linalg.qr("))
print("size", path.stat().st_size)
PY

python3 -m py_compile "$SOAP"
echo '=== diff stat ==='
git diff --stat -- "$SOAP"
echo '=== diff ==='
git diff -- "$SOAP"
"""


def main() -> int:
    info = parse_machine_info()
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
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
        _, stdout, stderr = target.exec_command(CMD, timeout=180)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        return stdout.channel.recv_exit_status()
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
