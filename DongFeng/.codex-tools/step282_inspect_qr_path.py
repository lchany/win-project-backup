#!/usr/bin/env python3
"""Inspect remote SOAP QR path and installed mx_driving_cloud linalg.py (no secrets printed)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
set -e
REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
echo '=== git ==='
cd "$REPO"
git rev-parse --abbrev-ref HEAD
git log -8 --oneline
echo '=== status soap/mx ==='
git status --short -- projects/mmdet3d_plugin/optimizers/soap.py mx_driving_cloud 2>/dev/null || true
echo '=== soap qr lines ==='
python3 - <<'PY'
from pathlib import Path
p=Path("/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang/projects/mmdet3d_plugin/optimizers/soap.py")
text=p.read_text(encoding="utf-8", errors="replace")
keys=("mx_driving_cloud","torch.linalg.qr","linalg.qr","SOAP_DIST_QR","SOAP_QR_DUMP","get_orthogonal_matrix_QR")
for i,line in enumerate(text.splitlines(),1):
    if any(k in line for k in keys):
        print(f"{i}:{line}")
PY
echo '=== site linalg ==='
docker exec mapqr-leicheng bash --noprofile --norc -lc '
python - <<'"'"'PY'"'"'
import inspect, mx_driving_cloud
import mx_driving_cloud.ops.linalg as L
print("mx", getattr(mx_driving_cloud,"__version__",None), getattr(mx_driving_cloud,"__file__",None))
print("linalg", L.__file__)
src=inspect.getsource(L)
print("has_QR_SOAP_FIXED", "QR_SOAP_FIXED_SHAPE" in src)
print("has_BYPASS", "MX_QR_VALIDATION_BYPASS" in src)
print("--- linalg.py ---")
print(src[:4000])
print("qr", L.qr)
PY
'
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
        _, stdout, stderr = target.exec_command(CMD, timeout=90)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        st = stdout.channel.recv_exit_status()
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        return st
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
