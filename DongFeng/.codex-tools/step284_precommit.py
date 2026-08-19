#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
set -euo pipefail
REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
cd "$REPO"
echo '=== HEAD/branch ==='
git rev-parse --short HEAD
git rev-parse --abbrev-ref HEAD
echo '=== status ==='
git status --short
echo '=== log ==='
git log -8 --oneline
echo '=== soap vs HEAD stat ==='
git diff --stat HEAD -- projects/mmdet3d_plugin/optimizers/soap.py
echo '=== soap vs 669a138 ==='
git diff --stat 669a138 -- projects/mmdet3d_plugin/optimizers/soap.py
echo '=== soap vs 669a138 full ==='
git diff 669a138 -- projects/mmdet3d_plugin/optimizers/soap.py
echo '=== soap vs HEAD names-only other files ==='
git diff --name-only HEAD
echo '=== soap qr/dump counts ==='
python3 - <<'PY'
from pathlib import Path
text=Path("projects/mmdet3d_plugin/optimizers/soap.py").read_text(encoding="utf-8", errors="replace")
print("mx_qr", text.count("mx_driving_cloud.linalg.qr("))
print("torch_qr", text.count("torch.linalg.qr("))
print("import_mx", "import mx_driving_cloud" in text)
print("dump", "SOAP_QR_DUMP" in text)
print("shape_log", "SOAP_QR_SHAPE" in text)
print("dist", "SOAP_DIST_QR" in text)
PY
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
        _, stdout, stderr = target.exec_command(CMD, timeout=60)
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
