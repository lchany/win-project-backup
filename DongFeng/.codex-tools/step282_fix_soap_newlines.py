#!/usr/bin/env python3
"""Rewrite SOAP from 669a138 blob keeping original newlines; only swap QR to mx_driving_cloud."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
set -e
REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
SOAP=projects/mmdet3d_plugin/optimizers/soap.py
cd "$REPO"
python3 - <<'PY'
from pathlib import Path
import subprocess
repo = Path("/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang")
rel = "projects/mmdet3d_plugin/optimizers/soap.py"
raw = subprocess.check_output(["git", "show", f"669a138:{rel}"], cwd=repo)
nl = b"\r\n" if b"\r\n" in raw else b"\n"
text = raw.decode("utf-8")
if "import mx_driving_cloud" not in text:
    if "import torch\n" in text:
        text = text.replace("import torch\n", "import torch\nimport mx_driving_cloud\n", 1)
    elif "import torch\r\n" in text:
        text = text.replace("import torch\r\n", "import torch\r\nimport mx_driving_cloud\r\n", 1)
    else:
        raise SystemExit("no import torch")
n = text.count("torch.linalg.qr(")
if n < 1:
    raise SystemExit("no qr calls")
text = text.replace("torch.linalg.qr(", "mx_driving_cloud.linalg.qr(")
out = text.encode("utf-8")
if nl == b"\r\n" and b"\r\n" not in out.replace(b"\r\n", b""):
    pass
(repo / rel).write_bytes(out)
print("nl", nl, "replaced", n, "size", len(out))
print("dump_left", "SOAP_QR_DUMP" in text)
print("torch_qr_left", "torch.linalg.qr(" in text)
print("mx_qr", text.count("mx_driving_cloud.linalg.qr("))
PY
python3 -m py_compile "$SOAP"
echo '=== vs 669a138 ==='
git diff --stat 669a138 -- "$SOAP"
git diff 669a138 -- "$SOAP"
echo '=== vs HEAD ==='
git diff --stat HEAD -- "$SOAP"
git diff HEAD -- "$SOAP" | python3 -c "import sys; t=sys.stdin.read(); print('diff_bytes', len(t)); print('mx_qr_in_diff', 'mx_driving_cloud.linalg.qr' in t); print('dump_removed', 'SOAP_QR_DUMP' in t)"
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
