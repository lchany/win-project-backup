#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
set -e
REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
cd "$REPO"
echo '=== HEAD ==='
git rev-parse --short HEAD
git rev-parse --abbrev-ref HEAD
echo '=== status ==='
git status --short
echo '=== untracked sample ==='
git status -u --short | head -n 80
echo '=== soap dump/test markers ==='
python3 - <<'PY'
from pathlib import Path
p=Path("projects/mmdet3d_plugin/optimizers/soap.py")
text=p.read_text(encoding="utf-8", errors="replace")
keys=("SOAP_QR_DUMP","SOAP_QR_SHAPE","SOAP_DIST_QR","factor_ind","qr_tensors","DEBUG","pytest","unittest")
for i,line in enumerate(text.splitlines(),1):
    if any(k in line for k in keys):
        print(f"{i}:{line}")
print("mx_qr", text.count("mx_driving_cloud.linalg.qr("))
print("torch_qr", text.count("torch.linalg.qr("))
PY
echo '=== repo test-ish tracked vs dirty ==='
git ls-files '*test*' '*dump*' '*qr*' 2>/dev/null | head -n 50
echo '=== dirty files matching test/diag ==='
git status --short | grep -Ei 'test|dump|qr|diagnos|codex|harness|repro' || true
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
