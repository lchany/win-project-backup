#!/usr/bin/env python3
"""Fetch patched soap.py from remote into local projects/ mirror."""
from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
set -euo pipefail
SOAP=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang/projects/mmdet3d_plugin/optimizers/soap.py
base64 -w0 "$SOAP"
echo
sha256sum "$SOAP"
"""

LOCAL_REL = Path("projects/mmdet3d_plugin/optimizers/soap.py")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    dest = root / LOCAL_REL
    dest.parent.mkdir(parents=True, exist_ok=True)

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
        print(redact(err, info), end="" if err.endswith("\n") else "\n")
        lines = [ln for ln in out.strip().splitlines() if ln and not ln[0].isdigit() or " " not in ln[:64]]
        # last line is sha256sum; everything before that joined is base64
        raw_lines = out.strip().splitlines()
        b64_line = raw_lines[0] if raw_lines else ""
        data = base64.b64decode(b64_line)
        dest.write_bytes(data)
        print(f"WROTE {dest} bytes={len(data)}")
        if len(raw_lines) > 1:
            print(redact(raw_lines[-1], info))
        return stdout.channel.recv_exit_status()
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
