#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


CMD = r"""
set -euo pipefail
echo '=== HOST PT_ELASTIC ==='
ps -fp 2217190,1584482 || true
echo '=== HOST PYTHON PIDS ==='
ps -fp 1584548,1584549,1584550,1584551,1584552,1584553,1584554,1584555,2217491,2217492,2217493,2217494,2217495,2217496,2217497,2217498 || true
echo '=== HOST PPIDS ==='
ps -o pid,ppid,user,lstart,cmd -p 1584548,1584549,1584550,1584551,1584552,1584553,1584554,1584555,2217491,2217492,2217493,2217494,2217495,2217496,2217497,2217498,2217190,1584482 || true
echo '=== DOCKER TOP mapqr-leicheng ==='
docker top mapqr-leicheng -eo pid,ppid,user,lstart,cmd || true
echo '=== DOCKER TOP shj ==='
docker top shj -eo pid,ppid,user,lstart,cmd || true
"""


def main() -> int:
    info = parse_machine_info()
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    target = None
    try:
        channel = jump.get_transport().open_channel(
            "direct-tcpip",
            (str(info["target_host"]), int(info["target_port"])),
            ("127.0.0.1", 0),
        )
        target = connect(str(info["target_host"]), int(info["target_port"]), str(info["target_user"]), str(info["target_password"]), sock=channel)
        _, stdout, stderr = target.exec_command(CMD, timeout=120)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        text = redact(out + err, info)
        print(text, end="" if text.endswith("\n") else "\n")
        return stdout.channel.recv_exit_status()
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
