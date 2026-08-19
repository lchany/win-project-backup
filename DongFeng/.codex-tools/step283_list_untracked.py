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
echo '=== soap cache vs worktree ==='
git diff --cached --stat -- projects/mmdet3d_plugin/optimizers/soap.py
echo '=== untracked at repo root ==='
git ls-files -o --exclude-standard | head -n 120
echo '=== repo diagnostics top ==='
ls -la diagnostics 2>/dev/null | head
echo '=== repo kernel_meta ==='
ls kernel_meta 2>/dev/null | head
echo '=== untracked py/sh in repo not vendor tests ==='
git ls-files -o --exclude-standard | grep -Ei '\.(py|sh)$' | grep -v '^mmdetection3d' | grep -v '^mmcv/' | head -n 80
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
        _, stdout, stderr = target.exec_command(CMD, timeout=45)
        print(redact(stdout.read().decode("utf-8", errors="replace") + stderr.read().decode("utf-8", errors="replace"), info))
        return stdout.channel.recv_exit_status()
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
