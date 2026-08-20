#!/usr/bin/env python3
"""STEP-335: read-only check for kernel-level memory diagnostic tools on remote."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


def run(client, cmd: str, timeout: int = 120) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> int:
    info = parse_machine_info()
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    target = None
    try:
        ch = jump.get_transport().open_channel(
            "direct-tcpip", (str(info["target_host"]), int(info["target_port"])), ("127.0.0.1", 0),
        )
        target = connect(
            str(info["target_host"]), int(info["target_port"]),
            str(info["target_user"]), str(info["target_password"]), sock=ch,
        )
        checks = r"""
echo ===hostname===; hostname
echo ===toolkit_tools===
docker exec mapqr-leicheng bash -lc 'ls /usr/local/Ascend/ascend-toolkit/latest/tools 2>/dev/null'
echo ===sanitizer_bins===
docker exec mapqr-leicheng bash -lc 'for b in mssanitizer msSanitizer msdebug msprof msopst ascend-dmi; do p=$(command -v $b 2>/dev/null); echo "$b=${p:-NOT_FOUND}"; done'
echo ===sanitizer_search===
docker exec mapqr-leicheng bash -lc 'find /usr/local/Ascend -maxdepth 4 -iname "*sanitizer*" -o -maxdepth 4 -iname "msdebug*" 2>/dev/null | head -10'
echo ===cann_version===
docker exec mapqr-leicheng bash -lc 'cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null | head -5; ls /usr/local/Ascend/ascend-toolkit 2>/dev/null'
echo ===mx_driving_loc===
docker exec mapqr-leicheng bash -lc 'python3 -c "import mx_driving_cloud,os;print(os.path.dirname(mx_driving_cloud.__file__))" 2>/dev/null || pip3 show mx-driving-cloud 2>/dev/null | head -3'
"""
        st, out, err = run(target, checks, 120)
        print(redact(out, info))
        if err.strip():
            print(redact(err, info), file=sys.stderr)
        return st
    finally:
        if target:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
