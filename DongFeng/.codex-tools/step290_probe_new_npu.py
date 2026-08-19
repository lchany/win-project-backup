#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

PROBE = r"""
set +e
echo hostname=$(hostname)
echo user=$(id -un)
echo uname=$(uname -s -m)
echo shared=$(test -d /mnt/sfs_turbo/workdir/wfc1_leicheng && echo yes || echo no)
echo docker=$(command -v docker >/dev/null && echo yes || echo no)
docker ps --format '{{.Names}}' 2>/dev/null | head
npu-smi info 2>/dev/null | sed -n '1,25p'
"""


def main() -> int:
    info = parse_machine_info()
    print("parsed_target_last_octet", str(info["target_host"]).rsplit(".", 1)[-1])
    print("parsed_target_port", info["target_port"])
    jump = connect(
        str(info["jump_host"]),
        int(info["jump_port"]),
        str(info["jump_user"]),
        str(info["jump_password"]),
    )
    print("jump_ok")
    target = None
    try:
        transport = jump.get_transport()
        channel = transport.open_channel(
            "direct-tcpip",
            (str(info["target_host"]), int(info["target_port"])),
            ("127.0.0.1", 0),
        )
        print("tcp_ok")
        target = connect(
            str(info["target_host"]),
            int(info["target_port"]),
            str(info["target_user"]),
            str(info["target_password"]),
            sock=channel,
        )
        print("ssh_ok")
        _, stdout, stderr = target.exec_command(PROBE, timeout=40)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        return stdout.channel.recv_exit_status()
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        try:
            from remote_exec import parse_machine_info, redact
            msg = redact(f"{type(exc).__name__}: {exc}", parse_machine_info())
        except Exception:
            msg = type(exc).__name__
        print(f"probe_failed: {msg}")
        raise SystemExit(2)
