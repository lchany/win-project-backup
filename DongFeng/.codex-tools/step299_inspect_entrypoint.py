#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

ROOT = Path(__file__).resolve().parents[1]
MACHINE_INFO = ROOT / "机器IP.md"


def parse_secondary_host() -> tuple[str, int, str, str]:
    text = MACHINE_INFO.read_text(encoding="utf-8")
    m = re.search(r"npu2训练机器[\s\S]*?((?:\d{1,3}\.){3}\d{1,3})\s+(\S+)\s+密码\s*[:：]\s*(\S+)", text)
    if not m:
        raise ValueError("unable to parse secondary NPU host line")
    host, user, password = m.groups()
    port = int(re.search(r"npu2训练机器[\s\S]*?端口\s*[:：]\s*(\d+)", text).group(1))
    return host, port, user, password


def main() -> int:
    info = parse_machine_info()
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    target = None
    try:
        transport = jump.get_transport()
        channel = transport.open_channel("direct-tcpip", (str(info["target_host"]), int(info["target_port"])), ("127.0.0.1", 0))
        target = connect(str(info["target_host"]), int(info["target_port"]), str(info["target_user"]), str(info["target_password"]), sock=channel)
        cmd = r"""docker inspect mapqr-leicheng --format 'entrypoint={{json .Config.Entrypoint}}
cmd={{json .Config.Cmd}}
tty={{.Config.Tty}}
openstdin={{.Config.OpenStdin}}
env={{json .Config.Env}}'"""
        _, stdout, stderr = target.exec_command(cmd, timeout=60)
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

