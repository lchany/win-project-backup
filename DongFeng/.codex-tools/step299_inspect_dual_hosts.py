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


def run(client, cmd: str, timeout: int = 120) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> int:
    info = parse_machine_info()
    sec_host, sec_port, sec_user, sec_password = parse_secondary_host()
    info_for_redact = dict(info)
    info_for_redact["secondary_host"] = sec_host
    info_for_redact["secondary_user"] = sec_user
    info_for_redact["secondary_password"] = sec_password

    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    primary = None
    secondary = None
    try:
        t = jump.get_transport()
        ch1 = t.open_channel("direct-tcpip", (str(info["target_host"]), int(info["target_port"])), ("127.0.0.1", 0))
        primary = connect(str(info["target_host"]), int(info["target_port"]), str(info["target_user"]), str(info["target_password"]), sock=ch1)
        ch2 = t.open_channel("direct-tcpip", (sec_host, sec_port), ("127.0.0.1", 0))
        secondary = connect(sec_host, sec_port, sec_user, sec_password, sock=ch2)

        cmds = {
            "primary": r"""
set +e
echo host=$(hostname)
echo '=== mapqr-leicheng ps ==='
docker ps -a --filter name=^/mapqr-leicheng$ --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
echo '=== mapqr-leicheng inspect ==='
docker inspect mapqr-leicheng --format 'image={{.Config.Image}}
user={{.Config.User}}
workdir={{.Config.WorkingDir}}
ipc={{.HostConfig.IpcMode}}
net={{.HostConfig.NetworkMode}}
restart={{.HostConfig.RestartPolicy.Name}}
privileged={{.HostConfig.Privileged}}
shmsize={{.HostConfig.ShmSize}}
binds={{json .HostConfig.Binds}}' 2>/dev/null
echo '=== shared dir writable ==='
test -d /mnt/sfs_turbo/workdir/wfc1_leicheng && echo shared=yes || echo shared=no
""",
            "secondary": r"""
set +e
echo host=$(hostname)
echo '=== existing mapqr-leicheng ==='
docker ps -a --filter name=^/mapqr-leicheng$ --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
echo '=== docker top names ==='
docker ps --format '{{.Names}}\t{{.Status}}'
echo '=== shared dir writable ==='
test -d /mnt/sfs_turbo/workdir/wfc1_leicheng && echo shared=yes || echo shared=no
mkdir -p /mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step299_container_migration && echo mkdir_ok
"""
        }

        for label, client in (("primary", primary), ("secondary", secondary)):
            print(f"=== {label} ===")
            st, out, err = run(client, cmds[label], timeout=120)
            print(redact(out + err, info_for_redact), end="" if (out + err).endswith("\n") else "\n")
            if st != 0:
                return st
        return 0
    finally:
        if secondary is not None:
            secondary.close()
        if primary is not None:
            primary.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())

