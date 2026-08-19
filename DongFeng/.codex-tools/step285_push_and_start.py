#!/usr/bin/env python3
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

LOCAL = Path(__file__).resolve().parent
FILES = [
    "step285_bad8_official_qr.py",
    "step285_launch_inside.sh",
    "step285_host_start.sh",
]
REL = "diagnostics/step285_bad8_official_qr"


def run(client, cmd: str, timeout: int = 60):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> int:
    info = parse_machine_info()
    shared = str(info["shared"]).rstrip("/")
    remote_dir = posixpath.join(shared, REL)
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
        st, out, err = run(target, f"mkdir -p {remote_dir} && chmod 755 {remote_dir}")
        print("mkdir", st, redact((out + err).strip(), info))
        sftp = target.open_sftp()
        try:
            for name in FILES:
                sftp.put(str(LOCAL / name), posixpath.join(remote_dir, name))
                print("put", name)
        finally:
            sftp.close()
        st, out, err = run(
            target,
            f"chmod +x {remote_dir}/step285_launch_inside.sh {remote_dir}/step285_host_start.sh && "
            f"bash {remote_dir}/step285_host_start.sh",
            timeout=30,
        )
        print("start", st)
        print(redact(out + err, info))
        return st
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
