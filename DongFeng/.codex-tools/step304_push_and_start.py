#!/usr/bin/env python3
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


LOCAL = Path(__file__).resolve().parent
REL = "diagnostics/step305_head27b1d6d_setdevice_legacylaunch_hccl64000_30step_20260819T1715"
FILES = ("step303_rank_device_preflight.py", "step303_launch_inside.sh")


def run(client, command: str, timeout: int = 60):
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> int:
    info = parse_machine_info()
    remote_dir = posixpath.join(str(info["shared"]).rstrip("/"), REL)
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    target = None
    try:
        channel = jump.get_transport().open_channel(
            "direct-tcpip", (str(info["target_host"]), int(info["target_port"])), ("127.0.0.1", 0)
        )
        target = connect(str(info["target_host"]), int(info["target_port"]), str(info["target_user"]), str(info["target_password"]), sock=channel)
        precheck = (
            "test \"$(docker ps --filter name=^/mapqr-leicheng$ --format '{{.Names}}')\" = mapqr-leicheng && "
            "! ss -ltn | awk '{print $4}' | grep -Eq ':(30141|30142|6400[0-9]|6401[0-9])$' && "
            f"mkdir -p {remote_dir}/logs && test ! -f {remote_dir}/launcher_rc.txt"
        )
        status, out, err = run(target, precheck)
        if status != 0:
            print(redact(out + err, info))
            print("remote precheck refused launch")
            return status
        sftp = target.open_sftp()
        try:
            for name in FILES:
                sftp.put(str(LOCAL / name), posixpath.join(remote_dir, name))
        finally:
            sftp.close()
        command = (
            f"chmod 755 {remote_dir}/step303_launch_inside.sh; "
            f"docker exec mapqr-leicheng bash -n {remote_dir}/step303_launch_inside.sh && "
            f"setsid -f sh -c 'docker exec -e STEP303_OUT={remote_dir} -e HCCL_IF_BASE_PORT=64000 "
            f"mapqr-leicheng bash --noprofile --norc {remote_dir}/step303_launch_inside.sh' </dev/null; echo started"
        )
        status, out, err = run(target, command, timeout=30)
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        return status
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
