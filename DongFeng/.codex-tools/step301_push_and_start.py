#!/usr/bin/env python3
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


LOCAL = Path(__file__).resolve().parent
INPUT = LOCAL.parent / "step260_qr_bad_tensors" / "rank0_step10_ind0_192x192_BAD.pt"
REL = "diagnostics/step301_bad_multivisible_setdevice_qr7_20260819T1645"


def run(client, cmd: str, timeout: int = 60):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
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
            "direct-tcpip",
            (str(info["target_host"]), int(info["target_port"])),
            ("127.0.0.1", 0),
        )
        target = connect(str(info["target_host"]), int(info["target_port"]), str(info["target_user"]), str(info["target_password"]), sock=channel)
        status, _, _ = run(target, f"mkdir -p {remote_dir} && test ! -f {remote_dir}/summary.json")
        if status != 0:
            print("refusing to reuse completed run directory")
            return status
        sftp = target.open_sftp()
        try:
            sftp.put(str(INPUT), posixpath.join(remote_dir, INPUT.name))
            for name in ("step301_bad_multivisible_setdevice_qr7.py", "step301_launch_inside.sh"):
                sftp.put(str(LOCAL / name), posixpath.join(remote_dir, name))
        finally:
            sftp.close()
        command = (
            f"chmod 755 {remote_dir}/step301_launch_inside.sh; "
            f"setsid -f sh -c 'docker exec -e STEP301_RUN_DIR={remote_dir} mapqr-leicheng "
            f"bash --noprofile --norc {remote_dir}/step301_launch_inside.sh "
            f"> {remote_dir}/driver.log 2>&1; echo $? > {remote_dir}/host_rc.txt' </dev/null; "
            f"echo started"
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
