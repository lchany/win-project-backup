#!/usr/bin/env python3
"""STEP-333: all-rank install diag + iter10-16 profile on back-8."""
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parent
DIAG = "diagnostics/step333_allrank_install_profile_back8_20260820T165500"
PORT = 30196


def run(client, command: str, timeout: int = 120) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> int:
    info = parse_machine_info()
    remote_root = posixpath.join(str(info["shared"]).rstrip("/"), DIAG)
    out_dir = posixpath.join(remote_root, "run")
    overlay_remote = posixpath.join(remote_root, "soap_overlay.py")
    launch_remote = posixpath.join(remote_root, "step333_launch_inside.sh")

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
        verify = (
            "hostname; "
            "test \"$(docker ps --filter name=^/mapqr-leicheng$ --format '{{.Names}}')\" = mapqr-leicheng"
        )
        status, out, err = run(target, verify, timeout=30)
        if status != 0:
            print(redact(out + err, info))
            return status

        run(target, f"mkdir -p {remote_root}/run/logs {remote_root}/run/profile_raw", timeout=30)
        sftp = target.open_sftp()
        try:
            sftp.put(str(ROOT / "projects/mmdet3d_plugin/optimizers/soap.py"), overlay_remote)
            sftp.put(str(LOCAL / "step333_launch_inside.sh"), launch_remote)
            for name in (
                "step331_cycle_profiler_hook.py",
                "step331_build_profile_config.py",
                "step331_analyze_qr_by_step.py",
                "step333_analyze_all_ranks.py",
            ):
                sftp.put(str(LOCAL / name), posixpath.join(out_dir, name))
        finally:
            sftp.close()

        prep = (
            f"python3 -c \"from pathlib import Path; "
            f"p=Path('{launch_remote}'); p.write_bytes(p.read_bytes().replace(b'\\\\r\\\\n', b'\\\\n')); p.chmod(0o755)\"; "
            f"docker exec mapqr-leicheng bash -n {launch_remote}"
        )
        run(target, prep, timeout=30)

        start_cmd = (
            f"setsid -f sh -c \"docker exec "
            f"-e ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 "
            f"-e STEP333_OUT={out_dir} "
            f"-e STEP333_SOAP_OVERLAY={overlay_remote} "
            f"-e SOAP_STALE_Q_K=4 -e MAX_ITERS=16 -e MASTER_PORT={PORT} "
            f"mapqr-leicheng bash --noprofile --norc {launch_remote}\" "
            f"</dev/null >{remote_root}/host_start.log 2>&1; echo started"
        )
        status, out, err = run(target, start_cmd, timeout=60)
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        print(f"REMOTE_ROOT={remote_root}")
        return status
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
