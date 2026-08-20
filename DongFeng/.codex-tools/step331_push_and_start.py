#!/usr/bin/env python3
"""Push STEP-331 profile job and start on back-8."""
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

LOCAL = Path(__file__).resolve().parent
DIAG = "diagnostics/step331_qr_step10_vs14_profile_back8_20260820T153000"
PORT = 30195
FILES = (
    "step331_launch_inside.sh",
    "step331_cycle_profiler_hook.py",
    "step331_build_profile_config.py",
    "step331_analyze_qr_by_step.py",
)


def run(client, cmd: str, timeout: int = 120) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> int:
    info = parse_machine_info()
    root = posixpath.join(str(info["shared"]).rstrip("/"), DIAG)
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    target = None
    try:
        ch = jump.get_transport().open_channel(
            "direct-tcpip", (str(info["target_host"]), int(info["target_port"])), ("127.0.0.1", 0)
        )
        target = connect(
            str(info["target_host"]), int(info["target_port"]),
            str(info["target_user"]), str(info["target_password"]), sock=ch,
        )
        pre = (
            "test \"$(docker ps --filter name=^/mapqr-leicheng$ --format '{{.Names}}')\" = mapqr-leicheng && "
            f"mkdir -p {root}/logs {root}/work {root}/profile_raw && "
            f"rm -f {root}/launcher_rc.txt {root}/work/train.log && "
            f"rm -rf {root}/profile_raw/*"
        )
        st, out, err = run(target, pre, 30)
        if st != 0:
            print(redact(out + err, info))
            return st

        sftp = target.open_sftp()
        try:
            for name in FILES:
                data = (LOCAL / name).read_bytes().replace(b"\r\n", b"\n")
                with sftp.open(posixpath.join(root, name), "wb") as f:
                    f.write(data)
        finally:
            sftp.close()

        cmd = (
            f"python3 - <<'PY'\nfrom pathlib import Path\np=Path('{root}/step331_launch_inside.sh')\n"
            f"p.write_bytes(p.read_bytes().replace(b'\\r\\n', b'\\n'))\nPY\n"
            f"chmod 755 {root}/step331_launch_inside.sh; "
            f"docker exec mapqr-leicheng bash -n {root}/step331_launch_inside.sh && "
            f"setsid -f sh -c 'docker exec -e ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 "
            f"-e STEP331_OUT={root} -e SOAP_STALE_Q_K=4 -e MAX_ITERS=16 -e MASTER_PORT={PORT} "
            f"mapqr-leicheng bash --noprofile --norc {root}/step331_launch_inside.sh' </dev/null; echo started"
        )
        st, out, err = run(target, cmd, 90)
        print(redact(out + err, info))
        return st
    finally:
        if target:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
