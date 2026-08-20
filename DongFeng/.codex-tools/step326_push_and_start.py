#!/usr/bin/env python3
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


LOCAL = Path(__file__).resolve().parent
REL = "diagnostics/step326_torch_qr_shortterm_30step_back8_20260819T233000"
FILES = ("step326_launch_inside.sh",)
MASTER_PORT = 30192


def run(client, command: str, timeout: int = 90):
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
            "direct-tcpip",
            (str(info["target_host"]), int(info["target_port"])),
            ("127.0.0.1", 0),
        )
        target = connect(str(info["target_host"]), int(info["target_port"]), str(info["target_user"]), str(info["target_password"]), sock=channel)

        precheck = (
            "test \"$(docker ps --filter name=^/mapqr-leicheng$ --format '{{.Names}}')\" = mapqr-leicheng && "
            f"! ss -ltn | awk '{{print $4}}' | grep -Eq ':({MASTER_PORT})$' && "
            f"mkdir -p {remote_dir}/logs && test ! -f {remote_dir}/launcher_rc.txt"
        )
        status, out, err = run(target, precheck, timeout=30)
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
            f"python3 - <<'PY'\n"
            f"from pathlib import Path\n"
            f"base = Path('{remote_dir}')\n"
            f"p = base / 'step326_launch_inside.sh'\n"
            f"p.write_bytes(p.read_bytes().replace(b'\\r\\n', b'\\n'))\n"
            f"(base / 'README.txt').write_text("
            f"'''STEP-326 short-term torch QR validation (30 iter).\n"
            f"Purpose: validate finite loss + timing vs GPU with fb979b2 NPU FP32 SOAP stack,\n"
            f"         but redirect mx_driving_cloud.linalg.qr to torch.linalg.qr at runtime.\n"
            f"Env: SOAP_QR_BACKEND=torch, MAX_ITERS=30, back-8 NPUs, clean HEAD 27b1d6d.\n"
            f"'''"
            f", encoding='utf-8')\n"
            f"PY\n"
            f"chmod 755 {remote_dir}/step326_launch_inside.sh; "
            f"docker exec mapqr-leicheng bash -n {remote_dir}/step326_launch_inside.sh && "
            f"setsid -f sh -c 'docker exec -e STEP326_OUT={remote_dir} -e SOAP_QR_BACKEND=torch -e MAX_ITERS=30 -e MASTER_PORT={MASTER_PORT} "
            f"mapqr-leicheng bash --noprofile --norc {remote_dir}/step326_launch_inside.sh' </dev/null; echo started"
        )
        status, out, err = run(target, command, timeout=90)
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        return status
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
