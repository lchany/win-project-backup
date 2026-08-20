#!/usr/bin/env python3
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


LOCAL = Path(__file__).resolve().parent
REL = "diagnostics/step324_qr_backend_torch_iter6_back8_20260819T223500"
FILES = ("step313_launch_inside.sh",)


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
            "direct-tcpip",
            (str(info["target_host"]), int(info["target_port"])),
            ("127.0.0.1", 0),
        )
        target = connect(str(info["target_host"]), int(info["target_port"]), str(info["target_user"]), str(info["target_password"]), sock=channel)

        precheck = (
            "test \"$(docker ps --filter name=^/mapqr-leicheng$ --format '{{.Names}}')\" = mapqr-leicheng && "
            "! ss -ltn | awk '{print $4}' | grep -Eq ':(30182)$' && "
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
            f"python3 - <<'PY'\n"
            f"from pathlib import Path\n"
            f"base = Path('{remote_dir}')\n"
            f"p = base / 'step313_launch_inside.sh'\n"
            f"p.write_bytes(p.read_bytes().replace(b'\\r\\n', b'\\n'))\n"
            f"(base / 'README.txt').write_text("
            f"'''STEP-324 QR backend control.\n"
            f"Purpose: run the same 10-step training as STEP-315, but monkey-patch mx_driving_cloud.linalg.qr to use torch.linalg.qr (SOAP_QR_BACKEND=torch).\n"
            f"Expected: if NaNs disappear, root cause is QR implementation convention difference affecting SOAP.\n"
            f"Notes:\n"
            f"- SOAP_QR_DUMP_MAX_CALLS is set small to limit dumps.\n"
            f"'''"
            f", encoding='utf-8')\n"
            f"PY\n"
            f"chmod 755 {remote_dir}/step313_launch_inside.sh; "
            f"docker exec mapqr-leicheng bash -n {remote_dir}/step313_launch_inside.sh && "
            f"setsid -f sh -c 'docker exec -e STEP313_OUT={remote_dir} -e SOAP_QR_BACKEND=torch -e SOAP_QR_DUMP_MAX_CALLS=4 "
            f"mapqr-leicheng bash --noprofile --norc {remote_dir}/step313_launch_inside.sh' </dev/null; echo started"
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

