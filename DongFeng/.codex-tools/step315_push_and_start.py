#!/usr/bin/env python3
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


LOCAL = Path(__file__).resolve().parent
REL = "diagnostics/step315_qr_dump_monkeypatch_iter6_back8_20260819T222000"
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
            "direct-tcpip", (str(info["target_host"]), int(info["target_port"])), ("127.0.0.1", 0)
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
            f"'''STEP-315 QR dump capture (monkey-patch) for back-8 training NaN investigation.\n"
            f"Purpose: capture QR input/output tensors around the first training NaN (observed from iter 6 onward).\n"
            f"Key change: generate `sitecustomize.py` on the fly to monkey-patch torch/mx QR.\n"
            f"Environment: mapqr-leicheng, ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15\n"
            f"Expected contents:\n"
            f"- logs/launcher.log\n"
            f"- qr_tensors/ : .pt dumps from SOAP_QR_DUMP_DIR\n"
            f"- launcher_rc.txt\n"
            f"Do not delete until QR standalone replay and root-cause summary are complete.\n"
            f"'''"
            f", encoding='utf-8')\n"
            f"PY\n"
            f"chmod 755 {remote_dir}/step313_launch_inside.sh; "
            f"docker exec mapqr-leicheng bash -n {remote_dir}/step313_launch_inside.sh && "
            f"setsid -f sh -c 'docker exec -e STEP313_OUT={remote_dir} "
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

