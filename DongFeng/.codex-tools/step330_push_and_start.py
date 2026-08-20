#!/usr/bin/env python3
"""STEP-330: push stale-Q A/B launcher and start k=0 then k=4 runs on remote."""
from __future__ import annotations

import base64
import posixpath
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

LOCAL = Path(__file__).resolve().parent
DIAG = "diagnostics/step330_stale_q_ab_30step_back8_20260820T143600"
K0_PORT = 30193
K4_PORT = 30194


def run(client, command: str, timeout: int = 120) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> int:
    info = parse_machine_info()
    root = posixpath.join(str(info["shared"]).rstrip("/"), DIAG)
    k0 = posixpath.join(root, "k0_run")
    k4 = posixpath.join(root, "k4_run")

    host_script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        ROOT={root}
        LAUNCH=$ROOT/step330_launch_inside.sh
        mkdir -p "$ROOT/k0_run/logs" "$ROOT/k4_run/logs"
        python3 - <<'PY'
        from pathlib import Path
        p = Path("{root}/step330_launch_inside.sh")
        p.write_bytes(p.read_bytes().replace(b"\\r\\n", b"\\n"))
        PY
        chmod 755 "$LAUNCH"
        docker exec mapqr-leicheng bash -n "$LAUNCH"
        for spec in "0:{k0}:{K0_PORT}" "4:{k4}:{K4_PORT}"; do
          IFS=: read -r k out port <<<"$spec"
          if [ -f "$out/launcher_rc.txt" ]; then
            echo "skip k=$k already has launcher_rc.txt"
            continue
          fi
          echo "starting SOAP_STALE_Q_K=$k port=$port out=$out"
          setsid -f sh -c "docker exec -e ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 -e STEP330_OUT=$out -e SOAP_STALE_Q_K=$k -e MAX_ITERS=30 -e MASTER_PORT=$port mapqr-leicheng bash --noprofile --norc $LAUNCH" </dev/null
          echo "started k=$k"
          while [ ! -f "$out/launcher_rc.txt" ]; do
            sleep 30
          done
          echo "finished k=$k rc=$(cat "$out/launcher_rc.txt")"
        done
        echo "STEP330_HOST_AB_DONE"
        """
    )

    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    target = None
    try:
        channel = jump.get_transport().open_channel(
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

        precheck = (
            "test \"$(docker ps --filter name=^/mapqr-leicheng$ --format '{{.Names}}')\" = mapqr-leicheng && "
            f"mkdir -p {root}/k0_run/logs {root}/k4_run/logs"
        )
        status, out, err = run(target, precheck, timeout=30)
        if status != 0:
            print(redact(out + err, info))
            return status

        sftp = target.open_sftp()
        try:
            sftp.put(str(LOCAL / "step330_launch_inside.sh"), posixpath.join(root, "step330_launch_inside.sh"))
        finally:
            sftp.close()

        b64 = base64.b64encode(host_script.encode()).decode()
        cmd = f"python3 - <<'PY'\nimport base64\nfrom pathlib import Path\np=Path('{root}/step330_host_ab.sh')\np.write_bytes(base64.b64decode('{b64}'))\np.chmod(0o755)\nPY\nsetsid -f bash {root}/step330_host_ab.sh >{root}/host_ab.log 2>&1; echo host_started"
        status, out, err = run(target, cmd, timeout=90)
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        return status
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
