#!/usr/bin/env python3
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

REL = "diagnostics/step280_qr_cpu_vs_mx"


def run(client, cmd: str, timeout: int = 30):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> int:
    info = parse_machine_info()
    d = posixpath.join(str(info["shared"]).rstrip("/"), REL)
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
        cmd = (
            f"echo PID=$(cat {d}/step280.pid 2>/dev/null || echo none); "
            f"echo RC=$(cat {d}/launcher_rc.txt 2>/dev/null || echo pending); "
            f"wc -l {d}/results.jsonl 2>/dev/null || true; "
            f"tail -n 12 {d}/step280_driver.log 2>/dev/null || true; "
            f"if [ -f {d}/summary.json ]; then python3 -c "
            f"\"import json; s=json.load(open('{d}/summary.json')); "
            f"print('SUMMARY pass',s.get('n_pass'),'fail',s.get('n_fail'),'shapes',s.get('fail_shapes')); "
            f"[print('FAIL',x) for x in s.get('fails',[])]\"; fi"
        )
        st, out, err = run(target, cmd, timeout=45)
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        return st
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
