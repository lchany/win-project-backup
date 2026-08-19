#!/usr/bin/env python3
from __future__ import annotations

import json
import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


REL = "diagnostics/step299_bad_single_visible_qr7_20260819T163125"


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
        target = connect(
            str(info["target_host"]), int(info["target_port"]),
            str(info["target_user"]), str(info["target_password"]), sock=channel,
        )
        command = (
            f"echo rc=$(cat {remote_dir}/launcher_rc.txt 2>/dev/null || echo pending); "
            f"tail -n 12 {remote_dir}/driver.log 2>/dev/null || true; "
            f"test -f {remote_dir}/summary.json && echo summary_ready=1 || echo summary_ready=0"
        )
        _, stdout, stderr = target.exec_command(command, timeout=30)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        return stdout.channel.recv_exit_status()
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
