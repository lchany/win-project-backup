#!/usr/bin/env python3
"""STEP-330: poll remote stale-Q A/B via grep on train.log."""
from __future__ import annotations

import json
import posixpath
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

DIAG = "diagnostics/step330_stale_q_ab_30step_back8_20260820T143600"
ITER_RE = re.compile(r"Iter \[(\d+)/\d+\].*?time: ([0-9.]+)")


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
        cmd = (
            f"for label in k0_run k4_run; do "
            f"echo \"===$label===\"; "
            f"test -f {root}/$label/launcher_rc.txt && echo rc=$(cat {root}/$label/launcher_rc.txt) || echo rc=running; "
            f"grep 'Iter \\[' {root}/$label/work/train.log 2>/dev/null | grep -oP 'Iter \\[\\K[0-9]+|time: \\K[0-9.]+' | paste - -; "
            f"done"
        )
        _, stdout, stderr = target.exec_command(cmd, timeout=60)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        print(redact(out, info))
        if err.strip():
            print(redact(err, info), file=sys.stderr)
        return 0
    finally:
        if target:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
