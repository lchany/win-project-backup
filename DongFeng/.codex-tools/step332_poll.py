#!/usr/bin/env python3
"""STEP-332: poll install-query test and summarize diag + iter times on remote."""
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

DIAG = "diagnostics/step332_install_query_test_back8_20260820T163000"


def main() -> int:
    info = parse_machine_info()
    root = posixpath.join(str(info["shared"]).rstrip("/"), DIAG)
    run_dir = posixpath.join(root, "run")
    diag_log = posixpath.join(run_dir, "logs/install_diag.jsonl")
    analyze = posixpath.join(root, "step332_analyze_install_diag.py")

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
        cmd = (
            f"echo '===status==='; "
            f"test -f {run_dir}/launcher_rc.txt && echo rc=$(cat {run_dir}/launcher_rc.txt) || echo rc=running; "
            f"tail -3 {run_dir}/logs/launcher.log 2>/dev/null || true; "
            f"echo '===iter_times==='; "
            f"grep 'Iter \\[' {run_dir}/work/train.log 2>/dev/null | grep -oP 'Iter \\[\\K[0-9]+|time: \\K[0-9.]+' | paste - -; "
            f"echo '===install_diag==='; "
            f"test -f {diag_log} && python3 {analyze} {diag_log} || echo diag_missing"
        )
        _, stdout, stderr = target.exec_command(cmd, timeout=120)
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
