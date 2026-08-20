#!/usr/bin/env python3
"""Poll STEP-333; on completion run all-rank diag + QR profile analysis on remote."""
from __future__ import annotations

import json
import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

DIAG = "diagnostics/step333_allrank_install_profile_back8_20260820T165500"


def run(client, cmd: str, timeout: int = 600) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> int:
    info = parse_machine_info()
    root = posixpath.join(str(info["shared"]).rstrip("/"), DIAG)
    run_dir = posixpath.join(root, "run")
    analyze_ranks = posixpath.join(run_dir, "step333_analyze_all_ranks.py")
    analyze_qr = posixpath.join(run_dir, "step331_analyze_qr_by_step.py")

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

        status_cmd = (
            f"echo '===status==='; "
            f"test -f {run_dir}/launcher_rc.txt && echo rc=$(cat {run_dir}/launcher_rc.txt) || echo rc=running; "
            f"tail -2 {run_dir}/logs/launcher.log 2>/dev/null || true; "
            f"echo '===iter_times==='; "
            f"grep 'Iter \\[' {run_dir}/work/train.log 2>/dev/null | grep -oP 'Iter \\[\\K[0-9]+|time: \\K[0-9.]+' | paste - -; "
            f"echo '===diag_files==='; "
            f"ls -la {run_dir}/logs/install_diag_rank*.jsonl 2>/dev/null | wc -l"
        )
        st, out, err = run(target, status_cmd, 120)
        print(redact(out, info))
        if "rc=running" in out:
            return 0

        remote_py = (
            "source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1"
        )
        analyze_cmd = (
            f"docker exec mapqr-leicheng bash --noprofile --norc -lc '"
            f"{remote_py} && "
            f"echo ===all_rank_diag=== && "
            f"python3 {analyze_ranks} \"{run_dir}/logs/install_diag_rank*.jsonl\" && "
            f"echo ===qr_profile=== && "
            f"python3 {analyze_qr} --profile-root {run_dir}/profile_raw "
            f"--train-log {run_dir}/work/train.log "
            f"--output-json {run_dir}/qr_step_attribution.json "
            f"--output-md {run_dir}/qr_step_attribution.md && "
            f"cat {run_dir}/qr_step_attribution.json'"
        )
        st, out, err = run(target, analyze_cmd, 600)
        print(redact(out, info))
        if err.strip():
            print(redact(err, info), file=sys.stderr)

        # Retain raw profile for follow-up (stack / re-analysis). Do NOT auto-delete.
        inv = (
            f"find {run_dir}/profile_raw -type f \\( -name '*.csv' -o -name '*.json' \\) 2>/dev/null | wc -l; "
            f"du -sh {run_dir}/profile_raw 2>/dev/null || true"
        )
        st2, cout, cerr = run(target, inv, 120)
        print(redact("===profile_retained===\n" + cout, info))
        return st
    finally:
        if target:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
