#!/usr/bin/env python3
"""Run STEP-333 analysis only (when train finished but launcher stuck)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step333_poll_and_analyze import DIAG, run, connect, parse_machine_info, redact
import posixpath


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
        remote_py = "source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1"
        cmd = (
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
        st, out, err = run(target, cmd, 600)
        print(redact(out, info))
        if err.strip():
            print(redact(err, info), file=sys.stderr)
        return st
    finally:
        if target:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
