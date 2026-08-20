#!/usr/bin/env python3
"""STEP-334: remote inspect + analyze QR Call Stack on profiler step 13 (train iter14)."""
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

DIAG = "diagnostics/step333_allrank_install_profile_back8_20260820T165500"
ANALYZE_SCRIPT = "step334_analyze_qr_stacks.py"


def run(client, cmd: str, timeout: int = 600) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> int:
    info = parse_machine_info()
    root = posixpath.join(str(info["shared"]).rstrip("/"), DIAG)
    run_dir = posixpath.join(root, "run")
    local = Path(__file__).resolve().parent
    remote_script = posixpath.join(run_dir, ANALYZE_SCRIPT)

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

        # 1) inventory: is operator_details still present?
        inv = (
            f"echo ===inventory===; "
            f"find {run_dir}/profile_raw -name operator_details.csv 2>/dev/null | head -3; "
            f"find {run_dir}/profile_raw -name kernel_details.csv 2>/dev/null | head -3; "
            f"ls -lh $(find {run_dir}/profile_raw -name operator_details.csv 2>/dev/null | head -1) 2>/dev/null; "
            f"test -f {run_dir}/qr_step_attribution.json && echo attribution=present || echo attribution=missing"
        )
        st, out, err = run(target, inv, 60)
        print(redact(out, info))
        if "operator_details.csv" not in out:
            print("OPERATOR_DETAILS_MISSING — cannot stack-attribute; need re-profile")
            return 2

        # 2) push analyzer
        sftp = target.open_sftp()
        try:
            sftp.put(str(local / ANALYZE_SCRIPT), remote_script)
        finally:
            sftp.close()

        # 3) run in container (needs pandas)
        cmd = (
            f"docker exec mapqr-leicheng bash --noprofile --norc -lc '"
            f"source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1 && "
            f"python3 {remote_script} "
            f"--profile-root {run_dir}/profile_raw "
            f"--focus-step 13 "
            f"--output-json {run_dir}/qr_stack_attribution.json "
            f"--output-md {run_dir}/qr_stack_attribution.md && "
            f"echo ===MD=== && cat {run_dir}/qr_stack_attribution.md'"
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
