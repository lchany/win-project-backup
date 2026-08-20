#!/usr/bin/env python3
"""Poll STEP-331 and run remote QR-by-step analysis."""
from __future__ import annotations

import json
import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

DIAG = "diagnostics/step331_qr_step10_vs14_profile_back8_20260820T153000"


def run(client, cmd: str, timeout: int = 600) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


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
        status_cmd = (
            f"test -f {root}/launcher_rc.txt && echo rc=$(cat {root}/launcher_rc.txt) || echo rc=running; "
            f"grep -c 'Iter \\[' {root}/work/train.log 2>/dev/null || echo 0; "
            f"find {root}/profile_raw -name kernel_details.csv 2>/dev/null | head -1"
        )
        st, out, err = run(target, status_cmd, 60)
        print(redact(out, info))
        if "rc=running" in out:
            return 0

        analyze = (
            f"docker exec mapqr-leicheng bash --noprofile --norc -lc '"
            f"source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1 && "
            f"python {root}/step331_analyze_qr_by_step.py "
            f"--profile-root {root}/profile_raw "
            f"--train-log {root}/work/train.log "
            f"--output-json {root}/qr_step_attribution.json "
            f"--output-md {root}/qr_step_attribution.md'"
        )
        st, out, err = run(target, analyze, 600)
        print(redact(out, info))
        if err.strip():
            print(redact(err, info), file=sys.stderr)
        if st != 0:
            return st

        cat = f"test -f {root}/qr_step_attribution.json && cat {root}/qr_step_attribution.json || echo missing"
        st, out, err = run(target, cat, 60)
        if out.strip() and out.strip() != "missing":
            data = json.loads(out)
            print("\n=== VERDICT ===")
            for v in data.get("verdict", []):
                print(v)
            f = data.get("focus", {})
            for key in ("iter10_submit", "iter14_install"):
                k = (f.get(key) or {}).get("kernel") or {}
                print(f"{key}: qr_kernel_ms={k.get('qr_kernel_ms')} train_iter={(f.get(key) or {}).get('train_iter')}")
        return 0
    finally:
        if target:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
