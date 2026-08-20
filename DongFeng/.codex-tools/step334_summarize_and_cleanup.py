#!/usr/bin/env python3
"""Summarize STEP-334 json focusing on real LinalgQr only.

Raw profile is RETAINED by default (user rule 2026-08-20: do not delete
immediately after analysis). Pass --delete-raw only when explicitly requested.
"""
from __future__ import annotations

import argparse
import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

DIAG = "diagnostics/step333_allrank_install_profile_back8_20260820T165500"


def run(client, cmd: str, timeout: int = 300) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delete-raw",
        action="store_true",
        help="Delete profile_raw csv/json after summarize. Default: keep.",
    )
    args = parser.parse_args()

    info = parse_machine_info()
    run_dir = posixpath.join(str(info["shared"]).rstrip("/"), DIAG, "run")
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
        cmd = f"""python3 - <<'PY'
import json
from pathlib import Path
p = Path("{run_dir}/qr_stack_attribution.json")
d = json.loads(p.read_text())
groups = [g for g in d.get("top_groups", []) if g.get("name") in ("aclnnLinalgQr", "aten::linalg_qr")]
by = {{}}
for g in groups:
    t = g["trigger_class"]
    by.setdefault(t, {{"count": 0, "device_self_ms": 0.0, "host_self_ms": 0.0, "boundaries": []}})
    by[t]["count"] += g["count"]
    by[t]["device_self_ms"] += g["device_self_ms"]
    by[t]["host_self_ms"] += g["host_self_ms"]
    if g.get("boundary") and g["boundary"] not in by[t]["boundaries"]:
        by[t]["boundaries"].append(g["boundary"])
out = {{
  "has_step_id": d.get("has_step_id"),
  "has_call_stack": d.get("has_call_stack"),
  "caveat": "operator_details has no Step Id; stacks cover whole profile window iters10-16, not iter14 alone",
  "linalg_qr_by_trigger": {{k: {{"count": v["count"], "device_self_ms": round(v["device_self_ms"],1), "host_self_ms": round(v["host_self_ms"],1), "boundaries": v["boundaries"][:3]}} for k,v in sorted(by.items(), key=lambda kv: -kv[1]["device_self_ms"])}},
  "top_linalg_groups": groups[:6],
  "verdict_raw": d.get("verdict"),
}}
Path("{run_dir}/qr_stack_linalg_only.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps(out, ensure_ascii=False, indent=2))
PY"""
        st, out, err = run(target, cmd, 60)
        print(redact(out, info))
        if err.strip():
            print(redact(err, info), file=sys.stderr)

        if args.delete_raw:
            cleanup = f"""
echo ===before===
find {run_dir}/profile_raw -type f \\( -name '*.csv' -o -name '*.json' -o -name '*.db' -o -name '*.trace' \\) 2>/dev/null | wc -l
find {run_dir}/profile_raw -type f \\( -name '*.csv' -o -name '*.json' -o -name '*.db' -o -name '*.trace' -o -name '*.txt' \\) -delete 2>/dev/null || true
find {run_dir}/profile_raw -type d -empty -delete 2>/dev/null || true
echo ===after===
find {run_dir}/profile_raw -type f \\( -name '*.csv' -o -name '*.json' \\) 2>/dev/null | wc -l
"""
            _, out2, _ = run(target, cleanup, 180)
            print(redact(out2, info))
        else:
            retain = f"""
echo ===profile_retained===
find {run_dir}/profile_raw -type f \\( -name '*.csv' -o -name '*.json' \\) 2>/dev/null | wc -l
du -sh {run_dir}/profile_raw 2>/dev/null || true
"""
            _, out2, _ = run(target, retain, 60)
            print(redact(out2, info))
        return st
    finally:
        if target:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
