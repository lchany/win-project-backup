#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
DIR=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step285_bad8_official_qr
echo '=== done ==='
test -f "$DIR/launcher_rc.txt" && echo rc=$(cat "$DIR/launcher_rc.txt") || echo rc=none
if [ -f "$DIR/step285.pid" ]; then
  old=$(cat "$DIR/step285.pid")
  if kill -0 "$old" 2>/dev/null; then echo running=1; else echo running=0; fi
fi
echo '=== log last ==='
tail -n 8 "$DIR/step285_driver.log"
echo '=== summary python ==='
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step285_bad8_official_qr/step285_summary.json")
if not p.is_file():
    print("summary missing")
    raise SystemExit(0)
s=json.loads(p.read_text())
print("n", s["n"], "ok", s["n_ok"], "fail", s["n_fail"])
print("visible", s.get("visible"))
print("by_npu", json.dumps(s["by_npu"]))
# npu0/npu1 recon
from collections import defaultdict
recons=defaultdict(list)
for r in s["results"]:
    if r.get("ok") and r.get("recon_max") is not None:
        recons[r["npu"]].append(r["recon_max"])
for npu, vals in sorted(recons.items()):
    print(f"npu{npu} recon_min={min(vals):.3e} recon_max={max(vals):.3e} n={len(vals)}")
print("npu0_ok", sum(1 for r in s["results"] if r["npu"]==0 and r.get("ok")))
print("npu1_ok", sum(1 for r in s["results"] if r["npu"]==1 and r.get("ok")))
print("npu1_tiny_recon", sum(1 for r in s["results"] if r["npu"]==1 and r.get("ok") and (r.get("recon_max") or 1)<1e-12))
print("npu1_large_recon", sum(1 for r in s["results"] if r["npu"]==1 and r.get("ok") and (r.get("recon_max") or 0)>1e-10))
print("npu2_7_crash507015", sum(1 for r in s["results"] if r["npu"]>=2 and r.get("error_has_507015")))
print("dump_Q_nonfinite_all8", "see dump_snapshots")
PY
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step285_bad8_official_qr/dump_snapshots.json")
if p.is_file():
    snaps=json.loads(p.read_text())
    print("snapshots", len(snaps))
    sums=sorted({round(s["A_sum"], 20) for s in snaps})
    print("A_sum_unique", sums)
    print("cpu_ok", all(s["cpu_torch_ok"] for s in snaps))
    print("dumpQ_finite", [s.get("dump_Q",{}).get("finite") for s in snaps])
    print("dumpQ_nan_cols", [ (s.get("dump_Q",{}).get("nan_col_start"), s.get("dump_Q",{}).get("nan_col_end"), s.get("dump_Q",{}).get("nan_col_count")) for s in snaps])
PY
"""


def main() -> int:
    info = parse_machine_info()
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
        _, stdout, stderr = target.exec_command(CMD, timeout=30)
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
