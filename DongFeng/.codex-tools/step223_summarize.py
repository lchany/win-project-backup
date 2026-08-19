#!/usr/bin/env python3
"""Summarize STEP-223 30-step A/B under stale-Q k=4."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ITER_RE = re.compile(
    r"Iter\s+\[(\d+)/(\d+)\].*?time:\s*([0-9.]+).*?memory:\s*(\d+).*?loss:\s*([0-9.naninf+-]+)",
    re.I,
)
GRAD_RE = re.compile(r"grad_norm:\s*([0-9.naninf+-]+)", re.I)


def parse_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        m = ITER_RE.search(line)
        if not m:
            continue
        it, total, t, mem, loss = m.groups()
        g = GRAD_RE.search(line)
        rows.append(
            {
                "iter": int(it),
                "total": int(total),
                "time": float(t),
                "memory": int(mem),
                "loss": float(loss) if loss.lower() not in {"nan", "inf", "-inf"} else float("nan"),
                "grad": float(g.group(1)) if g and g.group(1).lower() not in {"nan", "inf", "-inf"} else (
                    float("inf") if g and "inf" in g.group(1).lower() else float("nan")
                ),
            }
        )
    return rows


def stats(times):
    if not times:
        return {}
    s = sorted(times)
    n = len(s)
    return {
        "n": n,
        "mean": sum(s) / n,
        "median": s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]),
        "p95": s[min(n - 1, max(0, math.ceil(0.95 * n) - 1))],
    }


def classify(rows):
    # SOAP frequency 10: treat iters 10,20,30 as cycle-ish windows; also neighbors sometimes heavy.
    soap = [r for r in rows if r["iter"] % 10 == 0]
    # exclude first 3 warm and soap iters for normal
    normal = [r for r in rows if r["iter"] > 3 and r["iter"] % 10 != 0]
    return soap, normal


def main() -> int:
    out = Path(sys.argv[1])
    report = {"out": str(out)}
    for name in ("baseline", "candidate"):
        run = out / f"{name}_run"
        log = run / "work" / "train.log"
        if not log.exists():
            # fallback: nested dated log
            logs = list((run / "work").glob("*.log"))
            log = logs[0] if logs else log
        exit_txt = (run / "logs" / "exit.txt").read_text(encoding="utf-8", errors="replace") if (run / "logs" / "exit.txt").exists() else ""
        rows = parse_log(log) if log.exists() else []
        soap, normal = classify(rows)
        times = [r["time"] for r in rows]
        peak = max((r["memory"] for r in rows), default=0)
        finite_loss = all(math.isfinite(r["loss"]) for r in rows) if rows else False
        # grad may be inf early; require finite for iters>3
        finite_grad_late = all(math.isfinite(r["grad"]) for r in rows if r["iter"] > 3) if rows else False
        mean_t = (sum(times) / len(times)) if times else None
        # samples/s: batch 16 * 8 / mean_time
        thr = (128.0 / mean_t) if mean_t else None
        report[name] = {
            "exit": exit_txt.strip(),
            "iters": len(rows),
            "full": stats(times),
            "normal": stats([r["time"] for r in normal]),
            "soap": stats([r["time"] for r in soap]),
            "peak_memory": peak,
            "finite_loss": finite_loss,
            "finite_grad_late": finite_grad_late,
            "throughput_samples_s": thr,
        }

    b, c = report.get("baseline"), report.get("candidate")
    verdict = "INCOMPLETE"
    bi = (b or {}).get("iters") or 0
    ci = (c or {}).get("iters") or 0
    need = bi  # both must match and be >0
    if b and c and bi == ci and bi > 0 and b.get("finite_loss") and c.get("finite_loss"):
        bt, ct = b["throughput_samples_s"], c["throughput_samples_s"]
        if bt and ct and ct > bt:
            verdict = "PASS_E2E_THROUGHPUT"
        else:
            verdict = "REJECT_STILL_NO_E2E_UNDER_STALE_Q"
    report["verdict"] = verdict
    # late-window secondary (iters 10..N excluding final if present)
    if b and c and bi == ci and bi >= 20:
        def late_mean(name):
            run = out / f"{name}_run"
            log = run / "work" / "train.log"
            if not log.exists():
                logs = list((run / "work").glob("*.log"))
                log = logs[0] if logs else None
            if not log:
                return None
            rows = parse_log(log)
            late = [r["time"] for r in rows if 10 <= r["iter"] <= bi]
            return (sum(late) / len(late)) if late else None
        bl, cl = late_mean("baseline"), late_mean("candidate")
        if bl and cl:
            report["late_window_mean"] = {"baseline": bl, "candidate": cl}
            report["late_throughput"] = {"baseline": 128.0 / bl, "candidate": 128.0 / cl}
            report["delta_late_throughput_pct"] = 100.0 * ((128.0 / cl) / (128.0 / bl) - 1.0)
    if b and c and b.get("throughput_samples_s") and c.get("throughput_samples_s"):
        report["delta_throughput_pct"] = 100.0 * (c["throughput_samples_s"] / b["throughput_samples_s"] - 1.0)
        report["delta_full_mean_pct"] = 100.0 * (c["full"]["mean"] / b["full"]["mean"] - 1.0)
        if b["normal"] and c["normal"] and b["normal"].get("mean") and c["normal"].get("mean"):
            report["delta_normal_mean_pct"] = 100.0 * (c["normal"]["mean"] / b["normal"]["mean"] - 1.0)
        if b["soap"] and c["soap"] and b["soap"].get("mean") and c["soap"].get("mean"):
            report["delta_soap_mean_pct"] = 100.0 * (c["soap"]["mean"] / b["soap"]["mean"] - 1.0)

    (out / "step223_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# STEP-223 DataContainer pin under stale-Q k=4",
        "",
        f"- Verdict: `{verdict}`",
    ]
    for name in ("baseline", "candidate"):
        d = report[name]
        lines += [
            f"## {name}",
            f"- iters `{d['iters']}` exit `{d['exit']}`",
            f"- full mean `{d['full'].get('mean')}` throughput `{d.get('throughput_samples_s')}`",
            f"- normal mean `{d['normal'].get('mean')}` soap mean `{d['soap'].get('mean')}`",
            f"- peak mem `{d['peak_memory']}` finite_loss `{d['finite_loss']}`",
            "",
        ]
    for k in ("delta_throughput_pct", "delta_full_mean_pct", "delta_normal_mean_pct", "delta_soap_mean_pct"):
        if k in report:
            lines.append(f"- {k}: `{report[k]:.3f}`")
    (out / "step223_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, **{k: report.get(k) for k in report if k.startswith("delta")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
