#!/usr/bin/env python3
"""Parse NPU/GPU launcher logs for 30-step loss/time compare (STEP-274)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ITER_RE = re.compile(
    r"Iter \[(\d+)/(\d+)\].*?time: ([0-9.]+).*?"
    r"(?:frame_0_loss_map_cls: ([0-9.]+).*?)?"
    r"(?:loss: ([0-9.]+))?",
    re.DOTALL,
)
# mmdet style: grab time and a scalar total if present
LINE = re.compile(
    r"Iter \[(\d+)/(\d+)\][^\n]*?time: ([0-9.]+)",
)


def parse_lines(text: str) -> list[dict]:
    rows = []
    for m in LINE.finditer(text):
        step = int(m.group(1))
        total = int(m.group(2))
        t = float(m.group(3))
        rows.append({"step": step, "total_iters": total, "time_s": t})
    # attach loss_total heuristic from same line block
    for line in text.splitlines():
        if "Iter [" not in line:
            continue
        m2 = re.search(r"Iter \[(\d+)/(\d+)\]", line)
        if not m2:
            continue
        step = int(m2.group(1))
        tm = re.search(r"time: ([0-9.]+)", line)
        if not tm:
            continue
        # total loss proxy: sum of frame_0_loss_* on line (first few) or explicit loss:
        loss_m = re.search(r"\bloss: ([0-9.]+)", line)
        loss_val = float(loss_m.group(1)) if loss_m else None
        if loss_val is None:
            parts = re.findall(r"frame_0_loss_[a-z0-9_]+: ([0-9.]+)", line)
            if parts:
                loss_val = sum(float(x) for x in parts[:8])
        for r in rows:
            if r["step"] == step and loss_val is not None:
                r["loss_total_proxy"] = loss_val
                break
    return sorted(rows, key=lambda x: x["step"])


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: step274_summarize.py <log> [log2...]", file=sys.stderr)
        return 2
    for p in sys.argv[1:]:
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        rows = parse_lines(text)
        print(f"=== {p} ({len(rows)} iters) ===")
        if not rows:
            continue
        times = [r["time_s"] for r in rows if 2 <= r["step"] <= 30]
        losses = [r.get("loss_total_proxy") for r in rows if r.get("loss_total_proxy") is not None and 1 <= r["step"] <= 30]
        print(f"steps 1-30: n={len([r for r in rows if r['step']<=30])}")
        if times:
            print(f"time mean(2-30): {sum(times)/len(times):.3f}s  sum(2-30): {sum(times):.1f}s")
        for r in rows:
            if r["step"] in (1, 10, 11, 12, 20, 21, 22, 30):
                print(f"  step {r['step']:2d}: time={r['time_s']:.3f}s loss_proxy={r.get('loss_total_proxy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
