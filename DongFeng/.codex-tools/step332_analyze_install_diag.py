#!/usr/bin/env python3
"""Summarize SOAP install diag JSONL (query_true vs query_false at install time)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "")
    if not path.is_file():
        print(f"MISSING {path}")
        return 1
    by_step: dict[int, list[dict]] = defaultdict(list)
    total_ready = 0
    total_wait = 0
    total_sync_ms = 0.0
    total_install_ms = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") != "qr_install":
            continue
        step = int(rec["step"])
        by_step[step].append(rec)
        total_ready += int(rec.get("n_query_true", 0))
        total_wait += int(rec.get("n_query_false", 0))
        total_sync_ms += float(rec.get("sync_ms", 0))
        total_install_ms += float(rec.get("install_ms", 0))
    print("=== SOAP install diag summary ===")
    print(f"records={sum(len(v) for v in by_step.values())} steps_with_install={len(by_step)}")
    print(f"total_query_true={total_ready} total_query_false={total_wait}")
    if total_ready + total_wait:
        pct = 100.0 * total_ready / (total_ready + total_wait)
        print(f"query_true_pct={pct:.2f}%")
    print(f"total_sync_ms={total_sync_ms:.1f} total_install_ms={total_install_ms:.1f}")
    print("--- per training step (aggregated) ---")
    for step in sorted(by_step):
        rows = by_step[step]
        r_true = sum(int(r.get("n_query_true", 0)) for r in rows)
        r_false = sum(int(r.get("n_query_false", 0)) for r in rows)
        sync_ms = sum(float(r.get("sync_ms", 0)) for r in rows)
        install_ms = sum(float(r.get("install_ms", 0)) for r in rows)
        n_params = len(rows)
        print(
            f"step={step:3d} params={n_params:4d} "
            f"query_true={r_true:5d} query_false={r_false:5d} "
            f"sync_ms={sync_ms:10.1f} install_ms={install_ms:10.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
