#!/usr/bin/env python3
"""Aggregate all-rank SOAP install diag JSONL files."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize_rank(path: Path) -> dict:
    by_step: dict[int, dict] = defaultdict(lambda: {
        "qr_install": 0,
        "query_true": 0,
        "query_false": 0,
        "sync_ms": 0.0,
        "install_ms": 0.0,
        "install_if_due_ms": 0.0,
        "install_if_due_calls": 0,
    })
    rank = None
    for rec in load_records(path):
        rank = rec.get("rank", rank)
        step = int(rec.get("step", -1))
        ev = rec.get("event")
        slot = by_step[step]
        if ev == "qr_install":
            slot["qr_install"] += 1
            slot["query_true"] += int(rec.get("n_query_true", 0))
            slot["query_false"] += int(rec.get("n_query_false", 0))
            slot["sync_ms"] += float(rec.get("sync_ms", 0))
            slot["install_ms"] += float(rec.get("install_ms", 0))
        elif ev == "stale_q_install_if_due":
            slot["install_if_due_calls"] += 1
            slot["install_if_due_ms"] += float(rec.get("wall_ms", 0))
    return {"rank": rank, "path": str(path), "by_step": dict(by_step)}


def main() -> int:
    log_glob = sys.argv[1] if len(sys.argv) > 1 else ""
    if not log_glob:
        print("usage: step333_analyze_all_ranks.py '<dir>/install_diag_rank*.jsonl'")
        return 1
    paths = sorted(Path(p) for p in glob(log_glob))
    if not paths:
        print(f"NO_FILES {log_glob}")
        return 1

    summaries = [summarize_rank(p) for p in paths]
    print("=== all-rank install diag ===")
    print(f"ranks={len(summaries)} files={len(paths)}")
    focus_steps = sorted({s for sm in summaries for s in sm["by_step"] if s in (4, 14)})
    if not focus_steps:
        focus_steps = sorted({s for sm in summaries for s in sm["by_step"]})

    for step in focus_steps:
        print(f"--- step={step} ---")
        header = (
            f"{'rank':>4} {'params':>6} {'q_true':>7} {'q_false':>7} "
            f"{'sync_ms':>10} {'install_ms':>11} {'if_due_ms':>11}"
        )
        print(header)
        for sm in sorted(summaries, key=lambda x: x.get("rank", 99)):
            slot = sm["by_step"].get(step)
            if not slot:
                continue
            print(
                f"{sm.get('rank', '?'):>4} "
                f"{slot['qr_install']:>6} "
                f"{slot['query_true']:>7} "
                f"{slot['query_false']:>7} "
                f"{slot['sync_ms']:>10.1f} "
                f"{slot['install_ms']:>11.1f} "
                f"{slot['install_if_due_ms']:>11.1f}"
            )

    totals = defaultdict(float)
    for sm in summaries:
        for slot in sm["by_step"].values():
            totals["query_false"] += slot["query_false"]
            totals["sync_ms"] += slot["sync_ms"]
            totals["install_ms"] += slot["install_ms"]
            totals["install_if_due_ms"] += slot["install_if_due_ms"]
    print("--- totals all ranks ---")
    print(
        f"query_false={int(totals['query_false'])} sync_ms={totals['sync_ms']:.1f} "
        f"install_ms={totals['install_ms']:.1f} install_if_due_ms={totals['install_if_due_ms']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
