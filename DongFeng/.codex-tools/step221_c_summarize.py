#!/usr/bin/env python3
"""Summarize a STEP-221 Stage C 30-step run from MMCV log.json."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


NORMAL_EXCLUDE = {1, 2, 3, 11, 12, 21, 22, 30}
SOAP_STEPS = {10, 20, 30}  # first cycle may land near 10 after warm start; report by time spikes too
# GPU-contract stable windows used by STEP-214 formal metrics:
# soap cycle pairs around 14/24 in resumed-from-iter0 30-step after warmup.
FORMAL_SOAP = {14, 24}
FORMAL_NORMAL = [15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29]
FORMAL_CYCLE = list(range(15, 25))


def load_steps(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "mode" in obj and obj.get("mode") == "train" and "iter" in obj and "time" in obj:
            rows.append(obj)
        elif "iter" in obj and "time" in obj and "loss" in obj:
            rows.append(obj)
    # unique by iter, keep last
    by_iter = {}
    for row in rows:
        by_iter[int(row["iter"])] = row
    return [by_iter[i] for i in sorted(by_iter)]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(label: str, path: Path, global_batch: int = 128) -> dict:
    steps = load_steps(path)
    per = []
    for row in steps:
        per.append({
            "iter": int(row["iter"]),
            "time_s": float(row["time"]),
            "loss": float(row.get("loss", float("nan"))),
            "grad_norm": (
                None if row.get("grad_norm") in (None, "nan")
                else float(row["grad_norm"])
            ),
            "memory_mib": int(row["memory"]) if "memory" in row else None,
        })
    by = {p["iter"]: p for p in per}

    def window(ids):
        vals = [by[i]["time_s"] for i in ids if i in by]
        return {
            "count": len(vals),
            "mean_s": mean(vals),
            "median_s": statistics.median(vals) if vals else float("nan"),
            "p95_s": sorted(vals)[max(0, math.ceil(0.95 * len(vals)) - 1)] if vals else float("nan"),
            "throughput_samples_per_s": (global_batch / mean(vals)) if vals else float("nan"),
        }

    soap_present = [i for i in FORMAL_SOAP if i in by]
    # Also detect SOAP by wall spike: time > 2x median of non-excluded.
    candidates = [p for p in per if p["iter"] not in {1, 2, 3, 30}]
    med = statistics.median([p["time_s"] for p in candidates]) if candidates else 0
    spike = [p["iter"] for p in candidates if p["time_s"] > max(10.0, 2.5 * med)]

    losses = [p["loss"] for p in per]
    grads = [p["grad_norm"] for p in per if p["grad_norm"] is not None]
    mems = [p["memory_mib"] for p in per if p["memory_mib"] is not None]

    return {
        "label": label,
        "input": str(path),
        "count": len(per),
        "per_step": per,
        "formal_normal": window(FORMAL_NORMAL),
        "formal_cycle_window": window(FORMAL_CYCLE),
        "formal_soap_steps": {str(i): by[i]["time_s"] for i in soap_present},
        "spike_iters": spike,
        "loss": {
            "all_finite": all(math.isfinite(x) for x in losses),
            "values": losses,
        },
        "grad": {
            "finite_or_none_ok": all(g is None or math.isfinite(g) for g in [p["grad_norm"] for p in per]),
            "values": [p["grad_norm"] for p in per],
        },
        "peak_memory_mib": max(mems) if mems else None,
        "global_batch": global_batch,
    }


def compare(base: dict, cand: dict) -> dict:
    bn = base["formal_normal"]["mean_s"]
    cn = cand["formal_normal"]["mean_s"]
    # Prefer spike-based soap times if formal 14/24 exist; else use spike iters.
    def soap_mean(report):
        vals = list(report["formal_soap_steps"].values())
        if not vals and report["spike_iters"]:
            by = {p["iter"]: p["time_s"] for p in report["per_step"]}
            vals = [by[i] for i in report["spike_iters"] if i in by]
        return mean(vals)

    bs, cs = soap_mean(base), soap_mean(cand)
    return {
        "normal_mean_base_s": bn,
        "normal_mean_cand_s": cn,
        "normal_delta_s": cn - bn,
        "normal_delta_pct": (cn / bn - 1.0) * 100.0 if bn else float("nan"),
        "soap_mean_base_s": bs,
        "soap_mean_cand_s": cs,
        "soap_net_save_s": bs - cs,
        "gates": {
            "count_30": base["count"] == 30 and cand["count"] == 30,
            "loss_finite": base["loss"]["all_finite"] and cand["loss"]["all_finite"],
            "normal_not_worse_than_5pct": (cn / bn - 1.0) <= 0.05 if bn else False,
            "soap_net_save_gt_5s": (bs - cs) > 5.0 if (bs == bs and cs == cs) else False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-logjson", required=True)
    parser.add_argument("--candidate-logjson", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = summarize("baseline_k0", Path(args.baseline_logjson))
    cand = summarize("candidate_k4", Path(args.candidate_logjson))
    report = {"baseline": base, "candidate": cand, "compare": compare(base, cand)}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["compare"], indent=2))
    print("SUMMARY_OK", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
