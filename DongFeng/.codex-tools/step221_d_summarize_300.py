#!/usr/bin/env python3
"""Summarize STEP-221 Stage D 300-step stale-Q A/B."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def load_steps(path: Path) -> list[dict]:
    by_iter = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "iter" in obj and "time" in obj and ("loss" in obj or obj.get("mode") == "train"):
            by_iter[int(obj["iter"])] = obj
    return [by_iter[i] for i in sorted(by_iter)]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(label: str, path: Path, max_iters: int, global_batch: int = 128) -> dict:
    steps = load_steps(path)
    per = []
    for row in steps:
        g = row.get("grad_norm")
        per.append({
            "iter": int(row["iter"]),
            "time_s": float(row["time"]),
            "loss": float(row.get("loss", float("nan"))),
            "grad_norm": None if g in (None, "nan") else float(g),
            "memory_mib": int(row["memory"]) if "memory" in row else None,
        })
    by = {p["iter"]: p for p in per}

    # Warmup / polluted: 1-3, and final checkpoint step.
    exclude = {1, 2, 3, max_iters}
    # SOAP steps in this contract after warmup land near multiples of 10 offset by 4:
    # 14,24,..., while step%10==4 after the first cycle in Stage C evidence.
    soap_ids = [i for i in range(1, max_iters + 1) if i >= 14 and i % 10 == 4 and i not in exclude]
    # Also detect by wall spike relative to median.
    body = [p for p in per if p["iter"] not in exclude]
    med = statistics.median([p["time_s"] for p in body]) if body else 0.0
    spike_ids = [p["iter"] for p in body if p["time_s"] > max(12.0, 2.2 * med)]
    soap_use = sorted(set(soap_ids) & set(by))
    if not soap_use:
        soap_use = spike_ids
    normal_ids = [i for i in range(1, max_iters + 1) if i in by and i not in exclude and i not in soap_use]

    def window(ids):
        vals = [by[i]["time_s"] for i in ids if i in by]
        return {
            "count": len(vals),
            "mean_s": mean(vals),
            "median_s": statistics.median(vals) if vals else float("nan"),
            "throughput_samples_per_s": (global_batch / mean(vals)) if vals else float("nan"),
            "iters": ids[:20] + (["..."] if len(ids) > 20 else []),
        }

    losses = [p["loss"] for p in per]
    mems = [p["memory_mib"] for p in per if p["memory_mib"] is not None]
    # Loss trend: early body vs late body
    body_loss = [by[i]["loss"] for i in sorted(by) if i not in exclude]
    early = body_loss[:40] if len(body_loss) >= 80 else body_loss[: max(1, len(body_loss) // 3)]
    late = body_loss[-40:] if len(body_loss) >= 80 else body_loss[-max(1, len(body_loss) // 3):]

    return {
        "label": label,
        "input": str(path),
        "count": len(per),
        "max_iters": max_iters,
        "per_step": per,
        "normal": window(normal_ids),
        "soap": window(soap_use),
        "soap_iters_used": soap_use[:30],
        "spike_iters": spike_ids[:30],
        "all_mean_s": mean([p["time_s"] for p in per]),
        "all_throughput_samples_per_s": (
            global_batch / mean([p["time_s"] for p in per]) if per else float("nan")
        ),
        "loss": {
            "all_finite": all(math.isfinite(x) for x in losses),
            "early_mean": mean(early),
            "late_mean": mean(late),
            "first4": losses[:4],
            "last4": losses[-4:],
        },
        "grad": {
            "finite_or_dynamic_ok": all(
                p["grad_norm"] is None or math.isfinite(p["grad_norm"]) for p in per
            ),
        },
        "peak_memory_mib": max(mems) if mems else None,
        "global_batch": global_batch,
    }


def compare(base: dict, cand: dict) -> dict:
    bn, cn = base["normal"]["mean_s"], cand["normal"]["mean_s"]
    bs, cs = base["soap"]["mean_s"], cand["soap"]["mean_s"]
    return {
        "normal_mean_base_s": bn,
        "normal_mean_cand_s": cn,
        "normal_delta_pct": (cn / bn - 1.0) * 100.0 if bn else float("nan"),
        "soap_mean_base_s": bs,
        "soap_mean_cand_s": cs,
        "soap_net_save_s": bs - cs,
        "all_throughput_base": base["all_throughput_samples_per_s"],
        "all_throughput_cand": cand["all_throughput_samples_per_s"],
        "throughput_ratio_cand_over_base": (
            cand["all_throughput_samples_per_s"] / base["all_throughput_samples_per_s"]
            if base["all_throughput_samples_per_s"] else float("nan")
        ),
        "loss_early_late": {
            "base": [base["loss"]["early_mean"], base["loss"]["late_mean"]],
            "cand": [cand["loss"]["early_mean"], cand["loss"]["late_mean"]],
        },
        "gates": {
            "count_match": base["count"] == base["max_iters"] and cand["count"] == cand["max_iters"],
            "loss_finite": base["loss"]["all_finite"] and cand["loss"]["all_finite"],
            "normal_not_worse_than_5pct": (cn / bn - 1.0) <= 0.05 if bn else False,
            "soap_net_save_gt_5s": (bs - cs) > 5.0 if (bs == bs and cs == cs) else False,
            "cand_loss_still_descending": cand["loss"]["late_mean"] < cand["loss"]["early_mean"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-logjson", required=True)
    parser.add_argument("--candidate-logjson", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-iters", type=int, default=300)
    args = parser.parse_args()
    base = summarize("baseline_k0", Path(args.baseline_logjson), args.max_iters)
    cand = summarize("candidate_k4", Path(args.candidate_logjson), args.max_iters)
    report = {"baseline": base, "candidate": cand, "compare": compare(base, cand)}
    # Strip per_step from the on-disk compare? Keep it for audit; file is small for 300.
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["compare"], indent=2))
    print("SUMMARY_OK", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
