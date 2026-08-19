#!/usr/bin/env python3
"""Compare paired STEP-204 876-step metrics with frozen windows."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def diff_stats(values):
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def relative(candidate, baseline):
    return (candidate - baseline) / max(abs(baseline), 1e-30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    baseline_raw = args.baseline.read_bytes()
    candidate_raw = args.candidate.read_bytes()
    baseline = json.loads(baseline_raw)
    candidate = json.loads(candidate_raw)
    bp = baseline["per_step"]
    cp = candidate["per_step"]
    assert [x["iter"] for x in bp] == [x["iter"] for x in cp] == list(range(1, 877))
    assert baseline["iter_contract"] == candidate["iter_contract"]
    assert baseline["global_batch"] == candidate["global_batch"] == 128

    windows = {}
    for name, bw in baseline["windows"].items():
        cw = candidate["windows"][name]
        assert bw["steps"] == cw["steps"]
        windows[name] = {
            "count": bw["count"],
            "baseline_mean_s": bw["mean"],
            "candidate_mean_s": cw["mean"],
            "mean_time_delta_s": cw["mean"] - bw["mean"],
            "mean_time_delta_pct": relative(cw["mean"], bw["mean"]) * 100,
            "baseline_throughput_samples_s": bw["throughput_samples_s"],
            "candidate_throughput_samples_s": cw["throughput_samples_s"],
            "throughput_delta_pct": relative(
                cw["throughput_samples_s"], bw["throughput_samples_s"]
            )
            * 100,
        }

    time_diffs = [c["time_s"] - b["time_s"] for b, c in zip(bp, cp)]
    loss_rel = [relative(c["loss"], b["loss"]) for b, c in zip(bp, cp)]
    grad_pairs = [
        (b, c)
        for b, c in zip(bp, cp)
        if b["grad_norm"] is not None
        and c["grad_norm"] is not None
        and math.isfinite(b["grad_norm"])
        and math.isfinite(c["grad_norm"])
    ]
    grad_rel = [relative(c["grad_norm"], b["grad_norm"]) for b, c in grad_pairs]
    max_loss = max(range(876), key=lambda i: abs(loss_rel[i]))
    max_grad = max(range(len(grad_pairs)), key=lambda i: abs(grad_rel[i]))
    gb, gc = grad_pairs[max_grad]
    result = {
        "schema_version": 1,
        "input_sha256": {
            "baseline": hashlib.sha256(baseline_raw).hexdigest(),
            "candidate": hashlib.sha256(candidate_raw).hexdigest(),
        },
        "windows": windows,
        "per_step_time_delta_s": {
            **diff_stats(time_diffs),
            "candidate_faster_count": sum(x < 0 for x in time_diffs),
            "equal_count": sum(x == 0 for x in time_diffs),
            "candidate_slower_count": sum(x > 0 for x in time_diffs),
        },
        "trajectory": {
            "loss_relative_delta": diff_stats(loss_rel),
            "loss_max_abs_relative": {
                "iter": bp[max_loss]["iter"],
                "baseline": bp[max_loss]["loss"],
                "candidate": cp[max_loss]["loss"],
                "relative": loss_rel[max_loss],
            },
            "finite_grad_relative_delta": diff_stats(grad_rel),
            "grad_max_abs_relative": {
                "iter": gb["iter"],
                "baseline": gb["grad_norm"],
                "candidate": gc["grad_norm"],
                "relative": grad_rel[max_grad],
            },
            "baseline_grad_missing": baseline["function_gate"]["grad_missing_steps"],
            "candidate_grad_missing": candidate["function_gate"]["grad_missing_steps"],
            "baseline_grad_nonfinite": baseline["function_gate"][
                "grad_nonfinite_dynamic_loss_scale_steps"
            ],
            "candidate_grad_nonfinite": candidate["function_gate"][
                "grad_nonfinite_dynamic_loss_scale_steps"
            ],
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
