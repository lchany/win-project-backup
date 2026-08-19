#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from pathlib import Path


def quantiles(values):
    ordered = sorted(values)

    def pct(q):
        pos = (len(ordered) - 1) * q
        lo = math.floor(pos)
        hi = math.ceil(pos)
        return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)

    return {
        "min": min(values),
        "q1": pct(0.25),
        "median": statistics.median(values),
        "q3": pct(0.75),
        "p95": pct(0.95),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    assert baseline["iter_contract"] == candidate["iter_contract"]
    assert baseline["function_gate"] == candidate["function_gate"]
    base = {row["iter"]: row for row in baseline["per_step"]}
    cand = {row["iter"]: row for row in candidate["per_step"]}
    assert base.keys() == cand.keys() == set(range(1, 31))

    normal = baseline["iter_contract"]["normal_steps"]
    savings = [base[i]["time_s"] - cand[i]["time_s"] for i in normal]
    loss_rel_pct = [(cand[i]["loss"] / base[i]["loss"] - 1.0) * 100.0 for i in base]
    grad_rel_pct = [
        (cand[i]["grad_norm"] / base[i]["grad_norm"] - 1.0) * 100.0
        for i in base
        if base[i]["grad_norm"] is not None and math.isfinite(base[i]["grad_norm"])
    ]
    base_normal = baseline["windows"]["stable_normal_15_29_excluding_24"]
    cand_normal = candidate["windows"]["stable_normal_15_29_excluding_24"]
    base_cycle = baseline["windows"]["cycle_15_24"]
    cand_cycle = candidate["windows"]["cycle_15_24"]
    base_soap = baseline["windows"]["soap_14_24"]
    cand_soap = candidate["windows"]["soap_14_24"]
    result = {
        "schema_version": 1,
        "contracts_equal": True,
        "dynamic_loss_scale_trajectory_equal": True,
        "normal_paired_savings_s": {
            "steps": normal,
            "values": savings,
            "summary": quantiles(savings),
            "positive_count": sum(x > 0 for x in savings),
            "count": len(savings),
        },
        "window_comparison": {
            "normal_mean_saving_s": base_normal["mean_s"] - cand_normal["mean_s"],
            "normal_mean_saving_pct": (1 - cand_normal["mean_s"] / base_normal["mean_s"]) * 100,
            "normal_throughput_gain_pct": (cand_normal["throughput_samples_s"] / base_normal["throughput_samples_s"] - 1) * 100,
            "cycle_mean_saving_s": base_cycle["mean_s"] - cand_cycle["mean_s"],
            "cycle_mean_saving_pct": (1 - cand_cycle["mean_s"] / base_cycle["mean_s"]) * 100,
            "soap_mean_saving_s": base_soap["mean_s"] - cand_soap["mean_s"],
            "soap_mean_saving_pct": (1 - cand_soap["mean_s"] / base_soap["mean_s"]) * 100,
        },
        "trajectory": {
            "loss_relative_pct": quantiles(loss_rel_pct),
            "loss_max_abs_relative_pct": max(loss_rel_pct, key=abs),
            "loss_max_abs_step": max(base, key=lambda i: abs((cand[i]["loss"] / base[i]["loss"] - 1) * 100)),
            "loss_exact_steps": [i for i in base if base[i]["loss"] == cand[i]["loss"]],
            "grad_relative_pct_finite_steps": quantiles(grad_rel_pct),
            "grad_max_abs_relative_pct": max(grad_rel_pct, key=abs),
            "grad_max_abs_step": max(
                (i for i in base if base[i]["grad_norm"] is not None and math.isfinite(base[i]["grad_norm"])),
                key=lambda i: abs((cand[i]["grad_norm"] / base[i]["grad_norm"] - 1) * 100),
            ),
        },
        "gpu_reference": {
            "normal_mean_s": 4.3241,
            "cycle_mean_s": 4.416,
            "candidate_npu_gpu_normal_time_ratio": cand_normal["mean_s"] / 4.3241,
            "candidate_npu_gpu_normal_throughput_ratio": 4.3241 / cand_normal["mean_s"],
            "candidate_npu_gpu_cycle_time_ratio": cand_cycle["mean_s"] / 4.416,
            "candidate_npu_gpu_cycle_throughput_ratio": 4.416 / cand_cycle["mean_s"],
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
