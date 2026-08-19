#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from pathlib import Path


def summary(values):
    ordered = sorted(values)

    def pct(q):
        pos = (len(ordered) - 1) * q
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)

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

    assert baseline["logged_iters"] == candidate["logged_iters"] == list(range(30, 37))
    assert baseline["grad_nonfinite_iters"] == candidate["grad_nonfinite_iters"] == [31]
    assert baseline["checkpoint"] == candidate["checkpoint"]
    base = {row["iter"]: row for row in baseline["per_step"]}
    cand = {row["iter"]: row for row in candidate["per_step"]}
    loss_rel = [(cand[i]["loss"] / base[i]["loss"] - 1.0) * 100.0 for i in base]
    finite_iters = [i for i in base if math.isfinite(base[i]["grad_norm"]) and math.isfinite(cand[i]["grad_norm"])]
    grad_rel = [(cand[i]["grad_norm"] / base[i]["grad_norm"] - 1.0) * 100.0 for i in finite_iters]
    time_saving = [base[i]["time_s"] - cand[i]["time_s"] for i in base]
    result = {
        "schema_version": 1,
        "contracts_equal": True,
        "logged_iters_equal": True,
        "dynamic_loss_scale_phase_equal": True,
        "checkpoint_continuity_equal": True,
        "loss_relative_pct": summary(loss_rel),
        "loss_max_abs_relative_pct": max(loss_rel, key=abs),
        "loss_max_abs_step": max(base, key=lambda i: abs((cand[i]["loss"] / base[i]["loss"] - 1.0) * 100.0)),
        "grad_relative_pct_finite_steps": summary(grad_rel),
        "grad_max_abs_relative_pct": max(grad_rel, key=abs),
        "grad_max_abs_step": max(finite_iters, key=lambda i: abs((cand[i]["grad_norm"] / base[i]["grad_norm"] - 1.0) * 100.0)),
        "time_saving_s_including_resume_and_checkpoint_steps": summary(time_saving),
        "candidate_faster_count": sum(value > 0 for value in time_saving),
        "step_count": len(time_saving),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
