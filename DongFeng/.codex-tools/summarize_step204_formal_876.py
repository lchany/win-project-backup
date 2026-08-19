#!/usr/bin/env python3
"""Summarize one STEP-204 876-step formal run without changing its artifacts."""

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


def scalar_stats(values):
    assert values
    return {
        "count": len(values),
        "min": min(values),
        "q1": percentile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "q3": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def time_stats(rows, global_batch):
    times = [float(row["time"]) for row in rows]
    result = scalar_stats(times)
    result.update(
        {
            "steps": [int(row["iter"]) for row in rows],
            "sum_s": sum(times),
            "iqr_s": result["q3"] - result["q1"],
            "throughput_samples_s": global_batch * len(rows) / sum(times),
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--global-batch", type=int, default=128)
    args = parser.parse_args()

    raw = args.input_jsonl.read_bytes()
    records = [
        obj
        for line in raw.splitlines()
        if (obj := json.loads(line)).get("mode") == "train" and "iter" in obj
    ]
    assert [int(row["iter"]) for row in records] == list(range(1, 877))
    assert all(math.isfinite(float(row["loss"])) for row in records)

    by_iter = {int(row["iter"]): row for row in records}
    unpolluted = [by_iter[i] for i in range(1, 876)]
    stable = [by_iter[i] for i in range(15, 876)]
    stable_normal = [row for row in stable if int(row["iter"]) % 10 != 4]
    stable_soap = [row for row in stable if int(row["iter"]) % 10 == 4]
    stable_cycles = [by_iter[i] for i in range(15, 875)]
    last_100_unpolluted = [by_iter[i] for i in range(776, 876)]
    last_10_cycles = [by_iter[i] for i in range(775, 875)]
    last_100_normal = [row for row in last_100_unpolluted if int(row["iter"]) % 10 != 4]
    last_10_cycle_soap = [row for row in last_10_cycles if int(row["iter"]) % 10 == 4]
    assert len(stable_normal) == 775
    assert len(stable_soap) == 86
    assert len(stable_cycles) == 860
    assert len(last_100_unpolluted) == 100
    assert len(last_10_cycles) == 100
    assert len(last_100_normal) == 90
    assert len(last_10_cycle_soap) == 10

    grad_missing = [int(row["iter"]) for row in records if "grad_norm" not in row]
    grad_nonfinite = [
        int(row["iter"])
        for row in records
        if "grad_norm" in row and not math.isfinite(float(row["grad_norm"]))
    ]
    finite_grads = [
        float(row["grad_norm"])
        for row in records
        if "grad_norm" in row and math.isfinite(float(row["grad_norm"]))
    ]
    losses = [float(row["loss"]) for row in records]

    result = {
        "schema_version": 1,
        "label": args.label,
        "input_name": args.input_jsonl.name,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "global_batch": args.global_batch,
        "iter_contract": {
            "count": 876,
            "checkpoint_polluted_step": 876,
            "checkpoint_polluted_step_excluded_from_primary_windows": True,
            "soap_definition": "iter >= 15 and iter % 10 == 4",
            "complete_cycle_definition": "15..874 split into 86 consecutive 10-step cycles",
            "last_100_unpolluted": [776, 875],
            "last_10_complete_cycles": [775, 874],
        },
        "function_gate": {
            "loss_all_finite": True,
            "grad_missing_steps": grad_missing,
            "grad_nonfinite_dynamic_loss_scale_steps": grad_nonfinite,
            "loss_distribution": scalar_stats(losses),
            "finite_grad_distribution": scalar_stats(finite_grads),
        },
        "windows": {
            "all_876_including_checkpoint_step": time_stats(records, args.global_batch),
            "all_1_875_excluding_checkpoint_step": time_stats(unpolluted, args.global_batch),
            "stable_15_875": time_stats(stable, args.global_batch),
            "stable_normal_15_875_excluding_mod10_eq4": time_stats(stable_normal, args.global_batch),
            "stable_soap_15_875_mod10_eq4": time_stats(stable_soap, args.global_batch),
            "stable_complete_cycles_15_874": time_stats(stable_cycles, args.global_batch),
            "last_100_unpolluted_776_875": time_stats(last_100_unpolluted, args.global_batch),
            "last_100_normal_within_776_875": time_stats(last_100_normal, args.global_batch),
            "last_10_complete_cycles_775_874": time_stats(last_10_cycles, args.global_batch),
            "last_10_cycle_soap_steps": time_stats(last_10_cycle_soap, args.global_batch),
        },
        "per_step": [
            {
                "iter": int(row["iter"]),
                "time_s": float(row["time"]),
                "loss": float(row["loss"]),
                "grad_norm": None if "grad_norm" not in row else float(row["grad_norm"]),
                "memory_mib": int(row["memory"]),
            }
            for row in records
        ],
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
