#!/usr/bin/env python3
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


def stats(rows, global_batch):
    times = [float(row["time"]) for row in rows]
    total = sum(times)
    return {
        "steps": [int(row["iter"]) for row in rows],
        "count": len(rows),
        "sum_s": total,
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "p95_s": percentile(times, 0.95),
        "q1_s": percentile(times, 0.25),
        "q3_s": percentile(times, 0.75),
        "iqr_s": percentile(times, 0.75) - percentile(times, 0.25),
        "throughput_samples_s": global_batch * len(rows) / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--global-batch", type=int, default=128)
    args = parser.parse_args()

    raw = args.input_jsonl.read_bytes()
    records = []
    for line in raw.splitlines():
        obj = json.loads(line)
        if obj.get("mode") == "train" and "iter" in obj:
            records.append(obj)
    assert [row["iter"] for row in records] == list(range(1, 31))
    assert all(math.isfinite(float(row["loss"])) for row in records)
    finite_grad_steps = [
        int(row["iter"])
        for row in records
        if "grad_norm" in row and math.isfinite(float(row["grad_norm"]))
    ]
    missing_grad_steps = [int(row["iter"]) for row in records if "grad_norm" not in row]
    nonfinite_grad_steps = [
        int(row["iter"])
        for row in records
        if "grad_norm" in row and not math.isfinite(float(row["grad_norm"]))
    ]
    assert missing_grad_steps == [1, 2]
    assert nonfinite_grad_steps == [3]
    assert finite_grad_steps == list(range(4, 31))

    by_iter = {int(row["iter"]): row for row in records}
    normal = [by_iter[i] for i in range(15, 30) if i != 24]
    soap = [by_iter[i] for i in (14, 24)]
    cycle = [by_iter[i] for i in range(15, 25)]
    all_excluding_checkpoint = [by_iter[i] for i in range(1, 30)]
    assert len(normal) == 14 and len(soap) == 2 and len(cycle) == 10

    per_step = [
        {
            "iter": int(row["iter"]),
            "time_s": float(row["time"]),
            "loss": float(row["loss"]),
            "grad_norm": None if "grad_norm" not in row else float(row["grad_norm"]),
            "memory_mib": int(row["memory"]),
        }
        for row in records
    ]
    result = {
        "schema_version": 1,
        "label": args.label,
        "input_name": args.input_jsonl.name,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "global_batch": args.global_batch,
        "iter_contract": {
            "count": 30,
            "iters": list(range(1, 31)),
            "checkpoint_polluted_step": 30,
            "checkpoint_polluted_step_excluded_from_performance_windows": True,
            "normal_steps": [i for i in range(15, 30) if i != 24],
            "soap_steps": [14, 24],
            "cycle_steps": list(range(15, 25)),
        },
        "function_gate": {
            "loss_finite_steps": list(range(1, 31)),
            "grad_finite_steps": finite_grad_steps,
            "grad_missing_dynamic_loss_scale_steps": missing_grad_steps,
            "grad_nonfinite_dynamic_loss_scale_steps": nonfinite_grad_steps,
        },
        "windows": {
            "all_30_including_checkpoint_step": stats(records, args.global_batch),
            "all_1_29_excluding_checkpoint_step": stats(all_excluding_checkpoint, args.global_batch),
            "stable_normal_15_29_excluding_24": stats(normal, args.global_batch),
            "soap_14_24": stats(soap, args.global_batch),
            "cycle_15_24": stats(cycle, args.global_batch),
        },
        "per_step": per_step,
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
