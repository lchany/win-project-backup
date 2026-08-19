#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_metrics", type=Path)
    parser.add_argument("resume_jsonl", type=Path)
    parser.add_argument("input_checkpoint", type=Path)
    parser.add_argument("output_checkpoint", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    source = json.loads(args.source_metrics.read_text(encoding="utf-8"))
    raw = args.resume_jsonl.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    rows = [row for row in rows if row.get("mode") == "train" and "iter" in row]
    assert [row["iter"] for row in rows] == list(range(30, 37))
    assert all(math.isfinite(float(row["loss"])) for row in rows)
    nonfinite_grad = [int(row["iter"]) for row in rows if not math.isfinite(float(row["grad_norm"]))]
    assert nonfinite_grad == [31]

    before = torch.load(str(args.input_checkpoint), map_location="cpu", weights_only=False, mmap=True)
    after = torch.load(str(args.output_checkpoint), map_location="cpu", weights_only=False, mmap=True)
    assert before.keys() == after.keys() == {"meta", "state_dict", "optimizer"}
    assert before["state_dict"].keys() == after["state_dict"].keys()
    assert before["optimizer"].keys() == after["optimizer"].keys()
    assert before["optimizer"]["state"].keys() == after["optimizer"]["state"].keys()

    def optimizer_steps(checkpoint):
        values = []
        for state in checkpoint["optimizer"]["state"].values():
            if "step" in state:
                value = state["step"]
                values.append(float(value.item()) if torch.is_tensor(value) else float(value))
        return values

    before_steps = optimizer_steps(before)
    after_steps = optimizer_steps(after)
    assert len(before_steps) == len(after_steps) == 559
    assert len(set(before_steps)) == len(set(after_steps)) == 1
    assert after_steps[0] - before_steps[0] == 6
    assert after["meta"]["iter"] - before["meta"]["iter"] == 6

    source_rows = {row["iter"]: row for row in source["per_step"]}
    recent_losses = [source_rows[i]["loss"] for i in range(25, 31)]
    result = {
        "schema_version": 1,
        "label": args.label,
        "resume_log_sha256": hashlib.sha256(raw).hexdigest(),
        "logged_iters": [row["iter"] for row in rows],
        "logged_iter_count": len(rows),
        "loss_all_finite": True,
        "grad_nonfinite_iters": nonfinite_grad,
        "grad_finite_iters": [row["iter"] for row in rows if math.isfinite(float(row["grad_norm"]))],
        "source_last_loss": source_rows[30]["loss"],
        "resume_first_loss": float(rows[0]["loss"]),
        "resume_first_loss_relative_pct": (float(rows[0]["loss"]) / source_rows[30]["loss"] - 1) * 100,
        "source_recent_loss_min": min(recent_losses),
        "source_recent_loss_max": max(recent_losses),
        "resume_loss_min": min(float(row["loss"]) for row in rows),
        "resume_loss_max": max(float(row["loss"]) for row in rows),
        "resume_loss_mean": statistics.mean(float(row["loss"]) for row in rows),
        "checkpoint": {
            "input_meta_iter": before["meta"]["iter"],
            "output_meta_iter": after["meta"]["iter"],
            "meta_iter_delta": after["meta"]["iter"] - before["meta"]["iter"],
            "state_dict_count": len(before["state_dict"]),
            "optimizer_state_count": len(before["optimizer"]["state"]),
            "optimizer_step_tensor_count": len(before_steps),
            "input_optimizer_step": before_steps[0],
            "output_optimizer_step": after_steps[0],
            "optimizer_step_delta": after_steps[0] - before_steps[0],
            "schema_keys_equal": True,
        },
        "per_step": [
            {
                "iter": int(row["iter"]),
                "time_s": float(row["time"]),
                "loss": float(row["loss"]),
                "grad_norm": float(row["grad_norm"]),
                "memory_mib": int(row["memory"]),
            }
            for row in rows
        ],
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
