#!/usr/bin/env python3
"""Write a compact, read-only schema summary for a STEP-204 checkpoint."""

import argparse
import collections
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(
        str(args.checkpoint), map_location="cpu", mmap=True, weights_only=False
    )
    assert set(checkpoint) == {"meta", "optimizer", "state_dict"}
    meta = checkpoint["meta"]
    state_dict = checkpoint["state_dict"]
    optimizer = checkpoint["optimizer"]
    steps = [
        value
        for state in optimizer["state"].values()
        for key, value in state.items()
        if key == "step"
    ]
    assert all(isinstance(value, int) for value in steps)
    tensor_dtypes = collections.Counter(
        str(value.dtype) for value in state_dict.values() if torch.is_tensor(value)
    )
    optimizer_value_types = collections.Counter(
        type(value).__name__
        for state in optimizer["state"].values()
        for value in state.values()
    )
    loss_scaler = meta.get("fp16", {}).get("loss_scaler", {})
    result = {
        "schema_version": 1,
        "label": args.label,
        "checkpoint_name": args.checkpoint.name,
        "top_level_keys": sorted(checkpoint),
        "meta": {
            "epoch": meta.get("epoch"),
            "iter": meta.get("iter"),
            "seed": meta.get("seed"),
            "mmcv_version": meta.get("mmcv_version"),
            "loss_scaler": {
                "scale": loss_scaler.get("scale"),
                "growth_factor": loss_scaler.get("growth_factor"),
                "backoff_factor": loss_scaler.get("backoff_factor"),
                "growth_interval": loss_scaler.get("growth_interval"),
                "growth_tracker": loss_scaler.get("_growth_tracker"),
                "dynamic": loss_scaler.get("dynamic"),
            },
        },
        "state_dict_count": len(state_dict),
        "state_dict_dtype_counts": dict(sorted(tensor_dtypes.items())),
        "optimizer_param_group_count": len(optimizer["param_groups"]),
        "optimizer_state_count": len(optimizer["state"]),
        "optimizer_step_count": len(steps),
        "optimizer_step_min": min(steps),
        "optimizer_step_max": max(steps),
        "optimizer_step_unique": sorted(set(steps)),
        "optimizer_value_type_counts": dict(sorted(optimizer_value_types.items())),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
