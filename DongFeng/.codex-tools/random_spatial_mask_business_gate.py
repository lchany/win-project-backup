#!/usr/bin/env python3
"""Validate the patched business method against the frozen original oracle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

import random_spatial_mask_mechanism_gate as gate
from projects.mmdet3d_plugin.models.utils.bev_encoder import BevEncoder


def call_and_trace(fn, x, case):
    calls = []
    real_randperm = torch.randperm

    def traced_randperm(*args, **kwargs):
        value = real_randperm(*args, **kwargs)
        calls.append({
            "args": [str(v) for v in args],
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
            "value": value.clone(),
        })
        return value

    torch.randperm = traced_randperm
    try:
        output = fn(x, case["ratio"], case["block"], case["enable"])
    finally:
        torch.randperm = real_randperm
    return output, calls


def call_oracle(x, ratio, block, enable):
    return gate.original_mask(x, ratio, block, enable)


def call_business(x, ratio, block, enable):
    return BevEncoder.random_spatial_mask(None, x, ratio, block, enable)


def exact_case(case, seed, device, dtype):
    x = gate.make_input(case, device, dtype)
    gate.sync(device)
    x_before = x.clone()
    torch.manual_seed(seed)
    if device.type == "npu":
        torch.npu.manual_seed(seed + 100003)
    cpu_pre = gate.cpu_rng_state()
    dev_pre = gate.device_rng_state(device)

    old, old_calls = call_and_trace(call_oracle, x, case)
    gate.sync(device)
    old_cpu_post = gate.cpu_rng_state()
    old_dev_post = gate.device_rng_state(device)
    old_follow_cpu = torch.rand(17)
    old_follow_dev = torch.rand(17, device=device) if device.type == "npu" else None
    gate.sync(device)

    gate.restore_states(cpu_pre, dev_pre, device)
    new, new_calls = call_and_trace(call_business, x, case)
    gate.sync(device)
    new_cpu_post = gate.cpu_rng_state()
    new_dev_post = gate.device_rng_state(device)
    new_follow_cpu = torch.rand(17)
    new_follow_dev = torch.rand(17, device=device) if device.type == "npu" else None
    gate.sync(device)

    calls_equal = len(old_calls) == len(new_calls)
    if calls_equal:
        for a, b in zip(old_calls, new_calls):
            calls_equal = calls_equal and a["args"] == b["args"]
            calls_equal = calls_equal and a["kwargs"] == b["kwargs"]
            calls_equal = calls_equal and bool(torch.equal(a["value"], b["value"]))
    checks = {
        "randperm_calls_and_values_equal": calls_equal,
        "mask_bitwise_equal": bool(torch.equal(old, new)),
        "product_bitwise_equal": bool(torch.equal(x * old, x * new)),
        "cpu_rng_state_equal": gate.state_equal(old_cpu_post, new_cpu_post),
        "device_rng_state_equal": gate.state_equal(old_dev_post, new_dev_post),
        "follow_cpu_equal": bool(torch.equal(old_follow_cpu, new_follow_cpu)),
        "follow_device_equal": True if old_follow_dev is None else bool(torch.equal(old_follow_dev, new_follow_dev)),
        "contract_equal": gate.tensor_contract(old, x) == gate.tensor_contract(new, x),
        "input_unchanged": bool(torch.equal(x, x_before)),
    }
    checks["all"] = all(checks.values())
    return {
        "case": case["name"], "seed": seed, "dtype": str(dtype),
        "randperm_calls": len(old_calls), "checks": checks,
        "old_contract": gate.tensor_contract(old, x),
        "new_contract": gate.tensor_contract(new, x),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.cpu_only:
        device = torch.device("cpu")
    else:
        import torch_npu  # noqa: F401
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    rows = []
    for dtype in (torch.float32, torch.float16):
        for case in gate.build_cases():
            for seed in (0, 1, 17, 2147483647):
                rows.append(exact_case(case, seed + rank * 1000003, device, dtype))
    all_exact = all(row["checks"]["all"] for row in rows)
    result = {
        "rank": rank, "local_rank": local_rank, "world_size": world_size,
        "device": str(device), "case_count": len(rows), "all_exact": all_exact,
        "rows": rows,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank{rank}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "rank", "local_rank", "world_size", "device", "case_count", "all_exact")},
        sort_keys=True), flush=True)
    if not all_exact:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
