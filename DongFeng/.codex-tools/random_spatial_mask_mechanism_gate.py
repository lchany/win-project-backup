#!/usr/bin/env python3
"""Repository-external exactness/performance gate for random_spatial_mask.

This file deliberately copies the current f922c38 implementation as the oracle.
It never imports or edits the business module.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def original_mask(x, mask_ratio=0.2, block_size=8, enable_prob=False, return_indices=False):
    B, C, H, W = x.shape
    mask = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)
    enable = (torch.rand(B, 1, 1, 1, device=x.device) < enable_prob).to(x.dtype)
    num_blocks_h = H // block_size
    num_blocks_w = W // block_size
    num_mask = int(num_blocks_h * num_blocks_w * mask_ratio)
    all_indices = []
    for b in range(B):
        idx = torch.randperm(num_blocks_h * num_blocks_w)[:num_mask]
        all_indices.append(idx.clone())
        for i in idx:
            h = (i // num_blocks_w) * block_size
            w = (i % num_blocks_w) * block_size
            mask[b, :, h:h + block_size, w:w + block_size] = 0
    effective_mask = enable * mask + (1 - enable)
    if return_indices:
        return effective_mask, all_indices
    return effective_mask


def _candidate_stages(x, mask_ratio=0.2, block_size=8, enable_prob=False, return_indices=False):
    B, C, H, W = x.shape
    enable = (torch.rand(B, 1, 1, 1, device=x.device) < enable_prob).to(x.dtype)
    num_blocks_h = H // block_size
    num_blocks_w = W // block_size
    num_blocks = num_blocks_h * num_blocks_w
    num_mask = int(num_blocks * mask_ratio)

    all_indices = []
    offset_indices = []
    for b in range(B):
        # This call intentionally has no device/generator argument.  Its count,
        # order, N and slice semantics are identical to the oracle.
        idx = torch.randperm(num_blocks)[:num_mask]
        all_indices.append(idx.clone())
        if idx.numel():
            offset_indices.append(idx + b * num_blocks)

    if offset_indices:
        flat_indices_cpu = torch.cat(offset_indices, dim=0)
    else:
        flat_indices_cpu = torch.empty(0, dtype=torch.int64)
    flat_indices_device = flat_indices_cpu.to(device=x.device)

    block_mask = torch.ones(B * num_blocks, device=x.device, dtype=x.dtype)
    if flat_indices_device.numel():
        block_mask.index_fill_(0, flat_indices_device, 0)
    block_mask = block_mask.view(B, 1, num_blocks_h, num_blocks_w)

    expanded = block_mask.repeat_interleave(block_size, dim=2)
    expanded = expanded.repeat_interleave(block_size, dim=3)
    full_h = num_blocks_h * block_size
    full_w = num_blocks_w * block_size
    if full_w < W:
        right = torch.ones((B, 1, full_h, W - full_w), device=x.device, dtype=x.dtype)
        expanded = torch.cat((expanded, right), dim=3)
    if full_h < H:
        bottom = torch.ones((B, 1, H - full_h, W), device=x.device, dtype=x.dtype)
        expanded = torch.cat((expanded, bottom), dim=2)
    mask = expanded.contiguous()
    effective_mask = enable * mask + (1 - enable)
    return effective_mask, all_indices, flat_indices_cpu, flat_indices_device


def candidate_mask(x, mask_ratio=0.2, block_size=8, enable_prob=False, return_indices=False):
    output, indices, _, _ = _candidate_stages(
        x, mask_ratio=mask_ratio, block_size=block_size,
        enable_prob=enable_prob, return_indices=return_indices)
    if return_indices:
        return output, indices
    return output


def cpu_rng_state():
    return torch.get_rng_state().clone()


def device_rng_state(device):
    if device.type == "npu":
        return torch.npu.get_rng_state(device).clone().cpu()
    return None


def restore_states(cpu_state, dev_state, device):
    torch.set_rng_state(cpu_state)
    if device.type == "npu":
        torch.npu.set_rng_state(dev_state, device)


def sync(device):
    if device.type == "npu":
        torch.npu.synchronize(device)


def state_equal(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return bool(torch.equal(a.cpu(), b.cpu()))


def tensor_contract(t, x):
    return {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype),
        "device_type": t.device.type,
        "contiguous": bool(t.is_contiguous()),
        "requires_grad": bool(t.requires_grad),
        "grad_fn_none": t.grad_fn is None,
        "base_none": t._base is None,
        "aliases_input": t.untyped_storage().data_ptr() == x.untyped_storage().data_ptr(),
    }


def make_input(case, device, dtype):
    B, H, W = case["B"], case["H"], case["W"]
    if case.get("noncontiguous", False):
        base = (torch.arange(B * 2 * W * H, dtype=torch.float32) % 997).reshape(B, 2, W, H) / 997
        return base.transpose(2, 3).to(device=device, dtype=dtype)
    base = (torch.arange(B * 2 * H * W, dtype=torch.float32) % 997).reshape(B, 2, H, W) / 997
    return base.to(device=device, dtype=dtype)


def run_exact_case(case, seed, device, dtype):
    x = make_input(case, device, dtype)
    sync(device)
    x_before = x.clone()
    torch.manual_seed(seed)
    if device.type == "npu":
        torch.npu.manual_seed(seed + 100003)
    pre_cpu = cpu_rng_state()
    pre_dev = device_rng_state(device)

    out_old, idx_old = original_mask(
        x, case["ratio"], case["block"], case["enable"], True)
    sync(device)
    post_old_cpu = cpu_rng_state()
    post_old_dev = device_rng_state(device)
    follow_old_cpu = torch.rand(17)
    follow_old_dev = torch.rand(17, device=device) if device.type == "npu" else None
    sync(device)

    restore_states(pre_cpu, pre_dev, device)
    out_new, idx_new = candidate_mask(
        x, case["ratio"], case["block"], case["enable"], True)
    sync(device)
    post_new_cpu = cpu_rng_state()
    post_new_dev = device_rng_state(device)
    follow_new_cpu = torch.rand(17)
    follow_new_dev = torch.rand(17, device=device) if device.type == "npu" else None
    sync(device)

    indices_equal = len(idx_old) == len(idx_new) and all(
        torch.equal(a, b) for a, b in zip(idx_old, idx_new))
    product_old = x * out_old
    product_new = x * out_new
    sync(device)
    checks = {
        "indices_equal": indices_equal,
        "mask_bitwise_equal": bool(torch.equal(out_old, out_new)),
        "product_bitwise_equal": bool(torch.equal(product_old, product_new)),
        "cpu_rng_state_equal": state_equal(post_old_cpu, post_new_cpu),
        "device_rng_state_equal": state_equal(post_old_dev, post_new_dev),
        "follow_cpu_equal": bool(torch.equal(follow_old_cpu, follow_new_cpu)),
        "follow_device_equal": True if follow_old_dev is None else bool(torch.equal(follow_old_dev, follow_new_dev)),
        "input_unchanged": bool(torch.equal(x, x_before)),
        "contract_equal": tensor_contract(out_old, x) == tensor_contract(out_new, x),
    }
    checks["all"] = all(checks.values())
    return {
        "case": case,
        "seed": seed,
        "dtype": str(dtype),
        "checks": checks,
        "old_contract": tensor_contract(out_old, x),
        "new_contract": tensor_contract(out_new, x),
        "randperm_calls": len(idx_old),
        "randperm_lengths": [int(v.numel()) for v in idx_old],
    }


def duplicate_index_gate(device, dtype):
    B, nh, nw, block = 2, 3, 4, 2
    supplied = [torch.tensor([0, 0, 5, 11, 11]), torch.tensor([2, 2, 3, 9])]
    reference = torch.ones((B, 1, nh, nw), device=device, dtype=dtype)
    for b, idxs in enumerate(supplied):
        for idx in idxs:
            reference[b, 0, idx // nw, idx % nw] = 0
    flat = torch.ones(B * nh * nw, device=device, dtype=dtype)
    selected = torch.cat([idx + b * nh * nw for b, idx in enumerate(supplied)]).to(device)
    flat.index_fill_(0, selected, 0)
    candidate = flat.view(B, 1, nh, nw)
    sync(device)
    return bool(torch.equal(reference, candidate))


def elapsed_ms(device, fn):
    sync(device)
    begin = time.perf_counter_ns()
    result = fn()
    sync(device)
    return (time.perf_counter_ns() - begin) / 1e6, result


def time_candidate_components(x, ratio, block, enable):
    B, _, H, W = x.shape
    nh, nw = H // block, W // block
    N = nh * nw
    num_mask = int(N * ratio)
    parts = {}
    parts["enable_ms"], enable_tensor = elapsed_ms(
        x.device, lambda: (torch.rand(B, 1, 1, 1, device=x.device) < enable).to(x.dtype))

    begin = time.perf_counter_ns()
    idxs = [torch.randperm(N)[:num_mask] for _ in range(B)]
    offset = [idx + b * N for b, idx in enumerate(idxs) if idx.numel()]
    flat_cpu = torch.cat(offset) if offset else torch.empty(0, dtype=torch.int64)
    parts["idx_prepare_cpu_ms"] = (time.perf_counter_ns() - begin) / 1e6
    parts["idx_h2d_ms"], flat_device = elapsed_ms(x.device, lambda: flat_cpu.to(x.device))

    def index_fill_stage():
        flat = torch.ones(B * N, device=x.device, dtype=x.dtype)
        if flat_device.numel():
            flat.index_fill_(0, flat_device, 0)
        return flat.view(B, 1, nh, nw)
    parts["index_fill_ms"], grid = elapsed_ms(x.device, index_fill_stage)

    def repeat_stage():
        return grid.repeat_interleave(block, 2).repeat_interleave(block, 3)
    parts["repeat_ms"], expanded = elapsed_ms(x.device, repeat_stage)

    def tail_stage():
        y = expanded
        full_h, full_w = nh * block, nw * block
        if full_w < W:
            y = torch.cat((y, torch.ones((B, 1, full_h, W - full_w), device=x.device, dtype=x.dtype)), 3)
        if full_h < H:
            y = torch.cat((y, torch.ones((B, 1, H - full_h, W), device=x.device, dtype=x.dtype)), 2)
        return y.contiguous()
    parts["tail_pad_contiguous_ms"], mask = elapsed_ms(x.device, tail_stage)
    parts["final_formula_ms"], _ = elapsed_ms(
        x.device, lambda: enable_tensor * mask + (1 - enable_tensor))
    parts["component_sum_ms"] = sum(v for k, v in parts.items() if k != "component_sum_ms")
    return parts


def performance_gate(device, dtype, warmup=5, repeats=15):
    case = {"B": 16, "H": 128, "W": 320, "block": 8, "ratio": 0.2, "enable": 0.2}
    x = torch.empty((16, 1, 128, 320), device=device, dtype=dtype)
    for _ in range(warmup):
        original_mask(x, 0.2, 8, 0.2)
        candidate_mask(x, 0.2, 8, 0.2)
    sync(device)
    old_ms, new_ms = [], []
    for i in range(repeats):
        # Alternate order so thermal/load drift does not systematically favor one side.
        funcs = ((original_mask, old_ms), (candidate_mask, new_ms))
        if i % 2:
            funcs = tuple(reversed(funcs))
        for fn, bucket in funcs:
            ms, _ = elapsed_ms(device, lambda fn=fn: fn(x, 0.2, 8, 0.2))
            bucket.append(ms)
    component_samples = []
    for _ in range(5):
        component_samples.append(time_candidate_components(x, 0.2, 8, 0.2))
    component_medians = {
        key: statistics.median([sample[key] for sample in component_samples])
        for key in component_samples[0]
    }
    old_med = statistics.median(old_ms)
    new_med = statistics.median(new_ms)
    return {
        "case": case,
        "dtype": str(dtype),
        "warmup": warmup,
        "repeats": repeats,
        "old_ms": old_ms,
        "candidate_ms": new_ms,
        "old_median_ms": old_med,
        "candidate_median_ms": new_med,
        "net_saving_ms": old_med - new_med,
        "speedup": old_med / new_med if new_med else None,
        "passes_22p7ms": (old_med - new_med) > 22.7,
        "candidate_component_medians_ms": component_medians,
    }


def build_cases():
    return [
        {"name": "real", "B": 16, "H": 128, "W": 320, "block": 8, "ratio": 0.2, "enable": 0.2},
        {"name": "b1_div_ratio0_enable0", "B": 1, "H": 16, "W": 24, "block": 8, "ratio": 0.0, "enable": 0.0},
        {"name": "b4_div_ratio1_enable1", "B": 4, "H": 16, "W": 24, "block": 8, "ratio": 1.0, "enable": 1.0},
        {"name": "b16_nondiv_trunc", "B": 16, "H": 17, "W": 25, "block": 8, "ratio": 0.237, "enable": 0.2},
        {"name": "h_lt_block", "B": 4, "H": 7, "W": 25, "block": 8, "ratio": 1.0, "enable": 1.0},
        {"name": "w_lt_block", "B": 4, "H": 17, "W": 7, "block": 8, "ratio": 1.0, "enable": 0.2},
        {"name": "both_lt_block", "B": 1, "H": 7, "W": 7, "block": 8, "ratio": 0.237, "enable": 1.0},
        {"name": "noncontiguous_input", "B": 4, "H": 17, "W": 25, "block": 8, "ratio": 0.237, "enable": 0.2, "noncontiguous": True},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.cpu_only:
        device = torch.device("cpu")
    else:
        import torch_npu  # noqa: F401
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")

    seeds = [0, 1, 17, 2147483647]
    dtypes = [torch.float32, torch.float16]
    exact = []
    for dtype in dtypes:
        for case in build_cases():
            for seed in seeds:
                exact.append(run_exact_case(case, seed + rank * 1000003, device, dtype))
    duplicate = {str(dtype): duplicate_index_gate(device, dtype) for dtype in dtypes}
    perf = None if args.cpu_only else performance_gate(device, torch.float32, args.warmup, args.repeats)
    all_exact = all(item["checks"]["all"] for item in exact)
    result = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_npu_imported": (not args.cpu_only),
        "case_count": len(exact),
        "all_exact": all_exact,
        "duplicate_index_all_exact": all(duplicate.values()),
        "duplicate_index": duplicate,
        "exact_cases": exact,
        "performance": perf,
    }
    path = out_dir / f"rank{rank}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "rank": rank, "world_size": world_size, "device": str(device),
        "all_exact": all_exact, "duplicate": all(duplicate.values()),
        "net_saving_ms": None if perf is None else perf["net_saving_ms"],
        "passes_22p7ms": None if perf is None else perf["passes_22p7ms"],
    }, sort_keys=True), flush=True)
    if not all_exact or not all(duplicate.values()):
        raise SystemExit(3)
    if perf is not None and not perf["passes_22p7ms"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
