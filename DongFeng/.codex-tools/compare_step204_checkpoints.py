#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import torch


def force_cpu(value):
    """Normalize old checkpoints whose map_location leaves nested NPU tensors."""
    if torch.is_tensor(value):
        return value.cpu()
    if isinstance(value, dict):
        return type(value)((key, force_cpu(item)) for key, item in value.items())
    if isinstance(value, list):
        return [force_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(force_cpu(item) for item in value)
    return value


def tensor_summary(a, b, chunk_size=1_000_000):
    assert a.shape == b.shape and a.dtype == b.dtype
    af = a.reshape(-1)
    bf = b.reshape(-1)
    exact = torch.equal(a, b)
    finite_a = True
    finite_b = True
    max_abs = 0.0
    diff_sq = 0.0
    base_sq = 0.0
    if a.is_floating_point() or a.is_complex():
        for start in range(0, af.numel(), chunk_size):
            aa = af[start : start + chunk_size].float()
            bb = bf[start : start + chunk_size].float()
            finite_a = finite_a and bool(torch.isfinite(aa).all())
            finite_b = finite_b and bool(torch.isfinite(bb).all())
            diff = aa - bb
            if diff.numel():
                max_abs = max(max_abs, float(diff.abs().max()))
                diff_sq += float((diff.double() * diff.double()).sum())
                base_sq += float((aa.double() * aa.double()).sum())
    elif not exact:
        max_abs = float((af.to(torch.int64) - bf.to(torch.int64)).abs().max())
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "numel": a.numel(),
        "exact": exact,
        "finite_baseline": finite_a,
        "finite_candidate": finite_b,
        "max_abs_diff": max_abs,
        "relative_l2": math.sqrt(diff_sq) / max(math.sqrt(base_sq), 1e-30),
        "diff_sq": diff_sq,
        "base_sq": base_sq,
    }


def compare_tree(a, b, path, tensors, scalar_differences):
    assert type(a) is type(b), (path, type(a), type(b))
    if torch.is_tensor(a):
        tensors[path] = tensor_summary(a, b)
    elif isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), path
        for key in a:
            compare_tree(a[key], b[key], f"{path}.{key}", tensors, scalar_differences)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for index, (left, right) in enumerate(zip(a, b)):
            compare_tree(left, right, f"{path}[{index}]", tensors, scalar_differences)
    elif a != b:
        scalar_differences.append({"path": path, "baseline": repr(a), "candidate": repr(b)})


def aggregate(tensors):
    values = list(tensors.values())
    total_diff_sq = sum(x["diff_sq"] for x in values)
    total_base_sq = sum(x["base_sq"] for x in values)
    max_item = max(tensors.items(), key=lambda item: item[1]["max_abs_diff"])
    rel_item = max(tensors.items(), key=lambda item: item[1]["relative_l2"])
    return {
        "tensor_count": len(values),
        "tensor_exact_count": sum(x["exact"] for x in values),
        "all_shapes_dtypes_equal": True,
        "all_finite_baseline": all(x["finite_baseline"] for x in values),
        "all_finite_candidate": all(x["finite_candidate"] for x in values),
        "global_relative_l2": math.sqrt(total_diff_sq) / max(math.sqrt(total_base_sq), 1e-30),
        "max_abs_diff": max_item[1]["max_abs_diff"],
        "max_abs_diff_path": max_item[0],
        "max_tensor_relative_l2": rel_item[1]["relative_l2"],
        "max_tensor_relative_l2_path": rel_item[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    baseline = force_cpu(
        torch.load(str(args.baseline), map_location="cpu", weights_only=False, mmap=True)
    )
    candidate = force_cpu(
        torch.load(str(args.candidate), map_location="cpu", weights_only=False, mmap=True)
    )
    assert baseline.keys() == candidate.keys() == {"meta", "state_dict", "optimizer"}
    meta_fields = ("epoch", "iter", "seed", "mmcv_version")
    meta = {key: {"baseline": baseline["meta"].get(key), "candidate": candidate["meta"].get(key)} for key in meta_fields}
    assert all(value["baseline"] == value["candidate"] for value in meta.values())

    state_tensors = {}
    state_scalars = []
    compare_tree(baseline["state_dict"], candidate["state_dict"], "state_dict", state_tensors, state_scalars)
    optimizer_tensors = {}
    optimizer_scalars = []
    compare_tree(baseline["optimizer"], candidate["optimizer"], "optimizer", optimizer_tensors, optimizer_scalars)
    result = {
        "schema_version": 1,
        "meta": meta,
        "state_dict": aggregate(state_tensors),
        "state_dict_scalar_differences": state_scalars,
        "optimizer": aggregate(optimizer_tensors),
        "optimizer_scalar_difference_count": len(optimizer_scalars),
        "optimizer_scalar_differences_first_100": optimizer_scalars[:100],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
