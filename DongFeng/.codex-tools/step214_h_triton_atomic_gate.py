#!/usr/bin/env python3
"""World8 structured FP32 atomic-add gate for Triton-Ascend v3.2.0rc4.

This deliberately avoids data-dependent/fully-indirect addresses.  It scales
the matching rc4 official repeated-address test to channel=32 and an MSDA-like
average collision count (32 programs x 27 adds = 864 contributions/output).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path


_LOCAL_RANK_TEXT = os.environ.get("LOCAL_RANK", "unlaunched")
if _LOCAL_RANK_TEXT.isdigit():
    _cache_root = Path(os.environ["OUTPUT_DIR"]) / "triton_cache" / f"rank{_LOCAL_RANK_TEXT}"
    os.environ["TRITON_CACHE_DIR"] = str(_cache_root)

import torch
import torch_npu
import triton
import triton.language as tl


EXPECTED_VISIBLE = "8,9,10,11,12,13,14,15"
EXPECTED_WORLD_SIZE = 8


@triton.jit
def structured_atomic_add_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    COLLISION_PROGRAMS: tl.constexpr,
    REPEAT_ADDS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    group = pid // COLLISION_PROGRAMS
    offsets = group * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    for _ in tl.static_range(REPEAT_ADDS):
        tl.atomic_add(out_ptr + offsets, 1.0, mask=mask)


def launch(out: torch.Tensor, collision_programs: int, repeat_adds: int, block_size: int) -> None:
    groups = triton.cdiv(out.numel(), block_size)
    grid = (groups * collision_programs,)
    structured_atomic_add_kernel[grid](
        out,
        out.numel(),
        BLOCK_SIZE=block_size,
        COLLISION_PROGRAMS=collision_programs,
        REPEAT_ADDS=repeat_adds,
    )


def event_measure(fn) -> tuple[float, float]:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    wall_start = time.perf_counter()
    start.record()
    fn()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return float(start.elapsed_time(end)), wall_ms


def stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
        "max_ms": max(values),
    }


def run_case(
    device: torch.device,
    name: str,
    n_elements: int,
    collision_programs: int,
    repeat_adds: int,
    block_size: int,
    warmup: int,
    samples: int,
) -> dict:
    if n_elements % 32 != 0:
        raise ValueError("channel=32 contract requires n_elements divisible by 32")
    out = torch.empty(n_elements, dtype=torch.float32, device=device)
    for _ in range(warmup):
        out.zero_()
        launch(out, collision_programs, repeat_adds, block_size)
    torch.npu.synchronize()

    kernel_events: list[float] = []
    kernel_walls: list[float] = []
    boundary_events: list[float] = []
    boundary_walls: list[float] = []
    exact_runs: list[bool] = []
    max_abs_diffs: list[float] = []
    expected_value = float(collision_programs * repeat_adds)

    for _ in range(samples):
        out.zero_()
        torch.npu.synchronize()
        event_ms, wall_ms = event_measure(
            lambda: launch(out, collision_programs, repeat_adds, block_size)
        )
        kernel_events.append(event_ms)
        kernel_walls.append(wall_ms)
        exact_runs.append(bool(torch.all(out == expected_value).cpu().item()))
        max_abs_diffs.append(float((out - expected_value).abs().max().cpu().item()))

    for _ in range(samples):
        event_ms, wall_ms = event_measure(
            lambda: (out.zero_(), launch(out, collision_programs, repeat_adds, block_size))
        )
        boundary_events.append(event_ms)
        boundary_walls.append(wall_ms)
        exact_runs.append(bool(torch.all(out == expected_value).cpu().item()))
        max_abs_diffs.append(float((out - expected_value).abs().max().cpu().item()))

    finite = bool(torch.isfinite(out).all().cpu().item())
    repeat_exact = all(exact_runs) and len(set(max_abs_diffs)) == 1 and max(max_abs_diffs) == 0.0
    return {
        "name": name,
        "n_elements": n_elements,
        "logical_shape": [n_elements // 32, 32],
        "channel": 32,
        "block_size": block_size,
        "collision_programs": collision_programs,
        "repeat_adds": repeat_adds,
        "contributions_per_output": collision_programs * repeat_adds,
        "grid_programs": triton.cdiv(n_elements, block_size) * collision_programs,
        "expected_value": expected_value,
        "finite": finite,
        "oracle_exact_all": all(exact_runs),
        "repeat_exact": repeat_exact,
        "max_abs_diff": max(max_abs_diffs),
        "kernel_event": stats(kernel_events),
        "kernel_wall": stats(kernel_walls),
        "zero_plus_kernel_event": stats(boundary_events),
        "zero_plus_kernel_wall": stats(boundary_walls),
    }


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def wait_for_release(path: Path, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"release file was not created: {path}")
        time.sleep(0.2)


def run(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    output_dir = Path(args.output_dir).resolve(strict=True)
    if world_size != EXPECTED_WORLD_SIZE or rank != local_rank or rank not in range(8):
        raise RuntimeError(f"invalid world/rank mapping: world={world_size}, rank={rank}, local={local_rank}")
    if visible != EXPECTED_VISIBLE:
        raise RuntimeError(f"visible devices mismatch: {visible!r}")
    expected_cache = output_dir / "triton_cache" / f"rank{local_rank}"
    if Path(os.environ["TRITON_CACHE_DIR"]).resolve() != expected_cache:
        raise RuntimeError("rank-local Triton cache contract mismatch")
    expected_cache.mkdir(parents=True, exist_ok=False)
    for directory in (output_dir / "ready", output_dir / "done", output_dir / "failure"):
        directory.mkdir(parents=True, exist_ok=True)

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    cases = [
        run_case(device, "rc4_official_32core_c32", 16384, 32, 1, 1024, args.warmup, args.samples),
        run_case(device, "msda_c32_collision864", 576 * 8 * 32, 32, 27, 1024, args.warmup, args.samples),
    ]
    exact = all(case["oracle_exact_all"] and case["repeat_exact"] and case["finite"] for case in cases)
    max_abs_diff = max(case["max_abs_diff"] for case in cases)
    if not exact or max_abs_diff != 0.0:
        raise AssertionError(f"atomic gate mismatch exact={exact}, max_abs_diff={max_abs_diff}")
    payload = {
        "pid": os.getpid(), "rank": rank, "local_rank": local_rank, "world_size": world_size,
        "visible": visible, "logical_device": str(device), "device_count": int(torch.npu.device_count()),
        "current_device": int(torch.npu.current_device()), "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "triton_module": str(Path(triton.__file__).resolve()),
        "triton_cache": str(expected_cache), "exact": exact, "max_abs_diff": max_abs_diff,
        "warmup": args.warmup, "samples": args.samples, "cases": cases,
    }
    atomic_json(output_dir / "ready" / f"rank{rank}.json", payload)
    wait_for_release(output_dir / "release_after_npu_smi", args.hold_timeout_seconds)
    atomic_json(output_dir / "done" / f"rank{rank}.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--hold-timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    rank_text = os.environ.get("RANK", "unknown")
    try:
        run(args)
        return 0
    except BaseException as exc:
        output_text = os.environ.get("OUTPUT_DIR")
        if output_text:
            failure_dir = Path(output_text) / "failure"
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / f"rank{rank_text}.txt").write_text(
                "".join(traceback.format_exception(exc)), encoding="utf-8"
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
