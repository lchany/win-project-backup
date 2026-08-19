#!/usr/bin/env python3
"""Eight-rank Triton-Ascend vector-add mechanism gate.

This file is prepared for a later, explicitly authorized NPU run.  Importing or
py-compiling it does not initialize an NPU.  Each torchrun rank uses one logical
device from ASCEND_RT_VISIBLE_DEVICES=8,...,15 and an isolated Triton cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


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
    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"WORLD_SIZE={world_size}, expected {EXPECTED_WORLD_SIZE}")
    if rank != local_rank or rank not in range(EXPECTED_WORLD_SIZE):
        raise RuntimeError(f"invalid one-node rank mapping rank={rank}, local_rank={local_rank}")
    if visible != EXPECTED_VISIBLE:
        raise RuntimeError(f"visible devices mismatch: {visible!r}")
    expected_cache = output_dir / "triton_cache" / f"rank{local_rank}"
    if Path(os.environ["TRITON_CACHE_DIR"]).resolve() != expected_cache:
        raise RuntimeError("rank-local Triton cache contract mismatch")
    expected_cache.mkdir(parents=True, exist_ok=False)
    ready_dir = output_dir / "ready"
    done_dir = output_dir / "done"
    failure_dir = output_dir / "failure"
    for directory in (ready_dir, done_dir, failure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    # Official tutorial size deliberately exercises a masked tail (98432 % 1024).
    n_elements = args.n_elements
    base = torch.arange(n_elements, dtype=torch.float32, device=device)
    x = base * 0.125 + float(rank)
    y = (base.remainder(257.0) - 128.0) * 0.25
    expected = x + y
    actual = triton_add(x, y)
    torch.npu.synchronize(local_rank)
    exact = bool(torch.equal(actual, expected))
    max_abs_diff = float((actual - expected).abs().max().cpu().item())
    if not exact or max_abs_diff != 0.0:
        raise AssertionError(f"vector add mismatch exact={exact}, max_abs_diff={max_abs_diff}")

    payload = {
        "pid": os.getpid(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "visible": visible,
        "logical_device": str(device),
        "device_count": int(torch.npu.device_count()),
        "current_device": int(torch.npu.current_device()),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "triton_module": str(Path(triton.__file__).resolve()),
        "triton_cache": str(expected_cache),
        "n_elements": n_elements,
        "block_size": 1024,
        "exact": exact,
        "max_abs_diff": max_abs_diff,
    }
    atomic_json(ready_dir / f"rank{rank}.json", payload)
    wait_for_release(output_dir / "release_after_npu_smi", args.hold_timeout_seconds)
    atomic_json(done_dir / f"rank{rank}.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-elements", type=int, default=98432)
    parser.add_argument("--hold-timeout-seconds", type=int, default=300)
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
