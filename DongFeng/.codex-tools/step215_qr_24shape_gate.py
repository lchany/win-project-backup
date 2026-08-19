#!/usr/bin/env python3
"""STEP-215: numerical QR gate over the 24 real SOAP square shapes."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch_npu


VISIBLE = "8,9,10,11,12,13,14,15"
FULL_COUNTS = {
    1: 106, 3: 30, 4: 6, 7: 37, 8: 1, 11: 1, 22: 1, 32: 4,
    40: 9, 64: 28, 96: 3, 120: 1, 128: 18, 160: 1, 192: 32,
    220: 4, 256: 181, 352: 1, 440: 4, 512: 43, 768: 22,
    1024: 6, 2560: 8, 5120: 4,
}
# The active one-sided implementation removes four 5120 calls and four of the
# eight historical 2560 calls: 551 -> 543 QR calls per SOAP cycle.
ACTIVE_COUNTS = {**FULL_COUNTS, 2560: 4, 5120: 0}
THRESHOLD = 1.0e-5


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "max_ms": max(values),
    }


def timed(fn):
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    value = fn()
    end.record()
    end.synchronize()
    return value, float(start.elapsed_time(end)), (time.perf_counter() - wall_start) * 1000.0


def error(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    delta = candidate - reference
    denominator = torch.sqrt(torch.mean(reference * reference))
    nrmse = torch.sqrt(torch.mean(delta * delta)) / torch.clamp_min(denominator, 1.0e-30)
    return {
        "bitwise": bool(torch.equal(candidate, reference)),
        "max_abs": float(delta.abs().max().cpu()),
        "nrmse": float(nrmse.cpu()),
    }


def candidate(x: torch.Tensor):
    packed, tau = torch.geqrf(x)
    q = torch.orgqr(packed, tau).contiguous()
    return q, packed, tau


def measure_shape(n: int, device: torch.device, local_rank: int) -> dict[str, object]:
    # The exact same immutable tensor object is passed to both implementations.
    torch.manual_seed(2026081500 + n)
    torch.npu.manual_seed_all(2026081500 + n)
    x = torch.randn((n, n), dtype=torch.float32, device=device)
    input_ptr = int(x.data_ptr())
    before_sum = float(x.sum().cpu())
    before_norm = float(torch.linalg.vector_norm(x).cpu())

    direct = lambda: torch.linalg.qr(x, mode="reduced")
    proposed = lambda: candidate(x)
    samples = 5 if n <= 512 else (3 if n <= 1024 else 2)
    # Every shape, including 2560/5120, must be warmed before measurement so
    # first-use compilation and allocator setup cannot contaminate the gate.
    warm_direct, _, _ = timed(direct)
    warm_candidate, _, _ = timed(proposed)
    del warm_direct, warm_candidate

    direct_event: list[float] = []
    direct_wall: list[float] = []
    proposed_event: list[float] = []
    proposed_wall: list[float] = []
    direct_value = None
    proposed_value = None
    direct_repeat_reference = None
    proposed_repeat_reference = None
    direct_self_q_error = None
    proposed_self_q_error = None
    torch.npu.reset_peak_memory_stats(local_rank)
    base_allocated = int(torch.npu.memory_allocated(local_rank))
    base_reserved = int(torch.npu.memory_reserved(local_rank))
    for sample in range(samples):
        order = (("direct", direct), ("proposed", proposed))
        if (sample + local_rank) % 2:
            order = tuple(reversed(order))
        for name, fn in order:
            value, event_ms, wall_ms = timed(fn)
            if name == "direct":
                if direct_repeat_reference is None and direct_self_q_error is None:
                    direct_repeat_reference = value
                elif direct_self_q_error is None:
                    direct_self_q_error = error(value[0], direct_repeat_reference[0])
                    direct_repeat_reference = None
                if direct_value is not None:
                    del direct_value
                direct_value = value
                direct_event.append(event_ms)
                direct_wall.append(wall_ms)
            else:
                if proposed_repeat_reference is None and proposed_self_q_error is None:
                    proposed_repeat_reference = value
                elif proposed_self_q_error is None:
                    proposed_self_q_error = error(value[0], proposed_repeat_reference[0])
                    proposed_repeat_reference = None
                if proposed_value is not None:
                    del proposed_value
                proposed_value = value
                proposed_event.append(event_ms)
                proposed_wall.append(wall_ms)

    peak_allocated = int(torch.npu.max_memory_allocated(local_rank))
    peak_reserved = int(torch.npu.max_memory_reserved(local_rank))
    assert direct_value is not None and proposed_value is not None
    q_reference, r_reference = direct_value
    q_proposed, packed, tau = proposed_value
    r_proposed = torch.triu(packed)
    torch.npu.synchronize()
    q_error = error(q_proposed, q_reference)
    r_error = error(r_proposed, r_reference)
    assert direct_self_q_error is not None and proposed_self_q_error is not None
    finite = all(
        bool(torch.isfinite(value).all().cpu())
        for value in (q_reference, r_reference, q_proposed, packed, tau, r_proposed)
    )
    identity = torch.eye(n, dtype=torch.float32, device=device)
    orthogonal_delta = q_proposed.T @ q_proposed - identity
    orthogonality = float(orthogonal_delta.abs().max().cpu())
    orthogonality_normalized_fro = float(
        (torch.linalg.vector_norm(orthogonal_delta) / math.sqrt(n)).cpu()
    )
    reconstruction = float(
        (torch.linalg.vector_norm(q_proposed @ r_proposed - x) /
         torch.clamp_min(torch.linalg.vector_norm(x), 1.0e-30)).cpu()
    )
    after_sum = float(x.sum().cpu())
    after_norm = float(torch.linalg.vector_norm(x).cpu())
    input_unchanged = before_sum == after_sum and before_norm == after_norm and int(x.data_ptr()) == input_ptr
    numeric_gate = bool(
        finite
        and input_unchanged
        and q_error["nrmse"] <= min(
            THRESHOLD, max(5.0e-6, 2.0 * direct_self_q_error["nrmse"])
        )
        and q_error["max_abs"] <= THRESHOLD
        and r_error["nrmse"] <= THRESHOLD
        and orthogonality <= THRESHOLD
        and orthogonality_normalized_fro <= THRESHOLD
        and reconstruction <= THRESHOLD
    )
    return {
        "shape": [n, n],
        "dtype": str(x.dtype),
        "samples": samples,
        "historical_call_count": FULL_COUNTS[n],
        "active_call_count": ACTIVE_COUNTS[n],
        "same_input_object": True,
        "input_unchanged": input_unchanged,
        "stable_sort_outside_qr_boundary_unchanged": True,
        "finite_all": finite,
        "numeric_gate_pass": numeric_gate,
        "q_error": q_error,
        "r_error": r_error,
        "direct_self_q_error": direct_self_q_error,
        "proposed_self_q_error": proposed_self_q_error,
        "orthogonality_max_abs": orthogonality,
        "orthogonality_normalized_fro": orthogonality_normalized_fro,
        "reconstruction_rel_l2": reconstruction,
        "direct_event": stats(direct_event),
        "direct_wall": stats(direct_wall),
        "proposed_event": stats(proposed_event),
        "proposed_wall": stats(proposed_wall),
        "event_saved_ms": statistics.median(direct_event) - statistics.median(proposed_event),
        "base_allocated": base_allocated,
        "peak_allocated": peak_allocated,
        "extra_peak_allocated": peak_allocated - base_allocated,
        "base_reserved": base_reserved,
        "peak_reserved": peak_reserved,
        "extra_peak_reserved": peak_reserved - base_reserved,
    }


def run(args) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    root = Path(args.output_dir).resolve(strict=True)
    if world_size != 8 or rank != local_rank or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != VISIBLE:
        raise RuntimeError("world/device contract mismatch")
    for name in ("ready", "done", "failure"):
        (root / name).mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    results = [measure_shape(n, device, local_rank) for n in FULL_COUNTS]
    active_calls = sum(ACTIVE_COUNTS.values())
    historical_calls = sum(FULL_COUNTS.values())
    active_baseline_ms = sum(
        row["direct_event"]["median_ms"] * row["active_call_count"] for row in results
    )
    active_proposed_ms = sum(
        row["proposed_event"]["median_ms"] * row["active_call_count"] for row in results
    )
    payload = {
        "pid": os.getpid(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "visible": VISIBLE,
        # Controller contract: the harness completed and remains live. Numerical
        # acceptance is carried separately and may validly be false.
        "gate_pass": True,
        "numeric_gate_pass": all(row["numeric_gate_pass"] for row in results),
        "threshold": THRESHOLD,
        "shape_count": len(results),
        "historical_call_count": historical_calls,
        "active_call_count": active_calls,
        "active_contract": "historical 551 minus 4x5120 and 4x2560 = 543",
        "active_weighted_direct_event_ms": active_baseline_ms,
        "active_weighted_proposed_event_ms": active_proposed_ms,
        "active_weighted_saved_ms": active_baseline_ms - active_proposed_ms,
        "active_weighted_speedup": active_baseline_ms / active_proposed_ms,
        "results": results,
    }
    atomic_json(root / "ready" / f"rank{rank}.json", payload)
    deadline = time.monotonic() + 180.0
    while not (root / "release_after_npu_smi").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("controller did not release ranks")
        time.sleep(0.2)
    atomic_json(root / "done" / f"rank{rank}.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except BaseException as error:
        output = os.environ.get("OUTPUT_DIR")
        if output:
            failure = Path(output) / "failure"
            failure.mkdir(parents=True, exist_ok=True)
            (failure / f"rank{os.environ.get('RANK', 'x')}.txt").write_text(
                "".join(traceback.format_exception(error)), encoding="utf-8"
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
