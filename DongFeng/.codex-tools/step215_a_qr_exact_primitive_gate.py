#!/usr/bin/env python3
"""STEP-215-A strict raw-Q gate for fixed-stack QR primitives."""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

_local_rank = os.environ.get("LOCAL_RANK", "x")
if _local_rank.isdigit():
    os.environ["TRITON_CACHE_DIR"] = str(
        Path(os.environ["OUTPUT_DIR"]) / "triton_cache" / f"rank{_local_rank}"
    )

import torch
import torch_npu


VISIBLE = "8,9,10,11,12,13,14,15"


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def event_once(fn):
    torch.npu.synchronize()
    begin = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    wall_start = time.perf_counter()
    begin.record()
    result = fn()
    end.record()
    end.synchronize()
    return result, float(begin.elapsed_time(end)), (time.perf_counter() - wall_start) * 1000.0


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    delta = actual - expected
    denominator = torch.sqrt(torch.mean(expected * expected))
    return {
        "bitwise": bool(torch.equal(actual, expected)),
        "max_abs": float(delta.abs().max().cpu()),
        "nrmse": float((torch.sqrt(torch.mean(delta * delta)) / denominator).cpu()),
    }


def measured_call(local_rank: int, fn):
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats(local_rank)
    base_allocated = int(torch.npu.memory_allocated(local_rank))
    base_reserved = int(torch.npu.memory_reserved(local_rank))
    result, event_ms, wall_ms = event_once(fn)
    torch.npu.synchronize()
    peak_allocated = int(torch.npu.max_memory_allocated(local_rank))
    peak_reserved = int(torch.npu.max_memory_reserved(local_rank))
    return result, {
        "event_ms": event_ms,
        "wall_ms": wall_ms,
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
    if world_size != 8 or rank != local_rank:
        raise RuntimeError("rank contract failed")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != VISIBLE:
        raise RuntimeError("device visibility contract failed")
    for directory in (root / "ready", root / "done", root / "failure"):
        directory.mkdir(parents=True, exist_ok=True)

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    torch.manual_seed(20260815)
    torch.npu.manual_seed_all(20260815)
    x = torch.randn((2560, 2560), dtype=torch.float32, device=device)
    identity = torch.eye(2560, dtype=torch.float32, device=device)

    def direct():
        return torch.linalg.qr(x, mode="reduced")

    def ormqr_candidate():
        packed, tau = torch.geqrf(x)
        q = torch.ormqr(packed, tau, identity, left=True, transpose=False)
        return q, packed, tau

    def householder_candidate():
        packed, tau = torch.geqrf(x)
        q = torch.linalg.householder_product(packed, tau)
        return q, packed, tau

    def orgqr_candidate():
        packed, tau = torch.geqrf(x)
        q = torch.orgqr(packed, tau)
        return q, packed, tau

    # One warm-up per path is permitted. Correctness is evaluated on the first
    # measured output, and the candidate list stops at the first unsupported API
    # or raw-Q mismatch.
    warm_direct = direct()
    del warm_direct
    (q_direct, r_direct), direct_measurement = measured_call(local_rank, direct)
    direct_finite = bool(torch.isfinite(q_direct).all().cpu()) and bool(
        torch.isfinite(r_direct).all().cpu()
    )

    candidates = [
        ("geqrf_ormqr_identity", hasattr(torch, "ormqr"), ormqr_candidate),
        (
            "geqrf_householder_product",
            hasattr(torch.linalg, "householder_product"),
            householder_candidate,
        ),
        ("geqrf_orgqr", hasattr(torch, "orgqr"), orgqr_candidate),
    ]
    results = []
    stop_reason = "all_candidates_raw_q_exact"
    stopped_after = None
    for name, supported, candidate in candidates:
        if not supported:
            results.append({"name": name, "supported": False})
            stop_reason = "api_unsupported"
            stopped_after = name
            break
        try:
            warm_candidate = candidate()
            del warm_candidate
            (q_candidate, packed, tau), measurement = measured_call(local_rank, candidate)
        except (AttributeError, NotImplementedError, RuntimeError) as exc:
            results.append(
                {
                    "name": name,
                    "supported": False,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc)[:500],
                }
            )
            stop_reason = "api_dispatch_unsupported"
            stopped_after = name
            break

        r_candidate = torch.triu(packed)
        torch.npu.synchronize()
        q_error = error_metrics(q_candidate, q_direct)
        r_error = error_metrics(r_candidate, r_direct)
        finite = all(
            bool(torch.isfinite(tensor).all().cpu())
            for tensor in (q_candidate, packed, tau, r_candidate)
        )
        eye = torch.eye(2560, dtype=torch.float32, device=device)
        orthogonality = float((q_candidate.T @ q_candidate - eye).abs().max().cpu())
        reconstruction = float(
            torch.linalg.vector_norm(q_candidate @ r_candidate - x).cpu()
            / torch.linalg.vector_norm(x).cpu()
        )
        results.append(
            {
                "name": name,
                "supported": True,
                "measurement": measurement,
                "q_error": q_error,
                "r_error": r_error,
                "finite_all": finite,
                "orthogonality_max_abs": orthogonality,
                "reconstruction_rel_l2": reconstruction,
                "q_shape": list(q_candidate.shape),
                "packed_shape": list(packed.shape),
                "tau_shape": list(tau.shape),
            }
        )
        del q_candidate, packed, tau, r_candidate, eye
        if not q_error["bitwise"]:
            stop_reason = "raw_q_bitwise_mismatch"
            stopped_after = name
            break

    payload = {
        "pid": os.getpid(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "visible": VISIBLE,
        "gate_pass": direct_finite,
        "input_shape": [2560, 2560],
        "input_dtype": str(x.dtype),
        "direct_finite": direct_finite,
        "direct_measurement": direct_measurement,
        "candidate_results": results,
        "candidate_eligible": any(
            item.get("q_error", {}).get("bitwise", False) for item in results
        ),
        "stop_reason": stop_reason,
        "stopped_after": stopped_after,
        "expanded_shape": False,
        "training_or_profiling": False,
    }
    write_json(root / "ready" / f"rank{rank}.json", payload)
    deadline = time.monotonic() + 120
    while not (root / "release_after_npu_smi").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("release timeout")
        time.sleep(0.2)
    write_json(root / "done" / f"rank{rank}.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except BaseException as exc:
        if os.environ.get("OUTPUT_DIR"):
            failure = Path(os.environ["OUTPUT_DIR"]) / "failure"
            failure.mkdir(parents=True, exist_ok=True)
            (failure / f"rank{os.environ.get('RANK', 'x')}.txt").write_text(
                "".join(traceback.format_exception(exc)), encoding="utf-8"
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
