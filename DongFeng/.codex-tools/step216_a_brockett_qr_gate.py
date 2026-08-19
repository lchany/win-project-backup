#!/usr/bin/env python3
"""Strict world8 local screen for the pinned TurboSOAP Brockett core.

The gate uses the qualified real SOAP optimizer state. Common stable sort,
``power_iter`` and trace normalization are built once and shared by baseline
and candidate. No tensor content is written to disk.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch_npu

from step216_a_brockett_policy import (
    assert_inventory,
    assert_bound_project_signature,
    dispatch_candidate,
    load_source_contract,
    load_verified_policy,
    project_roundtrip_views,
    sha256_file,
    verify_identity,
    verify_source_package,
)
from step216_a_world8_controller import VISIBLE


ORTH_LIMIT = 2.0e-5
REPEAT_LIMIT = 1.0e-5
PROJECT_EFFECT_LIMIT = 5.0e-3
RAYLEIGH_ABSOLUTE_LIMIT = 0.3
MIN_CYCLE_SAVING_MS = 227.0
ALLOCATED_PEAK_DIFFERENCE_LIMIT = 256 * 1024 * 1024
RESERVED_PEAK_DIFFERENCE_LIMIT = 512 * 1024 * 1024
CYCLE_SAMPLES = 3
_WEIGHT_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed_cycle(device_index: int, fn: Callable[[], list[torch.Tensor]]) -> tuple[list[torch.Tensor], dict[str, float | int]]:
    torch.npu.synchronize()
    base_allocated = int(torch.npu.memory_allocated(device_index))
    base_reserved = int(torch.npu.memory_reserved(device_index))
    torch.npu.reset_peak_memory_stats(device_index)
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    value = fn()
    end.record()
    end.synchronize()
    return value, {
        "event_ms": float(start.elapsed_time(end)),
        "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
        "peak_allocated_delta_from_base_bytes": max(
            0, int(torch.npu.max_memory_allocated(device_index)) - base_allocated
        ),
        "peak_reserved_delta_from_base_bytes": max(
            0, int(torch.npu.max_memory_reserved(device_index)) - base_reserved
        ),
    }


def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(candidate - reference)
    denominator = torch.clamp_min(torch.linalg.vector_norm(reference), 1.0e-30)
    return float((numerator / denominator).cpu())


def squared_norm(value: torch.Tensor) -> float:
    return float(torch.sum(value.float() * value.float()).cpu())


def orthogonality(q: torch.Tensor) -> tuple[float, float]:
    n = int(q.shape[0])
    delta = q.T @ q - torch.eye(n, dtype=q.dtype, device=q.device)
    return float(delta.abs().max().cpu()), float(
        (torch.linalg.vector_norm(delta) / math.sqrt(n)).cpu()
    )


def rayleigh_diag_error(c: torch.Tensor, q: torch.Tensor) -> float:
    transformed = q.T @ c @ q
    total_sq = torch.sum(transformed * transformed)
    diagonal = torch.diagonal(transformed)
    off_sq = torch.clamp_min(total_sq - torch.sum(diagonal * diagonal), 0.0)
    return float(torch.sqrt(off_sq / torch.clamp_min(total_sq, 1.0e-30)).cpu())


def brockett_weights(n: int, q: torch.Tensor, exponent: float) -> torch.Tensor:
    key = (str(q.device), n)
    value = _WEIGHT_CACHE.get(key)
    if value is None:
        value = torch.linspace(float(n), 1.0, n, dtype=torch.float32, device=q.device)
        powered = value.pow(exponent)
        value = (powered / torch.mean(powered)).contiguous()
        _WEIGHT_CACHE[key] = value
    return value


def brockett_cubic_from_power_iter(
    power_iter: torch.Tensor,
    trace_norm: torch.Tensor,
    q_sorted: torch.Tensor,
    policy: dict[str, Any],
) -> torch.Tensor:
    """Reuse common C@Q and trace norm; no candidate-side n^3 C@Q."""
    algorithm = policy["algorithm"]
    retraction = policy["retraction"]
    n = int(q_sorted.shape[0])
    cq = power_iter / trace_norm
    weights = brockett_weights(n, q_sorted, algorithm["basis_weight_exponent"])
    gradient = cq * weights[None, :]
    tangent_inner = q_sorted.T @ gradient
    symmetric_inner = 0.5 * (tangent_inner + tangent_inner.T)
    direction = gradient - q_sorted @ symmetric_inner
    x = q_sorted + algorithm["eta"] * direction
    gram = x.T @ x
    upper_sq = torch.max(torch.sum(torch.abs(gram), dim=1))
    scale = torch.clamp_max(
        retraction["row_sum_target"]
        / (torch.sqrt(upper_sq) + retraction["scale_epsilon"]),
        1.0,
    )
    z = scale * x
    gram = (scale * scale) * gram
    gram2 = gram @ gram
    return (
        retraction["outer_scale"]
        * (
            retraction["coefficient_z"] * z
            + z
            @ (
                retraction["coefficient_g"] * gram
                + retraction["coefficient_g2"] * gram2
            )
        )
    ).contiguous()


def baseline_q(power_iter: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(power_iter, mode="reduced")[0].contiguous()


@torch.no_grad()
def select_q(factor: dict[str, Any], path: str, policy: dict[str, Any]) -> tuple[torch.Tensor, str]:
    power_iter = factor["power_iter"]
    approved = dispatch_candidate(
        int(power_iter.shape[0]),
        square=power_iter.ndim == 2 and power_iter.shape[0] == power_iter.shape[1],
        dtype=str(power_iter.dtype),
        contiguous=bool(power_iter.is_contiguous()),
        requires_grad=bool(power_iter.requires_grad or torch.is_grad_enabled()),
    )
    if path == "baseline" or not approved:
        route = "linalg_qr_baseline" if path == "baseline" else "linalg_qr_fallback"
        return baseline_q(power_iter), route
    if path != "candidate":
        raise ValueError(f"unknown path {path}")
    return brockett_cubic_from_power_iter(
        power_iter, factor["trace_norm"], factor["q_sorted"], policy
    ), "brockett_cubic_core"


def build_factors(optimizer: Any, parameters: list[torch.Tensor], policy: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Any]]:
    factors: list[dict[str, Any]] = []
    states = [optimizer.state[parameter] for parameter in parameters]
    inventory: Counter[int] = Counter()
    for state_index, state in enumerate(states):
        gg = state["GG"]
        q_values = state["Q"]
        if not isinstance(gg, list) or not isinstance(q_values, list) or len(gg) != len(q_values):
            raise RuntimeError("GG/Q list schema changed")
        for axis, (c, q_old) in enumerate(zip(gg, q_values)):
            if not torch.is_tensor(q_old):
                continue
            if not torch.is_tensor(c) or c.shape != q_old.shape or c.ndim != 2 or c.shape[0] != c.shape[1]:
                raise RuntimeError("active GG/Q is not same-shape square")
            c = c.float()
            q_old = q_old.float()
            estimate = torch.diagonal(q_old.T @ c @ q_old)
            sort_index = torch.argsort(estimate, descending=True, stable=True)
            q_sorted = q_old.index_select(1, sort_index).contiguous()
            power_iter = (c @ q_sorted).contiguous()
            trace_norm = torch.clamp_min(
                torch.abs(torch.trace(c)) / int(c.shape[0]),
                policy["algorithm"]["matrix_normalization_floor"],
            )
            dimension = int(c.shape[0])
            if not dispatch_candidate(
                dimension,
                square=True,
                dtype=str(power_iter.dtype),
                contiguous=power_iter.is_contiguous(),
                requires_grad=power_iter.requires_grad,
            ):
                raise RuntimeError("active factor unexpectedly selected fallback")
            inventory[dimension] += 1
            factors.append({
                "factor_index": len(factors),
                "state_index": state_index,
                "axis": axis,
                "dimension": dimension,
                "c": c,
                "q_sorted": q_sorted,
                "power_iter": power_iter,
                "trace_norm": trace_norm,
            })
    assert_inventory(dict(inventory))
    if len(factors) != 543:
        raise RuntimeError("factor count is not 543")
    return factors, states


def cycle(path: str, factors: list[dict[str, Any]], policy: dict[str, Any]) -> list[torch.Tensor]:
    outputs = []
    for factor in factors:
        q, route = select_q(factor, path, policy)
        expected = "linalg_qr_baseline" if path == "baseline" else "brockett_cubic_core"
        if route != expected:
            raise RuntimeError(f"measured route changed: {route}")
        outputs.append(q)
    return outputs


def compare_repeat(current: list[torch.Tensor], previous: list[torch.Tensor]) -> float:
    if len(current) != 543 or len(previous) != 543:
        raise RuntimeError("repeat output count changed")
    return max(relative_l2(left, right) for left, right in zip(current, previous))


def make_q_lists(states: list[Any], outputs: list[torch.Tensor], factors: list[dict[str, Any]]) -> list[list[Any]]:
    q_lists = [list(state["Q"]) for state in states]
    if len(outputs) != len(factors):
        raise RuntimeError("Q output/factor count mismatch")
    for factor, output in zip(factors, outputs):
        q_lists[factor["state_index"]][factor["axis"]] = output
    return q_lists


@torch.no_grad()
def real_project_effect(
    optimizer: Any,
    states: list[Any],
    baseline_outputs: list[torch.Tensor],
    candidate_outputs: list[torch.Tensor],
    factors: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_q_lists = make_q_lists(states, baseline_outputs, factors)
    candidate_q_lists = make_q_lists(states, candidate_outputs, factors)
    rows = []
    delta_sq_global = 0.0
    reference_sq_global = 0.0
    finite_all = True
    for state_index, state in enumerate(states):
        exp_avg = state["exp_avg"]
        original_coordinates, baseline_projected, candidate_projected = project_roundtrip_views(
            optimizer,
            exp_avg,
            state,
            baseline_q_lists[state_index],
            candidate_q_lists[state_index],
        )
        finite = all(
            bool(torch.isfinite(value).all().cpu())
            for value in (exp_avg, original_coordinates, baseline_projected, candidate_projected)
        )
        finite_all = finite_all and finite
        difference = candidate_projected - baseline_projected
        delta_sq = squared_norm(difference)
        reference_sq = squared_norm(baseline_projected)
        rel = math.sqrt(delta_sq / max(reference_sq, 1.0e-30))
        delta_sq_global += delta_sq
        reference_sq_global += reference_sq
        rows.append({
            "state_index": state_index,
            "shape": [int(value) for value in exp_avg.shape],
            "finite": finite,
            "rel_l2": rel,
        })
    global_rel = math.sqrt(delta_sq_global / max(reference_sq_global, 1.0e-30))
    worst = max(row["rel_l2"] for row in rows)
    return {
        "state_count": len(rows),
        "finite_all": finite_all,
        "per_tensor_rel_l2_worst": worst,
        "global_rel_l2": global_rel,
        "gate_pass": finite_all and worst <= PROJECT_EFFECT_LIMIT and global_rel <= PROJECT_EFFECT_LIMIT,
        "rows": rows,
    }


@torch.no_grad()
def factor_diagnostics(
    factors: list[dict[str, Any]],
    baseline_outputs: list[torch.Tensor],
    candidate_outputs: list[torch.Tensor],
) -> tuple[list[dict[str, Any]], bool]:
    rows = []
    passed = True
    for factor, baseline, candidate in zip(factors, baseline_outputs, candidate_outputs):
        baseline_orth_max, baseline_orth_fro = orthogonality(baseline)
        candidate_orth_max, candidate_orth_fro = orthogonality(candidate)
        baseline_rayleigh = rayleigh_diag_error(factor["c"], baseline)
        candidate_rayleigh = rayleigh_diag_error(factor["c"], candidate)
        finite = all(
            bool(torch.isfinite(value).all().cpu())
            for value in (factor["power_iter"], baseline, candidate)
        )
        rayleigh_relative_limit = max(
            baseline_rayleigh * 1.05, baseline_rayleigh + 1.0e-5
        )
        row_pass = all([
            finite,
            baseline_orth_max <= ORTH_LIMIT,
            baseline_orth_fro <= ORTH_LIMIT,
            candidate_orth_max <= ORTH_LIMIT,
            candidate_orth_fro <= ORTH_LIMIT,
            candidate_rayleigh <= RAYLEIGH_ABSOLUTE_LIMIT,
            candidate_rayleigh <= rayleigh_relative_limit,
        ])
        passed = passed and row_pass
        rows.append({
            "factor_index": factor["factor_index"],
            "state_index": factor["state_index"],
            "axis": factor["axis"],
            "dimension": factor["dimension"],
            "finite": finite,
            "gate_pass": row_pass,
            "baseline_orthogonality_max_abs": baseline_orth_max,
            "baseline_orthogonality_normalized_fro": baseline_orth_fro,
            "candidate_orthogonality_max_abs": candidate_orth_max,
            "candidate_orthogonality_normalized_fro": candidate_orth_fro,
            "baseline_rayleigh_offdiag": baseline_rayleigh,
            "candidate_rayleigh_offdiag": candidate_rayleigh,
            "rayleigh_relative_limit": rayleigh_relative_limit,
            "raw_q_rel_l2_diagnostic_only": relative_l2(candidate, baseline),
        })
    return rows, passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve(strict=True)
    failure = output / "failure" / f"rank{os.environ.get('RANK', 'unknown')}.txt"
    failure.parent.mkdir(exist_ok=True)
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size != 8 or rank not in range(8) or local_rank not in range(8):
            raise RuntimeError("world8/rank mapping contract failed")
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != VISIBLE:
            raise RuntimeError("visible NPU contract changed")
        if os.environ.get("TASK_QUEUE_ENABLE") is not None:
            raise RuntimeError("TASK_QUEUE_ENABLE must remain absent")
        if not hasattr(torch, "npu") or torch_npu is None:
            raise RuntimeError("torch_npu is not loaded")

        source_contract_path = Path(args.source_contract).resolve(strict=True)
        source_root = source_contract_path.parent
        source_contract = load_source_contract(source_contract_path)
        verify_source_package(source_contract, source_root)
        runtime = source_contract["runtime_artifacts"]
        repo = Path(args.repo).resolve(strict=True)
        soap = verify_identity(
            repo / "projects/mmdet3d_plugin/optimizers/soap.py", runtime["soap"]
        )
        config = verify_identity(args.config, runtime["config"])
        checkpoint = verify_identity(args.checkpoint, runtime["checkpoint"])
        adapter_path = verify_identity(args.adapter, runtime["adapter"])
        community_config = verify_identity(
            source_root / runtime["community_config"]["name"],
            runtime["community_config"],
        )
        policy = load_verified_policy(community_config)

        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        dist.init_process_group("hccl")
        adapter_module = import_module(adapter_path, "step216_pinned_real_soap_adapter")
        if (
            adapter_module.EXPECTED_CONFIG_SHA256 != runtime["config"]["sha256"]
            or adapter_module.EXPECTED_CHECKPOINT_SHA256 != runtime["checkpoint"]["sha256"]
            or adapter_module.EXPECTED_CHECKPOINT_BYTES != runtime["checkpoint"]["bytes"]
            or adapter_module.EXPECTED_STATE_KEYS
            != set(source_contract["soap_runtime_schema"]["state_keys"])
        ):
            raise RuntimeError("adapter embedded identity/schema differs from source contract")
        adapter = adapter_module.create_adapter({
            "repo": repo, "config": config, "checkpoint": checkpoint, "device": device
        })
        trial = adapter.build_trial({"device": device})
        optimizer = trial["optimizer"]
        parameters = trial["parameters"]
        assert_bound_project_signature(
            optimizer.project,
            source_contract["soap_runtime_schema"]["project_bound_signature"],
        )
        assert_bound_project_signature(
            optimizer.project_back,
            source_contract["soap_runtime_schema"]["project_back_bound_signature"],
        )
        factors, states = build_factors(optimizer, parameters, policy)
        dispatch_assertions = {
            "approved_2560": dispatch_candidate(2560, square=True, dtype="torch.float32", contiguous=True, requires_grad=False),
            "fallback_5120": not dispatch_candidate(5120, square=True, dtype="torch.float32", contiguous=True, requires_grad=False),
            "fallback_unknown257": not dispatch_candidate(257, square=True, dtype="torch.float32", contiguous=True, requires_grad=False),
            "fallback_fp16": not dispatch_candidate(256, square=True, dtype="torch.float16", contiguous=True, requires_grad=False),
            "fallback_grad": not dispatch_candidate(256, square=True, dtype="torch.float32", contiguous=True, requires_grad=True),
        }
        if not all(dispatch_assertions.values()):
            raise RuntimeError("dispatch assertions failed")

        ready = output / "ready"
        ready.mkdir(exist_ok=True)
        atomic_json(ready / f"rank{rank}.json", {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "visible": VISIBLE,
            "container_pid": os.getpid(),
            "factor_count": len(factors),
            "state_count": len(states),
            "source_contract_sha256": sha256_file(source_contract_path),
            "gate_pass": True,
        })
        release = output / "release_after_npu_smi"
        deadline = time.monotonic() + int(os.environ.get("STEP216_TIMEOUT_SECONDS", "1200"))
        while not release.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("release_after_npu_smi was not created")
            time.sleep(0.1)

        warmed: set[int] = set()
        for factor in factors:
            if factor["dimension"] in warmed:
                continue
            warm_baseline, _ = select_q(factor, "baseline", policy)
            warm_candidate, route = select_q(factor, "candidate", policy)
            if route != "brockett_cubic_core":
                raise RuntimeError("warmup selected fallback")
            del warm_baseline, warm_candidate
            warmed.add(factor["dimension"])
        torch.npu.synchronize()

        timings: dict[str, list[dict[str, float | int]]] = {"baseline": [], "candidate": []}
        previous: dict[str, list[torch.Tensor] | None] = {"baseline": None, "candidate": None}
        repeat_worst = {"baseline": 0.0, "candidate": 0.0}
        for sample in range(CYCLE_SAMPLES):
            order = ("baseline", "candidate") if (sample + rank) % 2 == 0 else ("candidate", "baseline")
            for path in order:
                outputs, timing = timed_cycle(
                    local_rank, lambda selected=path: cycle(selected, factors, policy)
                )
                timings[path].append(timing)
                if previous[path] is not None:
                    repeat_worst[path] = max(
                        repeat_worst[path], compare_repeat(outputs, previous[path] or [])
                    )
                previous[path] = outputs

        baseline_outputs = previous["baseline"]
        candidate_outputs = previous["candidate"]
        if baseline_outputs is None or candidate_outputs is None:
            raise RuntimeError("final cycle outputs are missing")
        factor_rows, factor_pass = factor_diagnostics(
            factors, baseline_outputs, candidate_outputs
        )
        effect = real_project_effect(
            optimizer, states, baseline_outputs, candidate_outputs, factors
        )
        paired_event_savings = [
            float(timings["baseline"][index]["event_ms"])
            - float(timings["candidate"][index]["event_ms"])
            for index in range(CYCLE_SAMPLES)
        ]
        paired_wall_savings = [
            float(timings["baseline"][index]["wall_ms"])
            - float(timings["candidate"][index]["wall_ms"])
            for index in range(CYCLE_SAMPLES)
        ]
        allocated_differences = [
            int(timings["candidate"][index]["peak_allocated_delta_from_base_bytes"])
            - int(timings["baseline"][index]["peak_allocated_delta_from_base_bytes"])
            for index in range(CYCLE_SAMPLES)
        ]
        reserved_differences = [
            int(timings["candidate"][index]["peak_reserved_delta_from_base_bytes"])
            - int(timings["baseline"][index]["peak_reserved_delta_from_base_bytes"])
            for index in range(CYCLE_SAMPLES)
        ]
        performance_pass = (
            statistics.median(paired_event_savings) > MIN_CYCLE_SAVING_MS
            and statistics.median(paired_wall_savings) > MIN_CYCLE_SAVING_MS
        )
        memory_pass = (
            max(allocated_differences) <= ALLOCATED_PEAK_DIFFERENCE_LIMIT
            and max(reserved_differences) <= RESERVED_PEAK_DIFFERENCE_LIMIT
        )
        repeat_pass = all(value <= REPEAT_LIMIT for value in repeat_worst.values())
        numeric_pass = factor_pass and effect["gate_pass"] and repeat_pass
        result = {
            "status": "PASS_LOCAL_SCREEN" if numeric_pass and memory_pass and performance_pass else "REJECT_LOCAL_SCREEN",
            "rank": rank,
            "source_contract_sha256": sha256_file(source_contract_path),
            "runtime_identities": {
                name: {"sha256": identity["sha256"], "bytes": identity["bytes"]}
                for name, identity in runtime.items()
            },
            "factor_count": len(factors),
            "state_count": len(states),
            "active_inventory": dict(
                sorted(Counter(row["dimension"] for row in factors).items())
            ),
            "dispatch_assertions": dispatch_assertions,
            "cycle_sample_count": CYCLE_SAMPLES,
            "timings": timings,
            "paired_event_cycle_savings_ms": paired_event_savings,
            "paired_wall_cycle_savings_ms": paired_wall_savings,
            "paired_event_cycle_saving_median_ms": statistics.median(paired_event_savings),
            "paired_wall_cycle_saving_median_ms": statistics.median(paired_wall_savings),
            "candidate_minus_baseline_peak_allocated_bytes": allocated_differences,
            "candidate_minus_baseline_peak_reserved_bytes": reserved_differences,
            "candidate_minus_baseline_peak_allocated_max_bytes": max(allocated_differences),
            "candidate_minus_baseline_peak_reserved_max_bytes": max(reserved_differences),
            "candidate_persistent_weight_cache_bytes": sum(
                int(value.numel() * value.element_size()) for value in _WEIGHT_CACHE.values()
            ),
            "repeat_rel_l2_worst": repeat_worst,
            "real_project_back_oldq_project_newq_effect": effect,
            "factors": factor_rows,
            "numeric_pass": numeric_pass,
            "memory_pass": memory_pass,
            "performance_pass": performance_pass,
            "thresholds": {
                "orthogonality_max_abs_and_normalized_fro": ORTH_LIMIT,
                "repeat_rel_l2": REPEAT_LIMIT,
                "real_project_effect_per_tensor_and_global_rel_l2": PROJECT_EFFECT_LIMIT,
                "rayleigh_offdiag_absolute": RAYLEIGH_ABSOLUTE_LIMIT,
                "minimum_paired_event_and_wall_cycle_saving_median_ms": MIN_CYCLE_SAVING_MS,
                "candidate_minus_baseline_peak_allocated_bytes": ALLOCATED_PEAK_DIFFERENCE_LIMIT,
                "candidate_minus_baseline_peak_reserved_bytes": RESERVED_PEAK_DIFFERENCE_LIMIT,
            },
            "scope": {
                "kind": "real_checkpoint_full_axis_local_operator_screen",
                "raw_q_distance_is_diagnostic_only": True,
                "validates_optimizer_state_trajectory": False,
                "validates_model_loss_or_ddp": False,
                "authorizes_business_change": False,
            },
        }
        done = output / "done"
        done.mkdir(exist_ok=True)
        atomic_json(done / f"rank{rank}.json", result)
        dist.barrier()
        adapter.destroy_trial(trial)
        dist.destroy_process_group()
        return 0
    except Exception as exc:
        failure.write_text(
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
