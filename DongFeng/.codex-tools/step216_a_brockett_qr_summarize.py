#!/usr/bin/env python3
"""Aggregate strict STEP-216-A rank results and controller evidence."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from step216_a_brockett_policy import (
    load_source_contract,
    sha256_file,
    verify_source_package,
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction)))]


def build_summary(root: Path, source_contract_path: Path) -> dict:
    contract = load_source_contract(source_contract_path)
    verify_source_package(contract, source_contract_path.parent)
    contract_sha = sha256_file(source_contract_path)
    ranks = [
        json.loads((root / "done" / f"rank{rank}.json").read_text(encoding="utf-8"))
        for rank in range(8)
    ]
    ready = [
        json.loads((root / "ready" / f"rank{rank}.json").read_text(encoding="utf-8"))
        for rank in range(8)
    ]
    controller = json.loads((root / "controller_status.json").read_text(encoding="utf-8"))
    if controller.get("status") != "PASS" or controller.get("logical_rank_count") != 8:
        raise RuntimeError("world8 controller did not pass")
    if set(controller.get("direct_rank_container_pids", [])) != {row["container_pid"] for row in ready}:
        raise RuntimeError("controller direct container PID evidence differs from ready")
    if len(set(controller.get("npu_host_pids", []))) != 8:
        raise RuntimeError("controller NPU host PID evidence is incomplete")
    if any(
        row["source_contract_sha256"] != contract_sha
        or row["factor_count"] != 543
        or row["state_count"] != 559
        or row["cycle_sample_count"] < 3
        for row in ranks
    ):
        raise RuntimeError("rank identity/count/sample contract failed")
    expected_runtime = {
        name: {"sha256": identity["sha256"], "bytes": identity["bytes"]}
        for name, identity in contract["runtime_artifacts"].items()
    }
    if any(row["runtime_identities"] != expected_runtime for row in ranks):
        raise RuntimeError("rank runtime identities differ from source contract")
    if len({json.dumps(row["active_inventory"], sort_keys=True) for row in ranks}) != 1:
        raise RuntimeError("rank inventories differ")

    by_shape: dict[int, list[dict]] = defaultdict(list)
    for rank in ranks:
        for factor in rank["factors"]:
            by_shape[int(factor["dimension"])].append(factor)
    shapes = []
    for dimension in sorted(by_shape):
        rows = by_shape[dimension]
        shapes.append({
            "dimension": dimension,
            "calls_per_rank": len(rows) // 8,
            "gate_pass_all": all(row["gate_pass"] for row in rows),
            "candidate_orthogonality_max_abs_worst": max(
                row["candidate_orthogonality_max_abs"] for row in rows
            ),
            "candidate_orthogonality_normalized_fro_worst": max(
                row["candidate_orthogonality_normalized_fro"] for row in rows
            ),
            "candidate_rayleigh_offdiag_worst": max(
                row["candidate_rayleigh_offdiag"] for row in rows
            ),
            "raw_q_rel_l2_diagnostic_median": statistics.median(
                row["raw_q_rel_l2_diagnostic_only"] for row in rows
            ),
        })

    event_savings = [row["paired_event_cycle_saving_median_ms"] for row in ranks]
    wall_savings = [row["paired_wall_cycle_saving_median_ms"] for row in ranks]
    allocated = [row["candidate_minus_baseline_peak_allocated_max_bytes"] for row in ranks]
    reserved = [row["candidate_minus_baseline_peak_reserved_max_bytes"] for row in ranks]
    effect_per_tensor = [
        row["real_project_back_oldq_project_newq_effect"]["per_tensor_rel_l2_worst"]
        for row in ranks
    ]
    effect_global = [
        row["real_project_back_oldq_project_newq_effect"]["global_rel_l2"]
        for row in ranks
    ]
    qualified = all(row["status"] == "PASS_LOCAL_SCREEN" for row in ranks)
    return {
        "decision": "GO_TO_STATEFUL_TWO_CYCLE_ONLY" if qualified else "REJECT_LOCAL_SCREEN",
        "qualified_all_8_ranks": qualified,
        "rank_count": 8,
        "state_count_per_rank": 559,
        "active_shape_count": 23,
        "factor_count_per_rank": 543,
        "cycle_sample_count_per_path": min(row["cycle_sample_count"] for row in ranks),
        "active_inventory": ranks[0]["active_inventory"],
        "source_contract_sha256": contract_sha,
        "runtime_identities": expected_runtime,
        "controller": {
            "status": controller["status"],
            "direct_rank_container_pids_bound": len(controller["direct_rank_container_pids"]),
            "npu_host_pids_bound": len(controller["npu_host_pids"]),
            "pid_namespace_mapping": controller["pid_namespace_mapping"],
            "physical_process_count": controller["physical_process_count"],
            "physical_pairs": controller["physical_pairs"],
        },
        "numeric_pass_all_ranks": all(row["numeric_pass"] for row in ranks),
        "memory_pass_all_ranks": all(row["memory_pass"] for row in ranks),
        "performance_pass_all_ranks": all(row["performance_pass"] for row in ranks),
        "paired_event_cycle_saving_rank_median_ms": statistics.median(event_savings),
        "paired_event_cycle_saving_rank_min_ms": min(event_savings),
        "paired_wall_cycle_saving_rank_median_ms": statistics.median(wall_savings),
        "paired_wall_cycle_saving_rank_min_ms": min(wall_savings),
        "candidate_minus_baseline_peak_allocated_rank_max_bytes": max(allocated),
        "candidate_minus_baseline_peak_reserved_rank_max_bytes": max(reserved),
        "candidate_persistent_weight_cache_bytes_max": max(
            row["candidate_persistent_weight_cache_bytes"] for row in ranks
        ),
        "real_project_effect_per_tensor_rel_l2_worst": max(effect_per_tensor),
        "real_project_effect_global_rel_l2_worst": max(effect_global),
        "repeat_rel_l2_worst": {
            path: max(row["repeat_rel_l2_worst"][path] for row in ranks)
            for path in ("baseline", "candidate")
        },
        "thresholds": ranks[0]["thresholds"],
        "scope": ranks[0]["scope"],
        "shapes": shapes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-contract", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve(strict=True)
    source_contract = Path(args.source_contract).resolve(strict=True)
    summary = build_summary(root, source_contract)
    temp = root / "summary.json.tmp"
    temp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(root / "summary.json")
    print(json.dumps({key: value for key, value in summary.items() if key != "shapes"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
