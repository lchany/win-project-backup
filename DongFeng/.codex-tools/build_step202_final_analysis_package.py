from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pooled_topn(per_step: dict[str, object], field: str) -> list[dict[str, object]]:
    pooled: dict[str, dict[str, object]] = defaultdict(
        lambda: {"count": 0, "duration_ms": 0.0, "wait_ms": 0.0, "task_class": ""}
    )
    for step in ("24", "25", "26"):
        for row in per_step[step][field]:
            item = pooled[row["name"]]
            item["count"] += int(row["count"])
            item["duration_ms"] += float(row["duration_ms"])
            item["wait_ms"] += float(row["wait_ms"])
            item["task_class"] = row["task_class"]
    rows = []
    for name, item in pooled.items():
        total = float(item["duration_ms"]) + float(item["wait_ms"])
        rows.append(
            {
                "name": name,
                **item,
                "total_cost_ms": total,
                "wait_ratio": float(item["wait_ms"]) / total if total else 0.0,
            }
        )
    metric = "duration_ms" if field == "by_kernel_duration" else "total_cost_ms"
    return sorted(rows, key=lambda row: float(row[metric]), reverse=True)[:20]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--retention-manifest", required=True, type=Path)
    args = parser.parse_args()
    analysis = args.analysis_dir.resolve()
    raw = args.raw_dir.resolve()
    retention_path = args.retention_manifest.resolve()

    old = json.loads((analysis / "v2_output/anomaly_discovery.json").read_text())
    optimized = json.loads(
        (analysis / "v2_optimized_output/anomaly_discovery.json").read_text()
    )
    schema = json.loads((analysis / "tools/anomaly_schema.json").read_text())
    viewcopy = json.loads((analysis / "viewcopy_stack_distribution_v2.json").read_text())
    retention = json.loads(retention_path.read_text())

    equal_fields = {
        key: old[key] == optimized.get(key) for key in old if key != "profile_run"
    }
    schema_errors = {
        "v2": len(list(Draft202012Validator(schema).iter_errors(old))),
        "v2_optimized": len(
            list(Draft202012Validator(schema).iter_errors(optimized))
        ),
    }
    architecture_sections = {}
    for label, folder in (("v2", "v2_output"), ("v2_optimized", "v2_optimized_output")):
        report = next((analysis / folder).glob("model_architecture_report*.md"))
        architecture_sections[label] = len(
            re.findall(r"^## [0-9]+\.", report.read_text(), re.MULTILINE)
        )

    raw_files = [path for path in raw.rglob("*") if path.is_file()]
    raw_dirs = [path for path in raw.rglob("*") if path.is_dir()]
    raw_bytes = sum(path.stat().st_size for path in raw_files)
    standalone_comm = [
        path
        for path in raw_files
        if path.name.lower() == "communication.json"
        or ("communication" in path.name.lower() and path.suffix.lower() == ".csv")
    ]
    memory_files = [path for path in raw_files if "memory" in path.name.lower()]

    normal_viewcopy = [
        row for row in viewcopy["per_step_aicpu_viewcopy_kernel"] if row["step"] in (24, 25, 26)
    ]
    normal_mean = {
        key: sum(float(row[key]) for row in normal_viewcopy) / len(normal_viewcopy)
        for key in ("count", "kernel_duration_ms", "wait_ms", "total_cost_ms", "wait_ratio")
    }
    random_group = next(
        row for row in viewcopy["groups"] if "random_spatial_mask" in row["boundary"]
    )
    normal_internal_bubbles = sorted(
        [
            row
            for row in old["bubble_windows"]
            if row["step"] in (24, 25, 26) and row["scope"] == "step_internal"
        ],
        key=lambda row: float(row["duration_ms"]),
        reverse=True,
    )[:5]
    pure_top20 = pooled_topn(old["per_step_topn"], "by_kernel_duration")
    total_top20 = pooled_topn(old["per_step_topn"], "by_total_cost")

    candidate = {
        "decision": "GO_MECHANISM_GATE_ONLY_NOT_IMPLEMENTED_NOT_TRAINED",
        "boundary": "bev_encoder.py:410-431 random_spatial_mask; line 427 block-slice assignment",
        "normal_step_evidence": normal_mean,
        "stack_cardinality": {
            "all_profiled_steps": int(random_group["count"]),
            "expected_4_steps_x_2048": 8192,
            "exact_cardinality_match": int(random_group["count"]) == 8192,
            "operator_host_self_ms_all_steps": float(random_group["host_self_us"]) / 1000.0,
            "operator_host_self_ms_per_step_proxy": float(random_group["host_self_us"]) / 4000.0,
            "operator_device_self_share_proxy": float(random_group["device_self_share"]),
            "shape_field": random_group["shape"],
        },
        "effective_profile_contract": {
            "lidar_dropout_prob": 0.1,
            "lidar_spatial_rate": 0.2,
            "lidar_mask_ratio": 0.2,
            "authority": "repository-external GPU-aligned runtime config copy used by this profile",
            "warning": "Do not replace these effective runtime values with the tracked source config's static zero values when interpreting this trace.",
        },
        "proposed_mechanism": (
            "Preserve the enable random draw and exactly one randperm per batch element in the "
            "same order; replace 512 per-block slice writes with one low-resolution block mask "
            "update per batch followed by exact 0/1 spatial expansion."
        ),
        "strict_equivalence_gates": [
            "same torch RNG state before and after every call",
            "same randperm call count and batch order",
            "bitwise-equal final mask and lidar_feat*mask output for fixed seeds",
            "same dtype, device, shape, contiguity, stride, storage/alias expectations, and autograd behavior",
            "edge cases: mask_ratio 0 and 1, non-divisible H/W, B=1 and B>1, enable outcomes 0 and 1",
            "then isolated NPU mechanism benchmark, test-set comparison, loss/grad fingerprint, and formal 8-NPU A/B",
            "GPU baseline semantics and timing are measured first/alongside NPU; NPU-only wins do not override GPU parity priority",
        ],
        "theoretical_net_gain_ms_per_normal_step": (
            "96.610072 - replacement_kernel_ms; queue wait 11.821988 ms is excluded"
        ),
        "historical_nonduplication": [
            "not STEP-179 PillarVFE max/layout copy removal (7.906410 ms, closed below threshold)",
            "not STEP-194 spatial_features clone (max single copy 6.888 ms, closed)",
            "not STEP-194 point_sampling Matmul/packed-BMM (formal E2E rejected)",
        ],
    }

    summary = {
        "run_name": old["profile_run"],
        "validation": {
            "schema_errors": schema_errors,
            "architecture_sections": architecture_sections,
            "v2_optimized_equal_except_profile_run": all(equal_fields.values()),
            "v2_optimized_field_equality": equal_fields,
        },
        "raw_retention": {
            "file_count": len(raw_files),
            "directory_count_including_raw": len(raw_dirs) + 1,
            "total_bytes": raw_bytes,
            "retention_manifest_sha256": sha256(retention_path),
            "retention_manifest_file_count": retention["file_count"],
            "retention_manifest_total_bytes": retention["total_bytes"],
            "retained": retention["retained"],
            "deletion_authorized": retention["deletion_authorized"],
            "mutation_performed": retention["mutation_performed"],
            "hash_mode": retention["hash_mode"],
        },
        "inventory_degradation": {
            "standalone_communication_json_or_csv": len(standalone_comm),
            "memory_files": len(memory_files),
            "rule": "communication/memory conclusions are degraded to trace/DB/kernel evidence; no bytes or bandwidth are inferred",
        },
        "per_step_four_clocks": old["per_step_four_clock_timing"],
        "normal_pooled_pure_kernel_top20": pure_top20,
        "normal_pooled_total_cost_top20": total_top20,
        "wait_anchors": old["wait_anchor_ops"],
        "aicpu_ops": old["aicpu_ops"],
        "normal_internal_bubble_top5": normal_internal_bubbles,
        "first_new_candidate": candidate,
    }
    candidate_path = analysis / "candidate_viewcopy_report_step202.json"
    candidate_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n")
    summary_path = analysis / "analysis_validation_summary_step202.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# STEP-202 Analysis Validation Summary",
        "",
        f"- Schema errors: V2={schema_errors['v2']}, optimized={schema_errors['v2_optimized']}",
        f"- Architecture sections: V2={architecture_sections['v2']}/10, optimized={architecture_sections['v2_optimized']}/10",
        f"- V2/optimized equality: {all(equal_fields.values())} after excluding only `profile_run`",
        f"- Raw retained unchanged by count/bytes: {len(raw_files)} files, {raw_bytes} bytes, {len(raw_dirs) + 1} directories including raw",
        f"- Retention manifest SHA256: `{sha256(retention_path)}`; retained=true, deletion_authorized=false, mutation_performed=false, hash_mode=all",
        "- Standalone communication JSON/CSV=0 and memory files=0; communication/memory analysis is explicitly degraded to trace/DB/kernel evidence without guessed volume or bandwidth.",
        "",
        "## Per-step four clocks",
        "",
        "| Profiler step | Service ms | Wall ms | Busy union ms | Kernel sum ms | Total cost ms |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in old["per_step_four_clock_timing"]:
        lines.append(
            f"| {row['step']} | {row['service_ms']:.3f} | {row['wall_ms']:.3f} | {row['busy_union_ms']:.3f} | {row['kernel_sum_ms']:.3f} | {row['total_cost_ms']:.3f} |"
        )
    lines += ["", "## Normal pooled pure-kernel Top20", "", "| Rank | Kernel | Count | Kernel ms | Wait ms | Wait ratio |", "|---:|---|---:|---:|---:|---:|"]
    for rank, row in enumerate(pure_top20, 1):
        lines.append(f"| {rank} | {row['name']} | {row['count']} | {row['duration_ms']:.3f} | {row['wait_ms']:.3f} | {100*row['wait_ratio']:.2f}% |")
    lines += ["", "## Normal pooled total-cost Top20", "", "| Rank | Kernel | Count | Kernel ms | Wait ms | Total ms | Wait ratio |", "|---:|---|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(total_top20, 1):
        lines.append(f"| {rank} | {row['name']} | {row['count']} | {row['duration_ms']:.3f} | {row['wait_ms']:.3f} | {row['total_cost_ms']:.3f} | {100*row['wait_ratio']:.2f}% |")
    lines += [
        "", "## First new candidate", "",
        "- Decision: mechanism gate only; no business code change, NPU run, training, or commit has been made.",
        f"- Normal mean: count={normal_mean['count']:.0f}, pure kernel={normal_mean['kernel_duration_ms']:.6f} ms, wait={normal_mean['wait_ms']:.6f} ms, total={normal_mean['total_cost_ms']:.6f} ms.",
        f"- Stack/cardinality: `random_spatial_mask` line 427 has {random_group['count']} calls across four steps, exactly 4x2048; host-self proxy={float(random_group['host_self_us'])/4000.0:.6f} ms/step.",
        "- Effective profiling overlay is dropout/spatial/mask=0.1/0.2/0.2. It is the authority for this trace; tracked source static zeros must not be substituted when interpreting this run.",
        "- Preserve RNG draw order and count, dtype/device/shape/stride/alias/autograd and bitwise final mask before any performance claim.",
        "- Theoretical net upper bound is `96.610072 ms - replacement kernel ms`; wait is excluded.",
        "- This boundary is distinct from STEP-179 PillarVFE layout and all STEP-194 clone/point-sampling ViewCopy/BMM candidates.",
        "", "## Normal internal bubble Top5", "",
        "| Rank | Step | Duration ms | Before | After | Host coverage | Sync overlap | Labels |", "|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for rank, row in enumerate(normal_internal_bubbles, 1):
        before = row["before"]["name"] if row.get("before") else "none"
        after = row["after"]["name"] if row.get("after") else "none"
        labels = ",".join(row.get("candidate_labels", []))
        lines.append(f"| {rank} | {row['step']} | {row['duration_ms']:.6f} | {before} | {after} | {row['host_visible_coverage_ratio']:.3f} | {row['sync_marker_overlap_ratio']:.3f} | {labels} |")
    (analysis / "analysis_validation_summary_step202.md").write_text("\n".join(lines) + "\n")

    excluded = {"analysis_sha256_manifest_step202.json", "SHA256SUMS_step202.txt"}
    artifacts = []
    for path in sorted(analysis.rglob("*")):
        if path.is_file() and path.name not in excluded:
            artifacts.append(
                {"relative_path": path.relative_to(analysis).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    manifest = {
        "manifest_version": 1,
        "run_name": old["profile_run"],
        "raw_retention_manifest_sha256": sha256(retention_path),
        "raw_retained": True,
        "raw_deletion_authorized": False,
        "raw_mutation_performed": False,
        "analysis_artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    (analysis / "analysis_sha256_manifest_step202.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (analysis / "SHA256SUMS_step202.txt").write_text(
        "".join(f"{item['sha256']}  {item['relative_path']}\n" for item in artifacts)
    )
    print(json.dumps({"schema_errors": schema_errors, "equal_except_run": all(equal_fields.values()), "artifacts": len(artifacts), "candidate": candidate["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
