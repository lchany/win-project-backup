from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import ijson


STEP_RE = re.compile(r"^ProfilerStep#(\d+)$")
AUTOGRAD_PREFIX = "autograd::engine::evaluate_function:"
DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def number(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def parse_steps(spec: str) -> set[int]:
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        else:
            result.add(int(part))
    return result


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        old_start, old_end = merged[-1]
        if start <= old_end:
            merged[-1] = (old_start, max(old_end, end))
        else:
            merged.append((start, end))
    return merged


def family(name: str, category: str) -> str:
    lower = name.lower()
    if category == "gpu_memcpy":
        if "htod" in lower:
            return "memory_h2d"
        if "dtoh" in lower:
            return "memory_d2h"
        if "dtod" in lower:
            return "memory_d2d"
        return "memory_copy_other"
    if category == "gpu_memset":
        return "memory_set"
    if any(token in lower for token in ("geqr", "orgqr", "linalg_qr")):
        return "soap_qr"
    if "ms_deformable" in lower or "multiscaledeformable" in lower:
        return "msda_backward" if "col2im" in lower or "backward" in lower else "msda_forward"
    if "nccl" in lower or "allreduce" in lower or "all_reduce" in lower or "reduce_scatter" in lower or "allgather" in lower:
        return "communication"
    if any(token in lower for token in ("cudnn", "convolution", "conv_depthwise", "implicit_convolve", "winograd")):
        return "convolution"
    if any(token in lower for token in ("gemm", "sgemm", "matmul", "cutlass")):
        return "matmul_bmm_addmm"
    if any(token in lower for token in ("direct_copy", "copy_kernel", "copy_")):
        return "tensor_copy_viewcopy"
    if "index" in lower or "gather" in lower or "scatter" in lower:
        return "index_gather_scatter"
    if any(token in lower for token in ("sort", "radix", "topk")):
        return "sort_topk"
    if any(token in lower for token in ("layer_norm", "layernorm", "batch_norm", "batchnorm", "group_norm")):
        return "normalization"
    if any(token in lower for token in ("reduce", "reduction", "sum_kernel", "mean_kernel")):
        return "reduction"
    if any(token in lower for token in ("elementwise", "vectorized_elementwise", "unrolled_elementwise")):
        return "elementwise"
    if any(token in lower for token in ("memset", "fill_kernel", "zero")):
        return "fill_zero"
    return "other"


def signature(args: dict[str, Any]) -> str:
    return json.dumps(
        {
            "Input Dims": args.get("Input Dims"),
            "Input type": args.get("Input type"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    return ordered[left] * (right - position) + ordered[right] * (position - left)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stable-steps", default="33-41,43-48")
    parser.add_argument("--soap-steps", default="22,32,42")
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    trace = args.trace.resolve(strict=True)
    inventory_path = args.inventory.resolve(strict=True)
    if trace.is_symlink() or inventory_path.is_symlink() or not trace.is_file() or not inventory_path.is_file():
        raise RuntimeError("inputs must be regular non-symlink files")
    if args.output.exists():
        raise RuntimeError("output already exists")
    inventory = json.load(inventory_path.open(encoding="utf-8"))
    stable_steps = parse_steps(args.stable_steps)
    soap_steps = parse_steps(args.soap_steps)

    markers: list[dict[str, Any]] = []
    occurrences: Counter[int] = Counter()
    for row in inventory["step_markers"]:
        match = STEP_RE.fullmatch(str(row.get("name", "")))
        if not match or row.get("phase") != "X":
            continue
        step_id = int(match.group(1))
        occurrence = occurrences[step_id]
        occurrences[step_id] += 1
        start = number(row.get("ts_us"))
        duration = number(row.get("duration_us"))
        if start is None or duration is None or duration <= 0:
            continue
        markers.append(
            {
                "step_id": step_id,
                "occurrence": occurrence,
                "window_id": f"{step_id}:{occurrence}",
                "start_us": start,
                "end_us": start + duration,
                "service_us": duration,
            }
        )
    markers.sort(key=lambda row: row["start_us"])
    starts = [float(row["start_us"]) for row in markers]
    selected_windows = {
        str(row["window_id"]): row
        for row in markers
        if int(row["step_id"]) in stable_steps | soap_steps and int(row["occurrence"]) == 0
    }
    if len([row for row in selected_windows.values() if int(row["step_id"]) in stable_steps]) != len(stable_steps):
        raise RuntimeError("not all requested stable steps have one complete marker")
    if len([row for row in selected_windows.values() if int(row["step_id"]) in soap_steps]) != len(soap_steps):
        raise RuntimeError("not all requested SOAP steps have one complete marker")

    def locate(ts: float) -> dict[str, Any] | None:
        index = bisect.bisect_right(starts, ts) - 1
        if index < 0:
            return None
        row = markers[index]
        return row if ts < float(row["end_us"]) else None

    boundaries: dict[str, dict[str, float | None]] = defaultdict(
        lambda: {"backward_start_us": None, "optimizer_start_us": None, "optimizer_end_us": None}
    )
    annotations: dict[str, Counter[str]] = defaultdict(Counter)
    cpu_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cpu_duration: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    runtime_counts: dict[str, Counter[str]] = defaultdict(Counter)
    runtime_duration: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    copy_signature_counts: dict[str, Counter[str]] = defaultdict(Counter)
    copy_signature_duration: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    copy_external_ids: dict[str, tuple[str, str]] = {}
    copy_linked_kernel_counts: Counter[tuple[str, str, str]] = Counter()
    copy_linked_kernel_duration: defaultdict[tuple[str, str, str], float] = defaultdict(float)
    device_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    device_counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    device_duration: dict[str, defaultdict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
    device_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    device_family_duration: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    device_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    device_category_duration: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    device_phase_counts: dict[str, Counter[str]] = defaultdict(Counter)
    device_phase_duration: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    first_device_seen_event_index: int | None = None
    event_index = 0

    with trace.open("rb") as handle:
        for event in ijson.items(handle, "traceEvents.item"):
            event_index += 1
            if not isinstance(event, dict) or str(event.get("ph", "")) != "X":
                continue
            ts = number(event.get("ts"))
            dur = number(event.get("dur"))
            if ts is None or dur is None or dur < 0:
                continue
            marker = locate(ts)
            if marker is None:
                continue
            window_id = str(marker["window_id"])
            if window_id not in selected_windows:
                continue
            category = str(event.get("cat", ""))
            name = str(event.get("name", ""))
            raw_args = event.get("args")
            event_args = raw_args if isinstance(raw_args, dict) else {}

            if category == "user_annotation":
                annotations[window_id][name] += 1
                if name.startswith("Optimizer.step#"):
                    start = ts
                    end = ts + dur
                    old_start = boundaries[window_id]["optimizer_start_us"]
                    old_end = boundaries[window_id]["optimizer_end_us"]
                    boundaries[window_id]["optimizer_start_us"] = start if old_start is None else min(float(old_start), start)
                    boundaries[window_id]["optimizer_end_us"] = end if old_end is None else max(float(old_end), end)
                continue
            if category == "cpu_op":
                cpu_counts[window_id][name] += 1
                cpu_duration[window_id][name] += dur
                if name.startswith(AUTOGRAD_PREFIX):
                    old = boundaries[window_id]["backward_start_us"]
                    boundaries[window_id]["backward_start_us"] = ts if old is None else min(float(old), ts)
                if name == "aten::copy_":
                    sig = signature(event_args)
                    copy_signature_counts[window_id][sig] += 1
                    copy_signature_duration[window_id][sig] += dur
                    external_id = event_args.get("External id")
                    if external_id is not None:
                        copy_external_ids[str(external_id)] = (window_id, sig)
                continue
            if category == "cuda_runtime":
                runtime_counts[window_id][name] += 1
                runtime_duration[window_id][name] += dur
                continue
            if category not in DEVICE_CATEGORIES:
                continue

            if first_device_seen_event_index is None:
                first_device_seen_event_index = event_index
            end = min(float(marker["end_us"]), ts + dur)
            start = max(float(marker["start_us"]), ts)
            if end <= start:
                continue
            effective_duration = end - start
            device_intervals[window_id].append((start, end))
            key = (category, name)
            device_counts[window_id][key] += 1
            device_duration[window_id][key] += effective_duration
            fam = family(name, category)
            device_family_counts[window_id][fam] += 1
            device_family_duration[window_id][fam] += effective_duration
            device_category_counts[window_id][category] += 1
            device_category_duration[window_id][category] += effective_duration
            bound = boundaries[window_id]
            backward_start = bound["backward_start_us"]
            optimizer_start = bound["optimizer_start_us"]
            optimizer_end = bound["optimizer_end_us"]
            if backward_start is None or optimizer_start is None or optimizer_end is None:
                phase = "unknown_missing_host_boundary"
            elif ts < float(backward_start):
                phase = "forward_plus_loss"
            elif ts < float(optimizer_start):
                phase = "backward"
            elif ts < float(optimizer_end):
                phase = "optimizer"
            else:
                phase = "tail_after_optimizer"
            device_phase_counts[window_id][phase] += 1
            device_phase_duration[window_id][phase] += effective_duration
            external_id = event_args.get("External id")
            linked = copy_external_ids.get(str(external_id)) if external_id is not None else None
            if linked is not None and linked[0] == window_id:
                link_key = (window_id, linked[1], name)
                copy_linked_kernel_counts[link_key] += 1
                copy_linked_kernel_duration[link_key] += effective_duration

    per_step: list[dict[str, Any]] = []
    for marker in selected_windows.values():
        window_id = str(marker["window_id"])
        intervals = device_intervals[window_id]
        merged = merge_intervals(intervals)
        busy = sum(end - start for start, end in merged)
        first = min((start for start, _ in intervals), default=None)
        last = max((end for _, end in intervals), default=None)
        service = float(marker["service_us"])
        step_id = int(marker["step_id"])
        bound = boundaries[window_id]
        backward_start = bound["backward_start_us"]
        optimizer_start = bound["optimizer_start_us"]
        optimizer_end = bound["optimizer_end_us"]
        per_step.append(
            {
                "step_id": step_id,
                "window_id": window_id,
                "group": "stable_normal" if step_id in stable_steps else "soap_periodic",
                "service_ms": service / 1000.0,
                "device_wall_ms": ((last - first) / 1000.0 if first is not None and last is not None else 0.0),
                "device_busy_union_ms": busy / 1000.0,
                "device_kernel_sum_ms": sum(end - start for start, end in intervals) / 1000.0,
                "device_total_cost_ms": None,
                "gpu_wait_metric": "N/A: Chrome trace has no Ascend Wait Time(us) semantic",
                "underfeed_ms": (service - busy) / 1000.0,
                "underfeed_ratio": ((service - busy) / service if service else None),
                "prelaunch_gap_ms": ((first - float(marker["start_us"])) / 1000.0 if first is not None else None),
                "tail_gap_ms": ((float(marker["end_us"]) - last) / 1000.0 if last is not None else None),
                "backward_start_offset_ms": ((float(backward_start) - float(marker["start_us"])) / 1000.0 if backward_start is not None else None),
                "optimizer_start_offset_ms": ((float(optimizer_start) - float(marker["start_us"])) / 1000.0 if optimizer_start is not None else None),
                "optimizer_end_offset_ms": ((float(optimizer_end) - float(marker["start_us"])) / 1000.0 if optimizer_end is not None else None),
                "device_category_duration_ms": {key: value / 1000.0 for key, value in device_category_duration[window_id].items()},
                "device_category_counts": dict(device_category_counts[window_id]),
                "device_phase_duration_ms": {key: value / 1000.0 for key, value in device_phase_duration[window_id].items()},
                "device_phase_counts": dict(device_phase_counts[window_id]),
                "cuda_runtime_sync_ms": sum(
                    value for name, value in runtime_duration[window_id].items() if "synchron" in name.lower()
                ) / 1000.0,
                "cuda_runtime_memcpy_async_ms": runtime_duration[window_id].get("cudaMemcpyAsync", 0.0) / 1000.0,
                "user_annotations": dict(annotations[window_id]),
            }
        )
    per_step.sort(key=lambda row: (row["group"], row["step_id"]))

    stable_window_ids = [str(row["window_id"]) for row in selected_windows.values() if int(row["step_id"]) in stable_steps]
    soap_window_ids = [str(row["window_id"]) for row in selected_windows.values() if int(row["step_id"]) in soap_steps]

    def aggregate_exact(window_ids: list[str]) -> list[dict[str, Any]]:
        counts: Counter[tuple[str, str]] = Counter()
        durations: defaultdict[tuple[str, str], float] = defaultdict(float)
        for window_id in window_ids:
            counts.update(device_counts[window_id])
            for key, value in device_duration[window_id].items():
                durations[key] += value
        rows = []
        denominator = len(window_ids)
        for key, duration in sorted(durations.items(), key=lambda item: item[1], reverse=True)[: args.top]:
            rows.append(
                {
                    "category": key[0],
                    "name": key[1],
                    "count_total": counts[key],
                    "count_per_step": counts[key] / denominator,
                    "duration_ms_total": duration / 1000.0,
                    "duration_ms_per_step": duration / 1000.0 / denominator,
                    "family": family(key[1], key[0]),
                }
            )
        return rows

    def aggregate_family(window_ids: list[str]) -> list[dict[str, Any]]:
        counts: Counter[str] = Counter()
        durations: defaultdict[str, float] = defaultdict(float)
        for window_id in window_ids:
            counts.update(device_family_counts[window_id])
            for key, value in device_family_duration[window_id].items():
                durations[key] += value
        denominator = len(window_ids)
        return [
            {
                "family": key,
                "count_total": counts[key],
                "count_per_step": counts[key] / denominator,
                "duration_ms_total": value / 1000.0,
                "duration_ms_per_step": value / 1000.0 / denominator,
            }
            for key, value in sorted(durations.items(), key=lambda item: item[1], reverse=True)
        ]

    all_signatures = set().union(*(copy_signature_counts[window_id] for window_id in stable_window_ids))
    copy_signatures = []
    for sig in all_signatures:
        counts = [copy_signature_counts[window_id][sig] for window_id in stable_window_ids]
        durations = [copy_signature_duration[window_id][sig] / 1000.0 for window_id in stable_window_ids]
        linked_counts: Counter[str] = Counter()
        linked_duration: defaultdict[str, float] = defaultdict(float)
        for window_id in stable_window_ids:
            for (candidate_window, candidate_sig, kernel_name), count in copy_linked_kernel_counts.items():
                if candidate_window == window_id and candidate_sig == sig:
                    linked_counts[kernel_name] += count
                    linked_duration[kernel_name] += copy_linked_kernel_duration[(candidate_window, candidate_sig, kernel_name)]
        copy_signatures.append(
            {
                "signature": json.loads(sig),
                "count_per_step": counts,
                "count_median": statistics.median(counts),
                "host_duration_ms_per_step": durations,
                "host_duration_ms_median": statistics.median(durations),
                "linked_device_count_total": sum(linked_counts.values()),
                "linked_device_duration_ms_total": sum(linked_duration.values()) / 1000.0,
                "linked_top_kernels": [
                    {
                        "name": name,
                        "count": linked_counts[name],
                        "duration_ms": duration / 1000.0,
                    }
                    for name, duration in sorted(linked_duration.items(), key=lambda item: item[1], reverse=True)[:20]
                ],
            }
        )
    copy_signatures.sort(key=lambda row: (row["count_median"], row["host_duration_ms_median"]), reverse=True)

    def group_clock_summary(group: str) -> dict[str, Any]:
        rows = [row for row in per_step if row["group"] == group]
        result: dict[str, Any] = {"step_count": len(rows), "member_steps": [row["step_id"] for row in rows]}
        for key in ("service_ms", "device_wall_ms", "device_busy_union_ms", "device_kernel_sum_ms", "underfeed_ms", "underfeed_ratio", "cuda_runtime_sync_ms", "cuda_runtime_memcpy_async_ms"):
            values = [float(row[key]) for row in rows if row[key] is not None]
            result[key] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p95": percentile(values, 0.95),
                "min": min(values),
                "max": max(values),
            }
        result["device_total_cost_ms"] = None
        result["gpu_wait_metric"] = "N/A"
        return result

    payload = {
        "trace": {"bytes": trace.stat().st_size, "sha256": inventory["trace"]["sha256"]},
        "inventory_sha256": sha256_file(inventory_path),
        "parser": {"ijson_backend": ijson.backend, "streaming": True},
        "capture_contract": {
            "profiler_step_window_count": len(markers),
            "unique_step_ids": sorted({int(row["step_id"]) for row in markers}),
            "duplicate_step_ids": {str(key): value for key, value in occurrences.items() if value > 1},
            "with_stack": False,
            "record_shapes_observed": inventory["shape_event_count"] > 0,
            "stable_selection_reason": "tail complete normal template after cold-start; periodic QR/SOAP windows excluded",
            "stable_steps": sorted(stable_steps),
            "soap_steps": sorted(soap_steps),
            "gpu_total_cost_wait": "N/A; do not compare against Ascend kernel Wait Time(us)",
        },
        "per_step_four_clock": per_step,
        "group_clock_summary": {
            "stable_normal": group_clock_summary("stable_normal"),
            "soap_periodic": group_clock_summary("soap_periodic"),
        },
        "stable_top_device_events": aggregate_exact(stable_window_ids),
        "soap_top_device_events": aggregate_exact(soap_window_ids),
        "stable_semantic_families": aggregate_family(stable_window_ids),
        "soap_semantic_families": aggregate_family(soap_window_ids),
        "stable_copy_signatures": copy_signatures[:200],
        "limitations": [
            "GPU trace has no Python call stack; source-line attribution is unavailable.",
            "Forward and loss have no authoritative boundary marker and remain one pre-backward region.",
            "Host cpu_op and cuda_runtime durations are nested and must not be arithmetically added as wall time.",
            "ProfilerStep service is instrumented; profiler-off logs remain authoritative for throughput.",
            "GPU Chrome trace has no Ascend Wait Time(us); total_cost/wait comparison is N/A.",
        ],
        "first_device_seen_event_index": first_device_seen_event_index,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "stable_steps": sorted(stable_steps),
        "soap_steps": sorted(soap_steps),
        "stable_service_median_ms": payload["group_clock_summary"]["stable_normal"]["service_ms"]["median"],
        "stable_busy_median_ms": payload["group_clock_summary"]["stable_normal"]["device_busy_union_ms"]["median"],
        "copy_signature_count": len(copy_signatures),
        "output_sha256": sha256_file(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
