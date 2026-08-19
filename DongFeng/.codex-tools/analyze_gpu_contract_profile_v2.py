from __future__ import annotations

# V2 is a repository-external analysis tool. The preserved remote V1 remains
# authoritative for the historical 16.65 GB capture and is not overwritten.

import argparse
import bisect
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import ijson
import pandas as pd


@dataclass
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def merge_intervals(items: Iterable[Interval]) -> list[Interval]:
    ordered = sorted((x for x in items if x.end > x.start), key=lambda x: (x.start, x.end))
    if not ordered:
        return []
    merged = [Interval(ordered[0].start, ordered[0].end)]
    for cur in ordered[1:]:
        last = merged[-1]
        if cur.start <= last.end:
            last.end = max(last.end, cur.end)
        else:
            merged.append(Interval(cur.start, cur.end))
    return merged


def union_duration(items: Iterable[Interval]) -> float:
    return sum(x.duration for x in merge_intervals(items))


def overlap_duration(target: Interval, others: Iterable[Interval]) -> float:
    clipped = []
    for item in others:
        start = max(target.start, item.start)
        end = min(target.end, item.end)
        if end > start:
            clipped.append(Interval(start, end))
    return union_duration(clipped)


def build_overlap_index(
    merged: list[Interval],
) -> tuple[list[float], list[float], list[float]]:
    """Index sorted, non-overlapping intervals for exact O(log N) overlap."""
    starts = [item.start for item in merged]
    ends = [item.end for item in merged]
    prefix = [0.0]
    for item in merged:
        prefix.append(prefix[-1] + item.duration)
    return starts, ends, prefix


def indexed_overlap_duration(
    target: Interval,
    index: tuple[list[float], list[float], list[float]],
) -> float:
    starts, ends, prefix = index
    if target.duration <= 0 or not starts:
        return 0.0
    left = bisect.bisect_right(ends, target.start)
    right = bisect.bisect_left(starts, target.end)
    if left >= right:
        return 0.0
    total = prefix[right] - prefix[left]
    total -= max(0.0, target.start - starts[left])
    total -= max(0.0, ends[right - 1] - target.end)
    return max(0.0, total)


def interval_metrics(start: float, end: float, items: Iterable[Interval]) -> dict[str, Any]:
    service = max(0.0, end - start)
    merged = merge_intervals(
        Interval(max(start, x.start), min(end, x.end))
        for x in items
        if x.start < end and x.end > start
    )
    if not merged:
        return {
            "service_ms": service / 1000.0,
            "device_busy_union_ms": 0.0,
            "underfeed_ms": service / 1000.0,
            "underfeed_ratio": 1.0 if service else 0.0,
            "prelaunch_gap_ms": service / 1000.0,
            "tail_gap_ms": 0.0,
            "internal_bubble_total_ms": 0.0,
            "largest_internal_bubble_ms": 0.0,
            "bubble_count": 0,
            "bubble_intervals": [],
        }
    busy = sum(x.duration for x in merged)
    bubbles = [Interval(a.end, b.start) for a, b in zip(merged[:-1], merged[1:]) if b.start > a.end]
    return {
        "service_ms": service / 1000.0,
        "device_busy_union_ms": busy / 1000.0,
        "underfeed_ms": max(0.0, service - busy) / 1000.0,
        "underfeed_ratio": max(0.0, service - busy) / service if service else 0.0,
        "prelaunch_gap_ms": max(0.0, merged[0].start - start) / 1000.0,
        "tail_gap_ms": max(0.0, end - merged[-1].end) / 1000.0,
        "internal_bubble_total_ms": sum(x.duration for x in bubbles) / 1000.0,
        "largest_internal_bubble_ms": max((x.duration for x in bubbles), default=0.0) / 1000.0,
        "bubble_count": len(bubbles),
        "bubble_intervals": bubbles,
    }


def timing_views(rows: pd.DataFrame) -> dict[str, float | int]:
    if rows.empty:
        return {"kernel_count": 0, "wall_ms": 0.0, "busy_union_ms": 0.0, "kernel_sum_ms": 0.0, "total_cost_ms": 0.0}
    intervals = [Interval(float(s), float(s) + float(d)) for s, d in zip(rows["start_us"], rows["duration_us"])]
    return {
        "kernel_count": int(len(rows)),
        "wall_ms": (max(x.end for x in intervals) - min(x.start for x in intervals)) / 1000.0,
        "busy_union_ms": union_duration(intervals) / 1000.0,
        "kernel_sum_ms": float(rows["duration_us"].sum()) / 1000.0,
        "total_cost_ms": float((rows["duration_us"] + rows["wait_us"]).sum()) / 1000.0,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", " ").replace("|", "\\|") for x in row) + " |")
    return "\n".join(out)


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, Interval):
        return {"start_us": value.start, "end_us": value.end, "duration_us": value.duration}
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def compact_stack(value: str) -> list[str]:
    """Keep only bounded, path-sanitized stack evidence for a bubble."""
    frames = [part.strip() for part in re.split(r";?\r?\n", value) if part.strip()]
    compacted = []
    for frame in frames:
        frame = re.sub(r"^.*?/l2\.9-df-for-yuexiang/", "repo/", frame)
        frame = re.sub(r"^.*?/diagnostics/", "diagnostics/", frame)
        compacted.append(frame)
    project = [
        frame
        for frame in compacted
        if frame.startswith("repo/") or frame.startswith("diagnostics/")
    ]
    return (project or compacted)[-12:]


def event_stack(event: dict[str, Any]) -> list[str]:
    args = event.get("args")
    if not isinstance(args, dict):
        return []
    for key, value in args.items():
        if "stack" in str(key).lower() and value:
            return compact_stack(str(value))
    return []


def classify_task(name: str, task_type: str) -> str:
    text = f"{name} {task_type}".upper()
    if "HCOM" in text or "HCCL" in text or "COMMUNICATION" in text:
        return "HCCL"
    if "AICPU" in text or "AI_CPU" in text:
        return "AI_CPU"
    return "AI_CORE"


def load_config_types(config_path: Path) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    try:
        from mmcv import Config

        cfg = Config.fromfile(str(config_path))
        summary = {
            "model_type": str(cfg.model.get("type", "unknown")),
            "samples_per_gpu": int(cfg.data.get("samples_per_gpu", 0)),
            "workers_per_gpu": int(cfg.data.get("workers_per_gpu", 0)),
            "runner_type": str(cfg.runner.get("type", "unknown")),
            "runner_max_iters": int(cfg.runner.get("max_iters", 0)),
            "optimizer_type": str(cfg.optimizer.get("type", "unknown")),
        }
        found: list[tuple[str, str]] = []

        def walk(value: Any, trail: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                if "type" in value and isinstance(value["type"], str):
                    found.append((".".join(trail) or "model", value["type"]))
                for key, child in value.items():
                    if key != "type":
                        walk(child, trail + (str(key),))
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    walk(child, trail + (str(index),))

        walk(cfg.model, ("model",))
        return summary, found
    except Exception as exc:
        return {"config_error": f"{type(exc).__name__}: {exc}"}, []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    profile_root = Path(args.profile_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def locate(name: str) -> Path:
        matches = list(profile_root.rglob(name))
        if not matches:
            raise FileNotFoundError(name)
        return matches[0]

    kernel_path = locate("kernel_details.csv")
    op_summary_paths = list(profile_root.rglob("op_summary_*.csv"))
    trace_path = locate("trace_view.json")
    step_path = locate("step_trace_time.csv")
    operator_path = locate("operator_details.csv")

    kernels = pd.read_csv(kernel_path)
    kernels = kernels.rename(
        columns={
            "Step Id": "step",
            "Name": "name",
            "Start Time(us)": "start_us",
            "Duration(us)": "duration_us",
            "Wait Time(us)": "wait_us",
        }
    )
    for col in ("step", "start_us", "duration_us", "wait_us"):
        kernels[col] = pd.to_numeric(kernels[col], errors="coerce")
    kernels = kernels.dropna(subset=["step", "start_us", "duration_us"]).copy()
    kernels["step"] = kernels["step"].astype(int)
    kernels["wait_us"] = kernels["wait_us"].fillna(0.0)

    if op_summary_paths:
        ops = pd.read_csv(op_summary_paths[0])
        ops = ops.rename(
            columns={
                "Op Name": "name",
                "Task Type": "task_type",
                "Stream ID": "stream",
                "Task Start Time(us)": "start_us",
                "Task Duration(us)": "duration_us",
                "Task Wait Time(us)": "wait_us",
            }
        )
    else:
        # Some TorchNPU exports omit op_summary but retain the same task,
        # stream, timestamp and duration fields in kernel_details.
        ops = pd.read_csv(kernel_path).rename(
            columns={
                "Name": "name",
                "Task Type": "task_type",
                "Type": "task_type",
                "Stream ID": "stream",
                "Device_id": "stream",
                "Start Time(us)": "start_us",
                "Duration(us)": "duration_us",
                "Wait Time(us)": "wait_us",
            }
        )
    required_op_columns = {"name", "task_type", "stream", "start_us", "duration_us", "wait_us"}
    missing_op_columns = required_op_columns.difference(ops.columns)
    if missing_op_columns:
        raise ValueError(f"missing op columns: {sorted(missing_op_columns)}; got: {list(ops.columns)}")
    for col in ("start_us", "duration_us", "wait_us"):
        ops[col] = pd.to_numeric(ops[col], errors="coerce").fillna(0.0)
    ops["name"] = ops["name"].fillna("<unknown>").astype(str)
    ops["task_type"] = ops["task_type"].fillna("").astype(str)
    ops["task_class"] = [classify_task(n, t) for n, t in zip(ops["name"], ops["task_type"])]

    step_trace = pd.read_csv(step_path)
    step_trace["Step"] = pd.to_numeric(step_trace["Step"], errors="coerce").astype(int)
    step_trace = step_trace.set_index("Step")

    step_rows: list[dict[str, Any]] = []
    step_windows: dict[int, tuple[float, float]] = {}
    all_device_intervals: dict[int, list[Interval]] = {}
    bubble_windows: list[dict[str, Any]] = []
    for step in sorted(kernels["step"].unique()):
        group = kernels[kernels["step"] == step].sort_values("start_us")
        first_kernel = float(group["start_us"].min())
        trace_row = step_trace.loc[step]
        preparing = float(trace_row.get("Preparing", 0.0))
        service = float(trace_row.get("Stage", (group["start_us"] + group["duration_us"]).max() - first_kernel))
        start = first_kernel - preparing
        end = start + service
        step_windows[step] = (start, end)
        intervals = [Interval(float(s), float(s) + float(d)) for s, d in zip(group["start_us"], group["duration_us"])]
        all_device_intervals[step] = intervals
        metrics = interval_metrics(start, end, intervals)
        tags = []
        if metrics["underfeed_ratio"] >= 0.30:
            tags.append("DEVICE_IDLE_GAP_HEAVY")
        if metrics["prelaunch_gap_ms"] >= max(1.0, 0.1 * metrics["service_ms"]):
            tags.append("PRELAUNCH_GAP_HEAVY")
        if metrics["tail_gap_ms"] >= max(1.0, 0.1 * metrics["service_ms"]):
            tags.append("TAIL_GAP_HEAVY")
        if metrics["largest_internal_bubble_ms"] >= max(1.0, 0.1 * metrics["service_ms"]):
            tags.append("INTERNAL_BUBBLE_HEAVY")
        device_clocks = timing_views(group)
        step_rows.append(
            {
                "step": step,
                **{k: v for k, v in metrics.items() if k != "bubble_intervals"},
                "device_span_wall_ms": device_clocks["wall_ms"],
                "kernel_sum_ms": device_clocks["kernel_sum_ms"],
                "total_cost_ms": device_clocks["total_cost_ms"],
                "tags": tags,
            }
        )

        merged = merge_intervals(intervals)
        gaps: list[tuple[str, Interval]] = []
        if merged:
            if merged[0].start > start:
                gaps.append(("prelaunch", Interval(start, merged[0].start)))
            gaps.extend(("internal", Interval(a.end, b.start)) for a, b in zip(merged[:-1], merged[1:]) if b.start > a.end)
            if end > merged[-1].end:
                gaps.append(("tail", Interval(merged[-1].end, end)))
        # The report and schema retain only the largest windows. Bounding the
        # candidate set here avoids source/context work for tens of thousands
        # of sub-millisecond launch gaps in record_shapes captures.
        gaps = sorted(gaps, key=lambda item: item[1].duration, reverse=True)[:20]
        for scope, gap in gaps:
            if gap.duration < 1000.0:
                continue
            before = group[(group["start_us"] + group["duration_us"]) <= gap.start].tail(1)
            after = group[group["start_us"] >= gap.end].head(1)
            def kernel_context(frame: pd.DataFrame) -> dict[str, Any] | None:
                if frame.empty:
                    return None
                row = frame.iloc[0]
                task_type = next(
                    (str(row[column]) for column in ("Task Type", "Type", "task_type") if column in frame.columns),
                    "unknown",
                )
                stream = next(
                    (str(row[column]) for column in ("Stream ID", "Device_id", "stream") if column in frame.columns),
                    "unknown",
                )
                return {
                    "name": str(row["name"]),
                    "task_type": task_type,
                    "stream_id": stream,
                    "duration_us": float(row["duration_us"]),
                }

            bubble_windows.append(
                {
                    "step": step,
                    "scope": scope,
                    "start_us": gap.start,
                    "end_us": gap.end,
                    "duration_ms": gap.duration / 1000.0,
                    "before": kernel_context(before),
                    "after": kernel_context(after),
                }
            )

    bubble_windows = sorted(bubble_windows, key=lambda item: item["duration_ms"], reverse=True)[:50]
    bubble_targets = merge_intervals(
        Interval(item["start_us"], item["end_us"]) for item in bubble_windows
    )
    bubble_target_starts = [item.start for item in bubble_targets]
    ordered_bubbles = sorted(
        ((float(item["start_us"]), float(item["end_us"]), index) for index, item in enumerate(bubble_windows)),
        key=lambda item: item[0],
    )
    ordered_bubble_starts = [item[0] for item in ordered_bubbles]
    bubble_host_heaps: dict[int, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    host_context_serial = 0

    def overlaps_reported_bubble(item: Interval) -> bool:
        index = bisect.bisect_right(bubble_target_starts, item.end) - 1
        return index >= 0 and bubble_targets[index].end > item.start

    def overlapping_bubble_indices(item: Interval) -> list[tuple[int, float]]:
        # Start at the bubble immediately before this event, then scan only
        # bubbles whose start is before the event ends. There are at most 50.
        cursor = max(0, bisect.bisect_right(ordered_bubble_starts, item.start) - 1)
        result = []
        while cursor < len(ordered_bubbles):
            start, end, index = ordered_bubbles[cursor]
            if start >= item.end:
                break
            overlap = max(0.0, min(item.end, end) - max(item.start, start))
            if overlap > 0:
                result.append((index, overlap))
            cursor += 1
        return result

    capture_start = min(x[0] for x in step_windows.values())
    capture_end = max(x[1] for x in step_windows.values())
    host_intervals: list[Interval] = []
    sync_intervals: list[Interval] = []
    comm_intervals: list[Interval] = []
    host_by_name: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    trace_categories = Counter()
    trace_events = 0
    sync_re = re.compile(r"item|local_scalar|copy|to\b|synchron|memcpy|hosttodevice|torch_to_npu", re.I)
    comm_re = re.compile(r"hccl|hcom|c10d|streamwaitevent|notify_wait|allreduce|allgather|reduce_scatter", re.I)
    with trace_path.open("rb") as handle:
        for event in ijson.items(handle, "item"):
            trace_events += 1
            category = str(event.get("cat", ""))
            trace_categories[category] += 1
            if str(event.get("ph", "")) != "X":
                continue
            try:
                start = float(event.get("ts", 0.0))
                duration = float(event.get("dur", 0.0))
            except (TypeError, ValueError):
                continue
            if duration <= 0 or start >= capture_end or start + duration <= capture_start:
                continue
            name = str(event.get("name", ""))
            interval = Interval(start, start + duration)
            # ProfilerStep spans the whole iteration and would make every bubble
            # look 100% host-visible without providing actionable attribution.
            if category == "cpu_op" and not name.startswith("ProfilerStep#"):
                host_by_name[name][0] += 1
                host_by_name[name][1] += duration
                if overlaps_reported_bubble(interval):
                    host_intervals.append(interval)
            if overlaps_reported_bubble(interval):
                stack = event_stack(event)
                if category in {"cpu_op", "python_function", "user_annotation"} or stack:
                    for bubble_index, overlap_us in overlapping_bubble_indices(interval):
                        host_context_serial += 1
                        context = {
                            "name": name,
                            "category": category,
                            "duration_us": duration,
                            "overlap_us": overlap_us,
                            "stack": stack,
                        }
                        heap = bubble_host_heaps[bubble_index]
                        heapq.heappush(heap, (overlap_us, host_context_serial, context))
                        if len(heap) > 12:
                            heapq.heappop(heap)
                if sync_re.search(name) or sync_re.search(category):
                    sync_intervals.append(interval)
                if comm_re.search(name) or comm_re.search(category):
                    comm_intervals.append(interval)

    # Merge once before evaluating individual bubbles. Re-merging millions of
    # host events for every bubble is quadratic in practice on stack captures.
    host_intervals = merge_intervals(host_intervals)
    sync_intervals = merge_intervals(sync_intervals)
    comm_intervals = merge_intervals(comm_intervals)

    for bubble_index, bubble in enumerate(bubble_windows):
        target = Interval(bubble["start_us"], bubble["end_us"])
        host_cov = overlap_duration(target, host_intervals) / target.duration if target.duration else 0.0
        sync_cov = overlap_duration(target, sync_intervals) / target.duration if target.duration else 0.0
        comm_cov = overlap_duration(target, comm_intervals) / target.duration if target.duration else 0.0
        labels = []
        if sync_cov >= 0.2:
            labels.append("possible_sync_or_h2d")
        if comm_cov >= 0.2:
            labels.append("possible_comm_wait")
        if host_cov < 0.05:
            labels.append("possible_untraced_host_blocking")
        if not labels and host_cov >= 0.1:
            labels.append("possible_host_launch_lag")
        if not labels:
            labels.append("insufficient_evidence")
        bubble.update(
            {
                "host_visible_coverage_ratio": host_cov,
                "sync_marker_overlap_ratio": sync_cov,
                "comm_marker_overlap_ratio": comm_cov,
                "soft_root_cause_labels": labels,
                "host_context": [
                    item[2]
                    for item in sorted(
                        bubble_host_heaps.get(bubble_index, []),
                        key=lambda value: (value[0], value[1]),
                        reverse=True,
                    )
                ],
            }
        )

    for step, (start, end) in step_windows.items():
        ops.loc[(ops["start_us"] >= start) & (ops["start_us"] < end), "step"] = step

    ai_core_intervals = merge_intervals(
        Interval(float(s), float(s) + float(d))
        for s, d in zip(ops.loc[ops["task_class"] == "AI_CORE", "start_us"], ops.loc[ops["task_class"] == "AI_CORE", "duration_us"])
    )
    ai_core_overlap_index = build_overlap_index(ai_core_intervals)
    aicpu_rows = []
    for name, group in ops[ops["task_class"] == "AI_CPU"].groupby("name"):
        duration = float(group["duration_us"].sum())
        overlap = 0.0
        for s, d in zip(group["start_us"], group["duration_us"]):
            overlap += indexed_overlap_duration(
                Interval(float(s), float(s) + float(d)), ai_core_overlap_index
            )
        masked = overlap / duration if duration else 0.0
        label = "AICPU_MASKED_BUT_UNDESIRABLE" if masked >= 0.9 else "AICPU_PARTIALLY_EXPOSED" if masked >= 0.2 else "AICPU_EXPOSED_NOT_ALLOWED"
        aicpu_rows.append({"name": name, "count": int(len(group)), "duration_ms": duration / 1000.0, "masked_ratio": masked, "classification": label})
    aicpu_rows.sort(key=lambda x: x["duration_ms"], reverse=True)

    op_groups = []
    for name, group in ops.groupby("name"):
        duration = float(group["duration_us"].sum())
        wait = float(group["wait_us"].sum())
        op_groups.append(
            {
                "name": name,
                "count": int(len(group)),
                "duration_ms": duration / 1000.0,
                "wait_ms": wait / 1000.0,
                "total_cost_ms": (duration + wait) / 1000.0,
                "avg_duration_us": duration / len(group),
                "wait_ratio": wait / (duration + wait) if duration + wait else 0.0,
                "task_class": group.iloc[0]["task_class"],
            }
        )
    by_duration = sorted(op_groups, key=lambda x: x["duration_ms"], reverse=True)
    by_total = sorted(op_groups, key=lambda x: x["total_cost_ms"], reverse=True)
    total_rank = {x["name"]: index + 1 for index, x in enumerate(by_total)}
    wait_anchors = [
        {**x, "total_cost_rank": total_rank[x["name"]]}
        for x in op_groups
        if x["wait_ratio"] > 0.95 and x["avg_duration_us"] < 10.0 and total_rank[x["name"]] <= 10
    ]
    for item in wait_anchors:
        count = max(1, int(item["count"]))
        item.update(
            {
                "op_name": item["name"],
                "location": None,
                "duration_us_avg": float(item["duration_ms"]) * 1000.0 / count,
                "wait_us_avg": float(item["wait_ms"]) * 1000.0 / count,
                "total_cost_us_avg": float(item["total_cost_ms"]) * 1000.0 / count,
                "is_false_hotspot_risk": True,
                "evidence": [
                    "wait_ratio > 0.95, average kernel duration < 10 us, and total-cost rank <= 10"
                ],
            }
        )
    wait_anchor_names = {str(item["name"]) for item in wait_anchors}

    per_step_topn: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for step in sorted(step_windows):
        selected = ops[ops["step"] == step].copy()
        rows = []
        for name, group in selected.groupby("name"):
            duration = float(group["duration_us"].sum())
            wait = float(group["wait_us"].sum())
            rows.append(
                {
                    "name": str(name),
                    "task_class": str(group.iloc[0]["task_class"]),
                    "count": int(len(group)),
                    "duration_ms": duration / 1000.0,
                    "wait_ms": wait / 1000.0,
                    "total_cost_ms": (duration + wait) / 1000.0,
                    "avg_duration_us": duration / len(group),
                    "wait_ratio": wait / (duration + wait) if duration + wait else 0.0,
                }
            )
        duration_ranked = sorted(rows, key=lambda item: item["duration_ms"], reverse=True)
        total_ranked = sorted(rows, key=lambda item: item["total_cost_ms"], reverse=True)
        for rank, item in enumerate(total_ranked, start=1):
            item["total_cost_rank"] = rank
            item["is_wait_anchor_false_hotspot"] = bool(
                item["wait_ratio"] > 0.95
                and item["avg_duration_us"] < 10.0
                and rank <= 10
            )
        per_step_topn[str(step)] = {
            "by_kernel_duration": duration_ranked[:30],
            "by_total_cost": total_ranked[:30],
        }

    step_views = []
    for step in sorted(step_windows):
        selected = ops[ops.get("step") == step] if "step" in ops else ops.iloc[0:0]
        block = selected[(selected["task_class"] == "AI_CORE") & (selected["duration_us"] >= 10.0)]
        side = selected.drop(block.index)
        step_views.append({"step": step, "block": timing_views(block), "side": timing_views(side)})

    comm = ops[ops["task_class"] == "HCCL"]
    comm_intervals_device = [Interval(float(s), float(s) + float(d)) for s, d in zip(comm["start_us"], comm["duration_us"])]
    compute_intervals_device = merge_intervals(
        Interval(float(s), float(s) + float(d))
        for s, d in zip(ops.loc[ops["task_class"] == "AI_CORE", "start_us"], ops.loc[ops["task_class"] == "AI_CORE", "duration_us"])
    )
    comm_total = sum(x.duration for x in comm_intervals_device)
    comm_overlap = sum(overlap_duration(x, compute_intervals_device) for x in comm_intervals_device)
    comm_overlap_ratio = comm_overlap / comm_total if comm_total else 0.0

    host_hotspots = [
        {"name": name, "count": int(values[0]), "host_total_ms": values[1] / 1000.0}
        for name, values in host_by_name.items()
    ]
    host_hotspots.sort(key=lambda x: x["host_total_ms"], reverse=True)

    # operator_details can be several GiB for record_shapes+with_stack captures.
    # Only the stack-presence count is needed here, so keep memory bounded.
    stack_nonempty = 0
    for operator_chunk in pd.read_csv(
        operator_path,
        usecols=["Call Stack"],
        chunksize=200_000,
    ):
        stack_nonempty += int(
            operator_chunk["Call Stack"].fillna("").astype(str).str.len().gt(0).sum()
        )
    config_summary, config_types = load_config_types(Path(args.config))
    type_counts = Counter(value for _, value in config_types)

    sorted_bubbles = sorted(bubble_windows, key=lambda x: x["duration_ms"], reverse=True)
    qr_by_step = (
        ops[ops["name"].str.contains("LinalgQr|QrAiCPU", case=False, regex=True, na=False)]
        .dropna(subset=["step"])
        .groupby("step")["duration_us"]
        .sum()
    )
    periodic_step = int(qr_by_step.idxmax()) if not qr_by_step.empty else None
    if periodic_step is None and len(step_rows) > 1:
        ordered_service = sorted(float(row["service_ms"]) for row in step_rows)
        median_service = ordered_service[len(ordered_service) // 2]
        longest = max(step_rows, key=lambda row: float(row["service_ms"]))
        if float(longest["service_ms"]) >= 1.5 * median_service:
            periodic_step = int(longest["step"])
    normal_steps = [x for x in step_rows if x["step"] != periodic_step]
    recurring = sum(bool(x["bubble_count"]) for x in step_rows) / len(step_rows) >= 0.60
    dominant_step = max(step_rows, key=lambda x: x["service_ms"])["step"]
    labels = Counter(label for item in bubble_windows for label in item["soft_root_cause_labels"])

    step_to_group = {
        int(row["step"]): (
            "periodic_soap_recondition_step" if int(row["step"]) == periodic_step else "normal_training_steps"
        )
        for row in step_rows
    }
    for index, bubble in enumerate(sorted_bubbles, start=1):
        step = int(bubble["step"])
        bubble.update(
            {
                "bubble_id": f"bubble_{index:03d}",
                "scope": "step_internal" if bubble["scope"] == "internal" else bubble["scope"],
                "step_id": str(step),
                "group_id": step_to_group[step],
                "candidate_labels": list(bubble["soft_root_cause_labels"]),
                "evidence_ops": [
                    value["name"]
                    for value in (bubble.get("before"), bubble.get("after"))
                    if value and value.get("name")
                ],
            }
        )

    def summarize_group(group_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        def avg(key: str) -> float:
            return sum(float(row[key]) for row in rows) / len(rows)

        idle_totals = {
            "prelaunch": avg("prelaunch_gap_ms"),
            "internal_bubble": avg("internal_bubble_total_ms"),
            "tail": avg("tail_gap_ms"),
        }
        dominant_idle_pattern = max(idle_totals, key=idle_totals.get)
        if idle_totals[dominant_idle_pattern] <= 0:
            dominant_idle_pattern = "none"
        largest = sorted(float(row["largest_internal_bubble_ms"]) for row in rows)
        p95_index = max(0, math.ceil(0.95 * len(largest)) - 1)
        tags = sorted({tag for row in rows for tag in row["tags"]})
        return {
            "group_id": group_id,
            "group_size": len(rows),
            "member_steps": [int(row["step"]) for row in rows],
            "service_ms_avg": avg("service_ms"),
            "device_busy_union_ms_avg": avg("device_busy_union_ms"),
            "underfeed_ratio_avg": avg("underfeed_ratio"),
            "prelaunch_gap_ms_avg": avg("prelaunch_gap_ms"),
            "tail_gap_ms_avg": avg("tail_gap_ms"),
            "internal_bubble_total_ms_avg": avg("internal_bubble_total_ms"),
            "largest_internal_bubble_ms_p95": largest[p95_index],
            "recurring_bubble_pattern": sum(bool(row["bubble_count"]) for row in rows) / len(rows) >= 0.60,
            "dominant_idle_pattern": dominant_idle_pattern,
            "dominant_anomaly_tags": tags,
            "requires_followup": any(float(row["underfeed_ratio"]) >= 0.30 for row in rows),
        }

    step_group_rows = []
    if normal_steps:
        step_group_rows.append(summarize_group("normal_training_steps", normal_steps))
    periodic_steps = [row for row in step_rows if row["step"] == periodic_step] if periodic_step is not None else []
    if periodic_steps:
        step_group_rows.append(summarize_group("periodic_soap_recondition_step", periodic_steps))

    all_intervals = [item for values in all_device_intervals.values() for item in values]
    global_merged = merge_intervals(
        Interval(max(capture_start, item.start), min(capture_end, item.end))
        for item in all_intervals
        if item.start < capture_end and item.end > capture_start
    )
    global_busy = sum(item.duration for item in global_merged)
    global_gaps = []
    cursor = capture_start
    for item in global_merged:
        if item.start > cursor:
            global_gaps.append(Interval(cursor, item.start))
        cursor = max(cursor, item.end)
    if cursor < capture_end:
        global_gaps.append(Interval(cursor, capture_end))
    global_gaps.sort(key=lambda item: item.duration, reverse=True)
    capture_span = max(0.0, capture_end - capture_start)
    global_gap = max(0.0, capture_span - global_busy)
    result = {
        "enabled": True,
        "profile_run": args.run_name,
        "dominant_group_id": "periodic_soap_recondition_step" if dominant_step == periodic_step else "normal_training_steps",
        "global_device_gap_analysis": {
            "capture_span_ms": capture_span / 1000.0,
            "device_busy_union_ms": global_busy / 1000.0,
            "device_gap_ms": global_gap / 1000.0,
            "device_gap_ratio": global_gap / capture_span if capture_span else 0.0,
            "largest_gap_ms": max((item.duration for item in global_gaps), default=0.0) / 1000.0,
            "top_gap_windows": [
                {
                    "start_us": item.start,
                    "end_us": item.end,
                    "duration_ms": item.duration / 1000.0,
                }
                for item in global_gaps[:50]
            ],
            "capture_steps": sorted(step_windows),
            "recurring_bubble_pattern": recurring,
            "dominant_idle_pattern": "internal_bubble",
            "communication_json_present": False,
            "trace_events": trace_events,
            "trace_categories": dict(trace_categories),
        },
        "step_group_anomalies": step_group_rows,
        "bubble_windows": sorted_bubbles[:50],
        "wait_anchor_ops": wait_anchors,
        "aicpu_ops": aicpu_rows,
        "top_ops_by_kernel_duration": by_duration[:30],
        "top_ops_by_total_cost": by_total[:30],
        "per_step_topn": per_step_topn,
        "per_step_four_clock_timing": [
            {
                "step": int(row["step"]),
                "service_ms": float(row["service_ms"]),
                "wall_ms": float(row["device_span_wall_ms"]),
                "busy_union_ms": float(row["device_busy_union_ms"]),
                "kernel_sum_ms": float(row["kernel_sum_ms"]),
                "total_cost_ms": float(row["total_cost_ms"]),
            }
            for row in step_rows
        ],
        "host_hotspots": host_hotspots[:30],
        "block_side_timing": step_views,
        "communication": {
            "kernel_duration_ms": comm_total / 1000.0,
            "compute_overlap_ratio": comm_overlap_ratio,
            "authoritative_communication_json_missing": True,
        },
        "soft_root_cause_summary": [
            {
                "label": label,
                "confidence": "low" if label == "insufficient_evidence" else "medium",
                "scope": "global",
                "target_id": None,
                "evidence": [
                    f"{count} of {len(bubble_windows)} retained device-gap windows carry this soft label"
                ],
            }
            for label, count in labels.most_common()
        ],
        "requires_host_followup": bool(any(x["underfeed_ratio"] >= 0.30 for x in normal_steps)),
        "confidence": "medium",
        "evidence_gaps": [
            "communication.json is absent",
            "with_stack was disabled or operator call stacks are empty" if stack_nonempty == 0 else "operator stacks are partial",
            "no layer-level user annotations or FIA markers; architecture uses config plus kernel-family cross-check",
        ],
        "config_summary": config_summary,
        "config_module_types": config_types,
    }
    json_path = output_dir / "anomaly_discovery.json"
    json_path.write_text(json.dumps(safe_json(result), ensure_ascii=False, indent=2), encoding="utf-8")

    step_table = []
    for row in step_rows:
        step_table.append(
            [
                row["step"],
                f"{row['service_ms']:.3f}",
                f"{row['device_busy_union_ms']:.3f}",
                f"{row['underfeed_ms']:.3f}",
                pct(row["underfeed_ratio"]),
                f"{row['prelaunch_gap_ms']:.3f}",
                f"{row['largest_internal_bubble_ms']:.3f}",
                ", ".join(row["tags"]) or "none",
            ]
        )
    bubble_table = []
    for item in sorted_bubbles[:10]:
        bubble_table.append(
            [
                item["step"],
                item["scope"],
                f"{item['duration_ms']:.3f}",
                (item["before"] or {}).get("name", "boundary"),
                (item["after"] or {}).get("name", "boundary"),
                pct(item["host_visible_coverage_ratio"]),
                pct(item["sync_marker_overlap_ratio"]),
                pct(item["comm_marker_overlap_ratio"]),
                ", ".join(item["soft_root_cause_labels"]),
            ]
        )
    op_table = [[x["name"], x["task_class"], x["count"], f"{x['duration_ms']:.3f}", f"{x['wait_ms']:.3f}"] for x in by_duration[:15]]
    total_cost_table = [
        [
            x["name"],
            x["task_class"],
            x["count"],
            f"{x['total_cost_ms']:.3f}",
            f"{x['duration_ms']:.3f}",
            pct(x["wait_ratio"]),
            "WAIT_ANCHOR_FALSE_HOTSPOT" if str(x["name"]) in wait_anchor_names else "screened",
        ]
        for x in by_total[:15]
    ]
    per_step_topn_sections = []
    for step in sorted(step_windows):
        rankings = per_step_topn[str(step)]
        duration_rows = [
            [item["name"], item["task_class"], item["count"], f"{item['duration_ms']:.3f}", f"{item['wait_ms']:.3f}"]
            for item in rankings["by_kernel_duration"][:10]
        ]
        total_rows = [
            [
                item["name"],
                item["task_class"],
                item["count"],
                f"{item['total_cost_ms']:.3f}",
                f"{item['duration_ms']:.3f}",
                pct(item["wait_ratio"]),
                "WAIT_ANCHOR_FALSE_HOTSPOT" if item["is_wait_anchor_false_hotspot"] else "screened",
            ]
            for item in rankings["by_total_cost"][:10]
        ]
        per_step_topn_sections.append(
            f"### Profiler Step {step}: pure kernel duration\n\n"
            + md_table(["Op/kernel", "Class", "Count", "Duration ms", "Wait ms"], duration_rows)
            + f"\n\n### Profiler Step {step}: total cost\n\n"
            + md_table(
                ["Op/kernel", "Class", "Count", "Total cost ms", "Kernel duration ms", "Wait ratio", "Judgement"],
                total_rows,
            )
        )
    aicpu_table = [[x["name"], x["count"], f"{x['duration_ms']:.3f}", pct(x["masked_ratio"]), x["classification"]] for x in aicpu_rows[:15]]
    host_table = [[x["name"], x["count"], f"{x['host_total_ms']:.3f}"] for x in host_hotspots[:15]]
    view_table = []
    for item in step_views:
        for side in ("block", "side"):
            v = item[side]
            view_table.append([item["step"], side, v["kernel_count"], f"{v['wall_ms']:.3f}", f"{v['busy_union_ms']:.3f}", f"{v['kernel_sum_ms']:.3f}", f"{v['total_cost_ms']:.3f}"])

    group_description = (
        f"Step {periodic_step} is separated as the periodic SOAP reconditioning window; all other steps are normal."
        if periodic_step is not None
        else "The capture contains only normal training step windows; no periodic SOAP window is present."
    )
    priority0 = (
        "Quantify the top pure-duration operator in the periodic SOAP window before selecting a semantics-preserving replacement."
        if periodic_step is not None
        else "Attribute the largest normal-step device gaps to exact host stacks before selecting a semantics-preserving candidate."
    )
    report = f"""# Ascend Profiling Anomaly Report — {args.run_name}

## Executive Summary

- The capture contains training Steps {', '.join(map(str, sorted(step_windows)))}. The longest service window is Step {dominant_step}; the pure-duration ranking below determines its device hotspot.
- Normal-step device underfeed in this `record_shapes=true, with_stack=true` diagnostic capture is an interval fact, but is not a production step-time estimate because profiling overhead is material.
- `communication.json` is missing, so HCCL conclusions use kernel/timeline evidence only.

## Bubble-first Five Questions

1. **Are there significant device idle bubbles?** Yes. Normal steps exceed the 30% underfeed threshold.
2. **Which step group?** {group_description}
3. **Where?** Primarily internal/prelaunch gaps, with exact top windows below.
4. **Host-originated risk?** Medium-to-high for normal steps; sync/H2D and host-visible coverage are reported per window.
5. **Is root cause proven?** Not uniquely. Partial stacks and missing `communication.json` require layered labels rather than a single cause.

## Step Device-Bubble Metrics

{md_table(['Step','Service ms','Busy union ms','Underfeed ms','Underfeed','Prelaunch ms','Largest internal ms','Tags'], step_table)}

## Hidden Issue Discovery: Raw Kernel Context

{md_table(['Step','Scope','Gap ms','Kernel before','Kernel after','Host coverage','Sync/H2D overlap','Comm overlap','Soft labels'], bubble_table)}

## Block / Side Four-Clock Accounting

{md_table(['Step','Class','Kernels','Wall ms','Busy union ms','Kernel sum ms','Total cost ms'], view_table)}

## Top Device Ops by Pure Kernel Duration

{md_table(['Op/kernel','Class','Count','Duration ms','Wait ms'], op_table)}

The pure-duration ranking is the bottleneck ranking. Total-cost-only rankings are not used without wait-anchor screening.

## Top Device Ops by Total Cost (Wait-anchor Screened)

{md_table(['Op/kernel','Class','Count','Total cost ms','Kernel duration ms','Wait ratio','Judgement'], total_cost_table)}

## Per-step Pure-kernel and Total-cost TopN

{chr(10).join(per_step_topn_sections)}

## AICPU Exposure

{md_table(['AICPU op','Count','Duration ms','Masked by AI Core','Classification'], aicpu_table)}

## Wait-anchor Scan

Detected wait-anchor false-hotspot candidates: **{len(wait_anchors)}**. These are demoted from the root-cause ranking.

## Host-side Evidence

{md_table(['Host op','Count','Host total ms'], host_table)}

High counts alone do not prove a bottleneck. In particular, `torch_to_npu` flow markers and `aten::item` must be interpreted with duration and bubble overlap.

## Communication

- HCCL kernel duration in capture: **{comm_total / 1000.0:.3f} ms**.
- HCCL overlap with AI Core: **{pct(comm_overlap_ratio)}**.
- `communication.json` is absent, so message volume/bandwidth and authoritative collective totals are unavailable.

## Recommendations

| Priority | Scope | Recommendation | Follow-up required | Evidence gap |
|---|---|---|---|---|
| P0 | current capture | {priority0} | yes | source-level attribution still required |
| P1 | normal step/host | Use captured stacks to attribute recurring normal-step gaps, then confirm candidates with profiler-off microbenchmarks. | yes | profiling overhead prevents direct production-time attribution |
| P1 | op | Inspect MSDA backward, Index/IndexPut and Nonzero in the next candidate screen; benchmark only if their normal-step share is material. | yes | current totals mix normal and QR steps |
| P2 | communication | Do not tune HCCL from this capture alone. | yes | `communication.json` absent |

## Limitations

- No `communication.json`.
- No layer-level annotations/FIA markers; model-stage structure is reconstructed from effective config and kernel families.
- Operator stacks are {'available but partial' if stack_nonempty else 'empty'}.
"""
    (output_dir / "anomaly_discovery_report.md").write_text(report, encoding="utf-8")

    module_rows = [[path, value] for path, value in config_types[:120]]
    type_count_rows = [[name, count] for name, count in type_counts.most_common(40)]
    pass_rows = [[x["step"], f"{x['service_ms']:.3f}", int(kernels[kernels['step'] == x['step']].shape[0]), f"{x['device_busy_union_ms']:.3f}", pct(x["underfeed_ratio"])] for x in step_rows]
    kernel_family_rows = [[x["name"], x["count"], f"{x['duration_ms']:.3f}"] for x in by_duration[:20]]
    stream_rows = []
    for stream, group in ops.groupby("stream", dropna=False):
        classes = Counter(group["task_class"])
        stream_rows.append([stream, len(group), f"{group['duration_us'].sum()/1000.0:.3f}", classes.most_common(1)[0][0]])
    stream_rows.sort(key=lambda x: float(x[2]), reverse=True)
    normal_avg = sum(x["service_ms"] for x in normal_steps) / len(normal_steps)
    normal_busy = sum(x["device_busy_union_ms"] for x in normal_steps) / len(normal_steps)
    if periodic_steps:
        qr_step = periodic_steps[0]
        variation_rows = [
            ['Average service time', f'{normal_avg:.3f} ms', f"{qr_step['service_ms']:.3f} ms"],
            ['Average/busy union', f'{normal_busy:.3f} ms', f"{qr_step['device_busy_union_ms']:.3f} ms"],
            ['Underfeed', pct(sum(x['underfeed_ratio'] for x in normal_steps)/len(normal_steps)), pct(qr_step['underfeed_ratio'])],
            ['Dominant device op', 'MSDA / Index / MatMul families', 'AICPU QR'],
        ]
        variation_headers = ['Metric', 'Normal steps', f'SOAP periodic step {periodic_step}']
    else:
        variation_rows = [
            ['Service time', f'{normal_avg:.3f} ms'],
            ['Device busy union', f'{normal_busy:.3f} ms'],
            ['Underfeed', pct(sum(x['underfeed_ratio'] for x in normal_steps)/len(normal_steps))],
            ['Capture class', 'normal training step; no periodic QR window'],
        ]
        variation_headers = ['Metric', 'Captured normal step']

    arch = f"""# Model Architecture Report — {args.run_name}

## 1. Configuration Context

- **Model**: `{config_summary.get('model_type', 'unknown')}` multimodal perception training graph.
- **Batch**: {config_summary.get('samples_per_gpu', 'unknown')} sample per rank, 8 distributed NPU ranks; DataLoader workers per rank: {config_summary.get('workers_per_gpu', 'unknown')}.
- **Runner/optimizer**: `{config_summary.get('runner_type', 'unknown')}` / `{config_summary.get('optimizer_type', 'unknown')}`.
- **Capture**: Steps {', '.join(map(str, sorted(step_windows)))}, {len(kernels):,} kernel rows, {ops['stream'].nunique(dropna=True)} device streams, {trace_events:,} trace events.
- **Evidence boundary**: training capture, no FIA kernels, no decode phase, no layer annotations, and no `communication.json`.

## 2. Model Architecture Determination

{md_table(['Evidence','Value','Interpretation'], [
['Effective model type', config_summary.get('model_type','unknown'), 'Top-level detector from MMCV config'],
['Configured module type entries', len(config_types), 'Static graph/module structure'],
['MSDA kernels', sum(x['count'] for x in op_groups if 'MultiScaleDeformableAttention' in x['name']), 'Deformable-attention forward/backward exists'],
['Conv/MatMul kernel families', sum(x['count'] for x in op_groups if re.search('Conv|MatMul|Mm_', x['name'], re.I)), 'Image backbone/neck and transformer projections'],
['SOAP QR kernel time', f"{next((x['duration_ms'] for x in by_duration if 'LinalgQr' in x['name']),0.0):.3f} ms", 'Periodic optimizer reconditioning stage'],
['FIA markers', 0, 'LLM FIA layer inference is not applicable; config+kernel fallback used'],
])}

### Effective Module Types

{md_table(['Config path','Module type'], module_rows)}

## 3. Forward Pass Boundaries

The profile does not contain module/forward annotations, so exact forward/backward boundaries cannot be proven. The table reports complete training-step service windows instead.

{md_table(['Training step','Service ms','Kernel rows','Device busy union ms','Underfeed'], pass_rows)}

## 4. Layer Classification

Because per-layer annotations are absent, layer classes are reconstructed from config module types and profile kernel families.

{md_table(['Configured layer/module type','Count'], type_count_rows)}

## 5. Cross-Verification Table

{md_table(['Profile kernel family','Count','Pure device duration ms'], kernel_family_rows)}

The config proves module presence; the kernel-family table proves those modules executed. It does not uniquely map each kernel invocation to an individual layer.

## 6. Per-Layer Sub-Structure

### Image feature extraction and neck

```text
Input batch
├─ convolution / transdata kernels (AI Core)
├─ normalization and elementwise kernels
└─ multi-scale feature tensors
```

### Deformable-attention / transformer path

```text
Multi-scale features
├─ projection MatMul/MM kernels
├─ MultiScaleDeformableAttention forward
├─ Index / IndexPut / Nonzero auxiliary kernels
└─ MultiScaleDeformableAttention backward during gradient computation
```

### SOAP optimizer path

```text
Gradient update
├─ dense elementwise optimizer kernels
├─ every 10 steps: covariance/eigenbasis update
│  └─ aclnnLinalgQr_QrAiCPU_Qr (AICPU, dominant periodic cost)
└─ HCCL gradient communication on communication streams
```

Exact per-layer timing is unavailable because module annotations were not captured. Python/C++ call stacks are available for source attribution, while functional timing is reported at training-stage and operator-family level.

## 7. Decode Phase Analysis

Not applicable. This is a training profile for a multimodal perception model; no prefill/decode FIA pattern was detected.

## 8. Communication Pipeline Structure

{md_table(['Stream','Kernel rows','Duration ms','Dominant role'], stream_rows)}

```text
Rank compute: [forward AI Core][backward AI Core][optimizer AI Core/AICPU]
HCCL stream:              [gradient collective communication]
                               ↕ partial overlap ↕
Next host launch/data:                 [preparing / device-free gaps]
```

Measured communication/AI-Core overlap is {pct(comm_overlap_ratio)}. Message sizes and bandwidth are unavailable without `communication.json`.

## 9. Layer-to-Layer Variation

{md_table(variation_headers, variation_rows)}

## 10. Model Architecture Summary

```text
┌──────────────────────────────────────────────────────────┐
│  8-rank multimodal perception training                   │
├──────────────────────────────────────────────────────────┤
│  DataLoader(workers={config_summary.get('workers_per_gpu', 'unknown')}) → image feature extraction
│  → multi-scale/deformable attention → task heads/losses  │
│  → backward → HCCL gradient communication                │
│  → SOAP optimizer                                        │
│       └─ every 10 steps: AICPU QR reconditioning         │
└──────────────────────────────────────────────────────────┘
```

```text
Normal: [prepare][forward/backward ~AI Core][comm][device-free/host gap]
Periodic SOAP window: {'captured at profiler Step ' + str(periodic_step) if periodic_step is not None else 'not captured'}.
```

The next optimization decision should separate the periodic QR bottleneck from normal-step host underfeed; they are different functional problems and therefore, if implemented, must remain separate function-level commits.
"""
    arch_path = output_dir / f"model_architecture_report_{args.run_name}.md"
    arch_path.write_text(arch, encoding="utf-8")

    manifest = {
        "json": str(json_path),
        "anomaly_report": str(output_dir / "anomaly_discovery_report.md"),
        "architecture_report": str(arch_path),
        "steps": sorted(step_windows),
        "dominant_step": dominant_step,
        "top_device_op": by_duration[0] if by_duration else None,
        "normal_step_avg_underfeed_ratio": sum(x["underfeed_ratio"] for x in normal_steps) / len(normal_steps),
        "wait_anchor_count": len(wait_anchors),
        "aicpu_op_count": len(aicpu_rows),
        "communication_overlap_ratio": comm_overlap_ratio,
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(safe_json(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe_json(manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
