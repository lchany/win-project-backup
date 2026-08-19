from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import ijson


TARGET_OPS = {
    "aten::copy_",
    "aten::fill_",
    "aten::item",
    "aten::_local_scalar_dense",
    "aten::slice",
    "aten::select",
    "aten::floor_divide",
    "aten::remainder",
    "aten::randperm",
    "aten::rand",
}


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


def sig(args: dict[str, Any]) -> str:
    return json.dumps(
        {"Input Dims": args.get("Input Dims"), "Input type": args.get("Input type")},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stable-steps", default="33,34,35,36,37,38,39,40,41,43,44,45,46,47,48")
    parser.add_argument("--soap-steps", default="22,32,42")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("output already exists")
    trace = args.trace.resolve(strict=True)
    inventory_path = args.inventory.resolve(strict=True)
    inventory = json.load(inventory_path.open(encoding="utf-8"))
    stable = {int(item) for item in args.stable_steps.split(",")}
    soap = {int(item) for item in args.soap_steps.split(",")}

    windows = []
    seen: Counter[int] = Counter()
    for row in inventory["step_markers"]:
        name = str(row.get("name", ""))
        if not name.startswith("ProfilerStep#") or row.get("phase") != "X":
            continue
        try:
            step = int(name.split("#", 1)[1])
        except ValueError:
            continue
        occurrence = seen[step]
        seen[step] += 1
        if occurrence or step not in stable | soap:
            continue
        start = number(row.get("ts_us")); duration = number(row.get("duration_us"))
        if start is None or duration is None:
            continue
        windows.append({"step": step, "id": f"{step}:0", "start": start, "end": start + duration})
    windows.sort(key=lambda row: row["start"])
    starts = [float(row["start"]) for row in windows]

    def locate(ts: float) -> dict[str, Any] | None:
        index = bisect.bisect_right(starts, ts) - 1
        if index < 0:
            return None
        row = windows[index]
        return row if ts < float(row["end"]) else None

    cpu_count: dict[str, Counter[str]] = defaultdict(Counter)
    cpu_duration: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    annotation_count: dict[str, Counter[str]] = defaultdict(Counter)
    annotation_duration: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    target_count: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    target_duration: dict[str, defaultdict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
    runtime_count: dict[str, Counter[str]] = defaultdict(Counter)
    runtime_duration: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    events_scanned = 0
    broke_at_first_device = False

    with trace.open("rb") as handle:
        for event in ijson.items(handle, "traceEvents.item"):
            events_scanned += 1
            if not isinstance(event, dict):
                continue
            category = str(event.get("cat", ""))
            if category in {"kernel", "gpu_memcpy", "gpu_memset"}:
                broke_at_first_device = True
                break
            if str(event.get("ph", "")) != "X":
                continue
            ts = number(event.get("ts")); duration = number(event.get("dur"))
            if ts is None or duration is None:
                continue
            window = locate(ts)
            if window is None:
                continue
            window_id = str(window["id"])
            name = str(event.get("name", ""))
            raw_args = event.get("args")
            event_args = raw_args if isinstance(raw_args, dict) else {}
            if category == "cpu_op":
                cpu_count[window_id][name] += 1
                cpu_duration[window_id][name] += duration
                if name in TARGET_OPS:
                    key = (name, sig(event_args))
                    target_count[window_id][key] += 1
                    target_duration[window_id][key] += duration
            elif category == "user_annotation":
                annotation_count[window_id][name] += 1
                annotation_duration[window_id][name] += duration
            elif category == "cuda_runtime":
                runtime_count[window_id][name] += 1
                runtime_duration[window_id][name] += duration

    stable_ids = [str(row["id"]) for row in windows if int(row["step"]) in stable]
    soap_ids = [str(row["id"]) for row in windows if int(row["step"]) in soap]

    def aggregate_names(ids: list[str], counts: dict[str, Counter[str]], durations: dict[str, defaultdict[str, float]]) -> list[dict[str, Any]]:
        total_count: Counter[str] = Counter()
        total_duration: defaultdict[str, float] = defaultdict(float)
        for window_id in ids:
            total_count.update(counts[window_id])
            for key, value in durations[window_id].items():
                total_duration[key] += value
        n = len(ids)
        return [
            {"name": key, "count_per_step": total_count[key] / n, "duration_ms_per_step": value / 1000.0 / n}
            for key, value in sorted(total_duration.items(), key=lambda item: item[1], reverse=True)
        ]

    all_target_keys = set().union(*(target_count[window_id] for window_id in stable_ids))
    target_rows = []
    for key in all_target_keys:
        counts = [target_count[window_id][key] for window_id in stable_ids]
        durations = [target_duration[window_id][key] / 1000.0 for window_id in stable_ids]
        target_rows.append(
            {
                "op": key[0],
                "signature": json.loads(key[1]),
                "count_per_step": counts,
                "count_median": statistics.median(counts),
                "count_min": min(counts),
                "count_max": max(counts),
                "host_duration_ms_per_step": durations,
                "host_duration_ms_median": statistics.median(durations),
            }
        )
    target_rows.sort(key=lambda row: (row["count_median"], row["host_duration_ms_median"]), reverse=True)

    payload = {
        "trace_sha256": inventory["trace"]["sha256"],
        "inventory_sha256": sha256_file(inventory_path),
        "parser": {"ijson_backend": ijson.backend, "stopped_at_first_device_category": broke_at_first_device},
        "events_scanned": events_scanned,
        "stable_steps": sorted(stable),
        "soap_steps": sorted(soap),
        "stable_cpu_ops": aggregate_names(stable_ids, cpu_count, cpu_duration)[:500],
        "soap_cpu_ops": aggregate_names(soap_ids, cpu_count, cpu_duration)[:500],
        "stable_user_annotations": aggregate_names(stable_ids, annotation_count, annotation_duration),
        "soap_user_annotations": aggregate_names(soap_ids, annotation_count, annotation_duration),
        "stable_cuda_runtime": aggregate_names(stable_ids, runtime_count, runtime_duration)[:200],
        "stable_target_signatures": target_rows[:1000],
        "limitations": [
            "No stack: target signature attribution uses source/config/count/shape/time evidence only.",
            "Host operator durations are nested and are not additive wall time.",
        ],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "events_scanned": events_scanned,
        "stopped_at_first_device": broke_at_first_device,
        "target_signature_count": len(target_rows),
        "output_sha256": sha256_file(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
