from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import ijson


STEP_RE = re.compile(r"(?:ProfilerStep|Iteration|Step)#?(\d+)", re.I)
STAGE_RE = re.compile(
    r"forward|backward|autograd|optimizer|zero_grad|dataloader|data loader|"
    r"allreduce|all_reduce|allgather|all_gather|reduce_scatter|broadcast|"
    r"nccl|memcpy|memset|cuda.*synchron|profilerstep|iteration",
    re.I,
)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top", type=int, default=200)
    args = parser.parse_args()

    trace = args.trace.resolve(strict=True)
    if trace.is_symlink() or not trace.is_file():
        raise RuntimeError("trace must be a regular non-symlink file")
    if args.output.exists():
        raise RuntimeError("output already exists")

    phase_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    name_counts: Counter[str] = Counter()
    category_name_counts: Counter[tuple[str, str]] = Counter()
    category_name_duration: defaultdict[tuple[str, str], float] = defaultdict(float)
    stage_name_counts: Counter[str] = Counter()
    stage_name_duration: defaultdict[str, float] = defaultdict(float)
    pid_counts: Counter[str] = Counter()
    tid_counts: Counter[str] = Counter()
    arg_key_counts: Counter[str] = Counter()
    samples: dict[str, list[dict[str, object]]] = defaultdict(list)
    step_markers: list[dict[str, object]] = []
    trace_events = 0
    complete_events = 0
    timestamp_min: float | None = None
    timestamp_max_end: float | None = None
    stack_event_count = 0
    shape_event_count = 0
    device_like_count = 0
    device_like_duration_us = 0.0

    with trace.open("rb") as handle:
        for event in ijson.items(handle, "traceEvents.item"):
            if not isinstance(event, dict):
                continue
            trace_events += 1
            ph = str(event.get("ph", ""))
            cat = str(event.get("cat", ""))
            name = str(event.get("name", ""))
            pid = str(event.get("pid", ""))
            tid = str(event.get("tid", ""))
            phase_counts[ph] += 1
            category_counts[cat] += 1
            name_counts[name] += 1
            pid_counts[pid] += 1
            tid_counts[tid] += 1
            dur = number(event.get("dur")) or 0.0
            ts = number(event.get("ts"))
            if ph == "X":
                complete_events += 1
                category_name_counts[(cat, name)] += 1
                category_name_duration[(cat, name)] += max(0.0, dur)
                if ts is not None:
                    timestamp_min = ts if timestamp_min is None else min(timestamp_min, ts)
                    end = ts + max(0.0, dur)
                    timestamp_max_end = end if timestamp_max_end is None else max(timestamp_max_end, end)

            raw_args = event.get("args")
            event_args = raw_args if isinstance(raw_args, dict) else {}
            keys = [str(key) for key in event_args]
            arg_key_counts.update(keys)
            if any("stack" in key.lower() for key in keys):
                stack_event_count += 1
            if any("shape" in key.lower() or "dim" in key.lower() for key in keys):
                shape_event_count += 1

            lower_cat = cat.lower()
            lower_name = name.lower()
            device_like = (
                "kernel" in lower_cat
                or "gpu" in lower_cat
                or "cuda_kernel" in lower_cat
                or "stream" in {key.lower() for key in keys}
            )
            if device_like and ph == "X":
                device_like_count += 1
                device_like_duration_us += max(0.0, dur)

            match = STEP_RE.search(name)
            if match or name in {"Record Window End", "Iteration Start: PyTorch Profiler"}:
                step_markers.append(
                    {
                        "name": name,
                        "category": cat,
                        "phase": ph,
                        "ts_us": ts,
                        "duration_us": dur if ph == "X" else None,
                        "pid": pid,
                        "tid": tid,
                    }
                )
            if STAGE_RE.search(name) or STAGE_RE.search(cat):
                stage_name_counts[name] += 1
                if ph == "X":
                    stage_name_duration[name] += max(0.0, dur)

            sample_key = cat or "<empty>"
            if len(samples[sample_key]) < 5:
                samples[sample_key].append(
                    {
                        "name": name,
                        "phase": ph,
                        "duration_us": dur if ph == "X" else None,
                        "pid_type": type(event.get("pid")).__name__,
                        "tid_type": type(event.get("tid")).__name__,
                        "arg_keys": keys[:30],
                    }
                )

    def top_counter(counter: Counter[str], limit: int) -> list[dict[str, object]]:
        return [{"name": key, "count": value} for key, value in counter.most_common(limit)]

    top_category_name_count = [
        {"category": cat, "name": name, "count": count, "duration_us": category_name_duration[(cat, name)]}
        for (cat, name), count in category_name_counts.most_common(args.top)
    ]
    top_category_name_duration = [
        {"category": cat, "name": name, "count": category_name_counts[(cat, name)], "duration_us": duration}
        for (cat, name), duration in sorted(category_name_duration.items(), key=lambda item: item[1], reverse=True)[: args.top]
    ]
    top_stage_duration = [
        {"name": name, "count": stage_name_counts[name], "duration_us": duration}
        for name, duration in sorted(stage_name_duration.items(), key=lambda item: item[1], reverse=True)[: args.top]
    ]
    payload = {
        "trace": {"bytes": trace.stat().st_size, "sha256": sha256_file(trace)},
        "parser": {"ijson_backend": ijson.backend, "prefix": "traceEvents.item"},
        "event_count": trace_events,
        "complete_event_count": complete_events,
        "capture_timestamp_min_us": timestamp_min,
        "capture_timestamp_max_end_us": timestamp_max_end,
        "capture_span_us": (
            timestamp_max_end - timestamp_min
            if timestamp_min is not None and timestamp_max_end is not None
            else None
        ),
        "phase_counts": dict(phase_counts.most_common()),
        "category_counts": dict(category_counts.most_common()),
        "pid_top": top_counter(pid_counts, 100),
        "tid_top": top_counter(tid_counts, 100),
        "arg_key_top": top_counter(arg_key_counts, 100),
        "stack_event_count": stack_event_count,
        "shape_event_count": shape_event_count,
        "device_like_count": device_like_count,
        "device_like_duration_us": device_like_duration_us,
        "step_markers": sorted(
            step_markers,
            key=lambda row: (float(row["ts_us"] or 0.0), str(row["name"])),
        ),
        "top_names_by_count": top_counter(name_counts, args.top),
        "top_category_name_by_count": top_category_name_count,
        "top_category_name_by_duration": top_category_name_duration,
        "top_stage_names_by_duration": top_stage_duration,
        "samples_by_category": dict(samples),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "event_count": trace_events,
        "capture_span_us": payload["capture_span_us"],
        "step_marker_count": len(step_markers),
        "stack_event_count": stack_event_count,
        "shape_event_count": shape_event_count,
        "ijson_backend": ijson.backend,
        "output_sha256": sha256_file(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
