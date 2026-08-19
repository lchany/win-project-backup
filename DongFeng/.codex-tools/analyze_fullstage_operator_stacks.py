from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


NUMERIC = (
    "Host Self Duration(us)",
    "Host Total Duration(us)",
    "Device Self Duration(us)",
    "Device Total Duration(us)",
)


def classify_stage(name: str, stack: str) -> str:
    op = name.lower()
    text = f"{name}\n{stack}".lower()
    # Prefer semantic operator names over incidental words in a deep Python
    # stack.  This keeps forward MSDA out of "backward" when an outer
    # autograd-related frame is present, while still recognizing gradient
    # kernels that have no usable stack.
    if re.search(r"all_reduce|allreduce|all_gather|reduce_scatter|hccl|hcom", op):
        return "communication"
    if re.search(r"backward|grad", op):
        return "backward"
    if "multiscaledeformableattn" in op:
        return "forward"
    if re.search(r"dataloader|data_loader|pipeline|__next__", text):
        return "data"
    if re.search(r"optimizer|soap|gradient_fingerprint|\.step\(", text):
        return "optimizer"
    if re.search(r"all_reduce|allreduce|all_gather|reduce_scatter|hccl|hcom", text):
        return "communication"
    if re.search(r"backward|autograd|grad_fn", text):
        return "backward"
    if re.search(r"loss|assigner|target", text):
        return "loss_target"
    if re.search(r"backbone|neck|encoder|decoder|head|forward|detector", text):
        return "forward"
    return "framework_or_unknown"


def compact_stack(value: str) -> list[str]:
    frames = [part.strip() for part in value.replace("\r", "").split(";\n") if part.strip()]
    project = [
        re.sub(r"^.*?/l2\.9-df-for-yuexiang/", "repo/", frame)
        for frame in frames
        if "/l2.9-df-for-yuexiang/" in frame or "/diagnostics/" in frame
    ]
    chosen = project[-12:] if project else frames[-8:]
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-details", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    totals: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "count": 0,
            "host_self_us": 0.0,
            "host_total_us": 0.0,
            "device_self_us": 0.0,
            "device_total_us": 0.0,
            "max_device_self_us": -1.0,
            "representative_shape": "",
            "representative_stack": [],
            "stage": "framework_or_unknown",
        }
    )
    rows_scanned = 0
    rows_with_stack = 0
    usecols = ["Name", "Input Shapes", "Call Stack", *NUMERIC]
    for chunk in pd.read_csv(
        args.operator_details,
        usecols=usecols,
        chunksize=100_000,
        low_memory=False,
    ):
        rows_scanned += len(chunk)
        for column in NUMERIC:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce").fillna(0.0)
        for row in chunk.itertuples(index=False, name=None):
            name = str(row[0] or "<empty>")
            shape = "" if pd.isna(row[1]) else str(row[1])
            stack = "" if pd.isna(row[2]) else str(row[2])
            if stack:
                rows_with_stack += 1
            values = [float(row[index]) for index in range(3, 7)]
            item = totals[name]
            if int(item["count"]) == 0:
                # A subset of device operators has no captured Python stack.
                # Preserve useful semantics encoded directly in names such as
                # ConvolutionBackward instead of leaving them as unknown.
                item["stage"] = classify_stage(name, "")
            item["count"] = int(item["count"]) + 1
            item["host_self_us"] = float(item["host_self_us"]) + values[0]
            item["host_total_us"] = float(item["host_total_us"]) + values[1]
            item["device_self_us"] = float(item["device_self_us"]) + values[2]
            item["device_total_us"] = float(item["device_total_us"]) + values[3]
            if stack and values[2] >= float(item["max_device_self_us"]):
                item["max_device_self_us"] = values[2]
                item["representative_shape"] = shape[:2000]
                item["representative_stack"] = compact_stack(stack)
                item["stage"] = classify_stage(name, stack)

    rows = [{"name": name, **values} for name, values in totals.items()]
    by_device = sorted(rows, key=lambda x: float(x["device_self_us"]), reverse=True)
    by_host = sorted(rows, key=lambda x: float(x["host_self_us"]), reverse=True)
    stage_totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"operator_names": 0, "rows": 0, "device_self_us": 0.0, "host_self_us": 0.0}
    )
    for row in rows:
        stage = str(row["stage"])
        stage_totals[stage]["operator_names"] += 1
        stage_totals[stage]["rows"] += int(row["count"])
        stage_totals[stage]["device_self_us"] += float(row["device_self_us"])
        stage_totals[stage]["host_self_us"] += float(row["host_self_us"])

    result = {
        "rows_scanned": rows_scanned,
        "rows_with_stack": rows_with_stack,
        "timing_semantics": (
            "Device/host self durations are additive. Total durations are nested upper bounds "
            "and must not be summed across operator names. Stage labels come from captured stacks."
        ),
        "stage_totals": dict(stage_totals),
        "top_by_device_self": by_device[:80],
        "top_by_host_self": by_host[:80],
    }
    Path(args.output_json).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def table(rows_to_render: list[dict[str, object]]) -> str:
        lines = [
            "| Rank | Operator | Stage | Count | Device self ms | Host self ms | Representative source |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
        for rank, row in enumerate(rows_to_render, 1):
            source = " → ".join(str(x) for x in row["representative_stack"][-3:])
            source = source.replace("|", "\\|")
            safe_name = str(row["name"]).replace("|", "\\|")
            lines.append(
                f"| {rank} | {safe_name} | {row['stage']} | "
                f"{row['count']} | {float(row['device_self_us']) / 1000.0:.3f} | "
                f"{float(row['host_self_us']) / 1000.0:.3f} | {source} |"
            )
        return "\n".join(lines)

    stage_lines = [
        "| Stage | Operator names | Rows | Device self ms | Host self ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, values in sorted(
        stage_totals.items(), key=lambda item: float(item[1]["device_self_us"]), reverse=True
    ):
        stage_lines.append(
            f"| {stage} | {values['operator_names']} | {values['rows']} | "
            f"{float(values['device_self_us']) / 1000.0:.3f} | "
            f"{float(values['host_self_us']) / 1000.0:.3f} |"
        )

    markdown = f"""# Full-stage Operator Stack Attribution

- Rows scanned: {rows_scanned:,}
- Rows with stack: {rows_with_stack:,}
- Timing rule: self durations are additive; nested total durations are not summed.

## Stage attribution

{chr(10).join(stage_lines)}

## Top operators by device self duration

{table(by_device[:30])}

## Top operators by host self duration

{table(by_host[:30])}
"""
    Path(args.output_md).write_text(markdown, encoding="utf-8")
    print(json.dumps({"rows_scanned": rows_scanned, "rows_with_stack": rows_with_stack}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
