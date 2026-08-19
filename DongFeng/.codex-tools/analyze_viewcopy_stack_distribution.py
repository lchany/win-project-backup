from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def project_frames(value: str) -> list[str]:
    frames = [part.strip() for part in value.replace("\r", "").split(";\n") if part.strip()]
    return [
        re.sub(r"^.*?/l2\.9-df-for-yuexiang/", "repo/", frame)
        for frame in frames
        if "/l2.9-df-for-yuexiang/" in frame
    ]


def md_escape(value: object) -> str:
    return str(value).replace("|", "\\|")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-details", required=True)
    parser.add_argument("--kernel-details", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    groups: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {
            "count": 0,
            "device_self_us": 0.0,
            "host_self_us": 0.0,
            "max_device_self_us": -1.0,
            "representative_stack": [],
        }
    )
    rows_scanned = 0
    matched_rows = 0
    for chunk in pd.read_csv(
        args.operator_details,
        usecols=[
            "Name",
            "Input Shapes",
            "Call Stack",
            "Device Self Duration(us)",
            "Host Self Duration(us)",
        ],
        chunksize=100_000,
        low_memory=False,
    ):
        rows_scanned += len(chunk)
        selected = chunk[chunk["Name"].fillna("").astype(str).eq("aclnnInplaceCopy")].copy()
        matched_rows += len(selected)
        for metric in ("Device Self Duration(us)", "Host Self Duration(us)"):
            selected[metric] = pd.to_numeric(selected[metric], errors="coerce").fillna(0.0)
        for row in selected.itertuples(index=False, name=None):
            shape = "<empty>" if pd.isna(row[1]) or not str(row[1]).strip() else str(row[1])[:1000]
            stack = "" if pd.isna(row[2]) else str(row[2])
            frames = project_frames(stack)
            boundary = frames[0] if frames else "<no_project_stack>"
            device_us = float(row[3])
            host_us = float(row[4])
            item = groups[(boundary, shape)]
            item["count"] = int(item["count"]) + 1
            item["device_self_us"] = float(item["device_self_us"]) + device_us
            item["host_self_us"] = float(item["host_self_us"]) + host_us
            if device_us >= float(item["max_device_self_us"]):
                item["max_device_self_us"] = device_us
                item["representative_stack"] = frames[-12:]

    group_rows = [
        {"boundary": boundary, "shape": shape, **values}
        for (boundary, shape), values in groups.items()
    ]
    group_rows.sort(key=lambda item: float(item["device_self_us"]), reverse=True)

    kernels = pd.read_csv(args.kernel_details, low_memory=False).rename(
        columns={
            "Step Id": "step",
            "Name": "name",
            "Duration(us)": "duration_us",
            "Wait Time(us)": "wait_us",
        }
    )
    for column in ("step", "duration_us", "wait_us"):
        kernels[column] = pd.to_numeric(kernels[column], errors="coerce")
    selected_kernels = kernels[
        kernels["name"].fillna("").astype(str).eq("aclnnInplaceCopy_ViewCopyAiCpu_ViewCopy")
    ].dropna(subset=["step", "duration_us"])
    selected_kernels["step"] = selected_kernels["step"].astype(int)
    selected_kernels["wait_us"] = selected_kernels["wait_us"].fillna(0.0)
    per_step = []
    for step, group in selected_kernels.groupby("step"):
        duration = float(group["duration_us"].sum())
        wait = float(group["wait_us"].sum())
        per_step.append(
            {
                "step": int(step),
                "count": int(len(group)),
                "kernel_duration_ms": duration / 1000.0,
                "wait_ms": wait / 1000.0,
                "total_cost_ms": (duration + wait) / 1000.0,
                "wait_ratio": wait / (duration + wait) if duration + wait else 0.0,
            }
        )

    total_device = sum(float(item["device_self_us"]) for item in group_rows)
    for item in group_rows:
        item["device_self_share"] = (
            float(item["device_self_us"]) / total_device if total_device else 0.0
        )
    result = {
        "rows_scanned": rows_scanned,
        "matched_operator_rows": matched_rows,
        "operator_device_self_ms": total_device / 1000.0,
        "operator_host_self_ms": sum(float(item["host_self_us"]) for item in group_rows) / 1000.0,
        "per_step_aicpu_viewcopy_kernel": sorted(per_step, key=lambda item: item["step"]),
        "groups": group_rows,
        "timing_semantics": (
            "Per-step kernel duration/wait is exact for the AICPU ViewCopy kernel. "
            "Stack groups use host aclnnInplaceCopy operator self timing and may include "
            "other InplaceCopy kernel variants; they are attribution shares, not additive "
            "AICPU-kernel savings claims."
        ),
    }
    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# AICPU ViewCopy Stack Distribution",
        "",
        f"- Operator rows scanned: {rows_scanned:,}",
        f"- `aclnnInplaceCopy` rows: {matched_rows:,}",
        f"- Operator device self: {total_device / 1000.0:.3f} ms",
        "- Kernel timing is exact; stack-group timing is an attribution proxy and is not claimed as removable kernel time.",
        "",
        "## Exact per-step AICPU ViewCopy kernel",
        "",
        "| Profiler Step | Count | Kernel ms | Wait ms | Total cost ms | Wait ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(per_step, key=lambda value: value["step"]):
        lines.append(
            f"| {item['step']} | {item['count']} | {item['kernel_duration_ms']:.3f} | "
            f"{item['wait_ms']:.3f} | {item['total_cost_ms']:.3f} | {100*item['wait_ratio']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Top stack/shape groups by operator device self",
            "",
            "| Rank | Boundary | Shape | Count | Device self ms | Host self ms | Share | Max call ms |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, item in enumerate(group_rows[:50], start=1):
        lines.append(
            f"| {rank} | {md_escape(item['boundary'])} | "
            f"{md_escape(item['shape'])} | {item['count']} | "
            f"{float(item['device_self_us'])/1000.0:.3f} | "
            f"{float(item['host_self_us'])/1000.0:.3f} | "
            f"{100*float(item['device_self_share']):.2f}% | "
            f"{float(item['max_device_self_us'])/1000.0:.3f} |"
        )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows_scanned": rows_scanned, "matched_rows": matched_rows, "groups": len(group_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
