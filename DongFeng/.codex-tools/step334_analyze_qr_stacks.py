#!/usr/bin/env python3
"""STEP-334: attribute QR operators to Python Call Stack (focus train iter14 / step 13)."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

QR_OP = re.compile(r"qr|geqrf|orgqr|ungqr|linalg.?qr", re.I)
SOAP_HINT = re.compile(
    r"soap\.py|_qr_finish|_stale_q_submit|_stale_q_install|"
    r"get_orthogonal_matrix_QR|update_preconditioner|_qr_install|_qr_plan",
    re.I,
)


def find_first(root: Path, name: str) -> Path | None:
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def split_frames(stack: str) -> list[str]:
    if not stack:
        return []
    parts = re.split(r";\n|;", stack.replace("\r", ""))
    return [p.strip() for p in parts if p.strip()]


def project_frames(stack: str) -> list[str]:
    frames = split_frames(stack)
    project = []
    for frame in frames:
        if "/l2.9-df-for-yuexiang/" in frame or "mmdet3d_plugin" in frame or "soap.py" in frame:
            frame = re.sub(r"^.*?/l2\.9-df-for-yuexiang/", "repo/", frame)
            project.append(frame)
    return project


def soap_boundary(frames: list[str]) -> str:
    for frame in reversed(frames):
        if SOAP_HINT.search(frame):
            return frame
    return frames[-1] if frames else "<no_project_stack>"


def classify_trigger(frames: list[str], boundary: str) -> str:
    text = "\n".join(frames + [boundary])
    if re.search(r"_qr_finish|_stale_q_submit", text):
        return "stale_q_submit_side_stream"
    if re.search(r"_stale_q_install|_qr_install", text):
        return "stale_q_install"
    if re.search(r"get_orthogonal_matrix_QR", text):
        return "sync_get_orthogonal_matrix_QR"
    if re.search(r"update_preconditioner", text):
        return "update_preconditioner"
    if re.search(r"soap\.py", text):
        return "soap_other"
    if frames:
        return "non_soap_project"
    return "no_stack"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", required=True)
    parser.add_argument("--focus-step", type=int, default=13)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    root = Path(args.profile_root)
    op_path = find_first(root, "operator_details.csv")
    if op_path is None:
        raise SystemExit(f"operator_details.csv not found under {root}")

    head = pd.read_csv(op_path, nrows=2, low_memory=False)
    cols = list(head.columns)
    has_step = "Step Id" in cols
    has_stack = "Call Stack" in cols
    usecols = [c for c in (
        "Name", "Step Id", "Call Stack", "Input Shapes",
        "Host Self Duration(us)", "Host Total Duration(us)",
        "Device Self Duration(us)", "Device Total Duration(us)",
    ) if c in cols]
    if "Name" not in usecols:
        raise SystemExit(f"Name column missing; columns={cols[:40]}")

    groups: dict[tuple[str, str, str], dict] = defaultdict(lambda: {
        "count": 0,
        "device_self_us": 0.0,
        "device_total_us": 0.0,
        "host_self_us": 0.0,
        "host_total_us": 0.0,
        "max_device_self_us": -1.0,
        "representative_stack": [],
        "sample_shapes": [],
    })
    rows_scanned = 0
    qr_rows = 0
    qr_with_stack = 0
    qr_focus_rows = 0
    name_hist: dict[str, int] = defaultdict(int)
    step_hist: dict[int, dict] = defaultdict(lambda: {
        "count": 0, "device_self_us": 0.0, "with_stack": 0,
    })

    for chunk in pd.read_csv(op_path, usecols=usecols, chunksize=50_000, low_memory=False):
        rows_scanned += len(chunk)
        names = chunk["Name"].fillna("").astype(str)
        selected = chunk[names.str.contains(QR_OP)].copy()
        if selected.empty:
            continue
        qr_rows += len(selected)
        for col in (
            "Host Self Duration(us)", "Host Total Duration(us)",
            "Device Self Duration(us)", "Device Total Duration(us)",
        ):
            if col in selected.columns:
                selected[col] = pd.to_numeric(selected[col], errors="coerce").fillna(0.0)
        if has_step:
            selected["Step Id"] = pd.to_numeric(selected["Step Id"], errors="coerce")

        for _, row in selected.iterrows():
            name = str(row["Name"])
            name_hist[name] += 1
            stack = ""
            if has_stack and "Call Stack" in row.index and not pd.isna(row["Call Stack"]):
                stack = str(row["Call Stack"])
            if stack:
                qr_with_stack += 1

            step_id = None
            if has_step and not pd.isna(row.get("Step Id")):
                step_id = int(row["Step Id"])

            device_self = float(row.get("Device Self Duration(us)", 0) or 0)
            device_total = float(row.get("Device Total Duration(us)", 0) or 0)
            host_self = float(row.get("Host Self Duration(us)", 0) or 0)
            host_total = float(row.get("Host Total Duration(us)", 0) or 0)

            if step_id is not None:
                slot = step_hist[step_id]
                slot["count"] += 1
                slot["device_self_us"] += device_self
                if stack:
                    slot["with_stack"] += 1
                if step_id != args.focus_step:
                    continue
                qr_focus_rows += 1
            else:
                # No Step Id: attribute all QR ops into focus bucket for stack classification
                qr_focus_rows += 1

            frames = project_frames(stack)
            boundary = soap_boundary(frames)
            trigger = classify_trigger(frames, boundary)
            key = (name, trigger, boundary)
            item = groups[key]
            item["count"] += 1
            item["device_self_us"] += device_self
            item["device_total_us"] += device_total
            item["host_self_us"] += host_self
            item["host_total_us"] += host_total
            if device_self >= float(item["max_device_self_us"]):
                item["max_device_self_us"] = device_self
                item["representative_stack"] = frames[-15:] if frames else split_frames(stack)[-10:]
            if "Input Shapes" in row.index and not pd.isna(row.get("Input Shapes")):
                s = str(row["Input Shapes"])[:200]
                if s not in item["sample_shapes"] and len(item["sample_shapes"]) < 3:
                    item["sample_shapes"].append(s)

    group_rows = [
        {
            "name": name,
            "trigger_class": trigger,
            "boundary": boundary,
            "count": int(v["count"]),
            "device_self_ms": float(v["device_self_us"]) / 1000.0,
            "device_total_ms": float(v["device_total_us"]) / 1000.0,
            "host_self_ms": float(v["host_self_us"]) / 1000.0,
            "host_total_ms": float(v["host_total_us"]) / 1000.0,
            "representative_stack": v["representative_stack"],
            "sample_shapes": v["sample_shapes"],
        }
        for (name, trigger, boundary), v in groups.items()
    ]
    group_rows.sort(key=lambda x: x["device_self_ms"], reverse=True)

    by_trigger: dict[str, dict] = defaultdict(lambda: {"count": 0, "device_self_ms": 0.0, "host_self_ms": 0.0})
    for row in group_rows:
        t = by_trigger[row["trigger_class"]]
        t["count"] += row["count"]
        t["device_self_ms"] += row["device_self_ms"]
        t["host_self_ms"] += row["host_self_ms"]

    top_trigger = None
    if by_trigger:
        top_trigger = max(by_trigger.items(), key=lambda kv: kv[1]["device_self_ms"])[0]

    verdict = []
    if qr_focus_rows == 0 and qr_rows == 0:
        verdict.append("No QR-matching operator rows in operator_details.csv.")
    elif qr_focus_rows == 0:
        verdict.append(
            f"QR ops exist ({qr_rows}) but none on profiler step {args.focus_step}; "
            "check Step Id alignment with kernel_details."
        )
    elif top_trigger == "stale_q_submit_side_stream":
        verdict.append(
            "iter14 QR stacks point to _qr_finish/_stale_q_submit (enqueue site), "
            "not install: Python trigger is submit; device wall shows on install step."
        )
    elif top_trigger == "stale_q_install":
        verdict.append("iter14 QR stacks point to install path (_stale_q_install/_qr_install).")
    elif top_trigger == "sync_get_orthogonal_matrix_QR":
        verdict.append("iter14 QR stacks point to synchronous get_orthogonal_matrix_QR.")
    elif top_trigger == "no_stack":
        verdict.append(
            "QR operators on focus step have empty Call Stack; cannot attribute to Python line."
        )
    else:
        verdict.append(f"Top trigger class on focus step: {top_trigger}")

    out = {
        "operator_details": str(op_path),
        "columns_sample": cols[:40],
        "has_step_id": has_step,
        "has_call_stack": has_stack,
        "rows_scanned": rows_scanned,
        "qr_rows_all_steps": qr_rows,
        "qr_rows_focus_step": qr_focus_rows,
        "qr_with_stack_seen": qr_with_stack,
        "focus_step": args.focus_step,
        "focus_train_iter": args.focus_step + 1,
        "qr_op_name_hist": dict(sorted(name_hist.items(), key=lambda kv: -kv[1])[:20]),
        "qr_by_step": {
            str(k): {
                "count": v["count"],
                "device_self_ms": v["device_self_us"] / 1000.0,
                "with_stack": v["with_stack"],
            }
            for k, v in sorted(step_hist.items())
        },
        "by_trigger_class": {
            k: {
                "count": v["count"],
                "device_self_ms": round(v["device_self_ms"], 3),
                "host_self_ms": round(v["host_self_ms"], 3),
            }
            for k, v in sorted(by_trigger.items(), key=lambda kv: -kv[1]["device_self_ms"])
        },
        "top_groups": group_rows[:20],
        "verdict": verdict,
    }
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# STEP-334 QR Call Stack attribution",
        "",
        f"- operator_details: `{op_path}`",
        f"- focus profiler step: {args.focus_step} (train iter {args.focus_step + 1})",
        f"- qr_rows_focus={qr_focus_rows}, qr_rows_all={qr_rows}, has_stack={has_stack}, has_step={has_step}",
        "",
        "## QR op name histogram",
        "",
    ]
    for name, cnt in list(out["qr_op_name_hist"].items())[:12]:
        lines.append(f"- `{name}`: {cnt}")
    lines.extend(["", "## By profiler step", ""])
    for sid, info in out["qr_by_step"].items():
        lines.append(
            f"- step {sid} (iter {int(sid)+1}): count={info['count']}, "
            f"device_self_ms={info['device_self_ms']:.1f}, with_stack={info['with_stack']}"
        )
    lines.extend(["", "## Trigger class (focus step)", ""])
    for k, v in out["by_trigger_class"].items():
        lines.append(
            f"- **{k}**: count={v['count']}, device_self_ms={v['device_self_ms']:.1f}, "
            f"host_self_ms={v['host_self_ms']:.1f}"
        )
    lines.extend(["", "## Top stack groups", ""])
    for i, g in enumerate(group_rows[:8], 1):
        lines.append(f"### {i}. {g['trigger_class']} — {g['name']}")
        lines.append(
            f"- count={g['count']}, device_self_ms={g['device_self_ms']:.1f}, "
            f"host_self_ms={g['host_self_ms']:.1f}"
        )
        lines.append(f"- boundary: `{g['boundary']}`")
        if g["representative_stack"]:
            lines.append("- stack (project frames):")
            for fr in g["representative_stack"]:
                lines.append(f"  - `{fr}`")
        lines.append("")
    lines.extend(["## Verdict", ""] + [f"- {v}" for v in verdict])
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": verdict,
        "by_trigger": out["by_trigger_class"],
        "qr_focus_rows": qr_focus_rows,
        "qr_rows_all": qr_rows,
        "top_boundary": group_rows[0]["boundary"] if group_rows else None,
        "name_hist_top": list(out["qr_op_name_hist"].items())[:5],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
