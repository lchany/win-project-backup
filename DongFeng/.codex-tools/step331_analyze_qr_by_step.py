#!/usr/bin/env python3
"""STEP-331: attribute QR/AICPU wall time to profiler steps (training iter = step_id + 1)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

QR_NAME = re.compile(r"qr|geqrf|orgqr|ungqr", re.I)
LINALG_HOST = re.compile(r"linalg.*qr|aten::.*qr", re.I)


def find_first(root: Path, name: str) -> Path | None:
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def load_kernel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.rename(
        columns={
            "Step Id": "step_id",
            "Name": "name",
            "Duration(us)": "duration_us",
            "Wait Time(us)": "wait_us",
            "Task Type": "task_type",
        }
    )
    for col in ("step_id", "duration_us", "wait_us"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["step_id", "name", "duration_us"])
    frame["step_id"] = frame["step_id"].astype(int)
    frame["wait_us"] = frame.get("wait_us", 0).fillna(0)
    frame["is_qr"] = frame["name"].astype(str).str.contains(QR_NAME)
    frame["train_iter"] = frame["step_id"] + 1
    return frame


def load_operator(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    frame = pd.read_csv(path, low_memory=False)
    rename = {}
    for src, dst in (
        ("Step Id", "step_id"),
        ("Name", "name"),
        ("Host Self Duration(us)", "host_self_us"),
        ("Host Total Duration(us)", "host_total_us"),
        ("Device Total Duration(us)", "device_total_us"),
    ):
        if src in frame.columns:
            rename[src] = dst
    frame = frame.rename(columns=rename)
    if "step_id" not in frame.columns or "name" not in frame.columns:
        return None
    frame["step_id"] = pd.to_numeric(frame["step_id"], errors="coerce")
    frame = frame.dropna(subset=["step_id", "name"])
    frame["step_id"] = frame["step_id"].astype(int)
    frame["train_iter"] = frame["step_id"] + 1
    frame["is_linalg_qr"] = frame["name"].astype(str).str.contains(LINALG_HOST)
    for col in ("host_self_us", "host_total_us", "device_total_us"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    return frame


def summarize_step(group: pd.DataFrame) -> dict:
    total_ms = float(group["duration_us"].sum()) / 1000.0
    qr = group[group["is_qr"]]
    qr_ms = float(qr["duration_us"].sum()) / 1000.0
    top = (
        group.groupby("name", as_index=False)["duration_us"]
        .sum()
        .sort_values("duration_us", ascending=False)
        .head(8)
    )
    return {
        "kernel_total_ms": total_ms,
        "qr_kernel_ms": qr_ms,
        "qr_kernel_count": int(len(qr)),
        "qr_share_pct": (100.0 * qr_ms / total_ms) if total_ms > 0 else 0.0,
        "top_kernels": [
            {
                "name": str(row.name),
                "duration_ms": float(row.duration_us) / 1000.0,
            }
            for row in top.itertuples(index=False)
        ],
        "qr_kernels": [
            {
                "name": str(row.name),
                "count": int(row.cnt),
                "duration_ms": float(row.duration_us) / 1000.0,
            }
            for row in (
                qr.groupby("name", as_index=False)
                .agg(cnt=("name", "size"), duration_us=("duration_us", "sum"))
                .sort_values("duration_us", ascending=False)
                .head(12)
                .itertuples(index=False)
            )
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--train-log", default="")
    args = parser.parse_args()

    root = Path(args.profile_root)
    kernel_path = find_first(root, "kernel_details.csv")
    if kernel_path is None:
        raise SystemExit(f"kernel_details.csv not found under {root}")

    op_path = find_first(root, "operator_details.csv")
    kernels = load_kernel(kernel_path)
    operators = load_operator(op_path) if op_path else None

    per_step = {}
    for step_id, group in kernels.groupby("step_id"):
        train_iter = int(step_id) + 1
        per_step[str(step_id)] = {
            "profiler_step_id": int(step_id),
            "train_iter": train_iter,
            **summarize_step(group),
        }

    op_by_step = {}
    if operators is not None:
        qr_ops = operators[operators["is_linalg_qr"]]
        for step_id, group in qr_ops.groupby("step_id"):
            op_by_step[str(int(step_id))] = {
                "linalg_qr_host_self_ms": float(group.get("host_self_us", 0).sum()) / 1000.0,
                "linalg_qr_host_total_ms": float(group.get("host_total_us", 0).sum()) / 1000.0,
                "linalg_qr_device_total_ms": float(group.get("device_total_us", 0).sum()) / 1000.0,
                "linalg_qr_op_count": int(len(group)),
                "sample_names": sorted(set(group["name"].astype(str).tolist()))[:8],
            }

    focus = {}
    for label, train_iter in (("iter10_submit", 10), ("iter14_install", 14)):
        step_id = train_iter - 1
        key = str(step_id)
        focus[label] = {
            "train_iter": train_iter,
            "profiler_step_id": step_id,
            "kernel": per_step.get(key),
            "operator_linalg_qr": op_by_step.get(key),
        }

    iter10_qr = focus["iter10_submit"]["kernel"]["qr_kernel_ms"] if focus["iter10_submit"]["kernel"] else 0.0
    iter14_qr = focus["iter14_install"]["kernel"]["qr_kernel_ms"] if focus["iter14_install"]["kernel"] else 0.0
    verdict = []
    if iter14_qr >= 20000 and iter10_qr < 5000:
        verdict.append(
            "QR device wall time concentrates on training iter14, not iter10: stale-Q deferred install pattern confirmed by trace."
        )
    elif iter10_qr >= 20000 and iter14_qr >= 20000:
        verdict.append(
            "QR wall time appears on BOTH iter10 and iter14: not explained by install-only; check per-param phase / stream drain."
        )
    elif iter10_qr >= 20000 and iter14_qr < 5000:
        verdict.append(
            "QR wall time concentrates on training iter10 (SOAP submit/cycle step): install hypothesis rejected for iter14 spike."
        )
    else:
        verdict.append(
            f"QR ms below decisive threshold (iter10={iter10_qr:.1f}, iter14={iter14_qr:.1f}); inspect top kernels / host ops."
        )

    train_times = {}
    if args.train_log:
        log = Path(args.train_log)
        if log.is_file():
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                m = re.search(r"Iter \[(\d+)/\d+\].*?time: ([0-9.]+)", line)
                if m:
                    train_times[int(m.group(1))] = float(m.group(2))

    out = {
        "kernel_details": str(kernel_path),
        "operator_details": str(op_path) if op_path else None,
        "profiler_steps_seen": sorted(int(k) for k in per_step),
        "per_step": per_step,
        "operator_linalg_qr_by_step": op_by_step,
        "focus": focus,
        "train_log_iter_time_s": train_times,
        "verdict": verdict,
    }
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# STEP-331 QR step attribution",
        "",
        f"- kernel_details: `{kernel_path}`",
        f"- operator_details: `{op_path}`" if op_path else "- operator_details: missing",
        "",
        "## Training log iter time (s)",
        "",
    ]
    for it in (10, 14):
        lines.append(f"- iter{it}: {train_times.get(it, 'n/a')}")
    lines.extend(["", "## Focus comparison", ""])
    for label in ("iter10_submit", "iter14_install"):
        block = focus[label]
        k = block.get("kernel") or {}
        o = block.get("operator_linalg_qr") or {}
        lines.append(
            f"### {label} (train iter {block['train_iter']}, profiler step {block['profiler_step_id']})"
        )
        lines.append(
            f"- kernel_total_ms={k.get('kernel_total_ms', 'n/a')}, "
            f"qr_kernel_ms={k.get('qr_kernel_ms', 'n/a')}, qr_share={k.get('qr_share_pct', 'n/a')}%"
        )
        if o:
            lines.append(
                f"- linalg_qr host_total_ms={o.get('linalg_qr_host_total_ms')}, "
                f"device_total_ms={o.get('linalg_qr_device_total_ms')}, count={o.get('linalg_qr_op_count')}"
            )
        lines.extend(["", "Top kernels:", ""])
        for item in (k.get("top_kernels") or [])[:5]:
            lines.append(f"- {item['name']}: {item['duration_ms']:.1f} ms")
        lines.append("")
    lines.extend(["## Verdict", ""] + [f"- {v}" for v in verdict])
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "iter10_qr_ms": iter10_qr, "iter14_qr_ms": iter14_qr}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
