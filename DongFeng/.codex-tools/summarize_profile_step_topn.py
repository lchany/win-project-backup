from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-details", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.kernel_details, low_memory=False).rename(
        columns={
            "Step Id": "step",
            "Name": "name",
            "Duration(us)": "duration_us",
            "Wait Time(us)": "wait_us",
        }
    )
    for column in ("step", "duration_us", "wait_us"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["step", "name", "duration_us"]).copy()
    frame["step"] = frame["step"].astype(int)
    frame["wait_us"] = frame["wait_us"].fillna(0.0)

    result: dict[str, list[dict[str, object]]] = {}
    markdown = [
        "# Per-step Pure Kernel Duration TopN",
        "",
        "排序只使用纯kernel duration；wait仅作假热点识别，不进入根因排名。Profiler Step 23对应训练日志step24（SOAP），24–26对应训练日志step25–27（普通步）。",
    ]
    for step, group in frame.groupby("step"):
        ranked = (
            group.groupby("name", as_index=False)
            .agg(count=("name", "size"), duration_us=("duration_us", "sum"), wait_us=("wait_us", "sum"))
            .sort_values("duration_us", ascending=False)
            .head(30)
        )
        rows = []
        markdown.extend(
            [
                "",
                f"## Profiler Step {step}",
                "",
                "| Rank | Kernel | Count | Duration ms | Wait ms |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(ranked.itertuples(index=False), 1):
            item = {
                "rank": rank,
                "name": str(row.name),
                "count": int(row.count),
                "duration_ms": float(row.duration_us) / 1000.0,
                "wait_ms": float(row.wait_us) / 1000.0,
            }
            rows.append(item)
            safe_name = item["name"].replace("|", "\\|")
            markdown.append(
                f"| {rank} | {safe_name} | {item['count']} | "
                f"{item['duration_ms']:.3f} | {item['wait_ms']:.3f} |"
            )
        result[str(step)] = rows

    Path(args.output_json).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(args.output_md).write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"steps": sorted(map(int, result)), "rows": len(frame)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
