#!/usr/bin/env python3
"""Aggregate STEP-215 eight-rank QR gate without exposing tensor contents."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve(strict=True)
    ranks = [json.loads((root / "done" / f"rank{i}.json").read_text()) for i in range(8)]
    dimensions = [row["shape"][0] for row in ranks[0]["results"]]
    shapes = []
    for index, dimension in enumerate(dimensions):
        rows = [rank["results"][index] for rank in ranks]
        direct = [row["direct_event"]["median_ms"] for row in rows]
        proposed = [row["proposed_event"]["median_ms"] for row in rows]
        shapes.append({
            "dimension": dimension,
            "historical_call_count": rows[0]["historical_call_count"],
            "active_call_count": rows[0]["active_call_count"],
            "numeric_gate_pass_all_ranks": all(row["numeric_gate_pass"] for row in rows),
            "finite_all_ranks": all(row["finite_all"] for row in rows),
            "direct_event_rank_median_ms": statistics.median(direct),
            "proposed_event_rank_median_ms": statistics.median(proposed),
            "speedup": statistics.median(direct) / statistics.median(proposed),
            "saved_event_rank_median_ms": statistics.median(direct) - statistics.median(proposed),
            "q_max_abs_worst": max(row["q_error"]["max_abs"] for row in rows),
            "q_nrmse_worst": max(row["q_error"]["nrmse"] for row in rows),
            "direct_self_q_nrmse_worst": max(row["direct_self_q_error"]["nrmse"] for row in rows),
            "proposed_self_q_nrmse_worst": max(row["proposed_self_q_error"]["nrmse"] for row in rows),
            "r_max_abs_worst": max(row["r_error"]["max_abs"] for row in rows),
            "r_nrmse_worst": max(row["r_error"]["nrmse"] for row in rows),
            "orthogonality_max_abs_worst": max(row["orthogonality_max_abs"] for row in rows),
            "orthogonality_normalized_fro_worst": max(row["orthogonality_normalized_fro"] for row in rows),
            "reconstruction_rel_l2_worst": max(row["reconstruction_rel_l2"] for row in rows),
            "extra_peak_allocated_max": max(row["extra_peak_allocated"] for row in rows),
            "extra_peak_reserved_max": max(row["extra_peak_reserved"] for row in rows),
        })
    weighted_direct = [rank["active_weighted_direct_event_ms"] for rank in ranks]
    weighted_proposed = [rank["active_weighted_proposed_event_ms"] for rank in ranks]
    summary = {
        "rank_count": len(ranks),
        "shape_count": len(shapes),
        "historical_call_count": ranks[0]["historical_call_count"],
        "active_call_count": ranks[0]["active_call_count"],
        "active_contract": ranks[0]["active_contract"],
        "threshold": ranks[0]["threshold"],
        "numeric_gate_pass_all_shapes_all_ranks": all(rank["numeric_gate_pass"] for rank in ranks),
        "active_weighted_direct_event_rank_median_ms": statistics.median(weighted_direct),
        "active_weighted_proposed_event_rank_median_ms": statistics.median(weighted_proposed),
        "active_weighted_saved_event_rank_median_ms": statistics.median(weighted_direct) - statistics.median(weighted_proposed),
        "active_weighted_speedup_ratio_of_rank_medians": statistics.median(weighted_direct) / statistics.median(weighted_proposed),
        "shapes": shapes,
    }
    temp = root / "summary.json.tmp"
    temp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(root / "summary.json")
    print(json.dumps({key: value for key, value in summary.items() if key != "shapes"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
