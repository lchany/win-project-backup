#!/usr/bin/env python3
"""Compare GPU and NPU losses at exactly matching training iterations."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf(?:inity)?)"
ITER_RE = re.compile(r"\bIter\s*\[\s*(\d+)\s*/", re.IGNORECASE)
METRIC_RE = re.compile(
    rf"(?P<key>[A-Za-z_][\w./-]*loss[\w./-]*|loss)\s*:\s*(?P<value>{NUMBER})(?![\w.])",
    re.IGNORECASE,
)


class InputError(ValueError):
    """Raised when an input cannot be interpreted as loss data."""


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric loss")
    return float(value)


def _json_rows(payload: Any) -> Iterable[tuple[int, dict[str, Any]]]:
    if isinstance(payload, dict) and "rows" in payload:
        payload = payload["rows"]

    if isinstance(payload, list):
        for index, row in enumerate(payload):
            if not isinstance(row, dict) or "iter" not in row:
                raise InputError(f"JSON row {index} must be an object containing 'iter'")
            yield int(row["iter"]), row
        return

    if isinstance(payload, dict):
        for raw_iter, raw_row in payload.items():
            try:
                iteration = int(raw_iter)
            except (TypeError, ValueError) as exc:
                raise InputError(f"invalid JSON iteration key: {raw_iter!r}") from exc
            row = raw_row if isinstance(raw_row, dict) else {"loss": raw_row}
            yield iteration, row
        return

    raise InputError("JSON root must be an iteration mapping, a row list, or {'rows': [...]}")


def _load_json(text: str) -> list[tuple[int, dict[str, Any]]]:
    try:
        payload = json.loads(text)
        return list(_json_rows(payload))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(f"invalid JSON loss data: {exc}") from exc


def _load_log(text: str) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for line in text.splitlines():
        match = ITER_RE.search(line)
        if not match:
            continue
        metrics = {
            metric.group("key"): metric.group("value")
            for metric in METRIC_RE.finditer(line)
        }
        rows.append((int(match.group(1)), metrics))
    if not rows:
        raise InputError("log contains no 'Iter [current/total]' records")
    return rows


def load_rows(path: Path, input_format: str) -> list[tuple[int, dict[str, Any]]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc

    chosen_format = input_format
    if chosen_format == "auto":
        chosen_format = "json" if path.suffix.lower() == ".json" else "log"
    return _load_json(text) if chosen_format == "json" else _load_log(text)


def _select_rows(
    rows: list[tuple[int, dict[str, Any]]], start: int, end: int
) -> list[tuple[int, dict[str, Any]]]:
    return [(iteration, row) for iteration, row in rows if start <= iteration <= end]


def _first_iteration(rows: list[tuple[int, dict[str, Any]]]) -> int:
    try:
        return min(iteration for iteration, _ in rows)
    except ValueError as exc:
        raise InputError("input has no iteration records") from exc


def _last_iteration(rows: list[tuple[int, dict[str, Any]]]) -> int:
    return max(iteration for iteration, _ in rows)


def _metric_value(row: dict[str, Any], key: str) -> tuple[float | None, str | None]:
    if key not in row:
        return None, "missing_loss"
    try:
        value = _as_float(row[key])
    except (TypeError, ValueError):
        return None, "invalid_loss"
    if not math.isfinite(value):
        return None, "non_finite_loss"
    return value, None


def _sub_loss_summary(
    expected: range,
    gpu_by_iter: dict[int, dict[str, Any]],
    npu_by_iter: dict[int, dict[str, Any]],
    loss_key: str,
) -> dict[str, dict[str, Any]]:
    metric_names = sorted(
        {
            key
            for row in (*gpu_by_iter.values(), *npu_by_iter.values())
            for key in row
            if "loss" in key.lower() and key != loss_key
        }
    )
    summaries: dict[str, dict[str, Any]] = {}
    for name in metric_names:
        compared = 0
        missing = 0
        invalid = 0
        zero_denominator = 0
        maximum: tuple[float, int] | None = None
        for iteration in expected:
            gpu_row = gpu_by_iter.get(iteration)
            npu_row = npu_by_iter.get(iteration)
            if gpu_row is None or npu_row is None or name not in gpu_row or name not in npu_row:
                missing += 1
                continue
            gpu_value, gpu_error = _metric_value(gpu_row, name)
            npu_value, npu_error = _metric_value(npu_row, name)
            if gpu_error or npu_error:
                invalid += 1
                continue
            assert gpu_value is not None and npu_value is not None
            if gpu_value == 0.0:
                zero_denominator += 1
                continue
            deviation = abs(npu_value - gpu_value) / abs(gpu_value)
            compared += 1
            if maximum is None or deviation > maximum[0]:
                maximum = (deviation, iteration)
        summaries[name] = {
            "compared_count": compared,
            "invalid_count": invalid,
            "max_relative_deviation": maximum[0] if maximum else None,
            "max_relative_deviation_iter": maximum[1] if maximum else None,
            "missing_count": missing,
            "zero_denominator_count": zero_denominator,
        }
    return summaries


def compare(
    gpu_rows: list[tuple[int, dict[str, Any]]],
    npu_rows: list[tuple[int, dict[str, Any]]],
    *,
    threshold: float,
    start_iter: int | None,
    end_iter: int | None,
    loss_key: str,
    include_sub_losses: bool,
) -> dict[str, Any]:
    if threshold < 0 or not math.isfinite(threshold):
        raise InputError("threshold must be a finite non-negative fraction")

    natural_start = min(_first_iteration(gpu_rows), _first_iteration(npu_rows))
    natural_end = max(_last_iteration(gpu_rows), _last_iteration(npu_rows))
    start = natural_start if start_iter is None else start_iter
    end = natural_end if end_iter is None else end_iter
    if start < 0 or end < start:
        raise InputError("iteration range must satisfy 0 <= start <= end")

    gpu_selected = _select_rows(gpu_rows, start, end)
    npu_selected = _select_rows(npu_rows, start, end)
    gpu_counts = Counter(iteration for iteration, _ in gpu_selected)
    npu_counts = Counter(iteration for iteration, _ in npu_selected)
    gpu_by_iter = {iteration: row for iteration, row in gpu_selected}
    npu_by_iter = {iteration: row for iteration, row in npu_selected}
    expected = range(start, end + 1)

    failures: list[dict[str, Any]] = []
    pass_count = 0
    maximum: tuple[float, int] | None = None
    for iteration in expected:
        reasons: list[str] = []
        if gpu_counts[iteration] == 0:
            reasons.append("gpu_missing")
        elif gpu_counts[iteration] > 1:
            reasons.append("gpu_duplicate")
        if npu_counts[iteration] == 0:
            reasons.append("npu_missing")
        elif npu_counts[iteration] > 1:
            reasons.append("npu_duplicate")

        deviation: float | None = None
        if not reasons:
            gpu_value, gpu_error = _metric_value(gpu_by_iter[iteration], loss_key)
            npu_value, npu_error = _metric_value(npu_by_iter[iteration], loss_key)
            if gpu_error:
                reasons.append(f"gpu_{gpu_error}")
            if npu_error:
                reasons.append(f"npu_{npu_error}")
            if not reasons:
                assert gpu_value is not None and npu_value is not None
                if gpu_value == 0.0:
                    if npu_value == 0.0:
                        deviation = 0.0
                    else:
                        reasons.append("gpu_zero_npu_nonzero")
                else:
                    deviation = abs(npu_value - gpu_value) / abs(gpu_value)
                if deviation is not None:
                    if maximum is None or deviation > maximum[0]:
                        maximum = (deviation, iteration)
                    if deviation > threshold:
                        reasons.append("threshold_exceeded")
        if reasons:
            failure: dict[str, Any] = {"iter": iteration, "reasons": reasons}
            if deviation is not None:
                failure["relative_deviation"] = deviation
            failures.append(failure)
        else:
            pass_count += 1

    summary: dict[str, Any] = {
        "status": "pass" if not failures else "fail",
        "threshold": threshold,
        "iter_start": start,
        "iter_end": end,
        "expected_count": len(expected),
        "pass_count": pass_count,
        "fail_count": len(failures),
        "failure_reason_counts": dict(
            sorted(Counter(reason for failure in failures for reason in failure["reasons"]).items())
        ),
        "first_failure": failures[0] if failures else None,
        "max_relative_deviation": maximum[0] if maximum else None,
        "max_relative_deviation_iter": maximum[1] if maximum else None,
    }
    if include_sub_losses:
        summary["sub_losses"] = _sub_loss_summary(
            expected, gpu_by_iter, npu_by_iter, loss_key
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, type=Path, help="GPU JSON or training log")
    parser.add_argument("--npu", required=True, type=Path, help="NPU training log or JSON")
    parser.add_argument("--gpu-format", choices=("auto", "json", "log"), default="auto")
    parser.add_argument("--npu-format", choices=("auto", "json", "log"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.02, help="relative fraction (default: 0.02)")
    parser.add_argument("--start-iter", type=int, help="inclusive comparison start")
    parser.add_argument("--end-iter", type=int, help="inclusive comparison end")
    parser.add_argument("--loss-key", default="loss", help="total-loss field name")
    parser.add_argument("--sub-losses", action="store_true", help="summarize all other loss fields")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = compare(
            load_rows(args.gpu, args.gpu_format),
            load_rows(args.npu, args.npu_format),
            threshold=args.threshold,
            start_iter=args.start_iter,
            end_iter=args.end_iter,
            loss_key=args.loss_key,
            include_sub_losses=args.sub_losses,
        )
    except InputError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, separators=(",", ":")))
        return 2
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
