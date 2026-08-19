#!/usr/bin/env python3
"""Standard-library controller for STEP-214-G rank/file/npu-smi handshake."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path


EXPECTED_RANKS = tuple(range(8))
EXPECTED_VISIBLE = "8,9,10,11,12,13,14,15"


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def validate_ready(root: Path) -> list[dict]:
    ready = root / "ready"
    names = sorted(path.name for path in ready.glob("rank*.json"))
    expected = [f"rank{i}.json" for i in EXPECTED_RANKS]
    if names != expected:
        raise RuntimeError(f"ready file set mismatch: {names}")
    rows = [json.loads((ready / name).read_text(encoding="utf-8")) for name in expected]
    if [row["rank"] for row in rows] != list(EXPECTED_RANKS):
        raise RuntimeError("rank sequence mismatch")
    if [row["local_rank"] for row in rows] != list(EXPECTED_RANKS):
        raise RuntimeError("local rank sequence mismatch")
    if not all(row["world_size"] == 8 for row in rows):
        raise RuntimeError("world size mismatch")
    if not all(row["visible"] == EXPECTED_VISIBLE for row in rows):
        raise RuntimeError("visible device contract mismatch")
    if not all(row["exact"] and row["max_abs_diff"] == 0.0 for row in rows):
        raise RuntimeError("vector add exactness mismatch")
    if len({row["pid"] for row in rows}) != 8:
        raise RuntimeError("rank PID uniqueness mismatch")
    return rows


def validate_npu_smi(text: str) -> list[tuple[int, int, int]]:
    matches = re.findall(
        r"^\|\s*([4-7])\s+([01])\s+\|\s*(\d+)\s+\|",
        text,
        flags=re.MULTILINE,
    )
    rows = [(int(npu), int(chip), int(pid)) for npu, chip, pid in matches]
    expected_pairs = {(npu, chip) for npu in range(4, 8) for chip in range(2)}
    if len(rows) != 8 or {(npu, chip) for npu, chip, _ in rows} != expected_pairs:
        raise RuntimeError(f"physical back8 npu-smi rows mismatch: {rows}")
    if len({pid for _, _, pid in rows}) != 8:
        raise RuntimeError("npu-smi PID uniqueness mismatch")
    return rows


def install_signal_deadline() -> None:
    def handler(signum, _frame):
        raise TimeoutError(f"controller received signal {signum}")

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def controller(root: Path, launcher_pid: int, timeout_seconds: int) -> None:
    release = root / "release_after_npu_smi"
    status_path = root / "controller_status.json"
    started = time.monotonic()
    status: dict = {"status": "STARTED", "release_created": False}
    try:
        deadline = started + timeout_seconds
        expected_names = {f"rank{i}.json" for i in EXPECTED_RANKS}
        while True:
            failures = list((root / "failure").glob("rank*.txt")) if (root / "failure").is_dir() else []
            if failures:
                raise RuntimeError(f"rank failure before live evidence: {[p.name for p in failures]}")
            names = {path.name for path in (root / "ready").glob("rank*.json")} if (root / "ready").is_dir() else set()
            if names == expected_names:
                break
            if not names.issubset(expected_names):
                raise RuntimeError(f"unexpected ready files: {sorted(names - expected_names)}")
            try:
                os.kill(launcher_pid, 0)
            except ProcessLookupError as exc:
                raise RuntimeError("torchrun exited before eight ready files") from exc
            if time.monotonic() >= deadline:
                raise TimeoutError(f"ready timeout: {len(names)}/8")
            time.sleep(0.25)

        ready_rows = validate_ready(root)
        remaining = max(1, int(deadline - time.monotonic()))
        if remaining < 5:
            raise TimeoutError("insufficient time remaining for live npu-smi")
        result = subprocess.run(
            ["npu-smi", "info"],
            check=True,
            capture_output=True,
            text=True,
            timeout=min(40, remaining),
        )
        (root / "npu_smi_while_live.txt").write_text(result.stdout, encoding="utf-8")
        physical_rows = validate_npu_smi(result.stdout)
        status.update(
            status="PASS",
            logical_rank_count=len(ready_rows),
            physical_process_count=len(physical_rows),
            physical_pairs=[[npu, chip] for npu, chip, _ in physical_rows],
        )
    except BaseException as exc:
        status.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        release.touch(exist_ok=True)
        status["release_created"] = True
        status["elapsed_seconds"] = time.monotonic() - started
        atomic_json(status_path, status)


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="step214_g_controller_") as temp:
        root = Path(temp)
        (root / "ready").mkdir()
        for rank in EXPECTED_RANKS:
            atomic_json(
                root / "ready" / f"rank{rank}.json",
                {
                    "rank": rank,
                    "local_rank": rank,
                    "world_size": 8,
                    "visible": EXPECTED_VISIBLE,
                    "pid": 10000 + rank,
                    "exact": True,
                    "max_abs_diff": 0.0,
                },
            )
        validate_ready(root)
        synthetic = "\n".join(
            f"| {npu}       {chip}                 | {20000 + npu * 2 + chip}       | rank | 133 |"
            for npu in range(4, 8)
            for chip in range(2)
        )
        validate_npu_smi(synthetic)
        release = root / "release_after_npu_smi"
        try:
            raise RuntimeError("intentional finally-path self-test")
        except RuntimeError:
            pass
        finally:
            release.touch()
        if not release.is_file():
            raise RuntimeError("release finally-path self-test failed")
    print("controller_file_protocol_self_test=PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--launcher-pid", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=105)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.output_dir or args.launcher_pid is None:
        parser.error("--output-dir and --launcher-pid are required")
    install_signal_deadline()
    controller(Path(args.output_dir).resolve(strict=True), args.launcher_pid, args.timeout_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
