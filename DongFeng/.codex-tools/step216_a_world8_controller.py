#!/usr/bin/env python3
"""Fail-closed world8/NPU PID controller for STEP-216-A."""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


RANKS = range(8)
VISIBLE = "8,9,10,11,12,13,14,15"
BACK8_PAIRS = {(physical, chip) for physical in range(4, 8) for chip in range(2)}


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def parse_npu_process_rows(text: str) -> list[tuple[int, int, int]]:
    return [
        (int(physical), int(chip), int(pid))
        for physical, chip, pid in re.findall(
            r"^\|\s*([0-7])\s+([01])\s+\|\s*(\d+)\s+\|", text, re.MULTILINE
        )
    ]


def npu_smi() -> str:
    return subprocess.run(
        ["npu-smi", "info"], check=True, capture_output=True, text=True, timeout=40
    ).stdout


def back8_rows(text: str) -> list[tuple[int, int, int]]:
    return [row for row in parse_npu_process_rows(text) if (row[0], row[1]) in BACK8_PAIRS]


def assert_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        handle.bind(("127.0.0.1", int(port)))


def nspid_chain_from_status(path: str | Path) -> list[int]:
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"^NSpid:\s+([0-9\s]+)$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError("/proc status does not expose NSpid")
    values = [int(value) for value in match.group(1).split()]
    if not values:
        raise RuntimeError("NSpid is empty")
    return values


def container_pid_for_host_pid(host_pid: int, proc_root: str | Path = "/proc") -> int:
    if int(host_pid) <= 1:
        raise ValueError("host PID must be greater than 1")
    return nspid_chain_from_status(Path(proc_root) / str(int(host_pid)) / "status")[-1]


def build_pid_namespace_mapping(
    ready_container_pids: list[int], npu_text: str, proc_root: str | Path = "/proc"
) -> list[dict[str, int]]:
    rows = back8_rows(npu_text)
    if len(rows) != 8 or {(physical, chip) for physical, chip, _ in rows} != BACK8_PAIRS:
        raise RuntimeError(f"live back8 physical mapping failed: {rows}")
    mapping = [
        {
            "physical": physical,
            "chip": chip,
            "host_pid": host_pid,
            "container_pid": container_pid_for_host_pid(host_pid, proc_root),
        }
        for physical, chip, host_pid in rows
    ]
    mapped = [row["container_pid"] for row in mapping]
    if len(set(mapped)) != 8 or set(mapped) != set(ready_container_pids):
        raise RuntimeError(
            f"npu-smi host PID NSpid mapping does not equal ready container PIDs: mapping={mapping}, ready={ready_container_pids}"
        )
    return mapping


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def load_ready(root: Path) -> list[dict[str, Any]]:
    return [json.loads((root / "ready" / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in RANKS]


def preflight(root: Path, port: int) -> int:
    assert_port_free(port)
    text = npu_smi()
    (root / "npu_smi_before.txt").write_text(text, encoding="utf-8")
    rows = back8_rows(text)
    if rows:
        raise RuntimeError(f"back8 is not idle: {rows}")
    atomic_json(root / "preflight_status.json", {"status": "PASS", "port": port, "back8_process_count": 0})
    return 0


def postflight(root: Path, port: int) -> int:
    assert_port_free(port)
    text = npu_smi()
    (root / "npu_smi_after.txt").write_text(text, encoding="utf-8")
    rows = back8_rows(text)
    if rows:
        raise RuntimeError(f"back8 still has processes: {rows}")
    controller = json.loads((root / "controller_status.json").read_text(encoding="utf-8")) if (root / "controller_status.json").is_file() else {}
    still_alive = [int(pid) for pid in controller.get("npu_host_pids", []) if pid_alive(int(pid))]
    if still_alive:
        raise RuntimeError(f"direct rank host PIDs still alive: {still_alive}")
    atomic_json(root / "postflight_status.json", {
        "status": "PASS", "port": port, "back8_process_count": 0,
        "direct_rank_host_pid_alive_count": 0,
    })
    return 0


def supervise(root: Path, launcher_pid: int, timeout_seconds: int, proc_root: str | Path = "/proc") -> int:
    release = root / "release_after_npu_smi"
    started = time.monotonic()
    deadline = started + timeout_seconds
    status: dict[str, Any] = {"status": "STARTED", "release_created": False}

    def interrupted(signum: int, _frame: object) -> None:
        raise TimeoutError(f"controller received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        expected = {f"rank{rank}.json" for rank in RANKS}
        while True:
            failures = list((root / "failure").glob("rank*.txt")) if (root / "failure").is_dir() else []
            if failures:
                raise RuntimeError(f"rank failed before ready: {[path.name for path in failures]}")
            names = {path.name for path in (root / "ready").glob("rank*.json")} if (root / "ready").is_dir() else set()
            if names == expected:
                break
            os.kill(launcher_pid, 0)
            if time.monotonic() > deadline:
                raise TimeoutError(f"ready {len(names)}/8")
            time.sleep(0.25)

        ready = load_ready(root)
        if [row["rank"] for row in ready] != list(RANKS) or [row["local_rank"] for row in ready] != list(RANKS):
            raise RuntimeError("rank/local-rank mapping changed")
        if not all(row["world_size"] == 8 and row["visible"] == VISIBLE and row["gate_pass"] for row in ready):
            raise RuntimeError("ready payload gate failed")
        container_pids = [int(row["container_pid"]) for row in ready]
        if len(set(container_pids)) != 8:
            raise RuntimeError("rank container PIDs are not unique")
        text = npu_smi()
        (root / "npu_smi_while_live.txt").write_text(text, encoding="utf-8")
        mapping = build_pid_namespace_mapping(container_pids, text, proc_root)
        status.update(
            status="LIVE_BINDING_PASS",
            logical_rank_count=8,
            physical_process_count=8,
            direct_rank_container_pids=container_pids,
            npu_host_pids=[row["host_pid"] for row in mapping],
            pid_namespace_mapping=mapping,
            physical_pairs=[[row["physical"], row["chip"]] for row in mapping],
        )
        release.touch(exist_ok=True)
        status["release_created"] = True

        while True:
            failures = list((root / "failure").glob("rank*.txt")) if (root / "failure").is_dir() else []
            if failures:
                raise RuntimeError(f"rank failure after release: {[path.name for path in failures]}")
            done = {path.name for path in (root / "done").glob("rank*.json")} if (root / "done").is_dir() else set()
            if done == expected:
                status["status"] = "PASS"
                break
            os.kill(launcher_pid, 0)
            if time.monotonic() > deadline:
                raise TimeoutError(f"done {len(done)}/8")
            time.sleep(0.5)
        return 0
    finally:
        release.touch(exist_ok=True)
        status["release_created"] = True
        status["elapsed_seconds"] = time.monotonic() - started
        atomic_json(root / "controller_status.json", status)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--launcher-pid", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--proc-root", default="/proc")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--postflight-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve(strict=True)
    if args.preflight_only == args.postflight_only and args.launcher_pid is None:
        raise ValueError("select exactly one flight mode or provide launcher PID")
    if args.preflight_only:
        return preflight(root, args.port)
    if args.postflight_only:
        return postflight(root, args.port)
    if args.launcher_pid is None:
        raise ValueError("launcher PID is required for supervision")
    return supervise(root, args.launcher_pid, args.timeout_seconds, args.proc_root)


if __name__ == "__main__":
    raise SystemExit(main())
