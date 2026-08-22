#!/usr/bin/env python3
"""Fail-closed live 8-rank and npu-smi process gate for STEP347."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import signal
import stat
import time
from pathlib import Path
from typing import Any


RANKS = tuple(range(8))
VISIBLE = "8,9,10,11,12,13,14,15"
BACK8_PAIRS = {(physical, chip) for physical in range(4, 8) for chip in range(2)}
BACK8_DEVICE_IDS = set(range(8, 16))


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def assert_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        handle.bind(("127.0.0.1", port))


def npu_smi() -> str:
    return subprocess.run(
        ["npu-smi", "info"], check=True, capture_output=True, text=True, timeout=40
    ).stdout


def parse_back8(text: str) -> list[tuple[int, int, int]]:
    rows = [
        (int(physical), int(chip), int(pid))
        for physical, chip, pid in re.findall(
            r"^\|\s*([0-7])\s+([01])\s+\|\s*(\d+)\s+\|", text, re.MULTILINE
        )
    ]
    return [row for row in rows if (row[0], row[1]) in BACK8_PAIRS]


def container_pid(host_pid: int) -> int:
    status = Path("/proc") / str(host_pid) / "status"
    match = re.search(r"^NSpid:\s+([0-9\s]+)$", status.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise RuntimeError(f"NSpid unavailable for host PID {host_pid}")
    return int(match.group(1).split()[-1])


def process_starttime(pid: int) -> int:
    fields = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8").split()
    return int(fields[21])


def wait_for_idle(deadline: float) -> str:
    while True:
        text = npu_smi()
        if not parse_back8(text):
            return text
        if time.monotonic() >= deadline:
            raise RuntimeError(f"back8 remains busy: {parse_back8(text)}")
        time.sleep(1.0)


def preflight(root: Path, port: int) -> int:
    assert_port_free(port)
    text = wait_for_idle(time.monotonic() + 5.0)
    (root / "npu_smi_before.txt").write_text(text, encoding="utf-8")
    atomic_json(root / "preflight_status.json", {"status": "PASS", "port": port, "back8_process_count": 0})
    return 0


def load_ready(root: Path) -> list[dict[str, Any]]:
    names = sorted(path.name for path in (root / "ready").glob("rank*.json"))
    expected = [f"rank{rank}.json" for rank in RANKS]
    if names != expected:
        raise RuntimeError(f"ready file set mismatch: {names}")
    return [json.loads((root / "ready" / name).read_text(encoding="utf-8")) for name in expected]


def validate_rank_device_mapping(
    rows: list[dict[str, Any]], device_by_container_pid: dict[int, int]
) -> list[dict[str, int]]:
    mapping = []
    for row in rows:
        rank = int(row["rank"])
        local_rank = int(row["local_rank"])
        container_pid_value = int(row["container_pid"])
        actual_device = device_by_container_pid.get(container_pid_value)
        expected_device = 8 + local_rank
        if rank != local_rank or actual_device != expected_device:
            raise RuntimeError(
                f"rank/local/device mismatch rank={rank} local_rank={local_rank} "
                f"expected_physical={expected_device} actual_physical={actual_device}"
            )
        mapping.append({"rank": rank, "local_rank": local_rank, "physical_device": actual_device})
    return mapping


def validate_ready_opp_transition(row: dict[str, Any]) -> bool:
    mode = row.get("mode")
    expected_startup = ["cloud"] if mode == "original" else ["overlay", "cloud"]
    expected_after = (
        ["cloud", "base", "cloud"]
        if mode == "original"
        else ["cloud", "base", "overlay", "cloud"]
    )
    expected_restored = ["cloud", "base"] if mode == "original" else ["overlay", "cloud", "base"]
    if mode not in {"original", "candidate"}:
        return False
    transition = row.get("custom_opp_transition")
    if not isinstance(transition, dict):
        return False
    hashes = [
        transition.get("startup_path_sha256"),
        transition.get("after_import_path_sha256"),
        transition.get("restored_path_sha256"),
    ]
    return (
        transition.get("startup_role_sequence") == expected_startup
        and transition.get("after_import_role_sequence") == expected_after
        and transition.get("restored_role_sequence") == expected_restored
        and len(expected_restored) == len(set(expected_restored))
        and transition.get("restored_exact") is True
        and transition.get("restored_paths_unique") is True
        and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes)
        and hashes[2] != hashes[0]
        and hashes[1] != hashes[0]
    )


def supervise(root: Path, launcher_pid: int, timeout_seconds: int) -> int:
    deadline = time.monotonic() + timeout_seconds
    release = root / "release_after_npu_smi"
    status: dict[str, Any] = {"status": "STARTED", "release_created": False}
    status["launcher_host_pid"] = launcher_pid
    status["launcher_starttime"] = process_starttime(launcher_pid)
    try:
        expected = {f"rank{rank}.json" for rank in RANKS}
        while True:
            failures = sorted(path.name for path in (root / "failure").glob("rank*.txt"))
            if failures:
                raise RuntimeError(f"rank failure before live gate: {failures}")
            names = {path.name for path in (root / "ready").glob("rank*.json")}
            if names == expected:
                break
            if not names.issubset(expected):
                raise RuntimeError(f"unexpected ready files: {sorted(names - expected)}")
            os.kill(launcher_pid, 0)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"ready timeout: {len(names)}/8")
            time.sleep(0.25)

        rows = load_ready(root)
        if [row["rank"] for row in rows] != list(RANKS):
            raise RuntimeError("global rank sequence mismatch")
        if [row["local_rank"] for row in rows] != list(RANKS):
            raise RuntimeError("local rank sequence mismatch")
        if not all(
            row["world_size"] == 8
            and row["visible"] == VISIBLE
            and row["device_count"] == 8
            and row["npu_available"] is True
            and row["gate_pass"] is True
            and validate_ready_opp_transition(row)
            for row in rows
        ):
            raise RuntimeError("rank torch_npu/device gate failed")
        ready_pids = [int(row["container_pid"]) for row in rows]
        if len(set(ready_pids)) != 8:
            raise RuntimeError("rank PIDs are not unique")

        text = npu_smi()
        (root / "npu_smi_while_live.txt").write_text(text, encoding="utf-8")
        physical = parse_back8(text)
        device_ids = {physical_id * 2 + chip_id for physical_id, chip_id, _ in physical}
        if (
            len(physical) != 8
            or {(a, b) for a, b, _ in physical} != BACK8_PAIRS
            or device_ids != BACK8_DEVICE_IDS
        ):
            raise RuntimeError(f"live back8 process mapping mismatch: {physical}")
        mapped = [container_pid(pid) for _, _, pid in physical]
        if len(set(mapped)) != 8 or set(mapped) != set(ready_pids):
            raise RuntimeError(f"npu-smi PIDs do not equal rank PIDs: mapped={mapped}, ready={ready_pids}")
        device_by_container_pid = {
            container_pid(host_pid): physical_id * 2 + chip_id
            for physical_id, chip_id, host_pid in physical
        }
        rank_device_mapping = validate_rank_device_mapping(rows, device_by_container_pid)
        status.update(
            status="LIVE_BINDING_PASS",
            logical_rank_count=8,
            physical_process_count=8,
            physical_pairs=[[a, b] for a, b, _ in physical],
            physical_device_ids=sorted(device_ids),
            rank_device_mapping=rank_device_mapping,
            direct_rank_container_pids=ready_pids,
            npu_host_pids=[pid for _, _, pid in physical],
            npu_host_pid_starttimes={str(pid): process_starttime(pid) for _, _, pid in physical},
        )
        release.touch()
        status["release_created"] = True

        while True:
            failures = sorted(path.name for path in (root / "failure").glob("rank*.txt"))
            if failures:
                raise RuntimeError(f"rank failure after release: {failures}")
            done = {path.name for path in (root / "done").glob("rank*.json")}
            if done == expected:
                status["status"] = "PASS"
                return 0
            os.kill(launcher_pid, 0)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"done timeout: {len(done)}/8")
            time.sleep(0.25)
    finally:
        release.touch(exist_ok=True)
        status["release_created"] = True
        atomic_json(root / "controller_status.json", status)


def same_process_alive(pid: int, starttime: int) -> bool:
    try:
        return process_starttime(pid) == starttime
    except FileNotFoundError:
        return False


def terminate_owned(root: Path, port: int) -> int | None:
    """Terminate only processes whose PID and starttime were recorded by this case."""
    ownership_path = root / "launcher_ownership.json"
    if ownership_path.is_symlink():
        raise RuntimeError("launcher ownership must not be a symlink")
    if not ownership_path.is_file():
        return None
    if not stat.S_ISREG(ownership_path.lstat().st_mode):
        raise RuntimeError("launcher ownership must be a regular file")
    ownership = json.loads(ownership_path.read_text(encoding="utf-8"))
    if int(ownership["port"]) != port:
        raise RuntimeError("launcher ownership port mismatch")
    launcher_pid = int(ownership["launcher_host_pid"])
    launcher_starttime = int(ownership["launcher_starttime"])
    launcher_pgid = int(ownership["launcher_pgid"])
    if launcher_pid <= 1:
        raise RuntimeError("unsafe launcher PID in ownership manifest")
    if launcher_pgid != launcher_pid:
        raise RuntimeError("owned launcher was not recorded as a dedicated process-group leader")

    tracked: dict[int, int] = {launcher_pid: launcher_starttime}
    controller_path = root / "controller_status.json"
    if controller_path.is_symlink():
        raise RuntimeError("controller status must not be a symlink")
    if controller_path.is_file():
        controller = json.loads(controller_path.read_text(encoding="utf-8"))
        if (
            int(controller["launcher_host_pid"]) != launcher_pid
            or int(controller["launcher_starttime"]) != launcher_starttime
        ):
            raise RuntimeError("controller ownership differs from launcher ownership")
        tracked.update({
            int(pid): int(starttime)
            for pid, starttime in controller.get("npu_host_pid_starttimes", {}).items()
        })

    if same_process_alive(launcher_pid, launcher_starttime):
        if os.getpgid(launcher_pid) != launcher_pgid:
            raise RuntimeError("live launcher process-group identity mismatch")
        os.killpg(launcher_pgid, signal.SIGTERM)
    for pid, starttime in tracked.items():
        if pid != launcher_pid and same_process_alive(pid, starttime):
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while any(same_process_alive(pid, starttime) for pid, starttime in tracked.items()):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    if same_process_alive(launcher_pid, launcher_starttime):
        if os.getpgid(launcher_pid) != launcher_pgid:
            raise RuntimeError("launcher process-group changed before SIGKILL")
        os.killpg(launcher_pgid, signal.SIGKILL)
    for pid, starttime in tracked.items():
        if pid != launcher_pid and same_process_alive(pid, starttime):
            os.kill(pid, signal.SIGKILL)
    return launcher_pid


def cleanup_owned_and_postflight(root: Path, port: int) -> int:
    launcher_pid = terminate_owned(root, port)
    controller_path = root / "controller_status.json"
    if launcher_pid is not None and controller_path.is_file() and not controller_path.is_symlink():
        postflight(root, port, launcher_pid)
    else:
        assert_port_free(port)
        text = npu_smi()
        if parse_back8(text):
            raise RuntimeError(f"back8 is not clean after owned cleanup: {parse_back8(text)}")
        (root / "npu_smi_after_cleanup.txt").write_text(text, encoding="utf-8")
    atomic_json(root / "finally_cleanup_status.json", {
        "status": "PASS", "port": port, "owned_launcher_seen": launcher_pid is not None,
        "back8_process_count": 0,
    })
    return 0


def postflight(root: Path, port: int, launcher_pid: int) -> int:
    deadline = time.monotonic() + 30.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            assert_port_free(port)
            text = npu_smi()
            if parse_back8(text):
                raise RuntimeError(f"back8 still has processes: {parse_back8(text)}")
            controller = json.loads((root / "controller_status.json").read_text(encoding="utf-8"))
            if int(controller["launcher_host_pid"]) != launcher_pid:
                raise RuntimeError("postflight launcher PID differs from supervised launcher")
            if same_process_alive(launcher_pid, int(controller["launcher_starttime"])):
                raise RuntimeError("launcher process remains alive with original starttime")
            lingering = [
                int(pid)
                for pid, starttime in controller.get("npu_host_pid_starttimes", {}).items()
                if same_process_alive(int(pid), int(starttime))
            ]
            if lingering:
                raise RuntimeError(f"rank host processes remain alive: {lingering}")
            (root / "npu_smi_after.txt").write_text(text, encoding="utf-8")
            atomic_json(root / "postflight_status.json", {
                "status": "PASS", "port": port, "back8_process_count": 0,
                "launcher_exited": True, "rank_host_processes_exited": True,
            })
            return 0
        except (OSError, RuntimeError) as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"postflight did not become clean: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--launcher-pid", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--postflight-only", action="store_true")
    parser.add_argument("--cleanup-owned", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve(strict=True)
    if sum((args.preflight_only, args.postflight_only, args.cleanup_owned)) > 1:
        parser.error("preflight, postflight, and cleanup-owned modes are mutually exclusive")
    if args.preflight_only:
        return preflight(root, args.port)
    if args.postflight_only:
        if args.launcher_pid is None:
            parser.error("postflight requires --launcher-pid")
        return postflight(root, args.port, args.launcher_pid)
    if args.cleanup_owned:
        return cleanup_owned_and_postflight(root, args.port)
    if args.launcher_pid is None:
        raise ValueError("--launcher-pid is required for supervision")
    return supervise(root, args.launcher_pid, args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
