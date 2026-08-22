#!/usr/bin/env python3
"""Host supervisor for the STEP358 rear-eight-card release math gate."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path, PurePosixPath

import step343_world8_controller as legacy


VISIBLE = "8,9,10,11,12,13,14,15"
RANKS = tuple(range(8))


def append_error(current: str | None, label: str, error: BaseException) -> str:
    detail = f"{label}: {type(error).__name__}: {error}"
    return f"{current}; {detail}" if current else detail


def best_effort_write_text(path: Path, text: str) -> str | None:
    """Persist controller evidence without replacing the primary run result."""

    try:
        path.write_text(text, encoding="utf-8")
    except BaseException as error:
        return f"{path.name}: {type(error).__name__}: {error}"
    return None


def terminate_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=20)


def snapshot_owned_npu_processes(
    root: Path,
    process: subprocess.Popen[bytes],
    worker: Path,
) -> dict[str, int]:
    """Record only rank processes whose exact command names this STEP358 case."""

    worker_token = str(worker.resolve(strict=True)).encode()
    root_token = str(root.resolve(strict=True)).encode()
    ownership = json.loads((root / "launcher_ownership.json").read_text(encoding="utf-8"))
    launcher_starttime = int(ownership["launcher_starttime"])
    physical = legacy.parse_back8(legacy.npu_smi())
    owned: dict[str, int] = {}
    for _, _, host_pid in physical:
        cmdline_path = Path("/proc") / str(host_pid) / "cmdline"
        try:
            tokens = [token for token in cmdline_path.read_bytes().split(b"\0") if token]
            if worker_token not in tokens or root_token not in tokens:
                continue
            owned[str(host_pid)] = legacy.process_starttime(host_pid)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    status_path = root / "controller_status.json"
    if status_path.is_symlink():
        raise RuntimeError("controller status must not be a symlink")
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            int(status["launcher_host_pid"]) != process.pid
            or int(status["launcher_starttime"])
            != launcher_starttime
        ):
            raise RuntimeError("partial cleanup snapshot launcher identity mismatch")
    else:
        status = {
            "status": "PARTIAL_OWNERSHIP_SNAPSHOT",
            "launcher_host_pid": process.pid,
            "launcher_starttime": launcher_starttime,
            "release_created": (root / "release_after_npu_smi").exists(),
        }
    previous = {
        str(pid): int(starttime)
        for pid, starttime in status.get("npu_host_pid_starttimes", {}).items()
    }
    for pid, starttime in previous.items():
        if pid in owned and owned[pid] != starttime:
            raise RuntimeError("owned rank PID starttime changed during snapshot")
    previous.update(owned)
    status["npu_host_pid_starttimes"] = previous
    status["npu_host_pids"] = sorted(int(pid) for pid in previous)
    legacy.atomic_json(status_path, status)
    return previous


def wait_for_results(root: Path, process: subprocess.Popen[bytes], timeout_seconds: int) -> dict:
    deadline = time.monotonic() + timeout_seconds
    expected = {f"rank{rank}.json" for rank in RANKS}
    while True:
        failures = sorted(path.name for path in (root / "failure").glob("rank*.txt"))
        if failures:
            raise RuntimeError(f"rank failure before live gate: {failures}")
        ready = {path.name for path in (root / "ready").glob("rank*.json")}
        if ready == expected:
            break
        if not ready.issubset(expected):
            raise RuntimeError(f"unexpected ready files: {sorted(ready - expected)}")
        if process.poll() is not None:
            raise RuntimeError(f"torchrun exited before ready: rc={process.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"ready timeout: {len(ready)}/8")
        time.sleep(0.25)

    rows = [json.loads((root / "ready" / f"rank{rank}.json").read_text()) for rank in RANKS]
    if not all(
        row.get("rank") == rank
        and row.get("local_rank") == rank
        and row.get("world_size") == 8
        and row.get("visible") == VISIBLE
        and row.get("npu_available") is True
        and row.get("device_count") == 8
        and row.get("gate_pass") is True
        and row.get("shadow_gate") is True
        and row.get("opp_first_shadow") is True
        and row.get("custom_opp_role_sequence") == ["shadow", "base"]
        and row.get("wrapper_contract", {}).get("gate") == "PASS"
        for rank, row in enumerate(rows)
    ):
        raise RuntimeError("ready rank/device/shadow gate failed")
    ready_pids = [int(row["container_pid"]) for row in rows]
    if len(set(ready_pids)) != 8:
        raise RuntimeError("ready rank PIDs are not unique")

    smi = legacy.npu_smi()
    (root / "npu_smi_while_live.txt").write_text(smi, encoding="utf-8")
    physical = legacy.parse_back8(smi)
    device_ids = {physical_id * 2 + chip_id for physical_id, chip_id, _ in physical}
    if (
        len(physical) != 8
        or {(a, b) for a, b, _ in physical} != legacy.BACK8_PAIRS
        or device_ids != legacy.BACK8_DEVICE_IDS
    ):
        raise RuntimeError(f"live rear-eight-card mapping mismatch: {physical}")
    mapped_pids = [legacy.container_pid(pid) for _, _, pid in physical]
    if set(mapped_pids) != set(ready_pids):
        raise RuntimeError("npu-smi PIDs do not equal rank PIDs")
    by_container = {
        legacy.container_pid(host_pid): physical_id * 2 + chip_id
        for physical_id, chip_id, host_pid in physical
    }
    rank_mapping = legacy.validate_rank_device_mapping(rows, by_container)
    controller = {
        "status": "LIVE_BINDING_PASS",
        "rank_count": 8,
        "physical_device_ids": sorted(device_ids),
        "rank_device_mapping": rank_mapping,
        "launcher_host_pid": process.pid,
        "launcher_starttime": legacy.process_starttime(process.pid),
        "direct_rank_container_pids": ready_pids,
        "npu_host_pids": [pid for _, _, pid in physical],
        "npu_host_pid_starttimes": {
            str(pid): legacy.process_starttime(pid) for _, _, pid in physical
        },
        "release_created": True,
    }
    legacy.atomic_json(root / "controller_status.json", controller)
    (root / "release_after_npu_smi").touch()

    while True:
        failures = sorted(path.name for path in (root / "failure").glob("rank*.txt"))
        if failures:
            raise RuntimeError(f"rank failure after release: {failures}")
        done = {path.name for path in (root / "done").glob("rank*.json")}
        if done == expected:
            controller["status"] = "PASS"
            legacy.atomic_json(root / "controller_status.json", controller)
            return controller
        if process.poll() is not None:
            raise RuntimeError(f"torchrun exited before done: rc={process.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"done timeout: {len(done)}/8")
        time.sleep(0.25)


def run(args: argparse.Namespace) -> int:
    for raw_path, label in (
        (args.shadow_root, "shadow root"),
        (args.installed_custom_opp, "installed custom OPP"),
        (args.input_dir, "STEP260 input directory"),
        (args.worker, "STEP358 worker"),
    ):
        if raw_path.is_symlink():
            raise RuntimeError(f"{label} must not be a symlink")
    root = args.output_dir.resolve(strict=True)
    shadow_root = args.shadow_root.resolve(strict=True)
    installed_custom_opp = PurePosixPath(str(args.installed_custom_opp))
    if not installed_custom_opp.is_absolute() or ".." in installed_custom_opp.parts:
        raise RuntimeError("installed custom OPP must be an absolute container path")
    worker = args.worker.resolve(strict=True)
    input_dir = args.input_dir.resolve(strict=True)
    shadow_custom_opp = (
        shadow_root / "mx_driving_cloud/packages/vendors/customize"
    ).resolve(strict=True)
    for path, label in (
        (shadow_root, "shadow root"),
        (shadow_custom_opp, "shadow custom OPP"),
        (input_dir, "STEP260 input directory"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"{label} must be a regular non-symlink directory")
    if shadow_custom_opp == installed_custom_opp:
        raise RuntimeError("shadow and installed custom OPP paths must differ")
    for name in ("ready", "done", "failure"):
        (root / name).mkdir(exist_ok=False)
    legacy.preflight(root, args.port)
    inner = (
        "torchrun --nnodes=1 --nproc-per-node=8 --master-addr=127.0.0.1 "
        f"--master-port={args.port} {shlex.quote(str(worker))} "
        f"--input-dir {shlex.quote(str(input_dir))} "
        f"--output-dir {shlex.quote(str(root))} "
        f"--shadow-root {shlex.quote(str(shadow_root))} "
        f"--installed-custom-opp {shlex.quote(str(installed_custom_opp))}"
        + (" --first-profiled-only" if args.first_profiled_only else "")
    )
    command = [
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        "900s",
        "docker",
        "exec",
        "-e",
        f"ASCEND_RT_VISIBLE_DEVICES={VISIBLE}",
        "-e",
        f"ASCEND_CUSTOM_OPP_PATH={shadow_custom_opp}",
        "-e",
        f"PYTHONPATH={shadow_root}",
        "-e",
        "TORCH_DEVICE_BACKEND_AUTOLOAD=0",
        *(
            ["-e", "STEP358_STATE_DIAGNOSTIC_ONLY=1"]
            if args.state_diagnostic_only
            else []
        ),
        "mapqr-leicheng",
        "bash",
        "--noprofile",
        "--norc",
        "-lc",
        inner,
    ]
    log = (root / "torchrun.log").open("xb")
    process: subprocess.Popen[bytes] | None = None
    controller_rc = 0
    controller_error: str | None = None
    postflight_rc = 123
    try:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            legacy.atomic_json(
                root / "launcher_ownership.json",
                {
                    "schema": "step358-launcher-ownership-v1",
                    "port": args.port,
                    "launcher_host_pid": process.pid,
                    "launcher_starttime": legacy.process_starttime(process.pid),
                    "launcher_pgid": os.getpgid(process.pid),
                },
            )
            wait_for_results(root, process, 820)
        except BaseException as error:
            controller_rc = 122
            controller_error = f"{type(error).__name__}: {error}"
            try:
                snapshot_owned_npu_processes(root, process, worker)
            except BaseException as snapshot_error:
                controller_error = append_error(
                    controller_error,
                    "ownership snapshot failed",
                    snapshot_error,
                )
        try:
            process.wait(timeout=30 if controller_rc == 0 else 0.1)
        except subprocess.TimeoutExpired:
            pass  # The unified finally block terminates and verifies cleanup.
        except BaseException as error:
            controller_rc = controller_rc or 122
            controller_error = append_error(
                controller_error, "launcher wait failed", error
            )
    finally:
        if process is not None:
            try:
                terminate_group(process)
            except BaseException as error:
                controller_rc = controller_rc or 122
                controller_error = append_error(
                    controller_error, "termination failed", error
                )
        try:
            log.close()
        except BaseException as error:
            controller_rc = controller_rc or 122
            controller_error = append_error(
                controller_error, "launcher log close failed", error
            )
        try:
            postflight_rc = legacy.cleanup_owned_and_postflight(root, args.port)
        except BaseException as error:
            postflight_rc = 123
            write_error = best_effort_write_text(
                root / "postflight_error.txt",
                f"{type(error).__name__}: {error}\n",
            )
            if write_error:
                controller_error = append_error(
                    controller_error,
                    "postflight evidence write failed",
                    RuntimeError(write_error),
                )
    if process is None:
        raise RuntimeError("launcher was not created")
    launcher_rc = process.returncode if process.returncode is not None else 124
    evidence_errors: list[str] = []
    for path, text in (
        (root / "controller_rc.txt", str(controller_rc) + "\n"),
        (root / "launcher_rc.txt", str(launcher_rc) + "\n"),
        (root / "postflight_rc.txt", str(postflight_rc) + "\n"),
    ):
        write_error = best_effort_write_text(path, text)
        if write_error:
            evidence_errors.append(write_error)
    if evidence_errors:
        controller_error = append_error(
            controller_error,
            "status evidence write failed",
            RuntimeError("; ".join(evidence_errors)),
        )
    if controller_error:
        best_effort_write_text(root / "controller_error.txt", controller_error + "\n")
    if controller_rc:
        return controller_rc
    if launcher_rc:
        return 121
    return postflight_rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--shadow-root", required=True, type=Path)
    parser.add_argument("--installed-custom-opp", required=True, type=Path)
    parser.add_argument("--state-diagnostic-only", action="store_true")
    parser.add_argument("--first-profiled-only", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
