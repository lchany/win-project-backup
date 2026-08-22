#!/usr/bin/env python3
"""Host-side supervisor; the container runs only torchrun and the QR worker."""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
from pathlib import Path

from step343_world8_controller import atomic_json, postflight, preflight, process_starttime, supervise


VISIBLE = "8,9,10,11,12,13,14,15"
SUPERVISOR_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


class SupervisorSignal(RuntimeError):
    """Turns an external termination request into normal Python unwinding."""


def install_signal_handlers() -> dict[signal.Signals, signal.Handlers]:
    previous = {signum: signal.getsignal(signum) for signum in SUPERVISOR_SIGNALS}
    state = {"raised": False}

    def handler(signum: int, _frame: object) -> None:
        if state["raised"]:
            return
        state["raised"] = True
        raise SupervisorSignal(f"host supervisor received signal {signum}")

    for signum in SUPERVISOR_SIGNALS:
        signal.signal(signum, handler)
    return previous


def restore_signal_handlers(previous: dict[signal.Signals, signal.Handlers]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def terminate_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def run(args: argparse.Namespace) -> int:
    root = Path(args.output_dir).resolve(strict=True)
    preflight(root, args.port)
    inner = (
        f"cd {shlex.quote(str(root))} && torchrun --nnodes=1 --nproc-per-node=8 --master-addr=127.0.0.1 "
        f"--master-port={args.port} {shlex.quote(args.worker)} --mode {args.mode} "
        f"--input-dir {shlex.quote(args.input_dir)} --output-dir {shlex.quote(str(root))} "
        f"--expected-kernel {shlex.quote(args.expected_kernel)} "
        f"--original-kernel {shlex.quote(args.original_kernel)} "
        f"--installed-custom-opp {shlex.quote(args.installed_custom_opp)}"
    )
    if args.overlay_custom_opp is not None:
        inner += f" --overlay-custom-opp {shlex.quote(args.overlay_custom_opp)}"
    command = [
        "timeout", "--signal=TERM", "--kill-after=30s", "240s", "docker", "exec",
        "-e", f"ASCEND_RT_VISIBLE_DEVICES={VISIBLE}",
        "-e", f"ASCEND_CUSTOM_OPP_PATH={args.custom_opp}",
        "-e", "TORCH_DEVICE_BACKEND_AUTOLOAD=0", "-e", "MASTER_ADDR=127.0.0.1",
        "-e", f"MASTER_PORT={args.port}", "mapqr-leicheng", "bash", "--noprofile", "--norc", "-lc", inner,
    ]
    previous_handlers = install_signal_handlers()
    log = None
    process: subprocess.Popen[bytes] | None = None
    controller_rc = 0
    controller_error = None
    try:
        log = (root / "torchrun.log").open("wb")
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        atomic_json(root / "launcher_ownership.json", {
            "schema": "step347-launcher-ownership-v1",
            "mode": args.mode,
            "port": args.port,
            "launcher_host_pid": process.pid,
            "launcher_starttime": process_starttime(process.pid),
            "launcher_pgid": os.getpgid(process.pid),
        })
        try:
            supervise(root, process.pid, 220)
        except BaseException as exc:
            controller_rc = 122
            controller_error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                process.wait(timeout=30 if controller_rc == 0 else 0.1)
            except subprocess.TimeoutExpired:
                terminate_group(process)
    finally:
        if process is not None:
            terminate_group(process)
        if log is not None:
            log.close()
        restore_signal_handlers(previous_handlers)
    if process is None:
        raise RuntimeError("launcher process was not created")
    launcher_rc = process.returncode if process.returncode is not None else 124
    try:
        postflight_rc = postflight(root, args.port, process.pid)
    except BaseException as exc:
        postflight_rc = 123
        (root / "postflight_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
    if controller_error is not None:
        (root / "controller_error.txt").write_text(controller_error + "\n", encoding="utf-8")
    (root / "launcher_rc.txt").write_text(str(launcher_rc) + "\n", encoding="utf-8")
    (root / "controller_rc.txt").write_text(str(controller_rc) + "\n", encoding="utf-8")
    (root / "postflight_rc.txt").write_text(str(postflight_rc) + "\n", encoding="utf-8")
    if controller_rc != 0:
        return 122
    if postflight_rc != 0:
        return 123
    return 121 if launcher_rc != 0 else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("original", "candidate"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--expected-kernel", required=True)
    parser.add_argument("--original-kernel", required=True)
    parser.add_argument("--custom-opp", required=True)
    parser.add_argument("--installed-custom-opp", required=True)
    parser.add_argument("--overlay-custom-opp")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
