#!/usr/bin/env python3
"""Thin world8 gate in front of the SHA-locked STEP204 training entry."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import runpy
import stat
import sys
import time
from pathlib import Path
from typing import Any


WORLD_SIZE = 8
VISIBLE = "8,9,10,11,12,13,14,15"
ORIGINAL_ENTRY_SHA256 = "8c5b315b1741a1557293db1df1bd6c6699494970bc136c434b5b84af9aad65fa"
GATE_TIMEOUT_SECONDS = 900


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value or "\0" in value:
        raise RuntimeError(f"STEP393 required environment missing: {name}")
    return value


def strict_path(name: str, *, file: bool = False, directory: bool = False) -> Path:
    raw = required_env(name)
    path = Path(raw)
    if not path.is_absolute() or path == Path("/") or path.is_symlink():
        raise RuntimeError(f"STEP393 unsafe path: {name}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise RuntimeError(f"STEP393 non-canonical path: {name}")
    status = resolved.lstat()
    if file and not stat.S_ISREG(status.st_mode):
        raise RuntimeError(f"STEP393 expected regular file: {name}")
    if directory and not stat.S_ISDIR(status.st_mode):
        raise RuntimeError(f"STEP393 expected directory: {name}")
    return resolved


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{os.urandom(8).hex()}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        payload = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_gate(path: Path, token_sha256: str) -> dict[str, Any]:
    deadline = time.monotonic() + GATE_TIMEOUT_SECONDS
    while True:
        try:
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_size > 4096:
                raise RuntimeError("STEP393 gate must be a bounded regular file")
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                opened = os.fstat(descriptor)
                data = os.read(descriptor, 4097)
                closed = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            after = path.lstat()
            identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
            if not identity(before) == identity(opened) == identity(closed) == identity(after):
                raise RuntimeError("STEP393 gate identity changed")
            if len(data) > 4096:
                raise RuntimeError("STEP393 gate exceeds limit")
            value = json.loads(data)
            if value != {"schema": "step393-host-gate-v1", "token_sha256": token_sha256}:
                raise RuntimeError("STEP393 gate payload mismatch")
            return value
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError("STEP393 host gate timeout")
            time.sleep(0.05)


def main() -> int:
    rank = int(required_env("RANK"))
    local_rank = int(required_env("LOCAL_RANK"))
    world_size = int(required_env("WORLD_SIZE"))
    if world_size != WORLD_SIZE or rank not in range(8) or local_rank != rank:
        raise RuntimeError("STEP393 rank/local/world contract mismatch")
    if required_env("ASCEND_RT_VISIBLE_DEVICES") != VISIBLE:
        raise RuntimeError("STEP393 visible device contract mismatch")
    if os.environ.get("LD_PRELOAD") or os.environ.get("PYTHONSTARTUP"):
        raise RuntimeError("STEP393 unsafe Python preload/startup environment")
    if "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
        raise RuntimeError("STEP393 Python startup customization was loaded")

    original = strict_path("STEP393_ORIGINAL_ENTRY", file=True)
    ready_dir = strict_path("STEP393_READY_DIR", directory=True)
    done_dir = strict_path("STEP393_DONE_DIR", directory=True)
    failure_dir = strict_path("STEP393_FAILURE_DIR", directory=True)
    ack_dir = strict_path("STEP393_GATE_ACK_DIR", directory=True)
    gate_file = Path(required_env("STEP393_GATE_FILE"))
    shadow_package = strict_path("STEP393_SHADOW_PACKAGE", directory=True)
    if (not gate_file.is_absolute() or gate_file.name != "start.gate"
            or gate_file.parent.resolve(strict=True) != ready_dir.parent):
        raise RuntimeError("STEP393 gate path contract mismatch")
    token_sha256 = required_env("STEP393_GATE_TOKEN_SHA256")
    if sha256_file(original) != ORIGINAL_ENTRY_SHA256:
        raise RuntimeError("STEP393 original entry SHA256 mismatch")
    if len(token_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in token_sha256):
        raise RuntimeError("STEP393 gate token SHA256 malformed")

    import torch
    import torch_npu

    if not torch.npu.is_available() or torch.npu.device_count() != 8:
        raise RuntimeError("STEP393 torch_npu/device-count gate failed")
    torch.npu.set_device(local_rank)
    if torch.npu.current_device() != local_rank:
        raise RuntimeError("STEP393 current NPU differs from local rank")
    # Materialize the rank's NPU context before the host samples npu-smi.  This
    # is a single startup synchronization, never a per-QR synchronization.
    startup_probe = torch.empty(1, dtype=torch.float32, device=f"npu:{local_rank}")
    startup_probe.fill_(float(rank))
    torch.npu.synchronize()
    del startup_probe
    spec = importlib.util.find_spec("mx_driving_cloud")
    if spec is None or spec.origin is None:
        raise RuntimeError("STEP393 mx_driving_cloud module missing")
    module_origin = Path(spec.origin).resolve(strict=True)
    if module_origin != shadow_package / "__init__.py":
        raise RuntimeError("STEP393 mx_driving_cloud did not resolve from shadow")

    task_queue = os.environ.get("TASK_QUEUE_ENABLE")
    ready = {
        "schema": "step393-rank-ready-v1", "rank": rank, "local_rank": local_rank,
        "world_size": world_size, "container_pid": os.getpid(), "visible": VISIBLE,
        "torch_version": str(torch.__version__), "torch_npu_version": str(torch_npu.__version__),
        "npu_available": True, "device_count": 8, "current_device": local_rank,
        "startup_context_synchronized": True,
        "module_origin": str(module_origin), "shadow_package": str(shadow_package),
        "instrumentation_requested": False, "fallback_not_observed": True,
        "task_queue_state": "production-preserved", "task_queue_present": task_queue is not None,
        "task_queue_value_sha256": hashlib.sha256(task_queue.encode()).hexdigest()
        if task_queue is not None else None,
    }
    write_new_json(ready_dir / f"rank{rank}.json", ready)
    try:
        read_gate(gate_file, token_sha256)
        write_new_json(ack_dir / f"rank{rank}.json", {
            "schema": "step393-rank-gate-ack-v1", "rank": rank,
            "container_pid": os.getpid(), "token_sha256": token_sha256,
        })
        old_argv = sys.argv
        sys.argv = [str(original), *old_argv[1:]]
        try:
            runpy.run_path(str(original), run_name="__main__")
        except SystemExit as error:
            code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
            if code != 0:
                raise
            write_new_json(done_dir / f"rank{rank}.json", {
                "schema": "step393-rank-done-v1", "rank": rank, "returncode": 0,
            })
            return 0
        else:
            write_new_json(done_dir / f"rank{rank}.json", {
                "schema": "step393-rank-done-v1", "rank": rank, "returncode": 0,
            })
        finally:
            sys.argv = old_argv
    except BaseException as error:
        try:
            write_new_json(failure_dir / f"rank{rank}.json", {
                "schema": "step393-rank-failure-v1", "rank": rank,
                "error_type": type(error).__name__,
            })
        except FileExistsError:
            pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
