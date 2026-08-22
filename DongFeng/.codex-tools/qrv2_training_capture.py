#!/usr/bin/env python3
"""Training-time MX/CPU QR capture with immediate immutable evidence writes.

This module is imported only by an isolated SOAP source tree.  The active
shared training worktree must never import or be modified by this tool.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import torch


SCHEMA = "qrv2-training-call-evidence-v1"
SUPPORTED_BACKENDS = frozenset({"mx", "cpu"})
EXPECTED_VISIBLE_DEVICES = tuple(range(8, 16))
_call_index = 0
_captured_count = 0


def _integer_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}")
    return value


def _rank() -> int:
    return _integer_env(
        "RANK",
        _integer_env("LOCAL_RANK", _integer_env("SLURM_PROCID", 0)),
    )


def _capture_root() -> Path:
    raw = os.environ.get("QR_CAPTURE_DIR", "")
    if not raw:
        raise RuntimeError("QR_CAPTURE_DIR is required")
    root = Path(raw)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("QR_CAPTURE_DIR must be an existing non-symlink directory")
    return root


def _backend() -> str:
    value = os.environ.get("QR_CAPTURE_BACKEND", "").strip().lower()
    if value not in SUPPORTED_BACKENDS:
        raise RuntimeError("QR_CAPTURE_BACKEND must be 'mx' or 'cpu'")
    return value


def _target_shape() -> tuple[int, int]:
    raw = os.environ.get("QR_CAPTURE_TARGET_SHAPE", "192x192").lower()
    fields = raw.split("x")
    if len(fields) != 2:
        raise RuntimeError("QR_CAPTURE_TARGET_SHAPE must look like 192x192")
    try:
        shape = (int(fields[0]), int(fields[1]))
    except ValueError as error:
        raise RuntimeError("QR_CAPTURE_TARGET_SHAPE must contain integers") from error
    if min(shape) <= 0:
        raise RuntimeError("QR_CAPTURE_TARGET_SHAPE dimensions must be positive")
    return shape


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    cpu = value.detach().contiguous().cpu()
    finite = torch.isfinite(cpu)
    nan_mask = torch.isnan(cpu)
    posinf_mask = torch.isposinf(cpu)
    neginf_mask = torch.isneginf(cpu)
    summary: dict[str, Any] = {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "stride": list(cpu.stride()),
        "numel": int(cpu.numel()),
        "finite": int(finite.sum().item()),
        "nan": int(nan_mask.sum().item()),
        "posinf": int(posinf_mask.sum().item()),
        "neginf": int(neginf_mask.sum().item()),
        "tensor_sha256": _tensor_sha256(cpu),
    }
    bad = ~finite
    if cpu.ndim == 2 and bool(bad.any().item()):
        coords = bad.nonzero(as_tuple=False)
        summary.update(
            {
                "bad_row_min": int(coords[:, 0].min().item()),
                "bad_row_max": int(coords[:, 0].max().item()),
                "bad_col_min": int(coords[:, 1].min().item()),
                "bad_col_max": int(coords[:, 1].max().item()),
                "fully_bad_columns": bad.all(dim=0).nonzero(as_tuple=False).flatten().tolist(),
            }
        )
    return summary


def _fsync_directory(root: Path) -> None:
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bytes(path: Path, payload: bytes) -> None:
    if path.is_symlink() or path.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite evidence: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _publish_json(path: Path, record: dict[str, Any]) -> None:
    payload = (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    _publish_bytes(path, payload)


def _publish_torch(path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink() or path.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite evidence: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_identity(backend: str) -> dict[str, Any]:
    torch_npu_module = sys.modules.get("torch_npu")
    rank = _rank()
    return {
        "backend": backend,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "rank": rank,
        "local_rank": _integer_env("LOCAL_RANK", rank),
        "world_size": _integer_env("WORLD_SIZE", 1, minimum=1),
        "torch_version": torch.__version__,
        "torch_npu_version": getattr(torch_npu_module, "__version__", None),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "run_id": os.environ.get("QR_CAPTURE_RUN_ID"),
        "source_commit": os.environ.get("QR_CAPTURE_SOURCE_COMMIT"),
        "soap_sha256": os.environ.get("QR_CAPTURE_SOAP_SHA256"),
        "config_sha256": os.environ.get("QR_CAPTURE_CONFIG_SHA256"),
        "checkpoint_sha256": os.environ.get("QR_CAPTURE_CHECKPOINT_SHA256"),
        "seed": os.environ.get("QR_CAPTURE_SEED"),
    }


def _validate_runtime_contract(value: torch.Tensor) -> None:
    rank = _rank()
    local_rank = _integer_env("LOCAL_RANK", rank)
    world_size = _integer_env("WORLD_SIZE", 1, minimum=1)
    if world_size != 8 or not (0 <= rank < 8) or local_rank != rank:
        raise RuntimeError("capture requires world_size=8 and rank=local_rank in [0,7]")
    raw_visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    try:
        visible = tuple(int(item.strip()) for item in raw_visible.split(",") if item.strip())
    except ValueError as error:
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES must contain integers") from error
    if visible != EXPECTED_VISIBLE_DEVICES:
        raise RuntimeError("capture requires exact visible devices 8,9,10,11,12,13,14,15")
    required = {
        "QR_CAPTURE_RUN_ID": r"[A-Za-z0-9._-]{1,128}",
        "QR_CAPTURE_SOURCE_COMMIT": r"[0-9a-f]{40}",
        "QR_CAPTURE_SOAP_SHA256": r"[0-9a-f]{64}",
        "QR_CAPTURE_CONFIG_SHA256": r"[0-9a-f]{64}",
        "QR_CAPTURE_CHECKPOINT_SHA256": r"[0-9a-f]{64}",
        "QR_CAPTURE_SEED": r"[0-9]+",
    }
    for name, pattern in required.items():
        if re.fullmatch(pattern, os.environ.get(name, "")) is None:
            raise RuntimeError(f"{name} is missing or invalid")
    if value.device.type == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("input is NPU but torch.npu is unavailable")
        if int(torch.npu.device_count()) != 8:
            raise RuntimeError("capture requires exactly eight visible NPU devices")
        if int(torch.npu.current_device()) != local_rank:
            raise RuntimeError("current NPU device does not match LOCAL_RANK")


def _should_capture(
    value: torch.Tensor, optimizer_step: int, factor_index: int
) -> bool:
    if optimizer_step != _integer_env("QR_CAPTURE_TARGET_STEP", 10):
        return False
    if factor_index != _integer_env("QR_CAPTURE_TARGET_FACTOR", 0):
        return False
    if tuple(value.shape) != _target_shape():
        return False
    return _captured_count < _integer_env("QR_CAPTURE_MAX_PER_RANK", 1, minimum=1)


def _cpu_qr(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if value.device.type != "cpu":
        raise RuntimeError("official CPU QR received a non-CPU tensor")
    return torch.linalg.qr(value, mode="reduced")


def qr(
    value: torch.Tensor,
    *,
    mx_qr: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    optimizer_step: int,
    factor_index: int,
    call_site: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the selected training backend and capture the target call."""

    global _call_index, _captured_count
    if not torch.is_tensor(value) or value.ndim != 2:
        raise RuntimeError("training QR input must be a rank-2 tensor")
    _validate_runtime_contract(value)
    backend = _backend()
    call_index = _call_index
    _call_index += 1
    capture = _should_capture(value, int(optimizer_step), int(factor_index))
    a_cpu: torch.Tensor | None = None
    input_path: Path | None = None
    prefix: str | None = None
    root: Path | None = None
    started_ns = time.time_ns()
    if capture:
        root = _capture_root()
        rank = _rank()
        prefix = (
            f"rank{rank}_step{int(optimizer_step)}_call{call_index}_"
            f"factor{int(factor_index)}_{value.shape[0]}x{value.shape[1]}_{backend}"
        )
        a_cpu = value.detach().to(device="cpu").contiguous()
        input_path = root / f"{prefix}_input.pt"
        _publish_torch(input_path, {"A": a_cpu})
        _publish_json(
            root / f"{prefix}_started.json",
            {
                "schema": SCHEMA,
                "status": "started",
                "optimizer_step": int(optimizer_step),
                "factor_index": int(factor_index),
                "call_index": call_index,
                "call_site": call_site,
                "started_ns": started_ns,
                "runtime": _runtime_identity(backend),
                "input": _tensor_summary(a_cpu),
                "input_file": input_path.name,
                "input_file_sha256": _sha256(input_path),
            },
        )
        _captured_count += 1

    try:
        if backend == "mx":
            q, r = mx_qr(value)
        else:
            if a_cpu is None:
                a_cpu = value.detach().to(device="cpu").contiguous()
            q_cpu, r_cpu = _cpu_qr(a_cpu)
            q = q_cpu.to(device=value.device, dtype=value.dtype)
            r = r_cpu.to(device=value.device, dtype=value.dtype)
        if not torch.is_tensor(q) or not torch.is_tensor(r):
            raise RuntimeError("QR backend did not return tensor Q/R")
        if capture:
            assert (
                root is not None
                and prefix is not None
                and a_cpu is not None
                and input_path is not None
            )
            q_cpu_actual = q.detach().to(device="cpu").contiguous()
            r_cpu_actual = r.detach().to(device="cpu").contiguous()
            output_path = root / f"{prefix}_output.pt"
            _publish_torch(output_path, {"Q": q_cpu_actual, "R": r_cpu_actual})
            _publish_json(
                root / f"{prefix}_complete.json",
                {
                    "schema": SCHEMA,
                    "status": "complete",
                    "optimizer_step": int(optimizer_step),
                    "factor_index": int(factor_index),
                    "call_index": call_index,
                    "call_site": call_site,
                    "started_ns": started_ns,
                    "completed_ns": time.time_ns(),
                    "runtime": _runtime_identity(backend),
                    "input": _tensor_summary(a_cpu),
                    "input_file": input_path.name,
                    "input_file_sha256": _sha256(input_path),
                    "q": _tensor_summary(q_cpu_actual),
                    "r": _tensor_summary(r_cpu_actual),
                    "output_file": output_path.name,
                    "output_file_sha256": _sha256(output_path),
                },
            )
        return q, r
    except BaseException as error:
        if capture:
            assert root is not None and prefix is not None
            _publish_json(
                root / f"{prefix}_failed.json",
                {
                    "schema": SCHEMA,
                    "status": "failed",
                    "optimizer_step": int(optimizer_step),
                    "factor_index": int(factor_index),
                    "call_index": call_index,
                    "call_site": call_site,
                    "started_ns": started_ns,
                    "failed_ns": time.time_ns(),
                    "runtime": _runtime_identity(backend),
                    "error_type": type(error).__name__,
                    "error": str(error)[:1000],
                },
            )
        raise


def _reset_for_tests() -> None:
    global _call_index, _captured_count
    _call_index = 0
    _captured_count = 0
