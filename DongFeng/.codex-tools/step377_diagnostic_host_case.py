#!/usr/bin/env python3
"""Diagnostic-only host adapter around the audited STEP358 world8 launcher."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import sys
import time
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


TOOLS = Path(__file__).resolve().parent
HOST_PATH = TOOLS / "step358_host_case.py"
WORLD8_PATH = TOOLS / "step343_world8_controller.py"
WORKER_ADAPTER = TOOLS / "step377_diagnostic_math_worker.py"
PROCESS_GUARD_PATH = TOOLS / "step377_process_guard.py"
EXPECTED_SHA256 = {
    HOST_PATH.name: "94e5a46059c0a57bb883999f2648f755158c879568ed3c33b6d4fde8cf1c7070",
    WORLD8_PATH.name: "ea0e587cd0b6c1b31fe753e3239a63c91597cee8f4ec917ad08ab7999bb82ce6",
    WORKER_ADAPTER.name: "f363ac8bd85bb6e56e0de9f1cc6eb8b321b9d6db296af61e3c12384bb4ce4c3d",
    PROCESS_GUARD_PATH.name: "7b4dcb578fd5227f51cf54b2acaa0591840261794b3296eeafa5731e76ad27c5",
}
RANKS = tuple(range(8))
VISIBLE = "8,9,10,11,12,13,14,15"
WRAPPER_SOURCE_SHA256 = "2e2171c4931e4796ecb1ec1a85d01846f25b3054e82b94fe7abc976e7cc02ee3"
GATE_NAME = "diagnostic_start_after_npu_smi"
GATE_ACK_DIR = "diagnostic_gate_ack"
ACK_KEYS = {"schema", "rank", "gate_device", "gate_inode", "token_sha256"}
SUMMARY_NAME = "step377_diagnostic_summary.json"
READY_KEYS = {
    "rank", "local_rank", "world_size", "visible", "npu_available",
    "device_count", "container_pid", "gate_pass", "shadow_gate",
    "opp_first_shadow", "module_file_sha256", "custom_opp_role_sequence",
    "wrapper_contract",
}
DONE_KEYS = {
    "rank", "local_rank", "world_size", "input_file_sha256", "call_count",
    "eligible_call_count", "mx_qr_call_count", "eligible_fallback_count",
    "all_contract_pass", "profiler_identity_pass", "state_diagnostic_only",
    "first_profiled_only", "calls",
}
CALL_KEYS = {
    "case_id", "shape", "dtype", "input_sha256", "eligible_mx_branch",
    "mx_qr_call_delta", "mx_qr_input", "expected_padded_shape",
    "wrapper_branch", "public_qr_mode", "cpu_fp32_projection_control_max",
    "input_unmodified", "elapsed_ms", "contract_pass", "shape_pass",
    "input_finite", "q_finite", "r_finite", "nonfinite_count", "finite_pass",
    "reconstruction", "orthogonality", "lower_triangle_exact_zero",
    "lower_triangle_required", "fp64", "full_rank_projection",
    "predicate_status", "failed_predicates", "not_evaluated_predicates",
    "diagnostic_scalars_finite", "diagnostic_nonfinite_scalar_count",
    "reconstruction_violation_count", "orthogonality_violation_count",
    "projection_pass",
}
COMPONENT_KEYS = {"violation_count", "max_abs", "max_bound", "max_scaled"}
FP64_KEYS = {
    "candidate_reconstruction_relative_fro", "candidate_orthogonality_relative_fro",
    "reference_reconstruction_relative_fro", "reference_orthogonality_relative_fro",
    "numerical_rank", "rank_threshold",
}
PROJECTION_KEYS = {
    "required", "candidate_to_reference", "reference_to_candidate", "control_max",
    "tolerance", "pass",
}
PROJECTION_DIRECTION_KEYS = {"relative_fro", "relative_max"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guard_dependencies() -> None:
    for path in (HOST_PATH, WORLD8_PATH, WORKER_ADAPTER, PROCESS_GUARD_PATH):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"STEP377 host dependency must be regular: {path.name}")
        if sha256_file(path) != EXPECTED_SHA256[path.name]:
            raise RuntimeError(f"STEP377 host dependency SHA mismatch: {path.name}")


def _exec_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load host dependency: {path.name}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    existed = name in sys.modules
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if existed:
            sys.modules[name] = previous  # type: ignore[assignment]
        else:
            sys.modules.pop(name, None)
    return module


def load_host() -> ModuleType:
    _guard_dependencies()
    world = _exec_module("_step377_isolated_world8", WORLD8_PATH)
    previous = sys.modules.get("step343_world8_controller")
    existed = "step343_world8_controller" in sys.modules
    sys.modules["step343_world8_controller"] = world
    try:
        return _exec_module("_step377_isolated_step358_host", HOST_PATH)
    finally:
        if existed:
            sys.modules["step343_world8_controller"] = previous  # type: ignore[assignment]
        else:
            sys.modules.pop("step343_world8_controller", None)


def load_process_guard() -> ModuleType:
    _guard_dependencies()
    return _exec_module("_step377_isolated_process_guard", PROCESS_GUARD_PATH)


def _input_hashes(input_dir: Path) -> dict[int, str]:
    root = input_dir.absolute()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("STEP260 input root must be a regular non-symlink directory")
    result = {}
    for rank in RANKS:
        path = root / f"rank{rank}_step10_ind0_192x192_BAD.pt"
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"STEP260 rank{rank} input must be regular")
        result[rank] = sha256_file(path)
    return result


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return False


def _validate_call(rank: int, call: Any) -> None:
    if not isinstance(call, dict):
        raise RuntimeError(f"rank{rank} call must be an object")
    if set(call) != CALL_KEYS:
        raise RuntimeError(f"rank{rank} call schema mismatch")
    exact = {
        "case_id": f"step260_rank{rank}_profiled",
        "shape": [192, 192],
        "dtype": "torch.float32",
        "eligible_mx_branch": True,
        "mx_qr_call_delta": 1,
        "expected_padded_shape": [192, 192],
        "wrapper_branch": "mx_fixed",
        "public_qr_mode": "complete",
        "input_unmodified": True,
        "contract_pass": True,
        "shape_pass": True,
        "input_finite": True,
        "q_finite": True,
        "r_finite": True,
        "finite_pass": True,
        "lower_triangle_exact_zero": True,
        "lower_triangle_required": True,
    }
    if any(type(call.get(key)) is not type(value) or call.get(key) != value for key, value in exact.items()):
        raise RuntimeError(f"rank{rank} call scalar contract mismatch")
    ledger = call.get("mx_qr_input")
    if ledger != {"shape": [192, 192], "dtype": "torch.float32", "contiguous": True}:
        raise RuntimeError(f"rank{rank} ledger/padding contract mismatch")
    counts = call.get("nonfinite_count")
    if counts != {"input": 0, "q": 0, "r": 0}:
        raise RuntimeError(f"rank{rank} nonfinite count mismatch")
    for name in ("reconstruction", "orthogonality"):
        value = call.get(name)
        if (
            not isinstance(value, dict) or set(value) != COMPONENT_KEYS
            or type(value.get("violation_count")) is not int or value["violation_count"] != 0
            or any(type(value.get(key)) is not float or not math.isfinite(value[key]) for key in COMPONENT_KEYS - {"violation_count"})
        ):
            raise RuntimeError(f"rank{rank} {name} predicate failed")
    expected_predicates = {
        "input_unmodified", "shape", "finite", "reconstruction",
        "orthogonality", "lower_triangle_exact_zero", "projection",
    }
    predicates = call.get("predicate_status")
    if (
        not isinstance(predicates, dict)
        or set(predicates) != expected_predicates
        or set(predicates.values()) != {"pass"}
        or call.get("failed_predicates") != []
        or call.get("not_evaluated_predicates") != []
        or call.get("diagnostic_scalars_finite") is not True
        or call.get("diagnostic_nonfinite_scalar_count") != 0
        or call.get("reconstruction_violation_count")
        != call["reconstruction"]["violation_count"]
        or call.get("orthogonality_violation_count")
        != call["orthogonality"]["violation_count"]
    ):
        raise RuntimeError(f"rank{rank} predicate summary mismatch")
    fp64 = call.get("fp64")
    projection = call.get("full_rank_projection")
    if (
        not isinstance(fp64, dict) or set(fp64) != FP64_KEYS
        or type(fp64.get("numerical_rank")) is not int
        or any(type(fp64.get(key)) is not float for key in FP64_KEYS - {"numerical_rank"})
        or not _all_finite(fp64)
    ):
        raise RuntimeError(f"rank{rank} FP64 diagnostic is invalid")
    if not isinstance(projection, dict) or set(projection) != PROJECTION_KEYS:
        raise RuntimeError(f"rank{rank} projection diagnostic is invalid")
    if type(projection.get("required")) is not bool or type(projection.get("pass")) is not bool:
        raise RuntimeError(f"rank{rank} projection diagnostic is invalid")
    for direction in ("candidate_to_reference", "reference_to_candidate"):
        value = projection.get(direction)
        if (
            not isinstance(value, dict) or set(value) != PROJECTION_DIRECTION_KEYS
            or any(type(value.get(key)) is not float or not math.isfinite(value[key]) for key in PROJECTION_DIRECTION_KEYS)
        ):
            raise RuntimeError(f"rank{rank} projection direction is invalid")
    if any(type(projection.get(key)) is not float or not math.isfinite(projection[key]) for key in ("control_max", "tolerance")):
        raise RuntimeError(f"rank{rank} projection thresholds are invalid")
    if projection["required"] and projection["pass"] is not True:
        raise RuntimeError(f"rank{rank} projection predicate failed")
    if projection.get("required") is not True or call.get("projection_pass") is not True:
        raise RuntimeError(f"rank{rank} projection summary mismatch")
    if (
        type(call.get("elapsed_ms")) is not float or not math.isfinite(call["elapsed_ms"])
        or type(call.get("cpu_fp32_projection_control_max")) is not float
        or not math.isfinite(call["cpu_fp32_projection_control_max"])
        or not _all_finite(projection)
    ):
        raise RuntimeError(f"rank{rank} diagnostic contains non-finite scalars")
    input_sha = call.get("input_sha256")
    if (
        not isinstance(input_sha, str)
        or len(input_sha) != 64
        or any(character not in "0123456789abcdef" for character in input_sha)
        or not _all_finite(call)
    ):
        raise RuntimeError(f"rank{rank} call diagnostic closure failed")


def validate_outputs(root: Path, input_hashes: dict[int, str]) -> dict[str, Any]:
    rank_summaries = []
    profile_inventory = []
    for rank in RANKS:
        done_path = root / "done" / f"rank{rank}.json"
        identity_path = root / f"profiler_identity_rank{rank}.json"
        for path in (done_path, identity_path):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"rank{rank} result file missing or symlinked")
        done = json.loads(done_path.read_text(encoding="utf-8"))
        expected_done = {
            "rank": rank, "local_rank": rank, "world_size": 8,
            "input_file_sha256": input_hashes[rank], "call_count": 1,
            "eligible_call_count": 1, "mx_qr_call_count": 1,
            "eligible_fallback_count": 0, "all_contract_pass": True,
            "profiler_identity_pass": True, "first_profiled_only": True,
            "state_diagnostic_only": False,
        }
        if (
            not isinstance(done, dict) or set(done) != DONE_KEYS
            or any(type(done.get(k)) is not type(v) or done.get(k) != v for k, v in expected_done.items())
        ):
            raise RuntimeError(f"rank{rank} done contract mismatch")
        calls = done.get("calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise RuntimeError(f"rank{rank} must contain exactly one call")
        _validate_call(rank, calls[0])
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        expected_identity = {
            "pass": True,
            "diagnostic_identity": "QrV2_vtv_direct_qa_legacy_probe_v6",
            "diagnostic_aic_task_reference_count": 1,
            "diagnostic_aiv_task_reference_count": 0,
            "original_task_reference_count": 0,
            "v4_task_reference_count": 0,
            "v5_task_reference_count": 0,
            "unknown_qrv2_task_reference_count": 0,
            "raw_profile_retained": True,
        }
        if not isinstance(identity, dict) or any(identity.get(k) != v for k, v in expected_identity.items()):
            raise RuntimeError(f"rank{rank} diagnostic identity mismatch")
        profile = root / f"profile_rank{rank}"
        if profile.is_symlink() or not profile.is_dir():
            raise RuntimeError(f"rank{rank} raw profile directory is invalid")
        files = [path for path in profile.rglob("*") if path.is_file() and not path.is_symlink()]
        if not files or any(path.is_symlink() for path in profile.rglob("*")):
            raise RuntimeError(f"rank{rank} raw profile is empty or contains symlinks")
        total_bytes = sum(path.stat().st_size for path in files)
        if total_bytes <= 0:
            raise RuntimeError(f"rank{rank} raw profile contains no retained bytes")
        profile_inventory.append({
            "rank": rank,
            "file_count": len(files),
            "total_bytes": total_bytes,
        })
        rank_summaries.append({"rank": rank, "call_count": 1, "identity_pass": True})
    return {"ranks": rank_summaries, "raw_profiles": profile_inventory}


def _verify_gate(path: Path, contract: dict[str, Any]) -> None:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 4096:
        raise RuntimeError("diagnostic gate must be a bounded regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        payload = os.read(descriptor, 4097)
    finally:
        os.close(descriptor)
    after = path.lstat()
    identity = (contract["device"], contract["inode"])
    if (
        (before.st_dev, before.st_ino) != identity
        or (opened.st_dev, opened.st_ino) != identity
        or (after.st_dev, after.st_ino) != identity
        or len(payload) > 4096
    ):
        raise RuntimeError("diagnostic gate inode changed")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise RuntimeError("diagnostic gate JSON is invalid") from error
    if not isinstance(value, dict) or set(value) != {"schema", "token"}:
        raise RuntimeError("diagnostic gate schema mismatch")
    token = value.get("token")
    if (
        value.get("schema") != "step377-diagnostic-start-v1"
        or not isinstance(token, str)
        or len(token) != 32
        or any(character not in "0123456789abcdef" for character in token)
        or hashlib.sha256(token.encode()).hexdigest() != contract["token_sha256"]
    ):
        raise RuntimeError("diagnostic gate token mismatch")


def _prepare_gate(path: Path) -> tuple[dict[str, Any], bytes]:
    token = uuid.uuid4().hex
    payload = (json.dumps({"schema": "step377-diagnostic-start-v1", "token": token}) + "\n").encode()
    return ({"path": str(path), "token_sha256": hashlib.sha256(token.encode()).hexdigest()}, payload)


def _publish_gate(path: Path, prepared: dict[str, Any], payload: bytes) -> dict[str, Any]:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        try:
            with os.fdopen(os.dup(descriptor), "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            status = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    contract = {
        "path": str(path),
        "token_sha256": prepared["token_sha256"],
        "device": status.st_dev,
        "inode": status.st_ino,
    }
    contract["published"] = True
    contract["dir_fsync_ok"] = True
    contract["dir_fsync_error"] = None
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            os.fsync(parent_fd)
        except OSError as error:
            contract["dir_fsync_ok"] = False
            contract["dir_fsync_error"] = f"{type(error).__name__}: {error}"
    finally:
        os.close(parent_fd)
    _verify_gate(path, contract)
    return contract


def _write_gate(path: Path) -> dict[str, Any]:
    prepared, payload = _prepare_gate(path)
    return _publish_gate(path, prepared, payload)


def _read_bounded_regular_json(path: Path, label: str) -> dict[str, Any]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 1024 * 1024:
        raise RuntimeError(f"{label} must be a bounded regular non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        payload = os.read(descriptor, 1024 * 1024 + 1)
    finally:
        os.close(descriptor)
    after = path.lstat()
    identity = (before.st_dev, before.st_ino)
    if (
        len(payload) > 1024 * 1024
        or (opened.st_dev, opened.st_ino) != identity
        or (after.st_dev, after.st_ino) != identity
    ):
        raise RuntimeError(f"{label} identity changed during read")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise RuntimeError(f"{label} JSON is invalid") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _validate_ready_row(rank: int, row: dict[str, Any]) -> None:
    if set(row) != READY_KEYS:
        raise RuntimeError("diagnostic ready file schema mismatch")
    module_hashes = row.get("module_file_sha256")
    wrapper = row.get("wrapper_contract")
    if not (
        row.get("rank") == rank and row.get("local_rank") == rank
        and row.get("world_size") == 8 and row.get("visible") == VISIBLE
        and row.get("gate_pass") is True and row.get("shadow_gate") is True
        and row.get("npu_available") is True and row.get("device_count") == 8
        and row.get("opp_first_shadow") is True
        and row.get("custom_opp_role_sequence") == ["shadow", "base"]
        and isinstance(wrapper, dict)
        and set(wrapper) == {"gate", "source_sha256", "threshold", "block_tiling"}
        and type(wrapper.get("gate")) is str and wrapper["gate"] == "PASS"
        and type(wrapper.get("source_sha256")) is str
        and wrapper["source_sha256"] == WRAPPER_SOURCE_SHA256
        and type(wrapper.get("threshold")) is int and wrapper["threshold"] == 80
        and type(wrapper.get("block_tiling")) is int and wrapper["block_tiling"] == 64
        and type(row.get("container_pid")) is int and row["container_pid"] > 0
        and isinstance(module_hashes, dict)
        and set(module_hashes) == {"cloud_init", "cloud_extension", "cloud_linalg"}
        and all(
            isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in module_hashes.values()
        )
    ):
        raise RuntimeError("diagnostic ready rank contract failed")


def _validate_ready_consistency(rows: list[dict[str, Any]]) -> dict[str, str]:
    if len(rows) != 8:
        raise RuntimeError("diagnostic ready row count mismatch")
    module_hashes = [row["module_file_sha256"] for row in rows]
    if any(value != module_hashes[0] for value in module_hashes[1:]):
        raise RuntimeError("diagnostic ready module hashes differ across ranks")
    return module_hashes[0]


def _validate_gate_acks(root: Path, gate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for rank in RANKS:
        row = _read_bounded_regular_json(root / GATE_ACK_DIR / f"rank{rank}.json", f"rank{rank} gate ack")
        if (
            set(row) != ACK_KEYS
            or row.get("schema") != "step377-diagnostic-gate-ack-v1"
            or type(row.get("rank")) is not int or row["rank"] != rank
            or type(row.get("gate_device")) is not int or row["gate_device"] != gate["device"]
            or type(row.get("gate_inode")) is not int or row["gate_inode"] != gate["inode"]
            or row.get("token_sha256") != gate["token_sha256"]
        ):
            raise RuntimeError(f"rank{rank} diagnostic gate acknowledgement mismatch")
        rows.append(row)
    return rows


def diagnostic_wait_for_results(host: ModuleType, guard: ModuleType, root: Path,
                                process: Any, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    expected = {f"rank{rank}.json" for rank in RANKS}
    while True:
        failures = sorted((root / "failure").glob("rank*.txt"))
        if failures:
            raise RuntimeError("rank failure before diagnostic live gate")
        ready = {path.name for path in (root / "ready").glob("rank*.json")}
        if ready == expected:
            break
        if not ready.issubset(expected) or process.poll() is not None or time.monotonic() >= deadline:
            raise RuntimeError("diagnostic ready gate failed")
        time.sleep(0.25)
    rows = []
    for rank in RANKS:
        path = root / "ready" / f"rank{rank}.json"
        row = _read_bounded_regular_json(path, f"rank{rank} ready")
        _validate_ready_row(rank, row)
        rows.append(row)
    module_hashes = _validate_ready_consistency(rows)
    ready_pids = [int(row["container_pid"]) for row in rows]
    if len(set(ready_pids)) != 8:
        raise RuntimeError("diagnostic ready PIDs are not unique")
    prepared_gate, gate_payload = _prepare_gate(root / GATE_NAME)
    binding = guard.stable_back8_binding(rows, host.legacy.npu_smi)
    rank_mapping = [
        {"rank": row["rank"], "local_rank": row["local_rank"], "physical_device": row["device_id"]}
        for row in binding["bindings"]
    ]
    device_ids = {row["device_id"] for row in binding["bindings"]}
    ack_dir = root / GATE_ACK_DIR
    ack_dir.mkdir(mode=0o700, exist_ok=False)
    launcher_path = root / "launcher_ownership.json"
    launcher_sha256 = sha256_file(launcher_path)
    rank_ownership = {
        "schema": "step377-rank-ownership-v1",
        "launcher_ownership_sha256": launcher_sha256,
        "gate_token_sha256": prepared_gate["token_sha256"],
        "case_path": str(Path(__file__).resolve()),
        "port": int(host._step377_port),
        "ranks": binding["bindings"],
    }
    rank_path = root / "rank_ownership.json"
    rank_commit = _write_new_json(rank_path, rank_ownership, committed_error_ok=True)
    host._step377_rank_ownership = rank_ownership
    host._step377_rank_ownership_sha256 = sha256_file(rank_path)
    guard.read_rank_ownership_json(
        rank_path, host._step377_rank_ownership_sha256,
        expected_launcher_sha256=launcher_sha256,
        case_path=Path(__file__).resolve(), port=int(host._step377_port))
    gate = _publish_gate(root / GATE_NAME, prepared_gate, gate_payload)
    host._step377_gate_token_sha256 = gate["token_sha256"]
    durability_errors = []
    if not rank_commit["dir_fsync_ok"]:
        durability_errors.append(f"rank_ownership_dir_fsync: {rank_commit['dir_fsync_error']}")
    if not gate["dir_fsync_ok"]:
        durability_errors.append(f"gate_dir_fsync: {gate['dir_fsync_error']}")
    while True:
        ack_names = {path.name for path in ack_dir.glob("rank*.json")}
        if ack_names == expected:
            gate_acks = _validate_gate_acks(root, gate)
            break
        if not ack_names.issubset(expected) or process.poll() is not None or time.monotonic() >= deadline:
            raise RuntimeError("diagnostic gate acknowledgement failed")
        time.sleep(0.25)
    controller = {
        "schema": "step377-diagnostic-world8-v1",
        "status": "DIAGNOSTIC_LIVE_BINDING_PASS",
        "diagnostic_only": True,
        "rank_count": 8,
        "physical_device_ids": sorted(device_ids),
        "rank_device_mapping": rank_mapping,
        "gate": gate,
        "gate_ack_count": len(gate_acks),
        "module_file_sha256": module_hashes,
        "launcher_ownership_sha256": launcher_sha256,
        "rank_ownership_sha256": host._step377_rank_ownership_sha256,
    }
    host.legacy.atomic_json(root / "controller_status.json", controller)
    while True:
        failures = sorted((root / "failure").glob("rank*.txt"))
        if failures:
            raise RuntimeError("rank failure after diagnostic start")
        done = {path.name for path in (root / "done").glob("rank*.json")}
        if done == expected:
            _verify_gate(root / GATE_NAME, gate)
            _validate_gate_acks(root, gate)
            controller["status"] = "DIAGNOSTIC_RANKS_DONE"
            host.legacy.atomic_json(root / "controller_status.json", controller)
            if durability_errors:
                error = RuntimeError("published artifact durability errors: " + "; ".join(durability_errors))
                setattr(error, "published_artifact_errors", tuple(durability_errors))
                raise error
            return controller
        if process.poll() is not None or time.monotonic() >= deadline:
            raise RuntimeError("diagnostic done gate failed")
        time.sleep(0.25)


def _write_new_json(path: Path, value: dict[str, Any], *,
                    committed_error_ok: bool = False) -> dict[str, Any]:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            descriptor = -1
            raise
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    result = {"published": True, "dir_fsync_ok": True, "dir_fsync_error": None}
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            os.fsync(parent_fd)
        except OSError as error:
            result["dir_fsync_ok"] = False
            result["dir_fsync_error"] = f"{type(error).__name__}: {error}"
    finally:
        os.close(parent_fd)
    if not result["dir_fsync_ok"] and not committed_error_ok:
        raise RuntimeError(f"committed JSON directory fsync failed: {result['dir_fsync_error']}")
    return result


def run(args: argparse.Namespace) -> int:
    _guard_dependencies()
    before = _input_hashes(args.input_dir)
    host = load_host()
    guard = load_process_guard()
    originals = {
        "wait_for_results": host.wait_for_results,
        "terminate_group": host.terminate_group,
        "snapshot_owned_npu_processes": host.snapshot_owned_npu_processes,
        "preflight": host.legacy.preflight,
        "cleanup_owned_and_postflight": host.legacy.cleanup_owned_and_postflight,
    }
    primary: BaseException | None = None
    result_code: int | None = None
    try:
        host.wait_for_results = lambda root, process, timeout: diagnostic_wait_for_results(
            host, guard, root, process, timeout
        )
        host._step377_port = args.port
        def safe_preflight(root: Path, port: int) -> int:
            guard.assert_stable_clear(host.legacy.npu_smi, Path(__file__).resolve(), port, None)
            guard.assert_port_free(port)
            return 0
        def cleanup_all(root: Path, port: int) -> None:
            path = root / "launcher_ownership.json"
            launcher_sha = sha256_file(path)
            ownership = guard.read_ownership_json(path, launcher_sha)
            _pid, _start, pgid, owned_port = guard.validate_ownership_manifest(ownership)
            if owned_port != port:
                raise RuntimeError("cleanup ownership port mismatch")
            errors = []
            rank_path = root / "rank_ownership.json"
            try:
                if rank_path.exists() and hasattr(host, "_step377_rank_ownership_sha256"):
                    rank, identities = guard.read_rank_ownership_json(
                        rank_path, host._step377_rank_ownership_sha256,
                        expected_launcher_sha256=launcher_sha,
                        case_path=Path(__file__).resolve(), port=port)
                    if hasattr(host, "_step377_gate_token_sha256") and rank["gate_token_sha256"] != host._step377_gate_token_sha256:
                        raise RuntimeError("rank ownership gate token SHA256 mismatch")
                    guard.terminate_owned(identities, guard.owned_identity_alive)
                elif guard.approved_step377_rank_workers():
                    raise RuntimeError("ownership_unestablished: approved residual lacks rank evidence")
            except Exception as error:
                errors.append(f"rank_cleanup: {error}")
            for label, action in (
                ("launcher_cleanup", lambda: guard.safe_group_cleanup(ownership, case_path=Path(__file__).resolve())),
                ("stable_clear", lambda: guard.assert_stable_clear(host.legacy.npu_smi, Path(__file__).resolve(), port, pgid)),
                ("port_free", lambda: guard.assert_port_free(port)),
            ):
                try:
                    action()
                except Exception as error:
                    errors.append(f"{label}: {error}")
            if errors:
                aggregate = RuntimeError("cleanup domain errors: " + "; ".join(errors))
                setattr(aggregate, "cleanup_errors", tuple(errors))
                raise aggregate
        def safe_terminate(_process: Any) -> None:
            cleanup_all(args.output_dir, args.port)
        def safe_postflight(root: Path, port: int) -> int:
            cleanup_all(root, port)
            return 0
        host.terminate_group = safe_terminate
        host.snapshot_owned_npu_processes = lambda *_args: {}
        host.legacy.preflight = safe_preflight
        host.legacy.cleanup_owned_and_postflight = safe_postflight
        delegated = argparse.Namespace(**vars(args))
        delegated.worker = WORKER_ADAPTER
        # The adapter owns this flag and appends it exactly once for STEP358.
        # Passing it through the host would be rejected by the adapter CLI.
        delegated.first_profiled_only = False
        delegated.state_diagnostic_only = False
        result_code = int(host.run(delegated))
    except BaseException as error:
        primary = error
    finally:
        try:
            after = _input_hashes(args.input_dir)
            if after != before:
                raise RuntimeError("STEP260 inputs changed during diagnostic world8 run")
        except BaseException as drift:
            if primary is None:
                primary = drift
            else:
                try:
                    setattr(primary, "input_drift_error", drift)
                except BaseException:
                    pass
        restore_errors = []
        for owner, name in (
            (host.legacy, "cleanup_owned_and_postflight"), (host.legacy, "preflight"),
            (host, "snapshot_owned_npu_processes"), (host, "terminate_group"),
            (host, "wait_for_results"),
        ):
            try:
                setattr(owner, name, originals[name])
            except BaseException as cleanup:
                restore_errors.append(cleanup)
        if restore_errors:
            if primary is None:
                primary = restore_errors[0]
            else:
                try:
                    setattr(primary, "cleanup_error", restore_errors[0])
                    setattr(primary, "cleanup_errors", tuple(restore_errors))
                except BaseException:
                    pass
    if primary is not None:
        raise primary
    if result_code is None:
        raise RuntimeError("diagnostic host returned no status")
    if result_code == 0:
        output = args.output_dir.resolve(strict=True)
        details = validate_outputs(output, before)
        forbidden = (output / "release_after_npu_smi", output / "release")
        if any(path.exists() or path.is_symlink() for path in forbidden):
            raise RuntimeError("release output appeared in diagnostic run")
        controller = _read_bounded_regular_json(output / "controller_status.json", "controller status")
        module_hashes = controller.get("module_file_sha256")
        gate = controller.get("gate")
        if (
            controller.get("schema") != "step377-diagnostic-world8-v1"
            or controller.get("status") != "DIAGNOSTIC_RANKS_DONE"
            or controller.get("diagnostic_only") is not True
            or type(controller.get("gate_ack_count")) is not int
            or controller["gate_ack_count"] != 8
            or not isinstance(module_hashes, dict)
            or set(module_hashes) != {"cloud_init", "cloud_extension", "cloud_linalg"}
            or any(
                not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in module_hashes.values()
            )
            or not isinstance(gate, dict)
            or not isinstance(gate.get("token_sha256"), str)
            or len(gate["token_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in gate["token_sha256"])
            or not isinstance(controller.get("launcher_ownership_sha256"), str)
            or not isinstance(controller.get("rank_ownership_sha256"), str)
            or any(not re.fullmatch(r"[0-9a-f]{64}", controller[key]) for key in
                   ("launcher_ownership_sha256", "rank_ownership_sha256"))
        ):
            raise RuntimeError("diagnostic controller completion contract mismatch")
        summary = {
            "schema": "step377-diagnostic-host-summary-v1",
            "status": "diagnostic_world8_pass",
            "diagnostic_only": True,
            "release_candidate": False,
            "rank_count": 8,
            "input_sha256": {str(rank): digest for rank, digest in before.items()},
            "raw_profiles_retained": True,
            "module_file_sha256": module_hashes,
            "gate_token_sha256": gate["token_sha256"],
            "launcher_ownership_sha256": controller["launcher_ownership_sha256"],
            "rank_ownership_sha256": controller["rank_ownership_sha256"],
            **details,
        }
        _write_new_json(output / SUMMARY_NAME, summary)
    return result_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--shadow-root", required=True, type=Path)
    parser.add_argument("--installed-custom-opp", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
