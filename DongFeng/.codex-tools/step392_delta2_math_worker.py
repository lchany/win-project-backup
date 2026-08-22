#!/usr/bin/env python3
"""Diagnostic-only identity adapter for the audited STEP358 math worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


TOOLS = Path(__file__).resolve().parent
WORKER_PATH = TOOLS / "step358_qrv2_release_math_worker.py"
COLD_CASE_PATH = TOOLS / "step343_qrv2_cold_case.py"
ORACLE_PATH = TOOLS / "qrv2_release_oracle.py"
EXPECTED_SHA256 = {
    WORKER_PATH.name: "f5e3bc0b4e333109c8c3c0003e3467b995fe6a3c061e911704ab06a29bfe10c7",
    COLD_CASE_PATH.name: "8a5abcd6e9654fc943847d6695bec1bd71fe2b2558a3ec7b903fe13a4eeb6508",
    ORACLE_PATH.name: "d92e02c3df761ddcc94580836615daa661c0e31c23eb8dc32a25dbd806bf6492",
}
DIAGNOSTIC_IDENTITY = "QrV2_qa_position_delta2_only_diagnostic_v1"
DIAGNOSTIC_AIC = DIAGNOSTIC_IDENTITY + "_0_mix_aic"
DIAGNOSTIC_AIV = DIAGNOSTIC_IDENTITY + "_0_mix_aiv"
LEGACY_V6_AIC = "QrV2_vtv_direct_qa_legacy_probe_v6_0_mix_aic"
LEGACY_V6_AIV = "QrV2_vtv_direct_qa_legacy_probe_v6_0_mix_aiv"
ORIGINAL_AIC = "QrV2_566c2e1c0e6c8c92152ad84416d77006_0_mix_aic"
ORIGINAL_AIV = "QrV2_566c2e1c0e6c8c92152ad84416d77006_0_mix_aiv"
V4_AIC = "QrV2_lifetime_alpha_sync_fix_v4_0_mix_aic"
V4_AIV = "QrV2_lifetime_alpha_sync_fix_v4_0_mix_aiv"
V5_AIC = "QrV2_matmul_position_fix_v5_0_mix_aic"
V5_AIV = "QrV2_matmul_position_fix_v5_0_mix_aiv"
DIAGNOSTIC_START_NAME = "diagnostic_start_after_npu_smi"
LEGACY_START_NAME = "release_after_npu_smi"
GATE_ACK_DIR = "diagnostic_gate_ack"
GATE_SCHEMA = "step392-diagnostic-start-v1"
ACK_SCHEMA = "step392-diagnostic-gate-ack-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guard_dependencies() -> None:
    for path in (WORKER_PATH, COLD_CASE_PATH, ORACLE_PATH):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"STEP392 dependency must be regular: {path.name}")
        if sha256_file(path) != EXPECTED_SHA256[path.name]:
            raise RuntimeError(f"STEP392 dependency SHA mismatch: {path.name}")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load STEP392 dependency: {path.name}")
    module = importlib.util.module_from_spec(spec)
    marker = object()
    previous = sys.modules.get(name, marker)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is marker:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous  # type: ignore[assignment]
    return module


def load_worker() -> ModuleType:
    _guard_dependencies()
    return _load_module("_step392_isolated_step358_worker", WORKER_PATH)


def _referenced_qrv2(worker: ModuleType, profile_root: Path) -> dict[str, int]:
    mappings, references, _dictionary_sources, _task_sources = (
        worker.legacy.collect_runtime_identity(profile_root)
    )
    by_identity: dict[str, list[tuple[int, int]]] = {}
    for hash_value, name in mappings.items():
        count = references.get(hash_value, 0)
        if count > 0 and name.startswith("QrV2"):
            by_identity.setdefault(name, []).append((hash_value, count))
    duplicate = {name: rows for name, rows in by_identity.items() if len(rows) != 1}
    if duplicate:
        raise RuntimeError(
            "STEP392 referenced QrV2 identity maps to multiple hashes: "
            + repr(sorted(duplicate))
        )
    return {name: sum(count for _hash, count in rows) for name, rows in by_identity.items()}


def install_worker_identity(worker: ModuleType) -> Callable[[], None]:
    original_verify = worker.verify_profile
    original_finalize = worker._finalize_call
    original_aic = worker.CANDIDATE_AIC
    original_aiv = worker.CANDIDATE_AIV
    worker.CANDIDATE_AIC = DIAGNOSTIC_AIC
    worker.CANDIDATE_AIV = DIAGNOSTIC_AIV

    def verify(profile_root: Path, *, expected_aic_references: int) -> dict[str, Any]:
        if expected_aic_references != 1:
            raise RuntimeError("STEP392 requires exactly one diagnostic task reference")
        result = original_verify(profile_root, expected_aic_references=1)
        referenced = _referenced_qrv2(worker, profile_root)
        if referenced != {DIAGNOSTIC_AIC: 1}:
            raise RuntimeError(
                "STEP392 task-referenced QrV2 identity set is not exact: "
                + repr(sorted(referenced.items()))
            )
        result.update(
            {
                "diagnostic_identity": DIAGNOSTIC_IDENTITY,
                "diagnostic_aic_task_reference_count": 1,
                "diagnostic_aiv_task_reference_count": 0,
                "original_task_reference_count": 0,
                "v4_task_reference_count": 0,
                "v5_task_reference_count": 0,
                "legacy_v6_task_reference_count": 0,
                "unknown_qrv2_task_reference_count": 0,
                "raw_profile_retained": True,
            }
        )
        return result

    worker.verify_profile = verify

    def finalize(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_finalize(*args, **kwargs)
        reconstruction = result["reconstruction"]
        orthogonality = result["orthogonality"]
        projection = result["full_rank_projection"]
        projection_pass = projection.get("pass") if projection.get("required") else True
        predicates = {
            "input_unmodified": "pass" if result["input_unmodified"] else "fail",
            "shape": "pass" if result["shape_pass"] else "fail",
            "finite": "pass" if result["finite_pass"] else "fail",
            "reconstruction": "pass" if reconstruction["violation_count"] == 0 else "fail",
            "orthogonality": "pass" if orthogonality["violation_count"] == 0 else "fail",
            "lower_triangle_exact_zero": "pass" if result["lower_triangle_exact_zero"] else "fail",
            "projection": "pass" if projection_pass else "fail",
        }
        _normalized, diagnostic_nonfinite_count = worker._normalize_json_diagnostic(result)
        result.update(
            {
                "predicate_status": predicates,
                "failed_predicates": sorted(key for key, value in predicates.items() if value == "fail"),
                "not_evaluated_predicates": [],
                "diagnostic_scalars_finite": diagnostic_nonfinite_count == 0,
                "diagnostic_nonfinite_scalar_count": diagnostic_nonfinite_count,
                "reconstruction_violation_count": reconstruction["violation_count"],
                "orthogonality_violation_count": orthogonality["violation_count"],
                "projection_pass": projection_pass,
            }
        )
        return result

    worker._finalize_call = finalize

    def restore() -> None:
        worker.verify_profile = original_verify
        worker._finalize_call = original_finalize
        worker.CANDIDATE_AIC = original_aic
        worker.CANDIDATE_AIV = original_aiv

    return restore


def install_diagnostic_wait(cold_case: ModuleType) -> Callable[[], None]:
    original_wait = cold_case.wait_release

    def wait(path: Path, timeout_seconds: int = 120) -> None:
        if path.name != LEGACY_START_NAME:
            raise RuntimeError("STEP392 worker received an unexpected start gate")
        diagnostic_path = path.with_name(DIAGNOSTIC_START_NAME)
        if path.exists() or path.is_symlink():
            raise RuntimeError("legacy release start gate must not exist")
        if diagnostic_path.exists() or diagnostic_path.is_symlink():
            raise RuntimeError("diagnostic start gate must be newly created")
        original_wait(diagnostic_path, timeout_seconds=timeout_seconds)
        if path.exists() or path.is_symlink():
            raise RuntimeError("legacy release start gate appeared")
        before = diagnostic_path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("diagnostic start gate must be a regular non-symlink file")
        descriptor = os.open(diagnostic_path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            opened = os.fstat(descriptor)
            payload = os.read(descriptor, 4097)
        finally:
            os.close(descriptor)
        after = diagnostic_path.lstat()
        identity = (before.st_dev, before.st_ino)
        if (
            (opened.st_dev, opened.st_ino) != identity
            or (after.st_dev, after.st_ino) != identity
            or not stat.S_ISREG(after.st_mode)
        ):
            raise RuntimeError("diagnostic start gate identity changed during validation")
        if len(payload) > 4096:
            raise RuntimeError("diagnostic start gate is too large")
        try:
            gate = json.loads(payload.decode("utf-8"))
        except (UnicodeError, ValueError) as error:
            raise RuntimeError("diagnostic start gate JSON is invalid") from error
        token = gate.get("token") if isinstance(gate, dict) else None
        if (
            not isinstance(gate, dict) or set(gate) != {"schema", "token"}
            or gate.get("schema") != GATE_SCHEMA or not isinstance(token, str)
            or len(token) != 32
            or any(character not in "0123456789abcdef" for character in token)
        ):
            raise RuntimeError("diagnostic start gate schema/token mismatch")
        rank_text = os.environ.get("LOCAL_RANK")
        if rank_text not in {str(rank) for rank in range(8)}:
            raise RuntimeError("STEP392 LOCAL_RANK is invalid for gate acknowledgement")
        ack_dir = diagnostic_path.parent / GATE_ACK_DIR
        ack_status = ack_dir.lstat()
        if not stat.S_ISDIR(ack_status.st_mode):
            raise RuntimeError("diagnostic gate acknowledgement directory is invalid")
        ack = {
            "schema": ACK_SCHEMA,
            "rank": int(rank_text),
            "gate_device": before.st_dev,
            "gate_inode": before.st_ino,
            "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        }
        ack_path = ack_dir / f"rank{rank_text}.json"
        ack_fd = os.open(ack_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(ack_fd, "w", encoding="utf-8") as stream:
            json.dump(ack, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    cold_case.wait_release = wait

    def restore() -> None:
        cold_case.wait_release = original_wait

    return restore


def _bind_module(name: str, module: ModuleType) -> Callable[[], None]:
    marker = object()
    previous = sys.modules.get(name, marker)
    sys.modules[name] = module

    def restore() -> None:
        if previous is marker:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous  # type: ignore[assignment]

    return restore


def _underlying_argv(args: argparse.Namespace) -> list[str]:
    return [
        str(WORKER_PATH),
        "--input-dir", str(args.input_dir),
        "--output-dir", str(args.output_dir),
        "--shadow-root", str(args.shadow_root),
        "--installed-custom-opp", str(args.installed_custom_opp),
        "--first-profiled-only",
    ]


def run(args: argparse.Namespace) -> int:
    _guard_dependencies()
    cold_case = _load_module("_step392_isolated_cold_case", COLD_CASE_PATH)
    worker = load_worker()
    restorers: list[Callable[[], None]] = []
    previous_argv = sys.argv
    primary: BaseException | None = None
    result: int | None = None
    try:
        restorers.append(_bind_module("step343_qrv2_cold_case", cold_case))
        restorers.append(install_diagnostic_wait(cold_case))
        restorers.append(install_worker_identity(worker))
        sys.argv = _underlying_argv(args)
        result = int(worker.main())
    except BaseException as error:
        primary = error
    finally:
        sys.argv = previous_argv
        cleanup_errors: list[BaseException] = []
        for restore in reversed(restorers):
            try:
                restore()
            except BaseException as cleanup:
                cleanup_errors.append(cleanup)
        if primary is not None and cleanup_errors:
            try:
                setattr(primary, "cleanup_errors", cleanup_errors)
            except BaseException:
                pass
            try:
                add_note = getattr(primary, "add_note", None)
                if add_note is not None:
                    for cleanup in cleanup_errors:
                        add_note(
                            "STEP392 cleanup failed: "
                            + type(cleanup).__name__
                            + ": "
                            + str(cleanup)
                        )
            except BaseException:
                pass
        elif primary is None and cleanup_errors:
            primary = cleanup_errors[0]
            try:
                setattr(primary, "cleanup_errors", cleanup_errors)
            except BaseException:
                pass
    if primary is not None:
        raise primary
    if result is None:
        raise RuntimeError("STEP392 underlying worker returned no result")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shadow-root", required=True, type=Path)
    parser.add_argument("--installed-custom-opp", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
