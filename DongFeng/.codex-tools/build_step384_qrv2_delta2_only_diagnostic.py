#!/usr/bin/env python3
"""Prepare/build-only adapter for the STEP384 delta2-only QRv2 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import secrets
import stat
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
ADAPTER_PATH = Path(__file__).resolve()
BASE_BUILDER_PATH = TOOLS_DIR / "build_qrv2_release.py"
AUDITED_ADAPTER_PATH = TOOLS_DIR / "build_qrv2_diagnostic_probe.py"
PATCHER_PATH = TOOLS_DIR / "step384_patch_qr_v2_delta2_only_diagnostic.py"
V4_PATCHER_PATH = TOOLS_DIR / "step338_patch_qr_v2_lifetime.py"
EXPECTED_DEPENDENCY_SHA256 = {
    AUDITED_ADAPTER_PATH: "fc65fecc58cefb86f64b6e71d64a21e5e4bc1416b42f1cd696aff6bbdedc299e",
    BASE_BUILDER_PATH: "d6f2aea68574422ba6e28f220c5581adf8b25019494882dea156a80b1b513e90",
    V4_PATCHER_PATH: "d4b260919440e6298c9caea5ca3a7c8bb7426a30ed84cc68d934250ee0ae9fe2",
    PATCHER_PATH: "2bdaf51e3b08388ca5fcb156e0602312b4f1de3dfc533da6e2d7778d10d3820c",
}

BUILD_READY = False
BIN_NAME = "QrV2_qa_position_delta2_only_diagnostic_v1"
MANIFEST_NAME = "release_manifest.json"
DIAGNOSTIC_BUILT_STATUS = "diagnostic_built_unvalidated"
FORBIDDEN_PACKAGE_STATUS = "forbidden_diagnostic_probe"
ATTEMPT_MARKER_NAME = "step384_build_attempt.json"
ATTEMPT_COMPLETION_TEMP_NAME = ".step384_build_attempt.completed.tmp"
ATTEMPT_MARKER_SCHEMA = "step384.nonconsumable-build-attempt.v1"
ATTEMPT_IDENTITY = "STEP384_qrv2_delta2_only_diagnostic_build_v1"

audited_adapter: Optional[ModuleType] = None
diagnostic_patcher: Optional[ModuleType] = None


def _require_build_ready() -> None:
    if BUILD_READY is not True:
        raise RuntimeError("STEP384 diagnostic build is not authorized: BUILD_READY is false")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_dependency_hashes() -> None:
    for path, expected in EXPECTED_DEPENDENCY_SHA256.items():
        if path.parent != TOOLS_DIR or not path.is_file() or path.is_symlink():
            raise RuntimeError(f"locked dependency path rejected: {path.name}")
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"locked dependency SHA drift: {path.name}: {actual}")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load locked dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _activate() -> tuple[ModuleType, ModuleType]:
    global audited_adapter, diagnostic_patcher
    _verify_dependency_hashes()
    if diagnostic_patcher is None:
        diagnostic_patcher = _load_module("_step384_locked_patcher", PATCHER_PATH)
    if audited_adapter is None:
        audited_adapter = _load_module(
            "_step384_audited_build_adapter", AUDITED_ADAPTER_PATH
        )
    audited_adapter.diagnostic_patcher = diagnostic_patcher
    audited_adapter.ADAPTER_PATH = ADAPTER_PATH
    audited_adapter.PATCHER_PATH = PATCHER_PATH
    audited_adapter.V4_PATCHER_PATH = V4_PATCHER_PATH
    audited_adapter.BIN_NAME = BIN_NAME
    audited_adapter._tool_hashes = _tool_hashes
    audited_adapter._validate_active_wiring = _validate_active_wiring
    return audited_adapter, diagnostic_patcher


def _tool_hashes() -> Dict[str, str]:
    return {
        "diagnostic_adapter_sha256": _sha256_file(ADAPTER_PATH),
        "audited_adapter_sha256": _sha256_file(AUDITED_ADAPTER_PATH),
        "base_builder_sha256": _sha256_file(BASE_BUILDER_PATH),
        "step384_patcher_sha256": _sha256_file(PATCHER_PATH),
        "v4_patcher_sha256": _sha256_file(V4_PATCHER_PATH),
    }


def _validate_active_wiring(base: ModuleType) -> None:
    audited, patcher = _activate()
    if Path(base.__file__).resolve() != BASE_BUILDER_PATH:
        raise RuntimeError("diagnostic base builder path drift")
    if Path(patcher.__file__).resolve() != PATCHER_PATH:
        raise RuntimeError("STEP384 patcher path drift")
    if Path(patcher.release_v4.__file__).resolve() != V4_PATCHER_PATH:
        raise RuntimeError("diagnostic v4 dependency path drift")
    if base.candidate_patcher is not patcher:
        raise RuntimeError("active patcher is not STEP384")
    for name in ("build_candidate", "verify_candidate_structure", "sha256_bytes", "write_new_file"):
        if getattr(base, name, None) is not getattr(patcher, name):
            raise RuntimeError(f"active patcher function drift: {name}")
    if base.BIN_NAME != BIN_NAME:
        raise RuntimeError("diagnostic bin identity drift")
    if base.EXPECTED_SOURCE_SHA256 != patcher.EXPECTED_SOURCE_SHA256:
        raise RuntimeError("diagnostic source SHA wiring drift")
    if base.EXPECTED_CANDIDATE_SHA256 != "e352ac31f97980bc5c92caac663414782b78fc3004ff36709b6be8000353b003":
        raise RuntimeError("diagnostic candidate SHA wiring drift")
    if patcher.EXPECTED_V4_CANDIDATE_SHA256 != "2213dbae5027c179f09c8e3b43be51587ba22e3eb2a7546311048efc3844614b":
        raise RuntimeError("reverse v4 SHA wiring drift")
    for name in ("package_release", "parse_args", "main"):
        if getattr(base, name, None) is not audited._forbidden_release_api:
            raise RuntimeError(f"base release API was not poisoned: {name}")


def _load_base() -> ModuleType:
    _require_build_ready()
    audited, _ = _activate()
    return audited._load_base()


def _marker_path(workdir: Path) -> Path:
    return workdir.absolute() / ATTEMPT_MARKER_NAME


def _json_payload(value: Dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


def _write_all(file_fd: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(file_fd, remaining)
        if written <= 0:
            raise OSError("short write while sealing STEP384 attempt marker")
        remaining = remaining[written:]


def _create_attempt_marker(workdir: Path, approved_root: Path) -> Dict[str, Any]:
    if workdir.absolute() != approved_root.absolute() / "work":
        raise ValueError("workdir must be exactly approved_root/work")
    parent_fd = os.open(
        workdir.absolute(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    file_fd: Optional[int] = None
    try:
        file_fd = os.open(
            ATTEMPT_MARKER_NAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        identity = os.fstat(file_fd)
        lock = {
            "schema": ATTEMPT_MARKER_SCHEMA,
            "status": "in_progress_nonconsumable",
            "attempt_identity": ATTEMPT_IDENTITY,
            "nonce": secrets.token_hex(32),
            "marker_dev": identity.st_dev,
            "marker_ino": identity.st_ino,
        }
        _write_all(file_fd, _json_payload(lock))
        os.fsync(file_fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise RuntimeError("attempt marker is not a regular file")
        os.fsync(parent_fd)
        return lock
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)


def _read_regular_at(parent_fd: int, name: str) -> tuple[bytes, os.stat_result]:
    file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        identity = os.fstat(file_fd)
        if not stat.S_ISREG(identity.st_mode):
            raise RuntimeError(f"STEP384 sealed input is not regular: {name}")
        chunks = []
        while True:
            block = os.read(file_fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks), identity
    finally:
        os.close(file_fd)


def _closure_summary(base: ModuleType, manifest: Dict[str, Any]) -> str:
    artifacts = manifest.get("artifacts", {})
    summary = {
        soc: {
            key: artifacts.get(soc, {}).get(key)
            for key in (
                "object_path", "object_size", "object_sha256", "json_path",
                "json_size", "json_sha256", "opc_log_path", "opc_log_size",
                "opc_log_sha256", "kernel_name", "bin_file_name",
            )
        }
        for soc in base.SOCS
    }
    return hashlib.sha256(_json_payload(summary)).hexdigest()


def _seal_completed_attempt(
    workdir: Path,
    lock: Dict[str, Any],
    manifest_sha256: str,
    closure_summary: str,
) -> None:
    parent_fd = os.open(
        workdir.absolute(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    temp_fd: Optional[int] = None
    committed = False
    temp_name = f"{ATTEMPT_COMPLETION_TEMP_NAME}.{lock['nonce']}"
    try:
        marker_payload, marker_identity = _read_regular_at(
            parent_fd, ATTEMPT_MARKER_NAME
        )
        if json.loads(marker_payload) != lock or (
            marker_identity.st_dev != lock["marker_dev"]
            or marker_identity.st_ino != lock["marker_ino"]
        ):
            raise RuntimeError("STEP384 in-progress marker identity drift")
        temp_fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        completion_identity = os.fstat(temp_fd)
        completed = {
            "schema": ATTEMPT_MARKER_SCHEMA,
            "status": "completed_consumable",
            "attempt_identity": ATTEMPT_IDENTITY,
            "nonce": lock["nonce"],
            "manifest_sha256": manifest_sha256,
            "artifact_closure_summary": closure_summary,
            "marker_dev": completion_identity.st_dev,
            "marker_ino": completion_identity.st_ino,
        }
        _write_all(temp_fd, _json_payload(completed))
        os.fsync(temp_fd)
        if not stat.S_ISREG(os.fstat(temp_fd).st_mode):
            raise RuntimeError("completion marker is not a regular file")
        os.close(temp_fd)
        temp_fd = None
        temp_payload, temp_identity = _read_regular_at(parent_fd, temp_name)
        if json.loads(temp_payload) != completed or (
            temp_identity.st_dev != completed["marker_dev"]
            or temp_identity.st_ino != completed["marker_ino"]
        ):
            raise RuntimeError("STEP384 completion seal verification failed")
        os.fsync(parent_fd)
        os.rename(
            temp_name,
            ATTEMPT_MARKER_NAME,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        committed = True
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        if committed:
            try:
                os.close(parent_fd)
            except OSError:
                pass
        else:
            os.close(parent_fd)


def _require_completed_attempt(base: ModuleType, workdir: Path, manifest: Dict[str, Any]) -> None:
    parent_fd = os.open(
        workdir.absolute(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        marker_payload, marker_identity = _read_regular_at(parent_fd, ATTEMPT_MARKER_NAME)
        manifest_payload, _ = _read_regular_at(parent_fd, MANIFEST_NAME)
    finally:
        os.close(parent_fd)
    try:
        marker = json.loads(marker_payload)
        persisted_manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("STEP384 completion marker or manifest is malformed") from error
    required = {
        "schema", "status", "attempt_identity", "nonce", "manifest_sha256",
        "artifact_closure_summary", "marker_dev", "marker_ino",
    }
    if set(marker) != required or marker.get("schema") != ATTEMPT_MARKER_SCHEMA or marker.get("status") != "completed_consumable" or marker.get("attempt_identity") != ATTEMPT_IDENTITY:
        raise RuntimeError("STEP384 build attempt is not completed and consumable")
    nonce = marker.get("nonce")
    if not isinstance(nonce, str) or len(nonce) != 64 or any(
        character not in "0123456789abcdef" for character in nonce
    ):
        raise RuntimeError("STEP384 completion nonce is malformed")
    if persisted_manifest != manifest:
        raise RuntimeError("STEP384 caller manifest differs from sealed manifest")
    if marker_identity.st_dev != marker["marker_dev"] or marker_identity.st_ino != marker["marker_ino"]:
        raise RuntimeError("STEP384 completion marker replacement detected")
    if hashlib.sha256(manifest_payload).hexdigest() != marker["manifest_sha256"]:
        raise RuntimeError("STEP384 sealed manifest SHA drift")
    audited_adapter._validate_built_artifact_closure(
        base, workdir, manifest, enrich=False
    )
    if _closure_summary(base, manifest) != marker["artifact_closure_summary"]:
        raise RuntimeError("STEP384 artifact closure summary drift")


def _delegate(name: str, *args: Any, **kwargs: Any) -> Any:
    _require_build_ready()
    audited, _ = _activate()
    return getattr(audited, name)(*args, **kwargs)


def _diagnostic_flags() -> Dict[str, Any]:
    return _delegate("_diagnostic_flags")


def _approved_workdir(*args: Any, **kwargs: Any) -> Any:
    return _delegate("_approved_workdir", *args, **kwargs)


def _validate_manifest(*args: Any, **kwargs: Any) -> Any:
    _require_build_ready()
    _activate()
    if kwargs.get("expected_status") == DIAGNOSTIC_BUILT_STATUS:
        _require_completed_attempt(args[0], Path(args[1]), args[2])
    return _delegate("_validate_manifest", *args, **kwargs)


def _validate_built_artifact_closure(*args: Any, **kwargs: Any) -> Any:
    return _delegate("_validate_built_artifact_closure", *args, **kwargs)


def _assert_no_release_outputs(*args: Any, **kwargs: Any) -> Any:
    return _delegate("_assert_no_release_outputs", *args, **kwargs)


def prepare_release(
    outer_zip: Path,
    workdir: Path,
    approved_root: Path,
    *,
    _base: Optional[ModuleType] = None,
) -> Dict[str, Any]:
    _require_build_ready()
    audited, _ = _activate()
    return audited.prepare_release(outer_zip, workdir, approved_root, _base=_base)


def build_release(
    workdir: Path,
    opc: Path,
    container_contract: Path,
    installed_cloud_root: Path,
    approved_root: Path,
    *,
    _base: Optional[ModuleType] = None,
) -> Dict[str, Any]:
    _require_build_ready()
    try:
        lock = _create_attempt_marker(workdir, approved_root)
    except FileExistsError as error:
        raise RuntimeError(
            "STEP384 workdir has a prior non-consumable build attempt"
        ) from error
    audited, _ = _activate()
    manifest = audited.build_release(
        workdir, opc, container_contract, installed_cloud_root, approved_root,
        _base=_base,
    )
    if manifest.get("status") != DIAGNOSTIC_BUILT_STATUS:
        raise RuntimeError("diagnostic build returned a non-consumable status")
    audited._validate_built_artifact_closure(
        _base if _base is not None else audited._load_base(),
        workdir,
        manifest,
        enrich=False,
    )
    manifest_sha256 = _sha256_file(workdir / MANIFEST_NAME)
    closure_summary = _closure_summary(
        _base if _base is not None else audited._load_base(), manifest
    )
    _seal_completed_attempt(workdir, lock, manifest_sha256, closure_summary)
    return manifest


def package_release(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("STEP384 diagnostic packaging is permanently forbidden")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--outer-zip", required=True, type=Path)
    prepare.add_argument("--workdir", required=True, type=Path)
    prepare.add_argument("--approved-root", required=True, type=Path)
    build = subparsers.add_parser("build")
    build.add_argument("--workdir", required=True, type=Path)
    build.add_argument("--opc", required=True, type=Path)
    build.add_argument("--container-contract", required=True, type=Path)
    build.add_argument("--installed-cloud-root", required=True, type=Path)
    build.add_argument("--approved-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _require_build_ready()
    args = parse_args(argv)
    if args.command == "prepare":
        prepare_release(args.outer_zip, args.workdir, args.approved_root)
    else:
        build_release(
            args.workdir, args.opc, args.container_contract,
            args.installed_cloud_root, args.approved_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
