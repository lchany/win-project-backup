#!/usr/bin/env python3
"""Build the STEP375 delta1-only QRv2 diagnostic probe.

This adapter deliberately exposes only prepare/build.  It reuses the audited
release builder mechanics while sealing every persisted artifact as a
non-packageable diagnostic probe.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Sequence

import step375_patch_qr_v2_vtv_direct_qa_legacy_probe_v6 as diagnostic_patcher


TOOLS_DIR = Path(__file__).resolve().parent
ADAPTER_PATH = Path(__file__).resolve()
BASE_BUILDER_PATH = TOOLS_DIR / "build_qrv2_release.py"
PATCHER_PATH = TOOLS_DIR / "step375_patch_qr_v2_vtv_direct_qa_legacy_probe_v6.py"
V4_PATCHER_PATH = Path(diagnostic_patcher.release_v4.__file__).resolve()
BASE_PATCHER_IMPORT = "step372_patch_qr_v2_matmul_position_v5"
BIN_NAME = diagnostic_patcher.CANDIDATE_IDENTITY
MANIFEST_NAME = "release_manifest.json"
DIAGNOSTIC_BUILT_STATUS = "diagnostic_built_unvalidated"
FORBIDDEN_PACKAGE_STATUS = "forbidden_diagnostic_probe"

_LOAD_COUNTER = itertools.count()


def _diagnostic_flags() -> Dict[str, Any]:
    return {
        "artifact_class": "diagnostic_probe",
        "diagnostic_only": True,
        "release_candidate": False,
        "package_forbidden": True,
    }


def _sha256_file(path: Path) -> str:
    return diagnostic_patcher.sha256_bytes(path.read_bytes())


def _tool_hashes() -> Dict[str, str]:
    return {
        "diagnostic_adapter_sha256": _sha256_file(ADAPTER_PATH),
        "base_builder_sha256": _sha256_file(BASE_BUILDER_PATH),
        "step375_patcher_sha256": _sha256_file(PATCHER_PATH),
        "v4_patcher_sha256": _sha256_file(V4_PATCHER_PATH),
    }


def _forbidden_release_api(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("diagnostic probe release and packaging APIs are forbidden")


def _approved_workdir(
    approved_root: Path, workdir: Path, *, require_new: bool
) -> tuple[Path, Path]:
    root_argument = approved_root.absolute()
    if root_argument.is_symlink():
        raise ValueError("approved root must not be a symlink")
    root = root_argument.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("approved root must be an existing directory")
    requested = workdir.absolute()
    expected = root / "work"
    if requested != expected:
        raise ValueError("workdir must be exactly approved_root/work")
    if requested.is_symlink():
        raise ValueError("diagnostic workdir must not be a symlink")
    if require_new:
        if requested.exists():
            raise FileExistsError("diagnostic workdir reuse is forbidden")
    elif not requested.is_dir():
        raise ValueError("diagnostic workdir must be an existing directory")
    return root, requested


def _load_base() -> ModuleType:
    """Load the base builder with its candidate import isolated and rebound."""

    module_name = f"_step376_qrv2_release_base_{next(_LOAD_COUNTER)}"
    spec = importlib.util.spec_from_file_location(module_name, BASE_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base builder: {BASE_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)

    marker = object()
    previous = sys.modules.get(BASE_PATCHER_IMPORT, marker)
    sys.modules[BASE_PATCHER_IMPORT] = diagnostic_patcher
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        if previous is marker:
            sys.modules.pop(BASE_PATCHER_IMPORT, None)
        else:
            sys.modules[BASE_PATCHER_IMPORT] = previous

    module.BIN_NAME = BIN_NAME
    module.package_release = _forbidden_release_api
    module.parse_args = _forbidden_release_api
    module.main = _forbidden_release_api
    _validate_active_wiring(module)
    return module


def _validate_active_wiring(base: ModuleType) -> None:
    for label, path in (
        ("adapter", ADAPTER_PATH),
        ("base builder", BASE_BUILDER_PATH),
        ("STEP375 patcher", PATCHER_PATH),
        ("v4 dependency", V4_PATCHER_PATH),
    ):
        if not path.is_file() or path.parent != TOOLS_DIR:
            raise RuntimeError(f"{label} escaped the approved tools directory")
    if Path(base.__file__).resolve() != BASE_BUILDER_PATH:
        raise RuntimeError("diagnostic base builder path drift")
    if Path(diagnostic_patcher.__file__).resolve() != PATCHER_PATH:
        raise RuntimeError("diagnostic patcher path drift")
    if Path(diagnostic_patcher.release_v4.__file__).resolve() != V4_PATCHER_PATH:
        raise RuntimeError("diagnostic v4 dependency path drift")
    if base.candidate_patcher is not diagnostic_patcher:
        raise RuntimeError("active patcher is not STEP375")
    expected_functions = (
        "build_candidate",
        "verify_candidate_structure",
        "sha256_bytes",
        "write_new_file",
    )
    for name in expected_functions:
        if getattr(base, name, None) is not getattr(diagnostic_patcher, name):
            raise RuntimeError(f"active patcher function drift: {name}")
    if base.BIN_NAME != BIN_NAME:
        raise RuntimeError("diagnostic bin identity drift")
    if base.EXPECTED_SOURCE_SHA256 != diagnostic_patcher.EXPECTED_SOURCE_SHA256:
        raise RuntimeError("diagnostic source SHA wiring drift")
    if base.EXPECTED_CANDIDATE_SHA256 != diagnostic_patcher.EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("diagnostic candidate SHA wiring drift")
    for name in ("package_release", "parse_args", "main"):
        if getattr(base, name, None) is not _forbidden_release_api:
            raise RuntimeError(f"base release API was not poisoned: {name}")


def _guard_tools(manifest: Dict[str, Any]) -> None:
    if manifest.get("tools") != _tool_hashes():
        raise RuntimeError("diagnostic tool SHA guard failed")


def _decorate_manifest(manifest: Dict[str, Any], *, status: str) -> Dict[str, Any]:
    manifest["status"] = status
    manifest["tools"] = _tool_hashes()
    flags = _diagnostic_flags()
    manifest.setdefault("policy", {}).update(flags)
    manifest.setdefault("candidate", {}).update(flags)
    manifest["candidate"]["identity"] = BIN_NAME
    manifest["candidate"]["bin_name"] = BIN_NAME
    manifest["package"] = {"status": FORBIDDEN_PACKAGE_STATUS}
    return manifest


def _manifest_path(workdir: Path) -> Path:
    return workdir.resolve() / MANIFEST_NAME


def _read_manifest(workdir: Path) -> Dict[str, Any]:
    return json.loads(_manifest_path(workdir).read_text(encoding="utf-8"))


def _assert_no_release_outputs(workdir: Path, manifest: Dict[str, Any]) -> None:
    root = workdir.resolve()
    release_dir = root / "release"
    if release_dir.exists() or release_dir.is_symlink():
        raise RuntimeError("diagnostic probe cannot create a release directory")

    allowed_wheel_raw = manifest.get("paths", {}).get("extracted_wheel")
    allowed_wheel = Path(allowed_wheel_raw).resolve() if allowed_wheel_raw else None
    for path in root.rglob("*"):
        if path.is_symlink() and path.suffix.lower() in {".whl", ".zip"}:
            raise RuntimeError(f"diagnostic probe package artifact forbidden: {path}")
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        resolved = path.resolve()
        if suffix == ".whl" and resolved != allowed_wheel:
            raise RuntimeError(f"new diagnostic wheel forbidden: {path}")
        if suffix == ".zip":
            raise RuntimeError(f"new diagnostic outer ZIP forbidden: {path}")


def _validate_manifest(
    base: ModuleType,
    workdir: Path,
    manifest: Dict[str, Any],
    *,
    expected_status: str,
) -> None:
    _validate_active_wiring(base)
    if manifest.get("status") != expected_status:
        raise RuntimeError(
            f"unexpected diagnostic manifest status: {manifest.get('status')!r}"
        )
    _guard_tools(manifest)
    flags = _diagnostic_flags()
    for layer in ("policy", "candidate"):
        section = manifest.get(layer, {})
        for key, value in flags.items():
            if section.get(key) != value:
                raise RuntimeError(f"diagnostic flag drift: {layer}.{key}")
    candidate = manifest.get("candidate", {})
    if candidate.get("identity") != BIN_NAME or candidate.get("bin_name") != BIN_NAME:
        raise RuntimeError("diagnostic candidate identity drift")
    if (
        manifest.get("original", {}).get("source_sha256")
        != diagnostic_patcher.EXPECTED_SOURCE_SHA256
    ):
        raise RuntimeError("diagnostic source SHA manifest drift")
    if candidate.get("source_sha256") != diagnostic_patcher.EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("diagnostic candidate SHA manifest drift")
    if manifest.get("package") != {"status": FORBIDDEN_PACKAGE_STATUS}:
        raise RuntimeError("diagnostic packaging guard drift")

    base._guard_originals(manifest)
    base._guard_build_inputs(workdir, manifest)
    original_source = Path(
        manifest["immutable_guards"]["extracted_original_source"]["path"]
    ).read_bytes()
    for soc_key in manifest["build_inputs"]:
        diagnostic_patcher.verify_candidate_structure(
            original_source,
            (workdir / "build" / soc_key / "qr_v2.cpp").read_bytes(),
        )
    _assert_no_release_outputs(workdir, manifest)
    if expected_status == DIAGNOSTIC_BUILT_STATUS:
        _validate_built_artifact_closure(base, workdir, manifest, enrich=False)


def _validate_built_artifact_closure(
    base: ModuleType,
    workdir: Path,
    manifest: Dict[str, Any],
    *,
    enrich: bool,
) -> None:
    base._assert_no_symlinks(workdir)
    audited: Dict[str, Dict[str, Any]] = {}
    for soc_key in base.SOCS:
        build_dir = workdir / "build" / soc_key
        object_path, json_path, metadata = base._validate_artifacts(build_dir)
        log_path = build_dir / "opc.log"
        if not log_path.is_file() or log_path.is_symlink():
            raise RuntimeError(f"{soc_key} OPC log closure failed")
        actual = {
            "object_path": str(object_path),
            "object_size": object_path.stat().st_size,
            "object_sha256": base.sha256_file(object_path),
            "json_path": str(json_path),
            "json_size": json_path.stat().st_size,
            "json_sha256": base.sha256_file(json_path),
            "opc_log_path": str(log_path),
            "opc_log_size": log_path.stat().st_size,
            "opc_log_sha256": base.sha256_file(log_path),
            "kernel_name": metadata["kernelName"],
            "bin_file_name": metadata["binFileName"],
            "concrete_entries": metadata["_audited_concrete_entries"],
        }
        artifact = manifest.get("artifacts", {}).get(soc_key)
        if not isinstance(artifact, dict):
            raise RuntimeError(f"{soc_key} artifact manifest missing")
        if enrich:
            artifact.update(
                {
                    "json_size": actual["json_size"],
                    "opc_log_path": actual["opc_log_path"],
                    "opc_log_size": actual["opc_log_size"],
                }
            )
        if artifact.get("status") != "built_structure_valid":
            raise RuntimeError(f"{soc_key} artifact status closure failed")
        for key, value in actual.items():
            if artifact.get(key) != value:
                raise RuntimeError(f"{soc_key} artifact closure failed for {key}")
        audited[soc_key] = {
            **actual,
            "object_bytes": object_path.read_bytes(),
            "json_bytes": json_path.read_bytes(),
        }

    canonical = audited[base.CANONICAL_SOC_KEY]
    alias = audited[base.ALIAS_SOC_KEY]
    for kind in ("object", "json"):
        if canonical[f"{kind}_sha256"] != alias[f"{kind}_sha256"]:
            raise RuntimeError(f"SoC alias {kind} SHA closure failed")
        if canonical[f"{kind}_bytes"] != alias[f"{kind}_bytes"]:
            raise RuntimeError(f"SoC alias {kind} byte closure failed")


def prepare_release(
    outer_zip: Path,
    workdir: Path,
    approved_root: Path,
    *,
    _base: Optional[ModuleType] = None,
) -> Dict[str, Any]:
    _, workdir = _approved_workdir(approved_root, workdir, require_new=True)
    base = _base if _base is not None else _load_base()
    _validate_active_wiring(base)
    original_write_json_new = base.write_json_new
    original_guard_tools = base._guard_tools
    manifest_written = False

    def diagnostic_write_json_new(root: Path, path: Path, value: Dict[str, Any]) -> None:
        nonlocal manifest_written
        if Path(path).name == MANIFEST_NAME:
            if value.get("status") != "prepared":
                raise RuntimeError("base prepare status drift")
            _decorate_manifest(value, status="prepared")
            manifest_written = True
        original_write_json_new(root, path, value)

    base.write_json_new = diagnostic_write_json_new
    base._guard_tools = _guard_tools
    try:
        manifest = base.prepare_release(outer_zip.resolve(), workdir.resolve())
    finally:
        base.write_json_new = original_write_json_new
        base._guard_tools = original_guard_tools

    if not manifest_written:
        raise RuntimeError("diagnostic prepare manifest was not atomically intercepted")
    persisted = _read_manifest(workdir)
    _validate_manifest(base, workdir, persisted, expected_status="prepared")
    return persisted


def build_release(
    workdir: Path,
    opc: Path,
    container_contract: Path,
    installed_cloud_root: Path,
    approved_root: Path,
    *,
    _base: Optional[ModuleType] = None,
) -> Dict[str, Any]:
    base = _base if _base is not None else _load_base()
    _, workdir = _approved_workdir(approved_root, workdir, require_new=False)
    before = _read_manifest(workdir)
    _validate_manifest(base, workdir, before, expected_status="prepared")

    original_write_json_atomic = base.write_json_atomic
    original_guard_tools = base._guard_tools
    sealed: Optional[Dict[str, Any]] = None

    def diagnostic_write_json_atomic(
        root: Path, path: Path, value: Dict[str, Any]
    ) -> None:
        nonlocal sealed
        if sealed is not None:
            raise RuntimeError("multiple diagnostic manifest seal attempts")
        if Path(path).resolve() != _manifest_path(workdir):
            raise RuntimeError("unexpected atomic write during diagnostic build")
        if _read_manifest(workdir) != before:
            raise RuntimeError("diagnostic manifest changed during build before sealing")
        if value.get("status") != "built":
            raise RuntimeError("base build status drift before diagnostic seal")
        candidate = copy.deepcopy(value)
        _decorate_manifest(candidate, status=DIAGNOSTIC_BUILT_STATUS)
        _guard_tools(candidate)
        _validate_built_artifact_closure(
            base, workdir, candidate, enrich=True
        )
        _assert_no_release_outputs(workdir, candidate)
        original_write_json_atomic(root, path, candidate)
        sealed = candidate

    base.write_json_atomic = diagnostic_write_json_atomic
    base._guard_tools = _guard_tools
    try:
        base.build_release(
            workdir,
            opc.resolve(),
            container_contract.resolve(),
            installed_cloud_root.resolve(),
        )
    finally:
        base.write_json_atomic = original_write_json_atomic
        base._guard_tools = original_guard_tools

    if sealed is None:
        raise RuntimeError("diagnostic build completed without a sealed manifest")
    persisted = _read_manifest(workdir)
    _validate_manifest(
        base,
        workdir,
        persisted,
        expected_status=DIAGNOSTIC_BUILT_STATUS,
    )
    return persisted


def package_release(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("diagnostic probe packaging is permanently forbidden")


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
    args = parse_args(argv)
    if args.command == "prepare":
        prepare_release(args.outer_zip, args.workdir, args.approved_root)
    elif args.command == "build":
        build_release(
            args.workdir,
            args.opc,
            args.container_contract,
            args.installed_cloud_root,
            args.approved_root,
        )
    else:  # pragma: no cover - argparse enforces this boundary.
        raise RuntimeError(f"unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
