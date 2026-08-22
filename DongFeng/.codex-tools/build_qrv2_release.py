#!/usr/bin/env python3
"""Prepare, build and package the audited MX QrV2 release candidate.

This tool intentionally contains no SSH or Docker control logic.  ``prepare``
expands the nested vendor archive into a brand-new work directory and creates
two isolated OPC inputs.  ``build`` must run inside the already-verified
``mapqr-leicheng`` container.  ``package`` creates a new wheel and a new outer
ZIP while recomputing wheel ``RECORD``; it never edits either source archive.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
import os
import re
import socket
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import step372_patch_qr_v2_matmul_position_v5 as candidate_patcher  # noqa: E402
from step372_patch_qr_v2_matmul_position_v5 import (  # noqa: E402
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_SOURCE_SHA256,
    build_candidate,
    sha256_bytes,
    verify_candidate_structure,
    write_new_file,
)


SCHEMA_VERSION = 1
EXPECTED_OUTER_SHA256 = "363fc46e0f3da952ef9c37cdfb67a190f557abc8a879d1438563c2d3eb807da7"
WHEEL_NAME = "mx_driving_cloud-26.0.7+CANN8.3.RC1.A3-cp311-cp311-linux_aarch64.whl"
EXPECTED_WHEEL_SHA256 = "23253f7fa2b9bfb1b6ff3c77df6620f6c559f68be154f6333246d73178eb5da9"
PACKAGE_PREFIX = "mx_driving_cloud/packages/vendors/customize"
SOURCE_REL = f"{PACKAGE_PREFIX}/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.cpp"
WRAPPER_REL = f"{PACKAGE_PREFIX}/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.py"
KERNEL_ROOT_REL = f"{PACKAGE_PREFIX}/op_impl/ai_core/tbe/kernel"
BIN_NAME = "QrV2_matmul_position_fix_v5"
BINARY_INFO_CONFIG_NAME = "binary_info_config.json"
ORIGINAL_BIN_NAME = "QrV2_566c2e1c0e6c8c92152ad84416d77006"
QRV2_SIMPLIFIED_KEYS = (
    "QrV2/d=0,p=0/0,2/0,2/0,2",
    "QrV2/d=1,p=0/0,2/0,2/0,2",
)
EXPECTED_CONTAINER = "mapqr-leicheng"
MAX_ARCHIVE_MEMBERS = 100_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 4 * 1024**3
SOCS = {
    "ascend910_93": "Ascend910_9362",
    "ascend910b": "Ascend910_9362",
}
CANONICAL_SOC_KEY = "ascend910_93"
ALIAS_SOC_KEY = "ascend910b"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CONTAINER_CONTRACT_SCHEMA = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"tree hash rejects symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _regular_file_inventory(path: Path, *, label: str) -> dict[str, Any]:
    argument = path.absolute()
    if argument.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    resolved = argument.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"{label} must be a regular file: {path}")
    file_stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size": file_stat.st_size,
        "mode": oct(stat.S_IMODE(file_stat.st_mode)),
    }


def _version_text_summary(path: Path) -> list[str]:
    if path.stat().st_size > 1024 * 1024:
        raise RuntimeError(f"CANN version file is unexpectedly large: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    nonempty = [line.strip()[:200] for line in lines if line.strip()]
    return nonempty[:16]


def _validate_container_contract(
    contract_path: Path, opc: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_inventory = _regular_file_inventory(contract_path, label="container contract")
    if contract_inventory["size"] > 1024 * 1024:
        raise RuntimeError("container contract is unexpectedly large")
    contract = json.loads(Path(contract_inventory["path"]).read_text(encoding="utf-8"))
    if contract.get("schema_version") != CONTAINER_CONTRACT_SCHEMA:
        raise RuntimeError("container contract schema mismatch")
    if contract.get("container_name") != EXPECTED_CONTAINER:
        raise RuntimeError("container contract exact-name gate failed")
    container_id = contract.get("inspect_container_id")
    if not isinstance(container_id, str) or not CONTAINER_ID_PATTERN.fullmatch(container_id):
        raise RuntimeError("container contract inspect_container_id must be 64 lowercase hex chars")
    inspect_hostname = contract.get("inspect_hostname")
    if not isinstance(inspect_hostname, str) or not inspect_hostname or len(inspect_hostname) > 255:
        raise RuntimeError("container contract inspect_hostname is invalid")
    actual_hostname = socket.gethostname()
    if actual_hostname != inspect_hostname:
        raise RuntimeError(
            f"container hostname mismatch: inspect={inspect_hostname!r}, actual={actual_hostname!r}"
        )

    opc_inventory = _regular_file_inventory(opc, label="OPC executable")
    expected_opc = contract.get("opc")
    if not isinstance(expected_opc, dict):
        raise RuntimeError("container contract OPC inventory is missing")
    if not isinstance(expected_opc.get("sha256"), str) or not SHA256_PATTERN.fullmatch(
        expected_opc["sha256"]
    ):
        raise RuntimeError("container contract OPC SHA-256 format is invalid")
    if expected_opc.get("path") != opc_inventory["path"]:
        raise RuntimeError("container contract OPC path mismatch")
    if expected_opc.get("sha256") != opc_inventory["sha256"]:
        raise RuntimeError("container contract OPC SHA-256 mismatch")

    expected_versions = contract.get("cann_version_files")
    if not isinstance(expected_versions, list) or not expected_versions:
        raise RuntimeError("container contract requires at least one CANN version file")
    version_inventory: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, expected in enumerate(expected_versions):
        if not isinstance(expected, dict):
            raise RuntimeError(f"CANN version contract entry {index} is invalid")
        if not isinstance(expected.get("sha256"), str) or not SHA256_PATTERN.fullmatch(
            expected["sha256"]
        ):
            raise RuntimeError(f"CANN version file {index} SHA-256 format is invalid")
        version_path = Path(str(expected.get("path", "")))
        inventory = _regular_file_inventory(version_path, label=f"CANN version file {index}")
        if inventory["path"] in seen_paths:
            raise RuntimeError(f"duplicate CANN version file: {inventory['path']}")
        seen_paths.add(inventory["path"])
        if expected.get("path") != inventory["path"]:
            raise RuntimeError(f"CANN version file {index} path mismatch")
        if expected.get("sha256") != inventory["sha256"]:
            raise RuntimeError(f"CANN version file {index} SHA-256 mismatch")
        inventory["text_summary"] = _version_text_summary(Path(inventory["path"]))
        version_inventory.append(inventory)

    runtime = {
        "contract": contract_inventory,
        "container_name": EXPECTED_CONTAINER,
        "inspect_container_id": container_id,
        "inspect_hostname": inspect_hostname,
        "actual_hostname": actual_hostname,
        "opc": opc_inventory,
        "cann_version_files": version_inventory,
    }
    return contract, runtime


def _validate_relative_regular_file(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"installed cloud inventory rejects symlink: {relative}")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"installed cloud path escapes root: {relative}") from error
    if not resolved.is_file():
        raise RuntimeError(f"installed cloud path is not a regular file: {relative}")
    return resolved


def installed_qrv2_inventory(installed_cloud_root: Path) -> dict[str, Any]:
    root_argument = installed_cloud_root.absolute()
    if root_argument.is_symlink():
        raise RuntimeError("installed cloud root must not be a symlink")
    root = root_argument.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("installed cloud root must be a directory")

    package_prefix = PurePosixPath("packages/vendors/customize")
    kernel_root = package_prefix / "op_impl/ai_core/tbe/kernel"
    relatives: list[PurePosixPath] = [
        package_prefix / "op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.cpp",
        package_prefix / "op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.py",
    ]
    for soc_key in SOCS:
        config_relative = kernel_root / "config" / soc_key / "qr_v2.json"
        config_path = _validate_relative_regular_file(root, config_relative)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        bin_list = config.get("binList")
        if not isinstance(bin_list, list) or len(bin_list) != 1:
            raise RuntimeError(f"installed {soc_key} QrV2 config binList contract failed")
        bin_info = bin_list[0].get("binInfo")
        if not isinstance(bin_info, dict) or set(bin_info) != {"jsonFilePath"}:
            raise RuntimeError(f"installed {soc_key} QrV2 config binInfo contract failed")
        json_relative = PurePosixPath(str(bin_info["jsonFilePath"]))
        _safe_member_path(json_relative.as_posix())
        if json_relative.suffix != ".json":
            raise RuntimeError(f"installed {soc_key} QrV2 metadata path must end in .json")
        if not json_relative.parts or json_relative.parts[0] != soc_key:
            raise RuntimeError(f"installed {soc_key} QrV2 config points to another SoC")
        relatives.extend(
            [
                config_relative,
                kernel_root / json_relative,
                kernel_root / PurePosixPath(json_relative.as_posix().removesuffix(".json") + ".o"),
            ]
        )

    files: dict[str, Any] = {}
    for relative in relatives:
        relative_text = relative.as_posix()
        if relative_text in files:
            raise RuntimeError(f"duplicate installed inventory path: {relative_text}")
        path = _validate_relative_regular_file(root, relative)
        file_stat = path.stat()
        if file_stat.st_size <= 0:
            raise RuntimeError(f"installed cloud inventory rejects empty file: {relative_text}")
        files[relative_text] = {
            "sha256": sha256_file(path),
            "size": file_stat.st_size,
            "mode": oct(stat.S_IMODE(file_stat.st_mode)),
            "regular": True,
            "symlink": False,
        }
    return {"root": str(root), "files": files}


def _assert_inventory_unchanged(
    before: dict[str, Any], after: dict[str, Any], *, label: str
) -> None:
    if before != after:
        raise RuntimeError(f"{label} inventory changed during OPC build")


def _assert_lexically_inside(root: Path, target: Path) -> None:
    root_absolute = Path(os.path.abspath(root))
    target_absolute = Path(os.path.abspath(target))
    try:
        target_absolute.relative_to(root_absolute)
    except ValueError as error:
        raise RuntimeError(f"output path escapes isolated root: {target}") from error


def _assert_output_parent(root: Path, target: Path) -> None:
    """Require both lexical and resolved containment for an output parent."""
    _assert_lexically_inside(root, target)
    resolved_root = root.resolve(strict=True)
    resolved_parent = target.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeError(f"output parent realpath escapes isolated root: {target}") from error


def _assert_no_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise RuntimeError(f"isolated root must not be a symlink: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden inside isolated root: {path}")


def write_new_inside(root: Path, path: Path, payload: bytes) -> None:
    _assert_lexically_inside(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_output_parent(root, path)
    write_new_file(path, payload)


def _safe_member_path(name: str) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name:
        raise RuntimeError(f"unsafe archive member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"archive member escapes extraction root: {name!r}")
    return path


def validate_archive(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise RuntimeError(f"archive has too many members: {len(infos)}")
    if sum(info.file_size for info in infos) > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise RuntimeError("archive exceeds the uncompressed-size limit")
    seen: set[str] = set()
    for info in infos:
        member = _safe_member_path(info.filename)
        normalized = member.as_posix().rstrip("/")
        if normalized in seen:
            raise RuntimeError(f"duplicate archive member: {info.filename}")
        seen.add(normalized)
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if file_type == stat.S_IFLNK:
            raise RuntimeError(f"archive symlink is forbidden: {info.filename}")
        if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise RuntimeError(f"non-regular archive member is forbidden: {info.filename}")
    return infos


def safe_extract(archive_path: Path, destination: Path, isolated_root: Path) -> None:
    _assert_lexically_inside(isolated_root, destination)
    destination.mkdir(parents=True, exist_ok=False)
    _assert_output_parent(isolated_root, destination / ".containment-check")
    with zipfile.ZipFile(archive_path) as archive:
        infos = validate_archive(archive)
        for info in infos:
            member = _safe_member_path(info.filename)
            target = destination.joinpath(*member.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                _assert_output_parent(isolated_root, target / ".containment-check")
                continue
            write_new_inside(isolated_root, target, archive.read(info))


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json_new(root: Path, path: Path, value: Any) -> None:
    write_new_inside(root, path, json_bytes(value))


def write_json_atomic(root: Path, path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".new")
    write_new_inside(root, temporary, json_bytes(value))
    _assert_output_parent(root, path)
    os.replace(temporary, path)


def input_descriptor() -> dict[str, Any]:
    tensor = {
        "shape": [-2],
        "format": "ND",
        "dtype": "float32",
    }
    return {
        "op_type": "QrV2",
        "op_list": [
            {
                "bin_filename": BIN_NAME,
                "inputs": [dict(tensor)],
                "outputs": [dict(tensor), dict(tensor)],
                "attrs": [],
            }
        ],
    }


def opc_command(workdir: Path, soc_key: str, opc: str) -> list[str]:
    build_dir = workdir / "build" / soc_key
    return [
        opc,
        str(build_dir / "qr_v2.py"),
        f"--input_param={build_dir / 'input_param.json'}",
        "--main_func=qr_v2",
        f"--bin_filename={BIN_NAME}",
        f"--output={build_dir / 'output'}",
        f"--debug_dir={build_dir / 'debug'}",
        f"--soc_version={SOCS[soc_key]}",
        "--op_mode=dynamic",
        "--simplified_key_mode=0",
        "--optional_input_mode=gen_placeholder",
        "--optional_output_mode=gen_placeholder",
        "--deterministic=false",
        "--log=info",
    ]


def _read_manifest(workdir: Path) -> dict[str, Any]:
    manifest_path = workdir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("release manifest schema mismatch")
    return manifest


def _guard_originals(manifest: dict[str, Any]) -> None:
    guards = manifest["immutable_guards"]
    for name, item in guards.items():
        path = Path(item["path"])
        actual = sha256_tree(path) if item.get("kind") == "tree" else sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"immutable guard failed for {name}: {actual}")


def _release_tool_hashes() -> dict[str, str]:
    patcher_path = Path(candidate_patcher.__file__).resolve()
    dependency_path = Path(candidate_patcher.release_v4.__file__).resolve()
    return {
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "patcher_sha256": sha256_file(patcher_path),
        "patcher_dependency_sha256": sha256_file(dependency_path),
    }


def _guard_tools(manifest: dict[str, Any]) -> None:
    actual = _release_tool_hashes()
    if actual != manifest["tools"]:
        raise RuntimeError(f"release tool hash guard failed: {actual}")


def _guard_build_inputs(workdir: Path, manifest: dict[str, Any]) -> None:
    for soc_key, expected in manifest["build_inputs"].items():
        build_dir = workdir / "build" / soc_key
        actual = {
            "source_sha256": sha256_file(build_dir / "qr_v2.cpp"),
            "wrapper_sha256": sha256_file(build_dir / "qr_v2.py"),
            "input_param_sha256": sha256_file(build_dir / "input_param.json"),
        }
        for key, digest in actual.items():
            if expected.get(key) != digest:
                raise RuntimeError(f"{soc_key} build-input guard failed for {key}: {digest}")


def prepare_release(outer_zip: Path, workdir: Path) -> dict[str, Any]:
    outer_argument = outer_zip.absolute()
    if outer_argument.is_symlink():
        raise ValueError("outer ZIP must not be a symlink")
    outer_zip = outer_argument.resolve(strict=True)
    if not outer_zip.is_file():
        raise ValueError("outer ZIP must be a non-symlink regular file")
    outer_sha = sha256_file(outer_zip)
    if outer_sha != EXPECTED_OUTER_SHA256:
        raise RuntimeError(f"outer ZIP SHA-256 mismatch: {outer_sha}")
    workdir = workdir.absolute()
    if workdir.exists():
        raise FileExistsError(f"work directory already exists: {workdir}")
    workdir.mkdir(parents=True)

    outer_root = workdir / "outer_original"
    safe_extract(outer_zip, outer_root, workdir)
    wheel_members = [path.name for path in outer_root.iterdir() if path.suffix == ".whl"]
    if wheel_members != [WHEEL_NAME]:
        raise RuntimeError(f"outer ZIP wheel-member contract failed: {wheel_members}")
    wheel_path = outer_root / WHEEL_NAME
    if not wheel_path.is_file() or wheel_path.is_symlink():
        raise RuntimeError(f"audited wheel missing from outer ZIP: {WHEEL_NAME}")
    wheel_sha = sha256_file(wheel_path)
    if wheel_sha != EXPECTED_WHEEL_SHA256:
        raise RuntimeError(f"inner wheel SHA-256 mismatch: {wheel_sha}")

    wheel_root = workdir / "wheel_original"
    safe_extract(wheel_path, wheel_root, workdir)
    wheel_tree_sha = sha256_tree(wheel_root)
    source_path = wheel_root / SOURCE_REL
    wrapper_path = wheel_root / WRAPPER_REL
    source = source_path.read_bytes()
    if sha256_bytes(source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("nested QrV2 source SHA-256 mismatch")
    candidate = build_candidate(source)
    structure = verify_candidate_structure(source, candidate)

    commands: dict[str, list[str]] = {}
    build_inputs: dict[str, Any] = {}
    for soc_key in SOCS:
        build_dir = workdir / "build" / soc_key
        (build_dir / "output").mkdir(parents=True)
        (build_dir / "debug").mkdir()
        write_new_inside(workdir, build_dir / "qr_v2.py", wrapper_path.read_bytes())
        write_new_inside(workdir, build_dir / "qr_v2.cpp", candidate)
        write_json_new(workdir, build_dir / "input_param.json", input_descriptor())
        commands[soc_key] = (
            opc_command(workdir, soc_key, "${OPC}")
            if soc_key == CANONICAL_SOC_KEY
            else ["alias-copy", CANONICAL_SOC_KEY, soc_key]
        )
        build_inputs[soc_key] = {
            "soc_version": SOCS[soc_key],
            "artifact_mode": (
                "canonical_opc" if soc_key == CANONICAL_SOC_KEY else "dav2201_alias_copy"
            ),
            "source_sha256": sha256_file(build_dir / "qr_v2.cpp"),
            "wrapper_sha256": sha256_file(build_dir / "qr_v2.py"),
            "input_param_sha256": sha256_file(build_dir / "input_param.json"),
            "status": "prepared",
        }

    config_hashes = {
        soc_key: sha256_file(
            wheel_root / KERNEL_ROOT_REL / "config" / soc_key / "qr_v2.json"
        )
        for soc_key in SOCS
    }
    binary_info_config_hashes = {
        soc_key: sha256_file(
            wheel_root
            / KERNEL_ROOT_REL
            / "config"
            / soc_key
            / BINARY_INFO_CONFIG_NAME
        )
        for soc_key in SOCS
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "policy": {
            "original_archives_read_only": True,
            "installed_package_inventory": "required_at_build",
            "required_container": EXPECTED_CONTAINER,
            "soc_artifacts_built_independently": False,
            "soc_alias_contract": {
                "npu_arch": "DAV_2201",
                "canonical": CANONICAL_SOC_KEY,
                "alias": ALIAS_SOC_KEY,
                "byte_identical_required": True,
            },
        },
        "paths": {
            "workdir": str(workdir),
            "outer_zip": str(outer_zip),
            "extracted_wheel": str(wheel_path),
            "wheel_root": str(wheel_root),
        },
        "immutable_guards": {
            "outer_zip": {"path": str(outer_zip), "sha256": outer_sha},
            "extracted_original_wheel": {"path": str(wheel_path), "sha256": wheel_sha},
            "extracted_original_source": {
                "path": str(source_path),
                "sha256": EXPECTED_SOURCE_SHA256,
            },
            "extracted_original_wheel_tree": {
                "path": str(wheel_root),
                "sha256": wheel_tree_sha,
                "kind": "tree",
            },
        },
        "original": {
            "outer_sha256": outer_sha,
            "wheel_name": WHEEL_NAME,
            "wheel_sha256": wheel_sha,
            "source_relative_path": SOURCE_REL,
            "source_sha256": EXPECTED_SOURCE_SHA256,
            "wrapper_relative_path": WRAPPER_REL,
            "wrapper_sha256": sha256_file(wrapper_path),
            "config_sha256": config_hashes,
            "binary_info_config_sha256": binary_info_config_hashes,
        },
        "tools": _release_tool_hashes(),
        "candidate": {
            "identity": candidate_patcher.CANDIDATE_IDENTITY,
            "bin_name": BIN_NAME,
            "source_sha256": EXPECTED_CANDIDATE_SHA256,
            "v4_candidate_sha256": candidate_patcher.EXPECTED_V4_CANDIDATE_SHA256,
            "reverse_v4_sha256": structure["reverse_v4_sha256"],
            "structure_assertions": structure,
        },
        "build_inputs": build_inputs,
        "opc_commands": commands,
        "artifacts": {soc_key: {"status": "pending"} for soc_key in SOCS},
        "package": {"status": "pending"},
    }
    write_json_new(workdir, workdir / "release_manifest.json", manifest)
    _assert_no_symlinks(workdir)
    _guard_originals(manifest)
    _guard_tools(manifest)
    _guard_build_inputs(workdir, manifest)
    return manifest


def _validate_artifacts(build_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    objects = [path for path in (build_dir / "output").rglob("*.o") if path.stat().st_size]
    metadata_files = [
        path for path in (build_dir / "output").rglob("*.json") if path.stat().st_size
    ]
    if len(objects) != 1 or len(metadata_files) != 1:
        raise RuntimeError(
            f"OPC artifact count mismatch: objects={len(objects)}, json={len(metadata_files)}"
        )
    object_path, metadata_path = objects[0], metadata_files[0]
    if object_path.name != f"{BIN_NAME}.o" or metadata_path.name != f"{BIN_NAME}.json":
        raise RuntimeError("OPC artifact filename contract failed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("binFileName") != BIN_NAME or metadata.get("kernelName") != BIN_NAME:
        raise RuntimeError("OPC artifact metadata name contract failed")
    kernel_list = metadata.get("kernelList")
    if kernel_list is not None:
        if not isinstance(kernel_list, list) or len(kernel_list) != 1:
            raise RuntimeError("OPC kernelList schema contract failed")
        kernel_entry = kernel_list[0]
        if (
            not isinstance(kernel_entry, dict)
            or kernel_entry.get("kernelName") != f"{BIN_NAME}_0"
        ):
            raise RuntimeError("OPC kernelList name contract failed")

    support_info = metadata.get("supportInfo")
    if not isinstance(support_info, dict):
        raise RuntimeError("OPC supportInfo schema contract failed")
    if support_info.get("opMode") != "dynamic":
        raise RuntimeError("OPC supportInfo opMode must be dynamic")
    simplified_key_mode = support_info.get("simplifiedKeyMode")
    if type(simplified_key_mode) is not int or simplified_key_mode != 0:
        raise RuntimeError("OPC supportInfo simplifiedKeyMode contract failed")
    if support_info.get("simplifiedKey") != list(QRV2_SIMPLIFIED_KEYS):
        raise RuntimeError("OPC supportInfo simplifiedKey contract failed")
    input_metadata = support_info.get("inputs")
    output_metadata = support_info.get("outputs")
    if not isinstance(input_metadata, list) or len(input_metadata) != 1:
        raise RuntimeError("OPC supportInfo input-count contract failed")
    if not isinstance(output_metadata, list) or len(output_metadata) != 2:
        raise RuntimeError("OPC supportInfo output-count contract failed")
    for kind, tensors in (("input", input_metadata), ("output", output_metadata)):
        for tensor in tensors:
            if not isinstance(tensor, dict):
                raise RuntimeError(f"OPC supportInfo {kind} tensor schema contract failed")
            if not _strict_json_equal(tensor.get("shape"), [-2]):
                raise RuntimeError(f"OPC supportInfo {kind} shape must be unknown-rank")
            if "ori_shape" in tensor and not _strict_json_equal(
                tensor["ori_shape"], [-2]
            ):
                raise RuntimeError(
                    f"OPC supportInfo {kind} ori_shape must be unknown-rank when present"
                )

    object_bytes = object_path.read_bytes()
    concrete_entries = {
        match.decode("ascii")
        for match in re.findall(
            rb"(?:^|\x00)(QrV2[A-Za-z0-9_]*_mix_ai[cv])(?=\x00|$)",
            object_bytes,
        )
    }
    expected_entries = {f"{BIN_NAME}_0_mix_aic", f"{BIN_NAME}_0_mix_aiv"}
    if concrete_entries != expected_entries:
        raise RuntimeError(
            "OPC object concrete-entry contract failed: "
            f"expected={sorted(expected_entries)}, actual={sorted(concrete_entries)}"
        )
    metadata["_audited_concrete_entries"] = sorted(concrete_entries)
    return object_path, metadata_path, metadata


def build_release(
    workdir: Path,
    opc: Path,
    container_contract: Path,
    installed_cloud_root: Path,
) -> dict[str, Any]:
    workdir = workdir.resolve(strict=True)
    manifest = _read_manifest(workdir)
    if manifest["status"] != "prepared":
        raise RuntimeError(f"build requires prepared status, got {manifest['status']}")
    _guard_originals(manifest)
    _assert_no_symlinks(workdir)
    _guard_tools(manifest)
    _guard_build_inputs(workdir, manifest)
    opc = opc.resolve(strict=True)
    if not opc.is_file() or not os.access(opc, os.X_OK):
        raise ValueError("OPC must be an executable regular file")
    _, runtime_before = _validate_container_contract(container_contract, opc)
    installed_before = installed_qrv2_inventory(installed_cloud_root)
    ascend_opp = os.environ.get("ASCEND_OPP_PATH")
    if not ascend_opp:
        raise RuntimeError("ASCEND_OPP_PATH is required")
    opp_tbe = (Path(ascend_opp) / "built-in/op_impl/ai_core/tbe").resolve(strict=True)
    if not (opp_tbe / "impl/util/platform_adapter.py").is_file():
        raise RuntimeError("ASCEND_OPP_PATH built-in TBE Python tree is incomplete")

    environment = os.environ.copy()
    environment["ASCEND_CUSTOM_OPP_PATH"] = str(
        workdir / "wheel_original" / PACKAGE_PREFIX
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(opp_tbe)

    for soc_key in SOCS:
        build_dir = workdir / "build" / soc_key
        source = build_dir.joinpath("qr_v2.cpp").read_bytes()
        original_source = Path(manifest["immutable_guards"]["extracted_original_source"]["path"]).read_bytes()
        verify_candidate_structure(original_source, source)
        command = opc_command(workdir, soc_key, str(opc))
        log_path = build_dir / "opc.log"
        _assert_output_parent(workdir, log_path)
        try:
            if soc_key == CANONICAL_SOC_KEY:
                with log_path.open("xb") as log:
                    completed = subprocess.run(
                        command,
                        cwd=build_dir,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                returncode = completed.returncode
            else:
                canonical_object, canonical_json, _ = _validate_artifacts(
                    workdir / "build" / CANONICAL_SOC_KEY
                )
                alias_object = build_dir / "output" / canonical_object.name
                alias_json = build_dir / "output" / canonical_json.name
                write_new_inside(workdir, alias_object, canonical_object.read_bytes())
                write_new_inside(workdir, alias_json, canonical_json.read_bytes())
                alias_log = {
                    "mode": "dav2201_alias_copy",
                    "npu_arch": "DAV_2201",
                    "source_soc_key": CANONICAL_SOC_KEY,
                    "target_soc_key": soc_key,
                    "source_object_sha256": sha256_file(canonical_object),
                    "source_json_sha256": sha256_file(canonical_json),
                }
                write_new_inside(workdir, log_path, json_bytes(alias_log))
                returncode = 0
        finally:
            installed_after_call = installed_qrv2_inventory(installed_cloud_root)
            _assert_inventory_unchanged(
                installed_before,
                installed_after_call,
                label=f"installed QrV2 after {soc_key}",
            )
            _, runtime_after_call = _validate_container_contract(container_contract, opc)
            _assert_inventory_unchanged(
                runtime_before,
                runtime_after_call,
                label=f"container/CANN/OPC after {soc_key}",
            )
        if returncode != 0:
            raise RuntimeError(f"OPC failed for {soc_key}: rc={returncode}; log={log_path}")
        _assert_no_symlinks(workdir)
        object_path, metadata_path, metadata = _validate_artifacts(build_dir)
        manifest["artifacts"][soc_key] = {
            "status": "built_structure_valid",
            "soc_version": SOCS[soc_key],
            "artifact_mode": manifest["build_inputs"][soc_key]["artifact_mode"],
            "artifact_source_soc_key": CANONICAL_SOC_KEY,
            "object_path": str(object_path),
            "object_size": object_path.stat().st_size,
            "object_sha256": sha256_file(object_path),
            "json_path": str(metadata_path),
            "json_sha256": sha256_file(metadata_path),
            "kernel_name": metadata["kernelName"],
            "bin_file_name": metadata["binFileName"],
            "concrete_entries": metadata["_audited_concrete_entries"],
            "opc_log_sha256": sha256_file(log_path),
        }
        _guard_originals(manifest)
        _guard_build_inputs(workdir, manifest)
    canonical_artifact = manifest["artifacts"][CANONICAL_SOC_KEY]
    alias_artifact = manifest["artifacts"][ALIAS_SOC_KEY]
    for key in ("object_sha256", "json_sha256"):
        if canonical_artifact[key] != alias_artifact[key]:
            raise RuntimeError(f"DAV_2201 SoC alias artifact differs for {key}")
    manifest["status"] = "built"
    manifest["policy"]["installed_package_modified"] = False
    _, runtime_after_final = _validate_container_contract(container_contract, opc)
    _assert_inventory_unchanged(
        runtime_before,
        runtime_after_final,
        label="container/CANN/OPC final",
    )
    manifest["build_runtime"] = {
        **runtime_before,
        "runtime_inventory_after": runtime_after_final,
        "runtime_inventory_closed": True,
        "controlled_pythonpath": str(opp_tbe),
        "ascend_custom_opp_path": environment["ASCEND_CUSTOM_OPP_PATH"],
        "installed_qrv2_before": installed_before,
        "installed_qrv2_after": installed_qrv2_inventory(installed_cloud_root),
        "installed_inventory_closed": True,
    }
    _assert_inventory_unchanged(
        installed_before,
        manifest["build_runtime"]["installed_qrv2_after"],
        label="installed QrV2 final",
    )
    write_json_atomic(workdir, workdir / "release_manifest.json", manifest)
    _guard_originals(manifest)
    _guard_tools(manifest)
    _guard_build_inputs(workdir, manifest)
    return manifest


def _wheel_hash(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


def _record_bytes(files: dict[str, bytes], record_path: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for path in sorted(files):
        if path == record_path:
            continue
        payload = files[path]
        writer.writerow((path, _wheel_hash(payload), str(len(payload))))
    writer.writerow((record_path, "", ""))
    return output.getvalue().encode("utf-8")


def _new_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 8, 21, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _write_zip_from_mapping(
    isolated_root: Path,
    output: Path,
    files: dict[str, bytes],
    originals: dict[str, zipfile.ZipInfo],
    directories: list[zipfile.ZipInfo],
    comment: bytes,
) -> None:
    _assert_output_parent(isolated_root, output)
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(output, "x", allowZip64=True) as archive:
        archive.comment = comment
        for directory in directories:
            archive.writestr(copy.copy(directory), b"")
        for name in sorted(files):
            info = copy.copy(originals[name]) if name in originals else _new_zip_info(name)
            archive.writestr(info, files[name])


def _read_regular_zip_files(
    path: Path,
) -> tuple[dict[str, bytes], dict[str, zipfile.ZipInfo], list[zipfile.ZipInfo], bytes]:
    with zipfile.ZipFile(path) as archive:
        infos = validate_archive(archive)
        files = {info.filename: archive.read(info) for info in infos if not info.is_dir()}
        metadata = {info.filename: copy.copy(info) for info in infos if not info.is_dir()}
        directories = [copy.copy(info) for info in infos if info.is_dir()]
        return files, metadata, directories, archive.comment


def _updated_config(original: bytes, soc_key: str) -> bytes:
    config = json.loads(original.decode("utf-8"))
    bin_list = config.get("binList")
    if not isinstance(bin_list, list) or len(bin_list) != 1:
        raise RuntimeError(f"{soc_key} QrV2 config binList contract failed")
    bin_info = bin_list[0].get("binInfo")
    if not isinstance(bin_info, dict) or set(bin_info) != {"jsonFilePath"}:
        raise RuntimeError(f"{soc_key} QrV2 config binInfo contract failed")
    bin_info["jsonFilePath"] = f"{soc_key}/qr_v2/{BIN_NAME}.json"
    return json_bytes(config)


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    """Compare decoded JSON without treating bool and int as interchangeable."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def _qrv2_binary_info_entry(soc_key: str, bin_name: str) -> dict[str, Any]:
    if soc_key not in SOCS:
        raise RuntimeError(f"unsupported binary-info SoC key: {soc_key}")
    bin_path = f"{soc_key}/qr_v2/{bin_name}.o"
    return {
        "dynamicRankSupport": True,
        "simplifiedKeyMode": 0,
        "binaryList": [
            {
                "coreType": 0,
                "simplifiedKey": QRV2_SIMPLIFIED_KEYS[0],
                "binPath": bin_path,
            },
            {
                "coreType": 0,
                "simplifiedKey": QRV2_SIMPLIFIED_KEYS[1],
                "binPath": bin_path,
            },
        ],
    }


def _json_string_count(value: Any, needle: str) -> int:
    if isinstance(value, dict):
        return sum(_json_string_count(item, needle) for item in value.values())
    if isinstance(value, list):
        return sum(_json_string_count(item, needle) for item in value)
    return int(isinstance(value, str) and value == needle)


def _validate_binary_info_config_delta(
    original: bytes, candidate: bytes, soc_key: str
) -> dict[str, Any]:
    try:
        original_object = json.loads(original.decode("utf-8"))
        candidate_object = json.loads(candidate.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{soc_key} binary-info JSON is invalid") from error
    if not isinstance(original_object, dict) or not isinstance(candidate_object, dict):
        raise RuntimeError(f"{soc_key} binary-info top-level object contract failed")

    expected_original_entry = _qrv2_binary_info_entry(soc_key, ORIGINAL_BIN_NAME)
    expected_candidate_entry = _qrv2_binary_info_entry(soc_key, BIN_NAME)
    if not _strict_json_equal(original_object.get("QrV2"), expected_original_entry):
        raise RuntimeError(f"{soc_key} original QrV2 binary-info entry contract failed")
    if not _strict_json_equal(candidate_object.get("QrV2"), expected_candidate_entry):
        raise RuntimeError(f"{soc_key} candidate QrV2 binary-info entry contract failed")

    expected_candidate_object = copy.deepcopy(original_object)
    expected_candidate_object["QrV2"] = expected_candidate_entry
    if not _strict_json_equal(candidate_object, expected_candidate_object):
        raise RuntimeError(f"{soc_key} binary-info changed outside the QrV2 entry")

    old_path = f"{soc_key}/qr_v2/{ORIGINAL_BIN_NAME}.o"
    new_path = f"{soc_key}/qr_v2/{BIN_NAME}.o"
    if _json_string_count(original_object, old_path) != 2:
        raise RuntimeError(f"{soc_key} original QrV2 binPath multiplicity contract failed")
    if _json_string_count(candidate_object, old_path) != 0:
        raise RuntimeError(f"{soc_key} old QrV2 binPath remains in candidate binary-info")
    if _json_string_count(candidate_object, new_path) != 2:
        raise RuntimeError(f"{soc_key} candidate QrV2 binPath multiplicity contract failed")
    return candidate_object


def _updated_binary_info_config(original: bytes, soc_key: str) -> bytes:
    try:
        candidate_object = json.loads(original.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{soc_key} binary-info JSON is invalid") from error
    expected_original_entry = _qrv2_binary_info_entry(soc_key, ORIGINAL_BIN_NAME)
    if not isinstance(candidate_object, dict) or not _strict_json_equal(
        candidate_object.get("QrV2"), expected_original_entry
    ):
        raise RuntimeError(f"{soc_key} original QrV2 binary-info entry contract failed")
    candidate_object["QrV2"] = _qrv2_binary_info_entry(soc_key, BIN_NAME)
    candidate = json_bytes(candidate_object)
    _validate_binary_info_config_delta(original, candidate, soc_key)
    return candidate


def verify_wheel_record(wheel_path: Path) -> None:
    files, _, _, _ = _read_regular_zip_files(wheel_path)
    record_paths = [name for name in files if name.endswith(".dist-info/RECORD")]
    if len(record_paths) != 1:
        raise RuntimeError(f"wheel RECORD count mismatch: {record_paths}")
    record_path = record_paths[0]
    expected = _record_bytes(files, record_path)
    if files[record_path] != expected:
        raise RuntimeError("wheel RECORD verification failed")


def _verify_wheel_delta(
    original: dict[str, bytes],
    candidate: dict[str, bytes],
    *,
    expected_added: set[str],
    expected_removed: set[str],
    allowed_modified: set[str],
) -> None:
    added = set(candidate) - set(original)
    removed = set(original) - set(candidate)
    modified = {
        name for name in set(original) & set(candidate) if original[name] != candidate[name]
    }
    if added != expected_added:
        raise RuntimeError(f"unexpected wheel members added: {sorted(added ^ expected_added)}")
    if removed != expected_removed:
        raise RuntimeError(f"unexpected wheel members removed: {sorted(removed ^ expected_removed)}")
    if modified != allowed_modified:
        raise RuntimeError(f"unexpected wheel members modified: {sorted(modified ^ allowed_modified)}")


def package_release(workdir: Path) -> dict[str, Any]:
    workdir = workdir.resolve(strict=True)
    manifest = _read_manifest(workdir)
    if manifest["status"] != "built":
        raise RuntimeError(f"package requires built status, got {manifest['status']}")
    runtime = manifest.get("build_runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("installed_inventory_closed") is not True
        or runtime.get("runtime_inventory_closed") is not True
    ):
        raise RuntimeError("package requires closed installed/runtime build evidence")
    runtime_before = {
        key: runtime[key]
        for key in (
            "contract",
            "container_name",
            "inspect_container_id",
            "inspect_hostname",
            "actual_hostname",
            "opc",
            "cann_version_files",
        )
    }
    _assert_inventory_unchanged(
        runtime_before,
        runtime["runtime_inventory_after"],
        label="manifest container/CANN/OPC",
    )
    _assert_inventory_unchanged(
        runtime["installed_qrv2_before"],
        runtime["installed_qrv2_after"],
        label="manifest installed QrV2",
    )
    _guard_originals(manifest)
    _assert_no_symlinks(workdir)
    _guard_tools(manifest)
    _guard_build_inputs(workdir, manifest)
    expected_alias_policy = {
        "npu_arch": "DAV_2201",
        "canonical": CANONICAL_SOC_KEY,
        "alias": ALIAS_SOC_KEY,
        "byte_identical_required": True,
    }
    if manifest.get("policy", {}).get("soc_alias_contract") != expected_alias_policy:
        raise RuntimeError("package SoC alias policy contract mismatch")
    canonical_manifest = manifest.get("artifacts", {}).get(CANONICAL_SOC_KEY, {})
    alias_manifest = manifest.get("artifacts", {}).get(ALIAS_SOC_KEY, {})
    if (
        canonical_manifest.get("artifact_mode") != "canonical_opc"
        or canonical_manifest.get("artifact_source_soc_key") != CANONICAL_SOC_KEY
        or alias_manifest.get("artifact_mode") != "dav2201_alias_copy"
        or alias_manifest.get("artifact_source_soc_key") != CANONICAL_SOC_KEY
    ):
        raise RuntimeError("package SoC alias provenance contract mismatch")
    canonical_object, canonical_json, _ = _validate_artifacts(
        workdir / "build" / CANONICAL_SOC_KEY
    )
    alias_object, alias_json, _ = _validate_artifacts(workdir / "build" / ALIAS_SOC_KEY)
    if canonical_object.read_bytes() != alias_object.read_bytes():
        raise RuntimeError("package SoC alias object is not byte-identical")
    if canonical_json.read_bytes() != alias_json.read_bytes():
        raise RuntimeError("package SoC alias metadata is not byte-identical")
    original_wheel = Path(manifest["paths"]["extracted_wheel"])
    files, metadata, wheel_directories, wheel_comment = _read_regular_zip_files(original_wheel)
    original_files = dict(files)
    signature_paths = [
        name for name in files if name.endswith((".dist-info/RECORD.jws", ".dist-info/RECORD.p7s"))
    ]
    if signature_paths:
        raise RuntimeError(f"signed wheel cannot be safely repacked: {signature_paths}")
    record_paths = [name for name in files if name.endswith(".dist-info/RECORD")]
    if len(record_paths) != 1:
        raise RuntimeError(f"wheel RECORD count mismatch: {record_paths}")
    record_path = record_paths[0]

    candidate_source = (workdir / "build" / "ascend910_93" / "qr_v2.cpp").read_bytes()
    original_source = Path(manifest["immutable_guards"]["extracted_original_source"]["path"]).read_bytes()
    verify_candidate_structure(original_source, candidate_source)
    other_candidate = (workdir / "build" / "ascend910b" / "qr_v2.cpp").read_bytes()
    if other_candidate != candidate_source:
        raise RuntimeError("SoC build inputs must use the identical audited C++ candidate")
    files[SOURCE_REL] = candidate_source

    expected_added: set[str] = set()
    expected_removed: set[str] = set()
    packaged_files: dict[str, Any] = {
        "source": {"path": SOURCE_REL, "sha256": EXPECTED_CANDIDATE_SHA256}
    }
    for soc_key in SOCS:
        object_path, json_path, _ = _validate_artifacts(workdir / "build" / soc_key)
        artifact_manifest = manifest["artifacts"].get(soc_key, {})
        if artifact_manifest.get("status") != "built_structure_valid":
            raise RuntimeError(f"{soc_key} manifest does not prove an OPC build")
        if artifact_manifest.get("object_sha256") != sha256_file(object_path):
            raise RuntimeError(f"{soc_key} object changed after build")
        if artifact_manifest.get("json_sha256") != sha256_file(json_path):
            raise RuntimeError(f"{soc_key} metadata changed after build")
        config_path = f"{KERNEL_ROOT_REL}/config/{soc_key}/qr_v2.json"
        original_config = files[config_path]
        original_config_object = json.loads(original_config)
        new_config = _updated_config(original_config, soc_key)
        new_config_object = json.loads(new_config)
        original_config_object["binList"][0]["binInfo"]["jsonFilePath"] = new_config_object[
            "binList"
        ][0]["binInfo"]["jsonFilePath"]
        if original_config_object != new_config_object:
            raise RuntimeError(f"{soc_key} config changed outside binInfo.jsonFilePath")
        old_json_rel = f"{KERNEL_ROOT_REL}/" + json.loads(original_config)["binList"][0][
            "binInfo"
        ]["jsonFilePath"]
        old_object_rel = old_json_rel.removesuffix(".json") + ".o"
        if old_json_rel not in files or old_object_rel not in files:
            raise RuntimeError(f"{soc_key} original QrV2 artifact pair is incomplete")

        binary_info_path = (
            f"{KERNEL_ROOT_REL}/config/{soc_key}/{BINARY_INFO_CONFIG_NAME}"
        )
        original_binary_info = files.get(binary_info_path)
        if original_binary_info is None:
            raise RuntimeError(f"{soc_key} binary-info config is missing")
        expected_binary_info_sha = manifest.get("original", {}).get(
            "binary_info_config_sha256", {}
        ).get(soc_key)
        if expected_binary_info_sha != sha256_bytes(original_binary_info):
            raise RuntimeError(f"{soc_key} original binary-info manifest SHA-256 mismatch")
        new_binary_info = _updated_binary_info_config(original_binary_info, soc_key)

        expected_removed.update((old_json_rel, old_object_rel))
        del files[old_json_rel]
        del files[old_object_rel]
        files[config_path] = new_config
        files[binary_info_path] = new_binary_info
        new_json_rel = f"{KERNEL_ROOT_REL}/{soc_key}/qr_v2/{BIN_NAME}.json"
        new_object_rel = f"{KERNEL_ROOT_REL}/{soc_key}/qr_v2/{BIN_NAME}.o"
        expected_added.update((new_json_rel, new_object_rel))
        files[new_json_rel] = json_path.read_bytes()
        files[new_object_rel] = object_path.read_bytes()
        packaged_files[soc_key] = {
            "config": {"path": config_path, "sha256": sha256_bytes(new_config)},
            "binary_info_config": {
                "path": binary_info_path,
                "sha256": sha256_bytes(new_binary_info),
            },
            "json": {"path": new_json_rel, "sha256": sha256_file(json_path)},
            "object": {"path": new_object_rel, "sha256": sha256_file(object_path)},
        }

    files[record_path] = _record_bytes(files, record_path)
    _verify_wheel_delta(
        original_files,
        files,
        expected_added=expected_added,
        expected_removed=expected_removed,
        allowed_modified={
            SOURCE_REL,
            record_path,
            *(f"{KERNEL_ROOT_REL}/config/{soc_key}/qr_v2.json" for soc_key in SOCS),
            *(
                f"{KERNEL_ROOT_REL}/config/{soc_key}/{BINARY_INFO_CONFIG_NAME}"
                for soc_key in SOCS
            ),
        },
    )
    release_dir = workdir / "release"
    release_dir.mkdir(exist_ok=False)
    _assert_output_parent(workdir, release_dir / ".containment-check")
    new_wheel = release_dir / WHEEL_NAME
    _write_zip_from_mapping(
        workdir, new_wheel, files, metadata, wheel_directories, wheel_comment
    )
    verify_wheel_record(new_wheel)
    packed_wheel_files, _, _, _ = _read_regular_zip_files(new_wheel)
    _verify_wheel_delta(
        original_files,
        packed_wheel_files,
        expected_added=expected_added,
        expected_removed=expected_removed,
        allowed_modified={
            SOURCE_REL,
            record_path,
            *(f"{KERNEL_ROOT_REL}/config/{soc_key}/qr_v2.json" for soc_key in SOCS),
            *(
                f"{KERNEL_ROOT_REL}/config/{soc_key}/{BINARY_INFO_CONFIG_NAME}"
                for soc_key in SOCS
            ),
        },
    )
    for soc_key in SOCS:
        binary_info_path = (
            f"{KERNEL_ROOT_REL}/config/{soc_key}/{BINARY_INFO_CONFIG_NAME}"
        )
        packed_binary_info = packed_wheel_files[binary_info_path]
        _validate_binary_info_config_delta(
            original_files[binary_info_path], packed_binary_info, soc_key
        )
        recorded = packaged_files[soc_key]["binary_info_config"]
        if recorded != {
            "path": binary_info_path,
            "sha256": sha256_bytes(packed_binary_info),
        }:
            raise RuntimeError(f"{soc_key} packaged binary-info manifest mismatch")

    outer_zip = Path(manifest["paths"]["outer_zip"])
    outer_files, outer_metadata, outer_directories, outer_comment = _read_regular_zip_files(
        outer_zip
    )
    if WHEEL_NAME not in outer_files:
        raise RuntimeError("outer ZIP no longer contains the audited wheel member")
    outer_files[WHEEL_NAME] = new_wheel.read_bytes()
    new_outer = release_dir / (outer_zip.stem + "-qrv2-matmul-position-fix-v5.zip")
    _write_zip_from_mapping(
        workdir, new_outer, outer_files, outer_metadata, outer_directories, outer_comment
    )
    with zipfile.ZipFile(new_outer) as archive:
        validate_archive(archive)
        if sha256_bytes(archive.read(WHEEL_NAME)) != sha256_file(new_wheel):
            raise RuntimeError("new outer ZIP does not contain the new wheel byte-for-byte")

    _guard_originals(manifest)
    _guard_tools(manifest)
    _guard_build_inputs(workdir, manifest)
    manifest["status"] = "packaged_unvalidated"
    manifest["package"] = {
        "status": "packaged_unvalidated",
        "wheel_path": str(new_wheel),
        "wheel_sha256": sha256_file(new_wheel),
        "outer_zip_path": str(new_outer),
        "outer_zip_sha256": sha256_file(new_outer),
        "record_path": record_path,
        "record_verified": True,
        "packaged_files": packaged_files,
        "device_validation": "pending",
        "loss_validation": "pending",
        "performance_validation": "pending",
    }
    write_json_atomic(workdir, workdir / "release_manifest.json", manifest)
    _assert_no_symlinks(workdir)
    _guard_originals(manifest)
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("outer_zip", type=Path)
    prepare.add_argument("workdir", type=Path)
    build = subparsers.add_parser("build")
    build.add_argument("workdir", type=Path)
    build.add_argument("--opc", type=Path, required=True)
    build.add_argument("--container-contract", type=Path, required=True)
    build.add_argument("--installed-cloud-root", type=Path, required=True)
    package = subparsers.add_parser("package")
    package.add_argument("workdir", type=Path)
    all_steps = subparsers.add_parser("all")
    all_steps.add_argument("outer_zip", type=Path)
    all_steps.add_argument("workdir", type=Path)
    all_steps.add_argument("--opc", type=Path, required=True)
    all_steps.add_argument("--container-contract", type=Path, required=True)
    all_steps.add_argument("--installed-cloud-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "prepare":
        manifest = prepare_release(args.outer_zip, args.workdir)
    elif args.action == "build":
        manifest = build_release(
            args.workdir,
            args.opc,
            args.container_contract,
            args.installed_cloud_root,
        )
    elif args.action == "package":
        manifest = package_release(args.workdir)
    else:
        prepare_release(args.outer_zip, args.workdir)
        build_release(
            args.workdir,
            args.opc,
            args.container_contract,
            args.installed_cloud_root,
        )
        manifest = package_release(args.workdir)
    print(json.dumps({"status": manifest["status"], "workdir": manifest["paths"]["workdir"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
