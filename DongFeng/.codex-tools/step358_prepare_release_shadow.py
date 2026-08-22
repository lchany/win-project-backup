#!/usr/bin/env python3
"""Physically extract and verify the audited QrV2 release wheel in a new shadow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_KERNEL = "QrV2_matmul_position_fix_v5"
EXPECTED_AIC = EXPECTED_KERNEL + "_0_mix_aic"
EXPECTED_AIV = EXPECTED_KERNEL + "_0_mix_aiv"
EXPECTED_CANDIDATE_SOURCE_SHA256 = (
    "e6ccbb84b0e0dbdc026ecdc6b6e07936fbd659401e35c38f7e9eb974d99bc3b7"
)
CANDIDATE_SOURCE_REL = (
    "packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.cpp"
)
QRV2_SIMPLIFIED_KEYS = (
    "QrV2/d=0,p=0/0,2/0,2/0,2",
    "QrV2/d=1,p=0/0,2/0,2/0,2",
)
BINARY_INFO_CONFIG_NAME = "binary_info_config.json"
SOCS = ("ascend910_93", "ascend910b")
MAX_MEMBERS = 20_000
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_path(name: str) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name:
        raise RuntimeError(f"unsafe wheel member: {name!r}")
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts:
        raise RuntimeError(f"wheel member escapes shadow: {name!r}")
    return member


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def _strict_json_equal(actual: Any, expected: Any) -> bool:
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


def _expected_binary_info(soc: str) -> dict[str, Any]:
    bin_path = f"{soc}/qr_v2/{EXPECTED_KERNEL}.o"
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


def _candidate_artifacts(package: Path) -> dict[str, Any]:
    source = package / CANDIDATE_SOURCE_REL
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"shadow candidate source missing or symlinked: {source}")
    if not source.resolve(strict=True).is_relative_to(package.resolve(strict=True)):
        raise RuntimeError("shadow candidate source realpath escapes package")
    source_sha256 = sha256_file(source)
    if source_sha256 != EXPECTED_CANDIDATE_SOURCE_SHA256:
        raise RuntimeError(f"shadow candidate source SHA-256 mismatch: {source_sha256}")

    vendor = package / "packages/vendors/customize"
    kernel_root = vendor / "op_impl/ai_core/tbe/kernel"
    config_root = kernel_root / "config"
    result: dict[str, Any] = {}
    for soc in SOCS:
        config = config_root / soc / "qr_v2.json"
        binary_info = config_root / soc / BINARY_INFO_CONFIG_NAME
        kernel_dir = kernel_root / soc / "qr_v2"
        kernel_json = kernel_dir / f"{EXPECTED_KERNEL}.json"
        kernel_object = kernel_dir / f"{EXPECTED_KERNEL}.o"
        for path in (config, binary_info, kernel_json, kernel_object):
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"shadow candidate artifact missing or symlinked: {path}")
            if not path.resolve(strict=True).is_relative_to(package.resolve(strict=True)):
                raise RuntimeError("shadow artifact realpath escapes package")
        config_payload = json.loads(config.read_text(encoding="utf-8"))
        bin_list = config_payload.get("binList")
        if not isinstance(bin_list, list) or len(bin_list) != 1:
            raise RuntimeError(f"{soc} config must contain exactly one binList row")
        bin_row = bin_list[0]
        if not isinstance(bin_row, dict):
            raise RuntimeError(f"{soc} config binList row schema mismatch")
        bin_info = bin_row.get("binInfo")
        if not isinstance(bin_info, dict) or set(bin_info) != {"jsonFilePath"}:
            raise RuntimeError(f"{soc} config binInfo schema mismatch")
        reference = bin_info["jsonFilePath"]
        expected_reference = f"{soc}/qr_v2/{EXPECTED_KERNEL}.json"
        if reference != expected_reference:
            raise RuntimeError(f"{soc} config does not uniquely select the release kernel")
        kernel_files = sorted(
            path.name for path in kernel_dir.iterdir() if path.is_file() and not path.is_symlink()
        )
        if kernel_files != [f"{EXPECTED_KERNEL}.json", f"{EXPECTED_KERNEL}.o"]:
            raise RuntimeError(f"{soc} qr_v2 directory contains stale or extra artifacts")
        kernel_payload = json.loads(kernel_json.read_text(encoding="utf-8"))
        if not isinstance(kernel_payload, dict) or (
            kernel_payload.get("kernelName") != EXPECTED_KERNEL
            or kernel_payload.get("binFileName") != EXPECTED_KERNEL
        ):
            raise RuntimeError(f"{soc} kernel JSON identity mismatch")
        kernel_list = kernel_payload.get("kernelList")
        if kernel_list is not None:
            if (
                not isinstance(kernel_list, list)
                or len(kernel_list) != 1
                or not isinstance(kernel_list[0], dict)
                or kernel_list[0].get("kernelName") != f"{EXPECTED_KERNEL}_0"
            ):
                raise RuntimeError(f"{soc} kernelList identity mismatch")
        support_info = kernel_payload.get("supportInfo")
        if not isinstance(support_info, dict):
            raise RuntimeError(f"{soc} kernel supportInfo schema mismatch")
        if support_info.get("opMode") != "dynamic":
            raise RuntimeError(f"{soc} kernel opMode must be dynamic")
        if type(support_info.get("simplifiedKeyMode")) is not int or support_info.get(
            "simplifiedKeyMode"
        ) != 0:
            raise RuntimeError(f"{soc} kernel simplifiedKeyMode mismatch")
        if not _strict_json_equal(
            support_info.get("simplifiedKey"), list(QRV2_SIMPLIFIED_KEYS)
        ):
            raise RuntimeError(f"{soc} kernel simplifiedKey mismatch")
        inputs = support_info.get("inputs")
        outputs = support_info.get("outputs")
        if not isinstance(inputs, list) or len(inputs) != 1:
            raise RuntimeError(f"{soc} kernel input-count mismatch")
        if not isinstance(outputs, list) or len(outputs) != 2:
            raise RuntimeError(f"{soc} kernel output-count mismatch")
        for tensor in [*inputs, *outputs]:
            if not isinstance(tensor, dict) or not _strict_json_equal(
                tensor.get("shape"), [-2]
            ):
                raise RuntimeError(f"{soc} kernel shape must be dynamic unknown-rank")
            if "ori_shape" in tensor and not _strict_json_equal(tensor["ori_shape"], [-2]):
                raise RuntimeError(f"{soc} kernel ori_shape must be unknown-rank when present")

        binary_info_payload = json.loads(binary_info.read_text(encoding="utf-8"))
        if not isinstance(binary_info_payload, dict) or not _strict_json_equal(
            binary_info_payload.get("QrV2"), _expected_binary_info(soc)
        ):
            raise RuntimeError(f"{soc} QrV2 binary-info contract mismatch")
        object_bytes = kernel_object.read_bytes()
        concrete_entries = {
            match.decode("ascii")
            for match in re.findall(
                rb"(?:^|\x00)(QrV2[A-Za-z0-9_]*_mix_ai[cv])(?=\x00|$)",
                object_bytes,
            )
        }
        if concrete_entries != {EXPECTED_AIC, EXPECTED_AIV}:
            raise RuntimeError(f"{soc} object concrete identity set is not exact")
        result[soc] = {
            "config_sha256": sha256_file(config),
            "binary_info_config_sha256": sha256_file(binary_info),
            "json_sha256": sha256_file(kernel_json),
            "object_sha256": sha256_file(kernel_object),
            "object_size": kernel_object.stat().st_size,
        }
    if result[SOCS[0]]["json_sha256"] != result[SOCS[1]]["json_sha256"]:
        raise RuntimeError("DAV_2201 alias JSONs differ in packaged wheel")
    if result[SOCS[0]]["object_sha256"] != result[SOCS[1]]["object_sha256"]:
        raise RuntimeError("DAV_2201 alias objects differ in packaged wheel")
    return result


def prepare(wheel: Path, shadow_root: Path, manifest_path: Path, expected_sha256: str) -> dict[str, Any]:
    wheel = wheel.absolute()
    if wheel.is_symlink() or not wheel.is_file():
        raise RuntimeError("release wheel must be a regular non-symlink file")
    actual_sha256 = sha256_file(wheel)
    if actual_sha256 != expected_sha256:
        raise RuntimeError("release wheel SHA mismatch")
    if shadow_root.exists() or shadow_root.is_symlink():
        raise FileExistsError(f"shadow root already exists: {shadow_root}")
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(f"shadow manifest already exists: {manifest_path}")
    shadow_root.mkdir(parents=True, mode=0o700)
    with zipfile.ZipFile(wheel) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS:
            raise RuntimeError("wheel member count exceeds limit")
        if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
            raise RuntimeError("wheel uncompressed size exceeds limit")
        seen: set[str] = set()
        for info in infos:
            member = _member_path(info.filename)
            normalized = member.as_posix().rstrip("/")
            if normalized in seen:
                raise RuntimeError(f"duplicate wheel member: {info.filename}")
            seen.add(normalized)
            file_type = stat.S_IFMT(info.external_attr >> 16)
            if file_type == stat.S_IFLNK or file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise RuntimeError(f"non-regular wheel member: {info.filename}")
            target = shadow_root.joinpath(*member.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                _write_new(target, archive.read(info))
    for path in shadow_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symlink appeared in physical shadow: {path}")
    package = shadow_root / "mx_driving_cloud"
    init_path = package / "__init__.py"
    ops_path = package / "ops/linalg.py"
    extensions = sorted(package.glob("_C*.so"))
    if not init_path.is_file() or not ops_path.is_file() or len(extensions) != 1:
        raise RuntimeError("full shadow lacks __init__, ops/linalg.py, or unique _C extension")
    artifacts = _candidate_artifacts(package)
    manifest = {
        "schema": "step358-release-shadow-v1",
        "wheel_path": str(wheel.resolve(strict=True)),
        "wheel_sha256": actual_sha256,
        "shadow_root": str(shadow_root.resolve(strict=True)),
        "package_root": str(package.resolve(strict=True)),
        "custom_opp": str((package / "packages/vendors/customize").resolve(strict=True)),
        "critical_files": {
            "init": {"path": str(init_path), "sha256": sha256_file(init_path)},
            "extension": {"path": str(extensions[0]), "sha256": sha256_file(extensions[0])},
            "linalg": {"path": str(ops_path), "sha256": sha256_file(ops_path)},
            "candidate_source": {
                "path": str((package / CANDIDATE_SOURCE_REL).resolve(strict=True)),
                "sha256": EXPECTED_CANDIDATE_SOURCE_SHA256,
            },
        },
        "artifacts": artifacts,
        "kernel": EXPECTED_KERNEL,
        "concrete_aic": EXPECTED_AIC,
        "physical_shadow_gate": "PASS",
    }
    _write_new(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--shadow-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-wheel-sha256", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{64}", args.expected_wheel_sha256) is None:
        parser.error("expected wheel SHA must be lowercase 64-hex")
    result = prepare(args.wheel, args.shadow_root, args.manifest, args.expected_wheel_sha256)
    print(json.dumps({"status": "PASS", "wheel_sha256": result["wheel_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
