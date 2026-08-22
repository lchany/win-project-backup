#!/usr/bin/env python3
"""Create a fail-closed QrV2 OPP overlay without modifying the installed package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


CANDIDATE_STEM = "QrV2_step338_lifetime_fix"
CANDIDATE_O_SIZE = 136872
CANDIDATE_JSON_SIZE = 2019
EXPECTED_CONFIG_SUFFIXES = {
    "config/ascend910_93/qr_v2.json",
    "config/ascend910b/qr_v2.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_bininfo_entries(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        bin_info = value.get("binInfo")
        if (
            isinstance(bin_info, dict)
            and "jsonFilePath" in bin_info
            and "simplifiedKey" in value
        ):
            yield value
        for child in value.values():
            yield from iter_bininfo_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_bininfo_entries(child)


def discover_qrv2_configs(installed: Path) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    matches: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for path in installed.rglob("*.json"):
        if "config" not in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for entry in iter_bininfo_entries(payload):
            raw = str(entry["binInfo"]["jsonFilePath"])
            if "qrv2" in raw.lower() or CANDIDATE_STEM.lower() in raw.lower():
                if path.is_symlink():
                    raise RuntimeError(f"QrV2 config must not be a symlink: {path}")
                matches.append((path, payload, entry))
    by_path: dict[Path, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for path, payload, entry in matches:
        by_path.setdefault(path.resolve(), []).append((payload, entry))
    real_paths = sorted(by_path)
    suffix_groups: dict[str, list[Path]] = {}
    for path in real_paths:
        suffix = "/".join(path.relative_to(installed.resolve()).parts[-3:])
        suffix_groups.setdefault(suffix, []).append(path)
    if len(real_paths) != 2 or set(suffix_groups) != EXPECTED_CONFIG_SUFFIXES:
        raise RuntimeError(
            f"expected exactly two QrV2 SOC configs {sorted(EXPECTED_CONFIG_SUFFIXES)}, "
            f"found {len(real_paths)} real paths with suffixes {sorted(suffix_groups)}"
        )
    result = []
    for suffix in sorted(EXPECTED_CONFIG_SUFFIXES):
        paths = suffix_groups[suffix]
        if len(paths) != 1:
            raise RuntimeError(f"duplicate QrV2 config suffix {suffix}: {list(map(str, paths))}")
        path = paths[0]
        rows = by_path[path]
        if len(rows) != 1:
            raise RuntimeError(f"expected one QrV2 binList item in {path}, found {len(rows)}")
        payload, entry = rows[0]
        result.append((path, payload, entry))
    return result


def resolve_installed_kernel_json(installed: Path, config: Path, raw: str) -> Path:
    raw_path = PurePosixPath(raw)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(Path(raw))
    else:
        candidates.extend((
            config.parent / raw,
            installed / raw,
            installed / "op_impl/ai_core/tbe/kernel" / raw,
        ))
    direct_existing = [path for path in candidates if path.is_file()]
    if any(path.is_symlink() for path in direct_existing):
        raise RuntimeError(f"installed kernel JSON must not be a symlink: {raw!r}")
    existing = {path.resolve() for path in direct_existing}
    if not existing:
        fallback = [path for path in installed.rglob(raw_path.name) if path.is_file()]
        if any(path.is_symlink() for path in fallback):
            raise RuntimeError(f"installed kernel JSON fallback must not be a symlink: {raw!r}")
        existing = {path.resolve() for path in fallback}
    if len(existing) != 1:
        raise RuntimeError(f"cannot uniquely resolve installed kernel JSON {raw!r}: {sorted(map(str, existing))}")
    resolved = existing.pop()
    try:
        resolved.relative_to(installed.resolve())
    except ValueError as exc:
        raise RuntimeError("installed kernel JSON escapes installed custom OPP") from exc
    return resolved


def changed_leaf_paths(before: Any, after: Any, prefix: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    if type(before) is not type(after):
        return [prefix]
    if isinstance(before, dict):
        if before.keys() != after.keys():
            return [prefix]
        result: list[tuple[object, ...]] = []
        for key in before:
            result.extend(changed_leaf_paths(before[key], after[key], prefix + (key,)))
        return result
    if isinstance(before, list):
        if len(before) != len(after):
            return [prefix]
        result = []
        for index, (left, right) in enumerate(zip(before, after)):
            result.extend(changed_leaf_paths(left, right, prefix + (index,)))
        return result
    return [] if before == after else [prefix]


def kernel_identity(payload: Any) -> tuple[str, str, str]:
    if not isinstance(payload, dict):
        raise RuntimeError("kernel JSON root must be an object")
    values = tuple(str(payload.get(key, "")) for key in ("kernelName", "binFileName", "binFileSuffix"))
    if not all(values):
        raise RuntimeError("kernel JSON lacks kernelName/binFileName/binFileSuffix")
    return values


def regular_file_contract(path: Path) -> dict[str, Any]:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"tracked installed artifact is not a regular non-symlink file: {path}")
    return {"mode": stat.S_IMODE(info.st_mode), "bytes": info.st_size, "sha256": sha256(path)}


def prepare(
    installed: Path,
    candidate_dir: Path,
    overlay: Path,
    manifest_path: Path,
    expected_o_sha256: str,
    expected_json_sha256: str,
) -> dict[str, Any]:
    installed = installed.resolve(strict=True)
    candidate_dir = candidate_dir.resolve(strict=True)
    if overlay.exists():
        raise FileExistsError(f"refusing to reuse overlay: {overlay}")
    candidate_o = candidate_dir / f"{CANDIDATE_STEM}.o"
    candidate_json = candidate_dir / f"{CANDIDATE_STEM}.json"
    if not candidate_o.is_file() or not candidate_json.is_file():
        raise FileNotFoundError(f"candidate pair missing in {candidate_dir}")
    if candidate_o.stat().st_size != CANDIDATE_O_SIZE or candidate_json.stat().st_size != CANDIDATE_JSON_SIZE:
        raise RuntimeError("candidate artifact size contract failed")
    if sha256(candidate_o) != expected_o_sha256 or sha256(candidate_json) != expected_json_sha256:
        raise RuntimeError("candidate artifact SHA256 contract failed")
    candidate_payload = json.loads(candidate_json.read_text(encoding="utf-8"))
    candidate_kernel, candidate_bin, candidate_suffix = kernel_identity(candidate_payload)
    if (candidate_kernel, candidate_bin, candidate_suffix) != (
        CANDIDATE_STEM, CANDIDATE_STEM, ".o"
    ):
        raise RuntimeError("candidate JSON kernel identity contract failed")

    configs = discover_qrv2_configs(installed)
    config_records: dict[str, Any] = {}
    installed_sources: dict[str, str] = {}
    installed_artifacts: dict[str, dict[str, str]] = {}
    expected_overlay_files: set[Path] = set()
    original_kernel_names: set[str] = set()
    simplified_keys: list[Any] = []
    original_json_shas: set[str] = set()
    original_o_shas: set[str] = set()

    for config, before, entry in configs:
        old_raw = PurePosixPath(str(entry["binInfo"]["jsonFilePath"]))
        if old_raw.is_absolute():
            raise RuntimeError("absolute jsonFilePath cannot be safely redirected by an isolated overlay")
        original_json = resolve_installed_kernel_json(installed, config, str(old_raw))
        original_payload = json.loads(original_json.read_text(encoding="utf-8"))
        original_kernel, original_bin, original_suffix = kernel_identity(original_payload)
        if original_kernel != original_bin or original_suffix != ".o":
            raise RuntimeError("installed kernel JSON identity contract failed")
        original_o = original_json.with_name(original_bin + original_suffix)
        if not original_o.is_file() or original_o.stat().st_size == 0:
            raise RuntimeError("installed kernel object matching binFileName is missing")
        if original_bin == CANDIDATE_STEM:
            raise RuntimeError("installed config already points at candidate")

        after = copy.deepcopy(before)
        after_entries = [
            candidate
            for candidate in iter_bininfo_entries(after)
            if candidate["binInfo"]["jsonFilePath"] == entry["binInfo"]["jsonFilePath"]
            and candidate["simplifiedKey"] == entry["simplifiedKey"]
        ]
        if len(after_entries) != 1:
            raise RuntimeError(
                f"cannot uniquely relocate target binInfo entry in {config}: "
                f"{len(after_entries)} matches"
            )
        simplified_key_before = copy.deepcopy(entry["simplifiedKey"])
        new_raw = str(old_raw.with_name(candidate_json.name))
        after_entries[0]["binInfo"]["jsonFilePath"] = new_raw
        if after_entries[0]["simplifiedKey"] != simplified_key_before:
            raise RuntimeError("simplifiedKey changed unexpectedly")
        changes = changed_leaf_paths(before, after)
        if len(changes) != 1 or changes[0][-1] != "jsonFilePath":
            raise RuntimeError(f"overlay config changed fields other than jsonFilePath: {changes}")

        config_rel = config.relative_to(installed)
        original_json_rel = original_json.relative_to(installed)
        original_o_rel = original_o.relative_to(installed)
        overlay_config = overlay / config_rel
        overlay_json = overlay / original_json_rel.parent / candidate_json.name
        overlay_o = overlay / original_json_rel.parent / candidate_o.name
        overlay_config.parent.mkdir(parents=True, exist_ok=True)
        overlay_json.parent.mkdir(parents=True, exist_ok=True)
        overlay_config.write_text(
            json.dumps(after, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
        shutil.copyfile(candidate_json, overlay_json)
        shutil.copyfile(candidate_o, overlay_o)
        expected_overlay_files.update(
            {overlay_config.resolve(), overlay_json.resolve(), overlay_o.resolve()}
        )
        if sha256(candidate_json) != sha256(overlay_json) or sha256(candidate_o) != sha256(overlay_o):
            raise RuntimeError("candidate copy SHA mismatch")
        round_trip = json.loads(overlay_config.read_text(encoding="utf-8"))
        if changed_leaf_paths(before, round_trip) != changes:
            raise RuntimeError("serialized config diff changed")

        config_contract = regular_file_contract(config)
        original_json_contract = regular_file_contract(original_json)
        original_o_contract = regular_file_contract(original_o)
        config_sha = str(config_contract["sha256"])
        original_json_sha = str(original_json_contract["sha256"])
        original_o_sha = str(original_o_contract["sha256"])
        original_kernel_names.add(original_kernel)
        simplified_keys.append(simplified_key_before)
        original_json_shas.add(original_json_sha)
        original_o_shas.add(original_o_sha)
        for relative, contract, kind in (
            (config_rel, config_contract, "config"),
            (original_json_rel, original_json_contract, "original_json"),
            (original_o_rel, original_o_contract, "original_object"),
        ):
            relative_text = str(relative)
            if relative_text in installed_sources:
                raise RuntimeError(f"installed artifact reused across SOC configs: {relative_text}")
            installed_sources[relative_text] = str(contract["sha256"])
            installed_artifacts[relative_text] = {
                "kind": kind,
                "before_sha256": str(contract["sha256"]),
                "expected_after_sha256": str(contract["sha256"]),
                "bytes": str(contract["bytes"]),
                "mode": oct(int(contract["mode"])),
            }
        config_records[str(config_rel)] = {
            "installed_before_sha256": config_sha,
            "installed_expected_after_sha256": config_sha,
            "overlay_after_sha256": sha256(overlay_config),
            "jsonFilePath_before": str(old_raw),
            "jsonFilePath_after": new_raw,
            "simplifiedKey_sha256": hashlib.sha256(
                json.dumps(simplified_key_before, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "changed_leaf_paths": [list(path) for path in changes],
            "original_json_relative_path": str(original_json_rel),
            "original_json_before_sha256": original_json_sha,
            "original_json_expected_after_sha256": original_json_sha,
            "original_o_relative_path": str(original_o_rel),
            "original_o_before_sha256": original_o_sha,
            "original_o_expected_after_sha256": original_o_sha,
        }

    if len(original_kernel_names) != 1:
        raise RuntimeError(f"installed SOC kernelName mismatch: {sorted(original_kernel_names)}")
    if simplified_keys[0] != simplified_keys[1]:
        raise RuntimeError("installed SOC simplifiedKey mismatch")
    if len(original_json_shas) != 1 or len(original_o_shas) != 1:
        raise RuntimeError("installed SOC kernel JSON/object bytes mismatch")
    original_kernel = original_kernel_names.pop()
    files = sorted(path for path in overlay.rglob("*") if path.is_file())
    if len(files) != 6 or {path.resolve() for path in files} != expected_overlay_files:
        raise RuntimeError(f"overlay must contain exactly six expected files: {[str(path) for path in files]}")
    manifest = {
        "schema": "step347-overlay-v1",
        "installed_custom_opp": str(installed),
        "overlay": str(overlay.resolve()),
        "configs": config_records,
        "original_kernel_name": original_kernel,
        "candidate_kernel_name": CANDIDATE_STEM,
        "files": {
            str(path.relative_to(overlay)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
        "candidate_source_files": {
            candidate_o.name: sha256(candidate_o),
            candidate_json.name: sha256(candidate_json),
        },
        "installed_artifacts": installed_artifacts,
        "installed_source_sha256": installed_sources,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def verify_installed(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    installed = Path(manifest["installed_custom_opp"]).resolve(strict=True)
    artifacts = manifest.get("installed_artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != 6:
        raise RuntimeError("manifest must describe exactly six tracked installed artifacts")
    sha_inventory = manifest.get("installed_source_sha256")
    expected_inventory = {
        relative: details.get("before_sha256") for relative, details in artifacts.items()
    }
    if sha_inventory != expected_inventory:
        raise RuntimeError("installed artifact manifest inventories disagree")
    actual: dict[str, dict[str, str]] = {}
    for relative, expected in artifacts.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe installed artifact relative path: {relative}")
        tracked_path = installed / relative_path
        try:
            tracked_path.parent.resolve(strict=True).relative_to(installed)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(f"installed artifact path escapes tracked root: {relative}") from exc
        contract = regular_file_contract(tracked_path)
        actual[relative] = {
            "sha256": str(contract["sha256"]),
            "bytes": str(contract["bytes"]),
            "mode": oct(int(contract["mode"])),
        }
        if (
            actual[relative]["sha256"] != expected["expected_after_sha256"]
            or actual[relative]["bytes"] != expected["bytes"]
            or actual[relative]["mode"] != expected["mode"]
        ):
            raise RuntimeError(f"tracked installed artifact changed: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed-custom-opp")
    parser.add_argument("--candidate-dir")
    parser.add_argument("--overlay")
    parser.add_argument("--manifest")
    parser.add_argument("--candidate-o-sha256")
    parser.add_argument("--candidate-json-sha256")
    parser.add_argument("--verify-installed", action="store_true")
    args = parser.parse_args()
    if args.verify_installed:
        if not args.manifest:
            parser.error("--verify-installed requires --manifest")
        verify_installed(Path(args.manifest))
        print(json.dumps({"installed_tracked_artifacts_gate": "PASS", "tracked_file_count": 6}, sort_keys=True))
        return 0
    missing = [
        option
        for option, value in (
            ("--installed-custom-opp", args.installed_custom_opp),
            ("--candidate-dir", args.candidate_dir),
            ("--overlay", args.overlay),
            ("--manifest", args.manifest),
        )
        if not value
    ]
    if missing:
        parser.error("overlay creation requires " + ", ".join(missing))
    if not args.candidate_o_sha256 or not args.candidate_json_sha256:
        parser.error("candidate SHA256 values are required when creating an overlay")
    manifest = prepare(
        Path(args.installed_custom_opp),
        Path(args.candidate_dir),
        Path(args.overlay),
        Path(args.manifest),
        args.candidate_o_sha256,
        args.candidate_json_sha256,
    )
    print(json.dumps({
        "overlay_gate": "PASS",
        "original_kernel_name": manifest["original_kernel_name"],
        "candidate_kernel_name": manifest["candidate_kernel_name"],
        "file_count": len(manifest["files"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
