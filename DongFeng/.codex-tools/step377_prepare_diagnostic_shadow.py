#!/usr/bin/env python3
"""Build an unpacked, diagnostic-only QrV2 shadow from STEP376 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


SOCS = ("ascend910_93", "ascend910b")
STATUS = "diagnostic_shadow_unvalidated"
EXPECTED_IDENTITY = "QrV2_vtv_direct_qa_legacy_probe_v6"
EXPECTED_SOURCE_SHA256 = "ef5db14e09170806acb7c5227fd619f3f5ffdc7d31f36e49058cc88987fce180"
EXPECTED_REVERSE_V4_SHA256 = "2213dbae5027c179f09c8e3b43be51587ba22e3eb2a7546311048efc3844614b"
EXPECTED_TOOLS = {
    "diagnostic_adapter_sha256": "fc65fecc58cefb86f64b6e71d64a21e5e4bc1416b42f1cd696aff6bbdedc299e",
    "base_builder_sha256": "d6f2aea68574422ba6e28f220c5581adf8b25019494882dea156a80b1b513e90",
    "step375_patcher_sha256": "98a655f89ac5efedd760067fdda595d9b5fe376b1e51fdc1b12d59c727711768",
    "v4_patcher_sha256": "d4b260919440e6298c9caea5ca3a7c8bb7426a30ed84cc68d934250ee0ae9fe2",
}
MAX_MEMBERS = 20_000
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200
INCOMPLETE_MARKER = ".step377-do-not-consume.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while True:
            left_chunk = left_stream.read(1024 * 1024)
            right_chunk = right_stream.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _regular(path: Path, label: str) -> Path:
    absolute = path.absolute()
    if absolute.is_symlink() or not absolute.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file")
    return absolute.resolve(strict=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.stat().st_size > 4 * 1024 * 1024:
        raise RuntimeError(f"{label} is unexpectedly large")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, ValueError) as error:
        raise RuntimeError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


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
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _open_directory_at(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    return os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )


def _write_new_at(parent_fd: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_fd,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_transaction_marker(root_fd: int, directory: str, status: str) -> None:
    directory_fd = os.open(
        directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=root_fd,
    )
    try:
        _write_new_at(
            directory_fd,
            INCOMPLETE_MARKER,
            (json.dumps({"consumable": False, "status": status}) + "\n").encode(),
        )
    finally:
        os.close(directory_fd)


def _open_existing_chain(root_fd: int, parts: tuple[str, ...]) -> int:
    current = os.dup(root_fd)
    try:
        for part in parts:
            following = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current,
            )
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def _read_regular_at(parent_fd: int, name: str, label: str) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_size > 4 * 1024 * 1024:
            raise RuntimeError(f"{label} must be a bounded regular file")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _json_at(parent_fd: int, name: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_read_regular_at(parent_fd, name, label).decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise RuntimeError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _replace_json_at(parent_fd: int, name: str, value: dict[str, Any]) -> None:
    temporary = name + ".step377-new"
    _write_new_at(
        parent_fd,
        temporary,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    )
    os.replace(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)


def _stream_member(root_fd: int, archive: zipfile.ZipFile, info: zipfile.ZipInfo, member: PurePosixPath) -> int:
    descriptors = [os.dup(root_fd)]
    try:
        for part in member.parts[:-1]:
            descriptors.append(_open_directory_at(descriptors[-1], part))
        if info.is_dir():
            descriptors.append(_open_directory_at(descriptors[-1], member.parts[-1]))
            return 0
        descriptor = os.open(
            member.parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=descriptors[-1],
        )
        actual = 0
        try:
            with archive.open(info) as source, os.fdopen(descriptor, "wb") as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    actual += len(chunk)
                    if actual > info.file_size or actual > MAX_UNCOMPRESSED_BYTES:
                        raise RuntimeError("wheel member expanded beyond declared size")
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
        except BaseException:
            try:
                os.unlink(member.parts[-1], dir_fd=descriptors[-1])
            except OSError:
                pass
            raise
        if actual != info.file_size:
            raise RuntimeError("wheel member actual size mismatch")
        return actual
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _extract_wheel(
    wheel: Path,
    shadow: Path,
    *,
    precreated: bool = False,
    parent_fd: int | None = None,
) -> None:
    if not precreated:
        shadow.mkdir(mode=0o700)
    if parent_fd is None:
        root_fd = os.open(shadow, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    else:
        root_fd = os.open(
            shadow.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    with zipfile.ZipFile(wheel) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS:
            raise RuntimeError("wheel member count exceeds limit")
        if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
            raise RuntimeError("wheel uncompressed size exceeds limit")
        for info in infos:
            if info.file_size and info.file_size / max(info.compress_size, 1) > MAX_COMPRESSION_RATIO:
                raise RuntimeError("wheel member compression ratio exceeds limit")
        seen: set[str] = set()
        actual_total = 0
        try:
            for info in infos:
                member = _member_path(info.filename)
                normalized = member.as_posix().rstrip("/")
                if normalized in seen:
                    raise RuntimeError(f"duplicate wheel member: {info.filename}")
                seen.add(normalized)
                file_type = stat.S_IFMT(info.external_attr >> 16)
                if file_type == stat.S_IFLNK or file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise RuntimeError(f"non-regular wheel member: {info.filename}")
                actual_total += _stream_member(root_fd, archive, info, member)
                if actual_total > MAX_UNCOMPRESSED_BYTES:
                    raise RuntimeError("wheel actual uncompressed size exceeds limit")
        finally:
            os.close(root_fd)
    if any(path.is_symlink() for path in shadow.rglob("*")):
        raise RuntimeError("symlink appeared in diagnostic shadow")


def _tree_inventory(root: Path, *, ignore_marker: bool = False) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if ignore_marker and relative == INCOMPLETE_MARKER:
            continue
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode):
            raise RuntimeError("tree inventory rejects symlink")
        if stat.S_ISDIR(status.st_mode):
            result[relative] = {"type": "directory", "mode": stat.S_IMODE(status.st_mode)}
        elif stat.S_ISREG(status.st_mode):
            result[relative] = {
                "type": "file",
                "mode": stat.S_IMODE(status.st_mode),
                "size": status.st_size,
                "sha256": sha256_file(path),
            }
        else:
            raise RuntimeError("tree inventory rejects non-regular entry")
    return result


def _validate_tree_delta(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    allowed_removed: set[str],
    allowed_added: set[str],
    allowed_modified: set[str],
) -> None:
    removed = set(before) - set(after)
    added = set(after) - set(before)
    modified = {path for path in set(before) & set(after) if before[path] != after[path]}
    if removed != allowed_removed or added != allowed_added or modified != allowed_modified:
        raise RuntimeError("diagnostic shadow tree delta escaped QrV2 allowlist")


def _assert_root_identity(root_fd: int, root: Path) -> None:
    descriptor_status = os.fstat(root_fd)
    path_status = root.lstat()
    if (
        not stat.S_ISDIR(path_status.st_mode)
        or stat.S_ISLNK(path_status.st_mode)
        or descriptor_status.st_ino != path_status.st_ino
        or descriptor_status.st_dev != path_status.st_dev
    ):
        raise RuntimeError("approved root changed during transaction")


def _artifact(
    manifest: dict[str, Any], manifest_path: Path, soc: str
) -> tuple[Path, Path, dict[str, Any]]:
    value = manifest.get("artifacts", {}).get(soc)
    if not isinstance(value, dict):
        raise RuntimeError(f"attempt3 artifact missing: {soc}")
    object_path = _regular(Path(str(value.get("object_path", ""))), f"{soc} object")
    json_path = _regular(Path(str(value.get("json_path", ""))), f"{soc} JSON")
    expected_root = (manifest_path.parent / "build" / soc).resolve(strict=True)
    for path in (object_path, json_path):
        try:
            path.relative_to(expected_root)
        except ValueError as error:
            raise RuntimeError(f"attempt3 {soc} artifact escapes build root") from error
    for path, key in ((object_path, "object_sha256"), (json_path, "json_sha256")):
        expected = value.get(key)
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise RuntimeError(f"attempt3 {soc} {key} mismatch")
    if value.get("object_size") != object_path.stat().st_size or value.get("json_size") != json_path.stat().st_size:
        raise RuntimeError(f"attempt3 {soc} artifact size mismatch")
    if value.get("kernel_name") != EXPECTED_IDENTITY or value.get("bin_file_name") != EXPECTED_IDENTITY:
        raise RuntimeError(f"attempt3 {soc} artifact identity mismatch")
    expected_entries = sorted(
        (EXPECTED_IDENTITY + "_0_mix_aic", EXPECTED_IDENTITY + "_0_mix_aiv")
    )
    if value.get("concrete_entries") != expected_entries:
        raise RuntimeError(f"attempt3 {soc} concrete identity mismatch")
    metadata = _load_json(json_path, f"{soc} candidate JSON")
    if metadata.get("kernelName") != EXPECTED_IDENTITY or metadata.get("binFileName") != EXPECTED_IDENTITY:
        raise RuntimeError(f"attempt3 {soc} candidate JSON identity mismatch")
    return object_path, json_path, value


def _update_soc(
    shadow_fd: int, package: Path, soc: str, identity: str, obj: Path, meta: Path
) -> dict[str, Any]:
    kernel_root = package / "packages/vendors/customize/op_impl/ai_core/tbe/kernel"
    kernel_dir = kernel_root / soc / "qr_v2"
    config = kernel_root / "config" / soc / "qr_v2.json"
    binary_info = kernel_root / "config" / soc / "binary_info_config.json"
    base_parts = (
        "mx_driving_cloud", "packages", "vendors", "customize", "op_impl",
        "ai_core", "tbe", "kernel",
    )
    kernel_fd = _open_existing_chain(shadow_fd, base_parts + (soc, "qr_v2"))
    config_fd = _open_existing_chain(shadow_fd, base_parts + ("config", soc))
    try:
        names = os.listdir(kernel_fd)
        old_json_names = [name for name in names if name.endswith(".json")]
        old_object_names = [name for name in names if name.endswith(".o")]
        if len(old_json_names) != 1 or len(old_object_names) != 1 or len(names) != 2:
            raise RuntimeError(f"{soc} must have exactly one old QrV2 JSON/object pair")
        for name in (*old_json_names, *old_object_names):
            status = os.stat(name, dir_fd=kernel_fd, follow_symlinks=False)
            if not stat.S_ISREG(status.st_mode):
                raise RuntimeError(f"{soc} old QrV2 artifact is not regular")
        config_value = _json_at(config_fd, "qr_v2.json", f"{soc} qr_v2 config")
        binary_value = _json_at(
            config_fd, "binary_info_config.json", f"{soc} binary-info"
        )
    except BaseException:
        os.close(kernel_fd)
        os.close(config_fd)
        raise
    old_json = [kernel_dir / name for name in old_json_names]
    old_object = [kernel_dir / name for name in old_object_names]
    new_json = kernel_dir / f"{identity}.json"
    new_object = kernel_dir / f"{identity}.o"
    try:
        rows = config_value.get("binList")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise RuntimeError(f"{soc} qr_v2 config must have one row")
        bin_info = rows[0].get("binInfo")
        if not isinstance(bin_info, dict) or set(bin_info) != {"jsonFilePath"}:
            raise RuntimeError(f"{soc} qr_v2 config path is not unique")
        qrv2 = binary_value.get("QrV2")
        if (
            not isinstance(qrv2, dict)
            or not isinstance(qrv2.get("binaryList"), list)
            or len(qrv2["binaryList"]) != 2
        ):
            raise RuntimeError(f"{soc} binary-info QrV2 list is invalid")
        for row in qrv2["binaryList"]:
            if not isinstance(row, dict) or not isinstance(row.get("binPath"), str):
                raise RuntimeError(f"{soc} binary-info QrV2 row is invalid")
        for name in (*old_json_names, *old_object_names):
            os.unlink(name, dir_fd=kernel_fd)
        _write_new_at(kernel_fd, new_json.name, meta.read_bytes())
        _write_new_at(kernel_fd, new_object.name, obj.read_bytes())
        config_value["binList"][0]["binInfo"]["jsonFilePath"] = f"{soc}/qr_v2/{identity}.json"
        for row in qrv2["binaryList"]:
            row["binPath"] = f"{soc}/qr_v2/{identity}.o"
        _replace_json_at(config_fd, "qr_v2.json", config_value)
        _replace_json_at(config_fd, "binary_info_config.json", binary_value)
    finally:
        os.close(kernel_fd)
        os.close(config_fd)
    shadow_root = package.parent
    old_paths = {path.relative_to(shadow_root).as_posix() for path in (*old_json, *old_object)}
    new_paths = {path.relative_to(shadow_root).as_posix() for path in (new_json, new_object)}
    return {
        "json_path": str(new_json.resolve(strict=True)),
        "json_sha256": sha256_file(meta),
        "object_path": str(new_object.resolve(strict=True)),
        "object_sha256": sha256_file(obj),
        "config_sha256": sha256_file(config),
        "binary_info_config_sha256": sha256_file(binary_info),
        "_tree_delta": {
            "removed": sorted(old_paths - new_paths),
            "added": sorted(new_paths - old_paths),
            "modified": sorted(
                (old_paths & new_paths)
                | {
                    config.relative_to(shadow_root).as_posix(),
                    binary_info.relative_to(shadow_root).as_posix(),
                }
            ),
        },
    }


def _prepare_impl(
    attempt3_manifest: Path,
    wheel: Path,
    approved_root: Path,
    output_manifest: Path,
    root_fd_holder: list[int],
) -> dict[str, Any]:
    root = approved_root.absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise RuntimeError("approved root must be an existing real directory")
    shadow = root / "shadow"
    partial = root / "shadow.partial"
    if shadow.exists() or shadow.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError("diagnostic shadow or partial transaction already exists")
    if output_manifest.absolute().parent != root or output_manifest.exists() or output_manifest.is_symlink():
        raise RuntimeError("output manifest must be a new direct child of approved root")
    if output_manifest.name in {"shadow", "shadow.partial", INCOMPLETE_MARKER}:
        raise RuntimeError("output manifest uses a reserved STEP377 name")
    manifest_path = _regular(attempt3_manifest, "attempt3 manifest")
    wheel_path = _regular(wheel, "immutable original wheel")
    manifest_sha_before = sha256_file(manifest_path)
    wheel_sha_before = sha256_file(wheel_path)
    source_manifest = _load_json(manifest_path, "attempt3 manifest")
    if source_manifest.get("status") != "diagnostic_built_unvalidated":
        raise RuntimeError("attempt3 manifest status mismatch")
    expected_flags = {
        "artifact_class": "diagnostic_probe",
        "diagnostic_only": True,
        "release_candidate": False,
        "package_forbidden": True,
    }
    policy = source_manifest.get("policy")
    if not isinstance(policy, dict) or any(policy.get(key) != value for key, value in expected_flags.items()):
        raise RuntimeError("attempt3 diagnostic policy mismatch")
    if source_manifest.get("package") != {"status": "forbidden_diagnostic_probe"}:
        raise RuntimeError("attempt3 package policy mismatch")
    guard = source_manifest.get("immutable_guards", {}).get("extracted_original_wheel")
    if not isinstance(guard, dict) or guard.get("path") != str(wheel_path) or guard.get("sha256") != wheel_sha_before:
        raise RuntimeError("immutable original wheel guard mismatch")
    candidate = source_manifest.get("candidate")
    if (
        not isinstance(candidate, dict)
        or candidate.get("identity") != EXPECTED_IDENTITY
        or candidate.get("source_sha256") != EXPECTED_SOURCE_SHA256
        or candidate.get("reverse_v4_sha256") != EXPECTED_REVERSE_V4_SHA256
        or any(candidate.get(key) != value for key, value in expected_flags.items())
    ):
        raise RuntimeError("attempt3 candidate contract mismatch")
    if source_manifest.get("tools") != EXPECTED_TOOLS:
        raise RuntimeError("attempt3 tool SHA contract mismatch")
    identity = candidate["identity"]
    artifacts = {soc: _artifact(source_manifest, manifest_path, soc) for soc in SOCS}
    if artifacts[SOCS[0]][2].get("object_sha256") != artifacts[SOCS[1]][2].get("object_sha256"):
        raise RuntimeError("attempt3 alias object SHA mismatch")
    if artifacts[SOCS[0]][2].get("json_sha256") != artifacts[SOCS[1]][2].get("json_sha256"):
        raise RuntimeError("attempt3 alias JSON SHA mismatch")
    if not _files_equal(artifacts[SOCS[0]][0], artifacts[SOCS[1]][0]):
        raise RuntimeError("attempt3 alias object bytes mismatch")
    if not _files_equal(artifacts[SOCS[0]][1], artifacts[SOCS[1]][1]):
        raise RuntimeError("attempt3 alias JSON bytes mismatch")

    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    root_fd_holder.append(root_fd)
    if os.fstat(root_fd).st_ino != root.stat().st_ino or os.fstat(root_fd).st_dev != root.stat().st_dev:
        raise RuntimeError("approved root changed during validation")
    try:
        os.mkdir("shadow.partial", mode=0o700, dir_fd=root_fd)
        _extract_wheel(wheel_path, partial, precreated=True, parent_fd=root_fd)
        _assert_root_identity(root_fd, root)
        _write_transaction_marker(root_fd, "shadow.partial", "step377_incomplete")
        package = partial / "mx_driving_cloud"
        if package.is_symlink() or not package.is_dir():
            raise RuntimeError("wheel lacks mx_driving_cloud package")
        before_tree = _tree_inventory(partial, ignore_marker=True)
        records = [
            path for path in partial.rglob("RECORD")
            if path.is_file() and not path.is_symlink()
        ]
        if len(records) != 1:
            raise RuntimeError("official wheel must contain exactly one regular RECORD")
        record_relative = records[0].relative_to(partial).as_posix()
        record_sha256 = sha256_file(records[0])
        partial_fd = os.open(
            "shadow.partial",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        try:
            result_artifacts = {
                soc: _update_soc(
                    partial_fd,
                    package,
                    soc,
                    identity,
                    artifacts[soc][0],
                    artifacts[soc][1],
                )
                for soc in SOCS
            }
        finally:
            os.close(partial_fd)
        _assert_root_identity(root_fd, root)
        allowed_removed: set[str] = set()
        allowed_added: set[str] = set()
        allowed_modified: set[str] = set()
        for value in result_artifacts.values():
            delta = value.pop("_tree_delta")
            allowed_removed.update(delta["removed"])
            allowed_added.update(delta["added"])
            allowed_modified.update(delta["modified"])
        after_tree = _tree_inventory(partial, ignore_marker=True)
        _validate_tree_delta(
            before_tree, after_tree, allowed_removed, allowed_added, allowed_modified
        )
        record_after = partial / record_relative
        if (
            not record_after.is_file()
            or record_after.is_symlink()
            or sha256_file(record_after) != record_sha256
        ):
            raise RuntimeError("wheel RECORD changed in diagnostic shadow")
    except BaseException:
        try:
            _write_transaction_marker(root_fd, "shadow.partial", "step377_failed")
        except BaseException:
            pass
        raise
    artifact_inputs = {
        soc: {
            "object_path": str(artifacts[soc][0]),
            "object_sha256": artifacts[soc][2]["object_sha256"],
            "json_path": str(artifacts[soc][1]),
            "json_sha256": artifacts[soc][2]["json_sha256"],
        }
        for soc in SOCS
    }
    artifact_inputs_unchanged = True
    for soc in SOCS:
        for index, key in ((0, "object_sha256"), (1, "json_sha256")):
            original = artifacts[soc][index]
            try:
                current = _regular(original, f"postflight {soc} artifact")
            except (OSError, RuntimeError):
                artifact_inputs_unchanged = False
                continue
            if current != original or sha256_file(current) != artifacts[soc][2][key]:
                artifact_inputs_unchanged = False
    try:
        manifest_unchanged = _regular(manifest_path, "postflight attempt3 manifest") == manifest_path
        wheel_unchanged = _regular(wheel_path, "postflight original wheel") == wheel_path
    except (OSError, RuntimeError):
        manifest_unchanged = False
        wheel_unchanged = False
    if (
        not manifest_unchanged
        or not wheel_unchanged
        or sha256_file(manifest_path) != manifest_sha_before
        or sha256_file(wheel_path) != wheel_sha_before
        or not artifact_inputs_unchanged
    ):
        raise RuntimeError("immutable STEP376 input changed during shadow preparation")
    for value in result_artifacts.values():
        for key in ("json_path", "object_path"):
            relative = Path(value[key]).relative_to(partial)
            value[key] = str(shadow / relative)
    _assert_root_identity(root_fd, root)
    output = {
        "schema": "step377-diagnostic-shadow-v1",
        "status": STATUS,
        "diagnostic_only": True,
        "package_forbidden": True,
        "source_overlay": False,
        "record_unchanged": True,
        "attempt3_manifest": {"path": str(manifest_path), "sha256": manifest_sha_before},
        "original_wheel": {"path": str(wheel_path), "sha256": wheel_sha_before},
        "shadow_root": str(shadow),
        "package_root": str(shadow / "mx_driving_cloud"),
        "candidate_identity": identity,
        "record": {"relative_path": record_relative, "sha256": record_sha256},
        "attempt3_artifact_inputs": artifact_inputs,
        "artifacts": result_artifacts,
    }
    os.rename("shadow.partial", "shadow", src_dir_fd=root_fd, dst_dir_fd=root_fd)
    _assert_root_identity(root_fd, root)
    if _tree_inventory(shadow, ignore_marker=True) != after_tree:
        raise RuntimeError("final diagnostic shadow changed after atomic publish")
    shadow_fd = os.open(
        "shadow", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd
    )
    try:
        marker_status = os.stat(INCOMPLETE_MARKER, dir_fd=shadow_fd, follow_symlinks=False)
        if not stat.S_ISREG(marker_status.st_mode):
            raise RuntimeError("final diagnostic shadow marker is invalid")
        os.unlink(INCOMPLETE_MARKER, dir_fd=shadow_fd)
    except BaseException:
        raise
    finally:
        os.close(shadow_fd)
    _write_new_at(
        root_fd,
        output_manifest.name,
        (json.dumps(output, indent=2, sort_keys=True) + "\n").encode(),
    )
    return output


def prepare(
    attempt3_manifest: Path,
    wheel: Path,
    approved_root: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    root_fd_holder: list[int] = []
    try:
        return _prepare_impl(
            attempt3_manifest,
            wheel,
            approved_root,
            output_manifest,
            root_fd_holder,
        )
    finally:
        for descriptor in root_fd_holder:
            os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt3-manifest", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--approved-root", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.attempt3_manifest, args.wheel, args.approved_root, args.output_manifest)
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
