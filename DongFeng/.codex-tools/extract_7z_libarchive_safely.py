from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import posixpath
import shutil
import stat
from pathlib import Path, PurePosixPath


ARCHIVE_OK = 0
ARCHIVE_EOF = 1
ARCHIVE_EXTRACT_NO_OVERWRITE = 0x0008
ARCHIVE_EXTRACT_SECURE_SYMLINKS = 0x0100
ARCHIVE_EXTRACT_SECURE_NODOTDOT = 0x0200
ARCHIVE_EXTRACT_SECURE_NOABSOLUTEPATHS = 0x10000
SECURE_OPTIONS = (
    ARCHIVE_EXTRACT_NO_OVERWRITE
    | ARCHIVE_EXTRACT_SECURE_SYMLINKS
    | ARCHIVE_EXTRACT_SECURE_NODOTDOT
    | ARCHIVE_EXTRACT_SECURE_NOABSOLUTEPATHS
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_archive() -> ctypes.CDLL:
    library = ctypes.util.find_library("archive")
    if not library:
        raise RuntimeError("system libarchive is unavailable")
    lib = ctypes.CDLL(library)
    lib.archive_read_new.restype = ctypes.c_void_p
    lib.archive_read_support_filter_all.argtypes = [ctypes.c_void_p]
    lib.archive_read_support_format_all.argtypes = [ctypes.c_void_p]
    lib.archive_read_open_filename.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.archive_read_open_filename.restype = ctypes.c_int
    lib.archive_read_next_header.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.archive_read_next_header.restype = ctypes.c_int
    lib.archive_read_data_skip.argtypes = [ctypes.c_void_p]
    lib.archive_read_data_skip.restype = ctypes.c_int
    lib.archive_read_free.argtypes = [ctypes.c_void_p]
    lib.archive_error_string.argtypes = [ctypes.c_void_p]
    lib.archive_error_string.restype = ctypes.c_char_p
    lib.archive_entry_pathname.argtypes = [ctypes.c_void_p]
    lib.archive_entry_pathname.restype = ctypes.c_char_p
    lib.archive_entry_size.argtypes = [ctypes.c_void_p]
    lib.archive_entry_size.restype = ctypes.c_longlong
    lib.archive_entry_filetype.argtypes = [ctypes.c_void_p]
    lib.archive_entry_filetype.restype = ctypes.c_uint
    lib.archive_entry_symlink.argtypes = [ctypes.c_void_p]
    lib.archive_entry_symlink.restype = ctypes.c_char_p
    lib.archive_entry_hardlink.argtypes = [ctypes.c_void_p]
    lib.archive_entry_hardlink.restype = ctypes.c_char_p
    lib.archive_write_disk_new.restype = ctypes.c_void_p
    lib.archive_write_disk_set_options.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.archive_write_disk_set_options.restype = ctypes.c_int
    lib.archive_write_free.argtypes = [ctypes.c_void_p]
    lib.archive_read_extract2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.archive_read_extract2.restype = ctypes.c_int
    return lib


def archive_error(lib: ctypes.CDLL, archive: int) -> str:
    raw = lib.archive_error_string(archive)
    return raw.decode("utf-8", "replace") if raw else "unknown libarchive error"


def open_reader(lib: ctypes.CDLL, archive_path: Path) -> int:
    reader = lib.archive_read_new()
    if not reader:
        raise RuntimeError("archive_read_new failed")
    lib.archive_read_support_filter_all(reader)
    lib.archive_read_support_format_all(reader)
    rc = lib.archive_read_open_filename(reader, os.fsencode(archive_path), 10240)
    if rc != ARCHIVE_OK:
        error = archive_error(lib, reader)
        lib.archive_read_free(reader)
        raise RuntimeError(f"archive open failed: {error}")
    return reader


def safe_member(name: str, filetype: int, symlink: bytes | None, hardlink: bytes | None) -> bool:
    portable = name.replace("\\", "/")
    norm = posixpath.normpath(portable)
    pure = PurePosixPath(norm)
    if not name or portable.startswith("/") or pure.is_absolute():
        return False
    if norm in (".", "..") or norm.startswith("../") or ".." in pure.parts:
        return False
    if pure.parts and ":" in pure.parts[0]:
        return False
    if symlink or hardlink:
        return False
    return filetype in (stat.S_IFREG, stat.S_IFDIR)


def inventory(lib: ctypes.CDLL, archive_path: Path) -> list[dict[str, object]]:
    reader = open_reader(lib, archive_path)
    rows: list[dict[str, object]] = []
    try:
        while True:
            entry = ctypes.c_void_p()
            rc = lib.archive_read_next_header(reader, ctypes.byref(entry))
            if rc == ARCHIVE_EOF:
                break
            if rc < ARCHIVE_OK:
                raise RuntimeError(f"archive header failed: {archive_error(lib, reader)}")
            raw_name = lib.archive_entry_pathname(entry) or b""
            name = raw_name.decode("utf-8", "replace")
            filetype = int(lib.archive_entry_filetype(entry))
            symlink = lib.archive_entry_symlink(entry)
            hardlink = lib.archive_entry_hardlink(entry)
            if not safe_member(name, filetype, symlink, hardlink):
                raise RuntimeError(f"unsafe archive member rejected: {name!r}")
            rows.append(
                {
                    "name": name,
                    "declared_bytes": max(0, int(lib.archive_entry_size(entry))),
                    "filetype": "regular" if filetype == stat.S_IFREG else "directory",
                }
            )
            if lib.archive_read_data_skip(reader) < ARCHIVE_OK:
                raise RuntimeError(f"archive skip failed: {archive_error(lib, reader)}")
    finally:
        lib.archive_read_free(reader)
    if not rows:
        raise RuntimeError("archive is empty")
    return rows


def extract(lib: ctypes.CDLL, archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=False, exist_ok=False)
    output_root = output_dir.resolve(strict=True)
    reader = open_reader(lib, archive_path)
    disk = lib.archive_write_disk_new()
    if not disk:
        lib.archive_read_free(reader)
        raise RuntimeError("archive_write_disk_new failed")
    if lib.archive_write_disk_set_options(disk, SECURE_OPTIONS) != ARCHIVE_OK:
        lib.archive_write_free(disk)
        lib.archive_read_free(reader)
        raise RuntimeError("failed to set secure extraction options")
    previous = Path.cwd()
    try:
        os.chdir(output_root)
        while True:
            entry = ctypes.c_void_p()
            rc = lib.archive_read_next_header(reader, ctypes.byref(entry))
            if rc == ARCHIVE_EOF:
                break
            if rc < ARCHIVE_OK:
                raise RuntimeError(f"archive header failed: {archive_error(lib, reader)}")
            name = (lib.archive_entry_pathname(entry) or b"").decode("utf-8", "replace")
            filetype = int(lib.archive_entry_filetype(entry))
            if not safe_member(
                name,
                filetype,
                lib.archive_entry_symlink(entry),
                lib.archive_entry_hardlink(entry),
            ):
                raise RuntimeError(f"unsafe archive member rejected during extraction: {name!r}")
            if lib.archive_read_extract2(reader, entry, disk) != ARCHIVE_OK:
                raise RuntimeError(f"archive extraction failed: {archive_error(lib, reader)}")
    finally:
        os.chdir(previous)
        lib.archive_write_free(disk)
        lib.archive_read_free(reader)


def verify_tree(output_dir: Path) -> list[dict[str, object]]:
    root = output_dir.resolve(strict=True)
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        resolved = path.resolve(strict=True)
        if os.path.commonpath((str(root), str(resolved))) != str(root):
            raise RuntimeError(f"extracted path escapes output root: {path}")
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise RuntimeError(f"unexpected extracted file type: {path}")
        if stat.S_ISREG(mode):
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not rows:
        raise RuntimeError("no regular files were extracted")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    archive_path = args.archive.resolve(strict=True)
    if archive_path.is_symlink() or not archive_path.is_file():
        raise RuntimeError("archive must be a regular non-symlink file")
    if args.output_dir.exists() or args.manifest.exists():
        raise RuntimeError("output directory and manifest must not already exist")

    lib = load_archive()
    archive_sha_before = sha256_file(archive_path)
    members = inventory(lib, archive_path)
    declared_bytes = sum(int(row["declared_bytes"]) for row in members)
    free_bytes = shutil.disk_usage(args.output_dir.parent).free
    if free_bytes < declared_bytes + 2 * 1024**3:
        raise RuntimeError("insufficient free space for safe extraction")

    extract(lib, archive_path, args.output_dir)
    extracted = verify_tree(args.output_dir)
    archive_sha_after = sha256_file(archive_path)
    if archive_sha_before != archive_sha_after:
        raise RuntimeError("archive changed during extraction")
    regular_declared = sum(int(row["declared_bytes"]) for row in members if row["filetype"] == "regular")
    actual_bytes = sum(int(row["bytes"]) for row in extracted)
    if actual_bytes != regular_declared:
        raise RuntimeError(f"declared/extracted byte mismatch: {regular_declared} != {actual_bytes}")

    payload = {
        "archive": {
            "bytes": archive_path.stat().st_size,
            "sha256_before": archive_sha_before,
            "sha256_after": archive_sha_after,
            "mutation_performed": False,
        },
        "secure_options": {
            "no_overwrite": True,
            "secure_symlinks": True,
            "secure_nodotdot": True,
            "secure_noabsolutepaths": True,
        },
        "members": members,
        "declared_bytes": declared_bytes,
        "free_bytes_before": free_bytes,
        "extracted_files": extracted,
        "extracted_regular_count": len(extracted),
        "extracted_regular_bytes": actual_bytes,
    }
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "entry_count": len(members),
        "declared_bytes": declared_bytes,
        "extracted_regular_count": len(extracted),
        "extracted_regular_bytes": actual_bytes,
        "archive_sha_unchanged": True,
        "manifest_sha256": sha256_file(args.manifest),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
