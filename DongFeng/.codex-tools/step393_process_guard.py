#!/usr/bin/env python3
"""STEP393 race-safe adapter for the SHA-locked STEP377 process guard."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import types
from pathlib import Path


_STEP377_SHA256 = "8f4886838c39f96e662ff2a5b3d17c79c9ee01d76bfe826f4b19fb63a66e8199"
_STEP377_NAME = "step377_process_guard.py"
_SOURCE_LIMIT = 1024 * 1024
_IDENTITY_ATTEMPTS = 3
_UNSTABLE_MESSAGES = frozenset({
    "process identity changed while reading /proc",
    "proc entry identity changed",
})


class IdentitySnapshotUnstable(RuntimeError):
    """A live target-PGID member did not yield one exact identity snapshot."""


class ProcessStatSnapshotUnstable(RuntimeError):
    """A PID directory remained in an invalid Linux release-state snapshot."""


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        stat.S_IFMT(value.st_mode),
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _load_locked_guard() -> types.ModuleType:
    """Load STEP377 into a private module only after an fd-bound SHA check."""
    directory = Path(__file__).resolve(strict=True).parent
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    source_fd = -1
    try:
        before = os.stat(_STEP377_NAME, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _SOURCE_LIMIT:
            raise RuntimeError("STEP377 guard must be a bounded regular file")
        source_fd = os.open(
            _STEP377_NAME,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        opened = os.fstat(source_fd)
        chunks: list[bytes] = []
        total = 0
        digest = hashlib.sha256()
        while True:
            chunk = os.read(source_fd, 65536)
            if not chunk:
                break
            total += len(chunk)
            if total > _SOURCE_LIMIT:
                raise RuntimeError("STEP377 guard exceeds source limit")
            digest.update(chunk)
            chunks.append(chunk)
        closed = os.fstat(source_fd)
        after = os.stat(_STEP377_NAME, dir_fd=directory_fd, follow_symlinks=False)
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        os.close(directory_fd)
    if not _file_identity(before) == _file_identity(opened) == _file_identity(closed) == _file_identity(after):
        raise RuntimeError("STEP377 guard identity changed while loading")
    if digest.hexdigest() != _STEP377_SHA256:
        raise RuntimeError("STEP377 guard SHA256 mismatch")

    source = b"".join(chunks)
    module_name = f"_step393_locked_step377_{id(source):x}"
    if module_name in sys.modules:
        raise RuntimeError("isolated STEP377 module name collision")
    module = types.ModuleType(module_name)
    module.__file__ = str(directory / _STEP377_NAME)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


_base = _load_locked_guard()


def _exact_target_identity(host_pid: int, process_fd: int, pgid: int):
    try:
        identity = _base._identity_from_open_dir(host_pid, process_fd)
    except RuntimeError as error:
        if str(error) in _UNSTABLE_MESSAGES:
            raise IdentitySnapshotUnstable(
                f"process {host_pid} exact identity snapshot was unstable"
            ) from error
        raise
    if identity.pgid != pgid:
        raise RuntimeError(
            f"target process {host_pid} changed process group while reading /proc"
        )
    return identity


def _parse_prefilter_stat(data: bytes) -> tuple[bytes, int, int]:
    """Parse enough stat fields to reject unrelated PGIDs before strict identity checks."""
    end = data.rfind(b")")
    if end < 2 or end + 2 >= len(data):
        raise RuntimeError("malformed proc stat")
    fields = data[end + 2 :].split()
    if len(fields) < 20:
        raise RuntimeError("truncated proc stat")
    state = fields[0]
    if len(state) != 1:
        raise RuntimeError("invalid proc state field")
    try:
        observed_pgid = int(fields[2])
        starttime = int(fields[19])
    except ValueError as error:
        raise RuntimeError("invalid proc identity fields") from error
    return state, starttime, observed_pgid


def _prefilter_pgid(host_pid: int, process_fd: int, target_pgid: int) -> int:
    last_error: RuntimeError | None = None
    for _attempt in range(_IDENTITY_ATTEMPTS):
        try:
            state, starttime, observed_pgid = _parse_prefilter_stat(
                _base._read_at(process_fd, "stat", 65536)
            )
        except RuntimeError as error:
            if str(error) != "invalid proc identity fields":
                raise
            last_error = error
            continue
        if state in (b"Z", b"X"):
            raise ProcessLookupError("process is a zombie or dead")
        if observed_pgid != target_pgid:
            return observed_pgid
        if starttime <= 0:
            last_error = RuntimeError("invalid proc identity fields")
            continue
        return observed_pgid
    raise ProcessStatSnapshotUnstable(
        f"process {host_pid} stat remained in an invalid release state"
    ) from last_error


def _target_identity_from_open_dir(host_pid: int, process_fd: int, pgid: int):
    if _prefilter_pgid(host_pid, process_fd, pgid) != pgid:
        return None

    last_error: IdentitySnapshotUnstable | None = None
    for _attempt in range(_IDENTITY_ATTEMPTS):
        try:
            return _exact_target_identity(host_pid, process_fd, pgid)
        except IdentitySnapshotUnstable as error:
            last_error = error
    assert last_error is not None
    raise last_error


def authorized_group_snapshot(
    manifest: dict[str, object], observed: dict[tuple[int, int], object],
    launcher_seen: bool, case_path: Path, proc_root: Path = Path("/proc"),
) -> tuple[tuple[object, ...], bool]:
    """Freeze signal authority once the owned launcher disappears."""
    launcher_pid, launcher_starttime, pgid, port = _base.validate_ownership_manifest(manifest)
    raw = enumerate_group_identities(pgid, proc_root)
    launcher = next((item for item in raw if item.host_pid == launcher_pid), None)
    if launcher is not None:
        if (
            launcher.starttime != launcher_starttime
            or launcher.pgid != pgid
            or not _base._docker_launcher_argv(launcher.argv, port)
        ):
            raise RuntimeError("launcher ownership changed")
        launcher_seen = True
    elif not launcher_seen:
        case_token = os.fsencode(str(case_path.resolve(strict=False)))
        if any(_base._approved_group_member_argv(item.argv, case_token, port) for item in raw):
            raise RuntimeError(
                "ownership_unestablished: approved STEP377 process exists without launcher"
            )
        return tuple(
            item for item in raw if (item.host_pid, item.starttime) in observed
        ), False
    else:
        fixed_members = []
        for identity in raw:
            fixed = observed.get((identity.host_pid, identity.starttime))
            if fixed is None:
                raise RuntimeError("new process appeared after launcher disappearance")
            if fixed != identity:
                raise RuntimeError("observed owned identity changed")
            fixed_members.append(identity)
        return tuple(fixed_members), True

    approved = []
    case_token = os.fsencode(str(case_path.resolve(strict=False)))
    for identity in raw:
        key = (identity.host_pid, identity.starttime)
        fixed = observed.get(key)
        if fixed is not None:
            if fixed != identity:
                raise RuntimeError("observed owned identity changed")
            approved.append(identity)
            continue
        if (
            identity.starttime < launcher_starttime
            or not _base._approved_group_member_argv(identity.argv, case_token, port)
        ):
            raise RuntimeError("unapproved process in owned process group")
        observed[key] = identity
        approved.append(identity)
    return tuple(approved), launcher_seen


def enumerate_group_identities(
    pgid: int, proc_root: Path = Path("/proc")
) -> tuple[object, ...]:
    """Enumerate one PGID without fully reading unrelated process identities."""
    if type(pgid) is not int or pgid <= 1:
        raise ValueError("process group must be greater than 1")
    members = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal() or int(entry.name) <= 1:
            continue
        host_pid = int(entry.name)
        process_fd = -1
        try:
            process_fd = os.open(
                entry, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
            identity = _target_identity_from_open_dir(host_pid, process_fd, pgid)
            if identity is not None:
                members.append(identity)
        except (FileNotFoundError, ProcessLookupError):
            # Only ENOENT/ESRCH and explicit Z/X classification are gone.
            continue
        finally:
            if process_fd >= 0:
                os.close(process_fd)
    return tuple(sorted(members, key=lambda value: value.host_pid))


# STEP377 cleanup resolves this symbol in its own isolated module globals.  All
# pidfd, ownership, PGID, argv, aggregation, and stable-clear behavior remains
# the locked implementation; only its group enumeration primitive is replaced.
_base.enumerate_group_identities = enumerate_group_identities
_base.authorized_group_snapshot = authorized_group_snapshot

for _name in dir(_base):
    if not _name.startswith("_") and _name not in globals():
        globals()[_name] = getattr(_base, _name)


def __getattr__(name: str):
    return getattr(_base, name)


if __name__ == "__main__":
    raise SystemExit(_base.main())
