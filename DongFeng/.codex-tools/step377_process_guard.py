#!/usr/bin/env python3
"""Fail-closed process/NPU identity gates for the disarmed STEP377 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import shlex
import socket
import stat
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


BACK8_PAIRS = frozenset((physical, chip) for physical in range(4, 8) for chip in range(2))
BACK8_DEVICE_IDS = frozenset(range(8, 16))
_PROCESS_ROW = re.compile(r"^\|\s*([0-7])\s+([01])\s*\|\s*([1-9][0-9]*)\s*\|.*\|\s*$")
_PROCESS_HEADER = re.compile(
    r"^\|\s*NPU\s+Chip\s*\|\s*Process\s+id\s*\|\s*Process\s+name\s*\|.*\|\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_IDLE_SENTINEL = re.compile(r"^\|\s*No running processes found in NPU\s+([0-7])\s*\|\s*$")
_LAUNCH_FLAGS = (b"--port", b"--output-dir", b"--input-dir", b"--shadow-root", b"--installed-custom-opp")


@dataclass(frozen=True, order=True)
class NpuRow:
    physical: int
    chip: int
    host_pid: int

    @property
    def device_id(self) -> int:
        return self.physical * 2 + self.chip


@dataclass(frozen=True)
class ProcessIdentity:
    host_pid: int
    starttime: int
    nspid: tuple[int, ...]
    pgid: int
    argv: tuple[bytes, ...] = ()

    @property
    def container_pid(self) -> int:
        return self.nspid[-1]


@dataclass(frozen=True)
class RankBinding:
    rank: int
    local_rank: int
    host_pid: int
    container_pid: int
    physical: int
    chip: int
    device_id: int
    starttime: int
    pgid: int
    nspid: tuple[int, ...]
    argv: tuple[str, ...]


def _parse_process_table(text: str) -> tuple[tuple[NpuRow, ...], frozenset[int]]:
    header = _PROCESS_HEADER.search(text)
    if header is None:
        raise RuntimeError("npu-smi process table header missing or changed")
    rows: list[NpuRow] = []
    idle_devices: list[int] = []
    live_devices: set[int] = set()
    for line in text[header.end():].splitlines():
        stripped = line.strip()
        if stripped.startswith("+"):
            continue
        if not stripped.startswith("|"):
            continue
        match = _PROCESS_ROW.fullmatch(line)
        if match is None:
            idle = _IDLE_SENTINEL.fullmatch(line)
            if idle is None:
                raise RuntimeError(f"malformed npu-smi process-table data row: {line!r}")
            idle_devices.append(int(idle.group(1)))
            continue
        row = NpuRow(*(int(value) for value in match.groups()))
        live_devices.add(row.physical)
        if (row.physical, row.chip) in BACK8_PAIRS:
            rows.append(row)
    if len(idle_devices) != len(set(idle_devices)):
        raise RuntimeError("duplicate npu-smi idle sentinel")
    if live_devices.intersection(idle_devices):
        raise RuntimeError("npu-smi physical device is both live and idle")
    return tuple(sorted(rows)), frozenset(idle_devices)


def _parse_target_rows(text: str) -> tuple[NpuRow, ...]:
    return _parse_process_table(text)[0]


def parse_back8_strict(text: str) -> tuple[NpuRow, ...]:
    """Parse one live snapshot requiring exactly one process on every back8 die."""
    rows = _parse_target_rows(text)
    if len(rows) != 8:
        raise RuntimeError(f"back8 row count mismatch: {len(rows)}")
    if frozenset((row.physical, row.chip) for row in rows) != BACK8_PAIRS:
        raise RuntimeError("back8 pair set mismatch")
    if len({row.host_pid for row in rows}) != 8:
        raise RuntimeError("back8 host PIDs are not unique")
    return rows


def parse_back8_idle(text: str) -> tuple[NpuRow, ...]:
    """Parse one idle snapshot, failing if any back8 process row exists."""
    rows, idle_devices = _parse_process_table(text)
    if rows:
        raise RuntimeError(f"back8 is not idle: {rows}")
    if not set(range(4, 8)).issubset(idle_devices):
        raise RuntimeError("back8 idle sentinel set mismatch")
    return rows


def parse_stat_identity_state(data: bytes) -> tuple[bytes, int, int]:
    """Read Linux /proc/PID/stat state, pgrp, and starttime."""
    end = data.rfind(b")")
    if end < 2 or end + 2 >= len(data):
        raise RuntimeError("malformed proc stat")
    fields = data[end + 2 :].split()
    if len(fields) < 20:
        raise RuntimeError("truncated proc stat")
    state = fields[0]
    if len(state) != 1:
        raise RuntimeError("invalid proc state field")
    pgrp = int(fields[2])  # suffix begins at field 3
    starttime = int(fields[19])
    if state not in (b"Z", b"X") and (pgrp <= 1 or starttime <= 0):
        raise RuntimeError("invalid proc identity fields")
    return state, starttime, pgrp


def parse_stat_identity(data: bytes) -> tuple[int, int]:
    """Read Linux /proc/PID/stat fields 5 (pgrp) and 22 (starttime)."""
    _state, starttime, pgrp = parse_stat_identity_state(data)
    if pgrp <= 1 or starttime <= 0:
        raise RuntimeError("invalid proc identity fields")
    return starttime, pgrp


def parse_stat_starttime(data: bytes) -> int:
    return parse_stat_identity(data)[0]


def parse_nspid(data: bytes, host_pid: int) -> tuple[int, ...]:
    horizontal = rb"[ \t\v\f\r]"
    matches = re.findall(
        rb"^NSpid:" + horizontal + rb"+([0-9]+(?:" + horizontal
        + rb"+[0-9]+)*)" + horizontal + rb"*$",
        data,
        re.MULTILINE,
    )
    if len(matches) != 1:
        raise RuntimeError("NSpid missing or duplicated")
    chain = tuple(int(value) for value in matches[0].split())
    if not chain or chain[0] != host_pid or any(value <= 1 for value in chain):
        raise RuntimeError("invalid NSpid chain")
    return chain


def _read_at(directory_fd: int, name: str, limit: int) -> bytes:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError("proc entry is not regular")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise RuntimeError("proc entry exceeds limit")
            chunks.append(chunk)
        closed = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (closed.st_dev, closed.st_ino):
            raise RuntimeError("proc entry identity changed")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _identity_from_open_dir(host_pid: int, process_fd: int) -> ProcessIdentity:
    state_a, start_a, pgid_a = parse_stat_identity_state(_read_at(process_fd, "stat", 65536))
    if state_a in (b"Z", b"X"):
        raise ProcessLookupError("process is a zombie or dead")
    nspid = parse_nspid(_read_at(process_fd, "status", 1048576), host_pid)
    argv_a = tuple(part for part in _read_at(process_fd, "cmdline", 1048576).split(b"\0") if part)
    argv_b = tuple(part for part in _read_at(process_fd, "cmdline", 1048576).split(b"\0") if part)
    state_b, start_b, pgid_b = parse_stat_identity_state(_read_at(process_fd, "stat", 65536))
    if state_b in (b"Z", b"X"):
        raise ProcessLookupError("process became a zombie or dead")
    if (start_a, pgid_a, argv_a) != (start_b, pgid_b, argv_b):
        raise RuntimeError("process identity changed while reading /proc")
    return ProcessIdentity(host_pid, start_a, nspid, pgid_a, argv_a)


def _matching_identity_from_open_dir(host_pid: int, process_fd: int, case_path: bytes,
                                     port: int) -> ProcessIdentity | None:
    """Classify argv before touching volatile identity fields, using one PID dirfd."""
    classified_argv = tuple(part for part in _read_at(process_fd, "cmdline", 1048576).split(b"\0") if part)
    if not _argv_matches(classified_argv, case_path, port):
        return None
    identity = _identity_from_open_dir(host_pid, process_fd)
    if identity.argv != classified_argv:
        raise RuntimeError("matching process argv changed while reading /proc")
    return identity


def _approved_identity_from_open_dir(host_pid: int, process_fd: int, case_path: bytes,
                                     port: int) -> ProcessIdentity | None:
    """Skip unrelated processes before reading their volatile identity fields."""
    classified_argv = tuple(part for part in _read_at(process_fd, "cmdline", 1048576).split(b"\0") if part)
    if not _approved_group_member_argv(classified_argv, case_path, port):
        return None
    identity = _identity_from_open_dir(host_pid, process_fd)
    if identity.argv != classified_argv:
        raise RuntimeError("approved process argv changed while reading /proc")
    return identity


def _rank_worker_identity_from_open_dir(host_pid: int, process_fd: int) -> ProcessIdentity | None:
    """Return only a live STEP377 rank worker; launchers never gain rank authority."""
    classified_argv = tuple(part for part in _read_at(process_fd, "cmdline", 1048576).split(b"\0") if part)
    if not _worker_argv(classified_argv):
        return None
    identity = _identity_from_open_dir(host_pid, process_fd)
    if identity.argv != classified_argv:
        raise RuntimeError("rank worker argv changed while reading /proc")
    return identity


def read_process_identity(host_pid: int, proc_root: Path = Path("/proc")) -> ProcessIdentity:
    if type(host_pid) is not int or host_pid <= 1:
        raise ValueError("host PID must be greater than 1")
    root_fd = os.open(proc_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    process_fd = -1
    try:
        process_fd = os.open(str(host_pid), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        return _identity_from_open_dir(host_pid, process_fd)
    finally:
        if process_fd >= 0:
            os.close(process_fd)
        os.close(root_fd)


def stable_back8_binding(
    ready_rows: Sequence[dict[str, object]],
    sample: Callable[[], str],
    identity_reader: Callable[[int], ProcessIdentity] = read_process_identity,
) -> dict[str, object]:
    """Bracket /proc identities with two identical npu-smi snapshots."""
    text_a = sample()
    rows_a = parse_back8_strict(text_a)
    identities = {row.host_pid: identity_reader(row.host_pid) for row in rows_a}
    text_b = sample()
    rows_b = parse_back8_strict(text_b)
    if rows_a != rows_b:
        raise RuntimeError("npu-smi mapping changed between samples")
    for pid, identity in identities.items():
        if identity_reader(pid) != identity:
            raise RuntimeError("process identity changed between samples")

    if len(ready_rows) != 8:
        raise RuntimeError("ready rank count mismatch")
    by_rank: dict[int, dict[str, object]] = {}
    for row in ready_rows:
        if not {"rank", "local_rank", "container_pid"}.issubset(row):
            raise RuntimeError("ready row schema mismatch")
        rank = row["rank"]
        local_rank = row["local_rank"]
        container_pid = row["container_pid"]
        if not all(type(value) is int for value in (rank, local_rank, container_pid)):
            raise RuntimeError("ready row types mismatch")
        if rank in by_rank:
            raise RuntimeError("duplicate rank")
        by_rank[rank] = row
    if set(by_rank) != set(range(8)):
        raise RuntimeError("rank set mismatch")
    by_container = {identity.container_pid: identity for identity in identities.values()}
    if len(by_container) != 8 or set(by_container) != {int(row["container_pid"]) for row in ready_rows}:
        raise RuntimeError("host/container PID mapping is not a bijection")
    by_host = {row.host_pid: row for row in rows_a}
    bindings: list[RankBinding] = []
    for rank in range(8):
        ready = by_rank[rank]
        local_rank = int(ready["local_rank"])
        if local_rank != rank:
            raise RuntimeError("rank/local_rank mapping mismatch")
        identity = by_container[int(ready["container_pid"])]
        npu = by_host[identity.host_pid]
        if not _worker_argv(identity.argv):
            raise RuntimeError("rank worker argv contract mismatch")
        if npu.device_id != 8 + local_rank:
            raise RuntimeError("rank/device mapping mismatch")
        bindings.append(RankBinding(rank, local_rank, identity.host_pid, identity.container_pid,
                                    npu.physical, npu.chip, npu.device_id,
                                    identity.starttime, identity.pgid, identity.nspid,
                                    tuple(os.fsdecode(item) for item in identity.argv)))
    return {
        "schema": "step377-back8-binding-v1",
        "sample_sha256": [hashlib.sha256(value.encode()).hexdigest() for value in (text_a, text_b)],
        "bindings": [asdict(binding) for binding in bindings],
    }


def _host_case_argv(argv: Sequence[bytes], case_path: bytes, port: int) -> bool:
    if len(argv) != 12 or Path(os.fsdecode(argv[0])).name != "python3":
        return False
    if argv[1] not in (case_path, os.fsencode(Path(os.fsdecode(case_path)).name)):
        return False
    return (tuple(argv[index] for index in range(2, 12, 2)) == _LAUNCH_FLAGS
            and argv[3] == str(port).encode() and all(argv[index] for index in range(3, 12, 2)))


def _docker_launcher_argv(argv: Sequence[bytes], port: int) -> bool:
    fixed = (b"timeout", b"--signal=TERM", b"--kill-after=30s", b"900s", b"docker", b"exec")
    if tuple(argv[:6]) != fixed or tuple(argv[-6:-1]) != (
        b"mapqr-leicheng", b"bash", b"--noprofile", b"--norc", b"-lc"
    ):
        return False
    env = argv[6:-6]
    if len(env) not in (8, 10) or any(env[index] != b"-e" for index in range(0, len(env), 2)):
        return False
    variables = {item.split(b"=", 1)[0]: item.split(b"=", 1)[1] for item in env[1::2] if b"=" in item}
    required = {b"ASCEND_RT_VISIBLE_DEVICES", b"ASCEND_CUSTOM_OPP_PATH", b"PYTHONPATH", b"TORCH_DEVICE_BACKEND_AUTOLOAD"}
    if (len(variables) != len(env) // 2 or not required.issubset(variables)
            or variables[b"ASCEND_RT_VISIBLE_DEVICES"] != b"8,9,10,11,12,13,14,15"
            or variables[b"TORCH_DEVICE_BACKEND_AUTOLOAD"] != b"0"
            or (set(variables) - required) not in (set(), {b"STEP358_STATE_DIAGNOSTIC_ONLY"})
            or variables.get(b"STEP358_STATE_DIAGNOSTIC_ONLY", b"1") != b"1"):
        return False
    try:
        inner = tuple(os.fsencode(item) for item in shlex.split(os.fsdecode(argv[-1])))
    except ValueError:
        return False
    optional = inner[-1:] == (b"--first-profiled-only",)
    core = inner[:-1] if optional else inner
    return (len(core) == 14 and core[:5] == (
        b"torchrun", b"--nnodes=1", b"--nproc-per-node=8", b"--master-addr=127.0.0.1",
        b"--master-port=" + str(port).encode())
        and Path(os.fsdecode(core[5])).name == "step377_diagnostic_math_worker.py"
        and core[6::2] == (b"--input-dir", b"--output-dir", b"--shadow-root", b"--installed-custom-opp")
        and all(core[index] for index in (7, 9, 11, 13)))


def _worker_argv(argv: Sequence[bytes]) -> bool:
    optional = argv[-1:] == (b"--first-profiled-only",)
    core = argv[:-1] if optional else argv
    unbuffered = core[1:2] == (b"-u",)
    script_index = 2 if unbuffered else 1
    flags_index = script_index + 1
    return (len(core) == 10 + int(unbuffered)
            and core.count(b"-u") == int(unbuffered)
            and Path(os.fsdecode(core[0])).name.startswith("python")
            and Path(os.fsdecode(core[script_index])).name == "step377_diagnostic_math_worker.py"
            and core[flags_index::2] == (b"--input-dir", b"--output-dir", b"--shadow-root", b"--installed-custom-opp")
            and all(core[index] for index in range(flags_index + 1, len(core), 2)))


def _approved_group_member_argv(argv: Sequence[bytes], case_path: bytes, port: int) -> bool:
    docker_child = tuple(argv[:2]) == (b"docker", b"exec") and _docker_launcher_argv(
        (b"timeout", b"--signal=TERM", b"--kill-after=30s", b"900s", *argv), port
    )
    return (_docker_launcher_argv(argv, port) or docker_child or _host_case_argv(argv, case_path, port)
            or _worker_argv(argv))


def _argv_matches(argv: Sequence[bytes], case_path: bytes, port: int) -> bool:
    return _host_case_argv(argv, case_path, port) or _docker_launcher_argv(argv, port)


def scan_case_processes_once(case_path: Path, port: int, expected_pgid: int | None,
                             proc_root: Path = Path("/proc")) -> tuple[ProcessIdentity, ...]:
    exact_path = os.fsencode(str(case_path.resolve(strict=False)))
    found: list[ProcessIdentity] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal() or int(entry.name) <= 1:
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            directory_fd = os.open(entry, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except (FileNotFoundError, ProcessLookupError):
            continue
        try:
            identity = _matching_identity_from_open_dir(pid, directory_fd, exact_path, port)
            if identity is not None:
                if expected_pgid is not None and identity.pgid != expected_pgid:
                    raise RuntimeError("matching launcher has unexpected process group")
                found.append(identity)
        except (FileNotFoundError, ProcessLookupError):
            continue
        finally:
            os.close(directory_fd)
    return tuple(sorted(found, key=lambda row: row.host_pid))


def stable_case_process_scan(case_path: Path, port: int, expected_pgid: int | None,
                             proc_root: Path = Path("/proc")) -> tuple[ProcessIdentity, ...]:
    first = scan_case_processes_once(case_path, port, expected_pgid, proc_root)
    second = scan_case_processes_once(case_path, port, expected_pgid, proc_root)
    if first != second:
        raise RuntimeError("case process scan changed between samples")
    return first


def enumerate_group_identities(pgid: int, proc_root: Path = Path("/proc")) -> tuple[ProcessIdentity, ...]:
    if type(pgid) is not int or pgid <= 1:
        raise ValueError("process group must be greater than 1")
    members: list[ProcessIdentity] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal() or int(entry.name) <= 1:
            continue
        pid = int(entry.name)
        try:
            directory_fd = os.open(entry, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                identity = _identity_from_open_dir(pid, directory_fd)
            finally:
                os.close(directory_fd)
            if identity.pgid == pgid:
                members.append(identity)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except RuntimeError as error:
            # During task release Linux can expose a final stat row whose pgrp
            # is no longer a usable identity. It cannot establish membership
            # in the owned group and must never gain signal authority.
            if str(error) != "invalid proc identity fields":
                raise
    return tuple(sorted(members, key=lambda value: value.host_pid))


def validate_ownership_manifest(manifest: dict[str, object]) -> tuple[int, int, int, int]:
    required = {"schema", "port", "launcher_host_pid", "launcher_starttime", "launcher_pgid"}
    if (set(manifest) != required or manifest.get("schema") != "step358-launcher-ownership-v1"
            or any(type(manifest[key]) is not int for key in required - {"schema"})):
        raise RuntimeError("ownership manifest schema mismatch")
    launcher_pid = int(manifest["launcher_host_pid"])
    launcher_starttime = int(manifest["launcher_starttime"])
    pgid = int(manifest["launcher_pgid"])
    port = int(manifest["port"])
    if launcher_pid <= 1 or launcher_starttime <= 0 or pgid != launcher_pid or not 1 <= port <= 65535:
        raise RuntimeError("ownership manifest identity mismatch")
    return launcher_pid, launcher_starttime, pgid, port


def _identities_from_ownership(manifest: dict[str, object],
                               proc_root: Path = Path("/proc")) -> tuple[ProcessIdentity, ...]:
    """Take one stable-per-member group snapshot; cross-snapshot changes are expected."""
    launcher_pid, launcher_starttime, pgid, port = validate_ownership_manifest(manifest)
    members = enumerate_group_identities(pgid, proc_root)
    launcher = next((item for item in members if item.host_pid == launcher_pid), None)
    if launcher is None:
        return members
    if (launcher.starttime != launcher_starttime or launcher.pgid != pgid
            or not _docker_launcher_argv(launcher.argv, port)):
        raise RuntimeError("launcher ownership changed")
    return members


def authorized_group_snapshot(
    manifest: dict[str, object], observed: dict[tuple[int, int], ProcessIdentity],
    launcher_seen: bool, case_path: Path, proc_root: Path = Path("/proc"),
) -> tuple[tuple[ProcessIdentity, ...], bool]:
    launcher_pid, launcher_starttime, pgid, port = validate_ownership_manifest(manifest)
    raw = enumerate_group_identities(pgid, proc_root)
    launcher = next((item for item in raw if item.host_pid == launcher_pid), None)
    if launcher is not None:
        if (launcher.starttime != launcher_starttime or launcher.pgid != pgid
                or not _docker_launcher_argv(launcher.argv, port)):
            raise RuntimeError("launcher ownership changed")
        launcher_seen = True
    elif not launcher_seen:
        # A reused PGID is not cleanup authority. Only previously fixed identities may survive this point.
        case_token = os.fsencode(str(case_path.resolve(strict=False)))
        if any(_approved_group_member_argv(item.argv, case_token, port) for item in raw):
            raise RuntimeError("ownership_unestablished: approved STEP377 process exists without launcher")
        return tuple(item for item in raw if (item.host_pid, item.starttime) in observed), False
    approved: list[ProcessIdentity] = []
    case_token = os.fsencode(str(case_path.resolve(strict=False)))
    for identity in raw:
        key = (identity.host_pid, identity.starttime)
        fixed = observed.get(key)
        if fixed is not None:
            if fixed != identity:
                raise RuntimeError("observed owned identity changed")
            approved.append(identity)
            continue
        if (identity.starttime < launcher_starttime
                or not _approved_group_member_argv(identity.argv, case_token, port)):
            raise RuntimeError("unapproved process in owned process group")
        observed[key] = identity
        approved.append(identity)
    return tuple(approved), launcher_seen


def assert_stable_clear(sample: Callable[[], str], case_path: Path, port: int, expected_pgid: int | None,
                        proc_root: Path = Path("/proc")) -> dict[str, object]:
    """Require two idle NPU samples and two empty exact case-process scans."""
    text_a = sample()
    parse_back8_idle(text_a)
    processes = stable_case_process_scan(case_path, port, expected_pgid, proc_root)
    text_b = sample()
    parse_back8_idle(text_b)
    if processes:
        raise RuntimeError(f"owned case processes remain: {processes}")
    return {
        "schema": "step377-stable-clear-v1",
        "back8_process_count": 0,
        "case_process_count": 0,
        "sample_sha256": [hashlib.sha256(text.encode()).hexdigest() for text in (text_a, text_b)],
    }


def signal_owned_pidfd(identity: ProcessIdentity, signum: int,
                       identity_reader: Callable[[int], ProcessIdentity] = read_process_identity) -> None:
    """Signal one owned process only through pidfd after signal-time revalidation."""
    if signum not in (signal.SIGTERM, signal.SIGKILL):
        raise ValueError("only TERM/KILL are allowed")
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if pidfd_open is None or pidfd_send_signal is None:
        raise RuntimeError("pidfd signaling unavailable; refusing PID-based fallback")
    descriptor = pidfd_open(identity.host_pid, 0)
    try:
        current = identity_reader(identity.host_pid)
        if current != identity:
            raise RuntimeError("owned process identity changed after pidfd_open")
        revalidated = identity_reader(identity.host_pid)
        if revalidated != identity:
            raise RuntimeError("owned process identity changed before pidfd signal")
        pidfd_send_signal(descriptor, signum, None, 0)
    finally:
        os.close(descriptor)


def owned_identity_alive(identity: ProcessIdentity,
                         identity_reader: Callable[[int], ProcessIdentity] = read_process_identity) -> bool:
    try:
        current = identity_reader(identity.host_pid)
    except (FileNotFoundError, ProcessLookupError):
        return False
    if current == identity:
        return True
    if current.starttime == identity.starttime:
        raise RuntimeError("owned process changed identity without exiting")
    return False


def terminate_owned(identities: Iterable[ProcessIdentity], alive: Callable[[ProcessIdentity], bool],
                    *, grace_seconds: float = 5.0,
                    signaler: Callable[[ProcessIdentity, int], None] = signal_owned_pidfd,
                    monotonic: Callable[[], float] = time.monotonic,
                    sleeper: Callable[[float], None] = time.sleep) -> None:
    """Best-effort TERM, bounded wait, then KILL; aggregate non-disappearance errors."""
    if not isinstance(grace_seconds, (int, float)) or not math.isfinite(grace_seconds) or grace_seconds < 0:
        raise ValueError("grace_seconds must be finite and non-negative")
    owned = tuple(identities)
    errors: list[str] = []
    for identity in owned:
        try:
            if alive(identity):
                signaler(identity, signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError):
            pass
        except Exception as error:
            errors.append(f"TERM pid={identity.host_pid}: {error}")
    if grace_seconds:
        sleeper(min(0.01, grace_seconds))
    deadline = monotonic() + grace_seconds
    while monotonic() < deadline:
        survivors = False
        for identity in owned:
            try:
                survivors |= alive(identity)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except Exception as error:
                marker = f"WAIT pid={identity.host_pid}: {error}"
                if marker not in errors:
                    errors.append(marker)
        if not survivors:
            break
        sleeper(min(0.05, max(0.0, deadline - monotonic())))
    for identity in owned:
        try:
            if alive(identity):
                signaler(identity, signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError):
            pass
        except Exception as error:
            errors.append(f"KILL pid={identity.host_pid}: {error}")
    if grace_seconds:
        sleeper(min(0.01, grace_seconds))
    final_deadline = monotonic() + grace_seconds
    while monotonic() < final_deadline:
        remaining = []
        for identity in owned:
            try:
                if alive(identity):
                    remaining.append(identity)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except Exception as error:
                marker = f"VERIFY pid={identity.host_pid}: {error}"
                if marker not in errors:
                    errors.append(marker)
        if not remaining:
            break
        sleeper(min(0.05, max(0.0, final_deadline - monotonic())))
    for identity in owned:
        try:
            if alive(identity):
                errors.append(f"SURVIVED pid={identity.host_pid}")
        except (FileNotFoundError, ProcessLookupError):
            pass
        except Exception as error:
            errors.append(f"VERIFY pid={identity.host_pid}: {error}")
    if errors:
        raise RuntimeError("owned cleanup errors: " + "; ".join(errors))


def safe_group_cleanup(manifest: dict[str, object], *, proc_root: Path = Path("/proc"),
                       grace_seconds: float = 5.0,
                       max_rounds: int = 1024,
                       case_path: Path | None = None,
                       signaler: Callable[[ProcessIdentity, int], None] = signal_owned_pidfd,
                       group_reader: Callable[[dict[str, object], Path], tuple[ProcessIdentity, ...]] | None = None,
                       alive: Callable[[ProcessIdentity], bool] = owned_identity_alive,
                       monotonic: Callable[[], float] = time.monotonic,
                       sleeper: Callable[[float], None] = time.sleep) -> dict[str, object]:
    """Clean a changing owned pgid, requiring two final empty enumerations.

    The caller must subsequently run ``assert_stable_clear`` so device and exact
    launcher observations independently confirm the postcondition.
    """
    if not isinstance(grace_seconds, (int, float)) or not math.isfinite(grace_seconds) or grace_seconds < 0:
        raise ValueError("grace_seconds must be finite and non-negative")
    if type(max_rounds) is not int or max_rounds < 2:
        raise ValueError("max_rounds must be an integer greater than one")
    validate_ownership_manifest(manifest)
    errors: list[str] = []
    term_seen: set[tuple[int, int]] = set()
    observed: dict[tuple[int, int], ProcessIdentity] = {}
    launcher_seen = False
    approved_case = case_path if case_path is not None else Path("step377_diagnostic_host_case.py")
    def read_group() -> tuple[ProcessIdentity, ...]:
        nonlocal launcher_seen
        if group_reader is not None:
            return group_reader(manifest, proc_root)
        members, launcher_seen = authorized_group_snapshot(
            manifest, observed, launcher_seen, approved_case, proc_root
        )
        return members
    deadline = monotonic() + float(grace_seconds)
    term_rounds = 0
    while term_rounds < max_rounds:
        term_rounds += 1
        try:
            members = read_group()
        except Exception as error:
            if "ownership_unestablished" in str(error):
                raise
            raise RuntimeError("owned group TERM snapshot failed") from error
        for identity in members:
            key = (identity.host_pid, identity.starttime)
            observed[key] = identity
            if key in term_seen:
                continue
            term_seen.add(key)
            try:
                signaler(identity, signal.SIGTERM)
            except (FileNotFoundError, ProcessLookupError):
                pass
            except Exception as error:
                errors.append(f"TERM pid={identity.host_pid}: {error}")
        if not members or monotonic() >= deadline:
            break
        sleeper(min(0.05, max(0.0, deadline - monotonic())))

    else:
        errors.append("TERM group enumeration exceeded max_rounds")

    kill_deadline = monotonic() + float(grace_seconds)
    empty_count = 0
    post_kill_scan_required = False
    for kill_round in range(1, max_rounds + 1):
        try:
            members = read_group()
        except Exception as error:
            if "ownership_unestablished" in str(error):
                raise
            raise RuntimeError("owned group KILL snapshot failed") from error
        if not members:
            empty_count += 1
            if empty_count == 2:
                break
        else:
            empty_count = 0
            if monotonic() >= kill_deadline and post_kill_scan_required:
                errors.extend(f"SURVIVED pid={identity.host_pid}" for identity in members)
                break
            for identity in members:
                observed[(identity.host_pid, identity.starttime)] = identity
                try:
                    signaler(identity, signal.SIGKILL)
                except (FileNotFoundError, ProcessLookupError):
                    pass
                except Exception as error:
                    errors.append(f"KILL pid={identity.host_pid}: {error}")
            post_kill_scan_required = True
            remaining = max(0.0, kill_deadline - monotonic())
            sleeper(min(0.05, remaining))
            continue
        post_kill_scan_required = False
        if monotonic() >= kill_deadline and empty_count == 0:
            errors.append("final empty process-group state was not observed twice")
            break
        sleeper(min(0.05, max(0.0, kill_deadline - monotonic())))
    else:
        errors.append("KILL group enumeration exceeded max_rounds")
    for identity in observed.values():
        try:
            if alive(identity):
                errors.append(f"IDENTITY_SURVIVED pid={identity.host_pid}")
        except (FileNotFoundError, ProcessLookupError):
            pass
        except Exception as error:
            errors.append(f"IDENTITY_VERIFY pid={identity.host_pid}: {error}")
    if errors:
        raise RuntimeError("owned group cleanup errors: " + "; ".join(errors))
    return {
        "schema": "step377-owned-group-clean-v1",
        "member_count": 0,
        "consecutive_empty_group_scans": 2,
        "external_stable_clear_required": True,
    }


def _read_bounded_json_sha(path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"expected {label} SHA256 is invalid")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_size > 65536:
        raise RuntimeError(f"{label} must be a bounded regular file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        digest = hashlib.sha256(); chunks = []; total = 0
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            total += len(chunk)
            if total > 65536:
                raise RuntimeError(f"{label} exceeds limit")
            digest.update(chunk); chunks.append(chunk)
        data = b"".join(chunks)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, stat.S_IFMT(value.st_mode),
                              value.st_mtime_ns, value.st_ctime_ns)
    if len(data) > 65536 or not identity(before) == identity(opened) == identity(closed) == identity(after):
        raise RuntimeError(f"{label} identity changed")
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError(f"{label} SHA256 mismatch")
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} JSON must be an object")
    return value


def read_ownership_json(path: Path, expected_sha256: str) -> dict[str, object]:
    value = _read_bounded_json_sha(path, expected_sha256, "ownership")
    validate_ownership_manifest(value)
    return value


def validate_rank_ownership_manifest(manifest: dict[str, object], *,
                                     expected_launcher_sha256: str, case_path: Path,
                                     port: int) -> tuple[ProcessIdentity, ...]:
    required = {"schema", "launcher_ownership_sha256", "gate_token_sha256",
                "case_path", "port", "ranks"}
    if set(manifest) != required or manifest.get("schema") != "step377-rank-ownership-v1":
        raise RuntimeError("rank ownership schema mismatch")
    for key in ("launcher_ownership_sha256", "gate_token_sha256"):
        if not isinstance(manifest.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", manifest[key]):
            raise RuntimeError(f"rank ownership {key} mismatch")
    if manifest["launcher_ownership_sha256"] != expected_launcher_sha256:
        raise RuntimeError("rank ownership launcher SHA256 mismatch")
    if manifest.get("case_path") != str(case_path.resolve(strict=False)) or manifest.get("port") != port:
        raise RuntimeError("rank ownership case/port mismatch")
    ranks = manifest.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 8:
        raise RuntimeError("rank ownership rank count mismatch")
    exact = {"rank", "local_rank", "host_pid", "container_pid", "physical", "chip",
             "device_id", "starttime", "pgid", "nspid", "argv"}
    identities: list[ProcessIdentity] = []
    for item in ranks:
        if not isinstance(item, dict) or set(item) != exact:
            raise RuntimeError("rank ownership row schema mismatch")
        integer_keys = exact - {"nspid", "argv"}
        if any(type(item[key]) is not int for key in integer_keys):
            raise RuntimeError("rank ownership row integer mismatch")
        nspid, argv = item["nspid"], item["argv"]
        if (not isinstance(nspid, list) or any(type(value) is not int for value in nspid)
                or not isinstance(argv, list) or any(type(value) is not str for value in argv)):
            raise RuntimeError("rank ownership row sequence mismatch")
        encoded = tuple(os.fsencode(value) for value in argv)
        identity = ProcessIdentity(item["host_pid"], item["starttime"], tuple(nspid),
                                   item["pgid"], encoded)
        if (item["rank"] != item["local_rank"] or item["device_id"] != 8 + item["rank"]
                or item["physical"] * 2 + item["chip"] != item["device_id"]
                or identity.host_pid <= 1 or identity.starttime <= 0 or identity.pgid <= 1
                or not identity.nspid or identity.nspid[0] != identity.host_pid
                or identity.container_pid != item["container_pid"] or not _worker_argv(encoded)):
            raise RuntimeError("rank ownership row identity mismatch")
        identities.append(identity)
    if ({item["rank"] for item in ranks} != set(range(8))
            or len({(item.host_pid, item.starttime) for item in identities}) != 8
            or len({item.container_pid for item in identities}) != 8):
        raise RuntimeError("rank ownership is not a strict bijection")
    return tuple(sorted(identities, key=lambda item: item.container_pid))


def read_rank_ownership_json(path: Path, expected_sha256: str, *,
                             expected_launcher_sha256: str, case_path: Path,
                             port: int) -> tuple[dict[str, object], tuple[ProcessIdentity, ...]]:
    value = _read_bounded_json_sha(path, expected_sha256, "rank ownership")
    identities = validate_rank_ownership_manifest(
        value, expected_launcher_sha256=expected_launcher_sha256, case_path=case_path, port=port
    )
    return value, identities


def approved_step377_processes(case_path: Path, port: int,
                               proc_root: Path = Path("/proc")) -> tuple[ProcessIdentity, ...]:
    case_token = os.fsencode(str(case_path.resolve(strict=False)))
    found = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal() or int(entry.name) <= 1:
            continue
        pid = int(entry.name)
        try:
            directory_fd = os.open(entry, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except (FileNotFoundError, ProcessLookupError):
            continue
        try:
            identity = _approved_identity_from_open_dir(pid, directory_fd, case_token, port)
            if identity is not None:
                found.append(identity)
        except (FileNotFoundError, ProcessLookupError):
            continue
        finally:
            os.close(directory_fd)
    return tuple(sorted(found, key=lambda item: item.host_pid))


def approved_step377_rank_workers(proc_root: Path = Path("/proc")) -> tuple[ProcessIdentity, ...]:
    """Find rank workers for no-manifest fail-closed checks, without signaling them."""
    found = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal() or int(entry.name) <= 1:
            continue
        pid = int(entry.name)
        try:
            directory_fd = os.open(entry, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except (FileNotFoundError, ProcessLookupError):
            continue
        try:
            identity = _rank_worker_identity_from_open_dir(pid, directory_fd)
            if identity is not None:
                found.append(identity)
        except (FileNotFoundError, ProcessLookupError):
            continue
        finally:
            os.close(directory_fd)
    return tuple(sorted(found, key=lambda item: item.host_pid))


def npu_smi_sample() -> str:
    return subprocess.run(["npu-smi", "info"], check=True, capture_output=True,
                          text=True, timeout=40).stdout


def assert_port_free(port: int) -> None:
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port out of range")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        handle.bind(("127.0.0.1", port))


def cleanup_owned_protocol(ownership_path: Path, expected_ownership_sha256: str,
                           case_path: Path, port: int,
                           grace_seconds: float, proc_root: Path = Path("/proc"),
                           rank_ownership_path: Path | None = None,
                           expected_rank_ownership_sha256: str | None = None,
                           expected_gate_token_sha256: str | None = None) -> dict[str, object]:
    ownership = read_ownership_json(ownership_path, expected_ownership_sha256)
    _pid, _start, pgid, manifest_port = validate_ownership_manifest(ownership)
    if manifest_port != port:
        raise RuntimeError("ownership port mismatch")
    if (rank_ownership_path is None) != (expected_rank_ownership_sha256 is None):
        raise RuntimeError("rank ownership path and SHA256 must be supplied together")
    if expected_gate_token_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_gate_token_sha256):
        raise ValueError("expected gate token SHA256 is invalid")
    if expected_gate_token_sha256 is not None and rank_ownership_path is None:
        raise RuntimeError("expected gate token SHA256 requires rank ownership")
    errors: list[str] = []
    rank_cleanup = None
    cleanup = None
    clear = None
    port_free = False
    try:
        if rank_ownership_path is not None:
            rank_manifest, identities = read_rank_ownership_json(
                rank_ownership_path, expected_rank_ownership_sha256,
                expected_launcher_sha256=expected_ownership_sha256, case_path=case_path, port=port
            )
            if (expected_gate_token_sha256 is not None
                    and rank_manifest["gate_token_sha256"] != expected_gate_token_sha256):
                raise RuntimeError("rank ownership gate token SHA256 mismatch")
            terminate_owned(identities, owned_identity_alive, grace_seconds=grace_seconds)
            rank_cleanup = {"schema": "step377-fixed-ranks-clean-v1", "rank_count": len(identities)}
        elif approved_step377_rank_workers(proc_root):
            raise RuntimeError("ownership_unestablished: approved STEP377 residual lacks rank evidence")
    except Exception as error:
        errors.append(f"rank_cleanup: {error}")
    try:
        cleanup = safe_group_cleanup(ownership, proc_root=proc_root, grace_seconds=grace_seconds,
                                     case_path=case_path)
    except Exception as error:
        errors.append(f"launcher_cleanup: {error}")
    try:
        clear = assert_stable_clear(npu_smi_sample, case_path, port, pgid, proc_root)
    except Exception as error:
        errors.append(f"stable_clear: {error}")
    try:
        assert_port_free(port)
        port_free = True
    except Exception as error:
        errors.append(f"port_free: {error}")
    if errors:
        aggregate = RuntimeError("cleanup domain errors: " + "; ".join(errors))
        setattr(aggregate, "cleanup_errors", tuple(errors))
        raise aggregate
    return {"schema": "step377-cleanup-owned-v1", "rank_cleanup": rank_cleanup,
            "launcher_cleanup": cleanup, "stable_clear": clear, "port_free": port_free}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    idle = commands.add_parser("snapshot-idle")
    cleanup = commands.add_parser("cleanup-owned")
    for command in (idle, cleanup):
        command.add_argument("--case-path", required=True, type=Path)
        command.add_argument("--port", required=True, type=int)
        command.add_argument("--proc-root", type=Path, default=Path("/proc"))
    idle.add_argument("--expected-pgid", required=True, type=int)
    cleanup.add_argument("--ownership", required=True, type=Path)
    cleanup.add_argument("--expected-ownership-sha256", required=True)
    cleanup.add_argument("--rank-ownership", type=Path)
    cleanup.add_argument("--expected-rank-ownership-sha256")
    cleanup.add_argument("--expected-gate-token-sha256")
    cleanup.add_argument("--grace-seconds", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "cleanup-owned":
        value = cleanup_owned_protocol(args.ownership, args.expected_ownership_sha256,
                                       args.case_path, args.port,
                                       args.grace_seconds, args.proc_root,
                                       args.rank_ownership, args.expected_rank_ownership_sha256,
                                       args.expected_gate_token_sha256)
    else:
        if args.expected_pgid is None or args.expected_pgid <= 1:
            raise RuntimeError("snapshot-idle requires expected pgid")
        value = assert_stable_clear(npu_smi_sample, args.case_path, args.port,
                                    args.expected_pgid, args.proc_root)
        assert_port_free(args.port)
        value = {"schema": "step377-snapshot-idle-v1", "stable_clear": value, "port_free": True}
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
