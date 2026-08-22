#!/usr/bin/env python3
"""Local RealBackend and remote host supervisor for STEP393."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import ipaddress
import json
import os
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STEP357 = HERE / "step357_build_qrv2_release_remote.py"
REMOTE_EXEC = HERE / "remote_exec.py"
GUARD_PATH = HERE / "step393_process_guard.py"
AUTHORITY_MAP = Path("/home/l30002999/import-md/hw-import-ip.md")
PROJECT_MACHINE_MAP = ROOT / "机器IP.md"
STEP357_SHA256 = "bf111e2e7eee407e3af26f0ed4e1aab1f833f0e068e66e463664b115c1879d91"
REMOTE_EXEC_SHA256 = "8dfcdda0630413db6cf3593756b81b6a633bc40fe1c761f8ea9a8c8a4e0ffaab"
GUARD_SHA256 = "65a15e832d742f3cca2171126ba11e933599632e531e3b41ccdfbf5ffe2c95c0"
CONTAINER = "mapqr-leicheng"
EXPECTED_HOSTNAME = "yfzy-zhsc-910c-1.novalocal"
ENTRY_BASENAME = "step393_training_entry.py"
RUNNER_BASENAME = "run_step393_delta2_shadow_30.sh"
CONFIG_BASENAME = "aligned_gpu_contract_npu_runtime.py"
CANONICAL_CONFIG_BASENAME = "step393_canonical_aligned_gpu_contract_npu_runtime.py"
BACKEND_BASENAME = "step393_remote_backend.py"
CONTRACT_DIR = "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/gpu_contract_alignment_f922c38_8npu_20260814T172611"
VISIBLE = "8,9,10,11,12,13,14,15"
IDLE_PGID = 2147483647
ACTIVE_ROOT: Path | None = None
ACTIVE_INSTALLED: Path | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_local_locked(path: Path, expected_sha256: str) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise RuntimeError(f"STEP393 local input must be regular: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        chunks = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    if not identity(before) == identity(opened) == identity(closed) == identity(after):
        raise RuntimeError(f"STEP393 local input identity changed: {path.name}")
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError(f"STEP393 local input SHA mismatch: {path.name}")
    return b"".join(chunks)


def upload_readback_script() -> str:
    return r'''import hashlib,json,os,stat,sys
r=sys.argv[1]; e=json.loads(sys.argv[2]); ident=json.loads(sys.argv[3])
s=os.stat(r,follow_symlinks=False)
assert [s.st_dev,s.st_ino]==ident and stat.S_ISDIR(s.st_mode)
out={}
def key(x): return (x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
for n,d in e.items():
 p=r+'/'+n; q=os.lstat(p)
 assert stat.S_ISREG(q.st_mode) and not stat.S_ISLNK(q.st_mode)
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  o=os.fstat(fd); h=hashlib.sha256()
  while True:
   b=os.read(fd,1048576)
   if not b: break
   h.update(b)
  c=os.fstat(fd)
 finally: os.close(fd)
 a=os.lstat(p)
 assert key(q)==key(o)==key(c)==key(a)
 out[n]=h.hexdigest(); assert out[n]==d
z=os.stat(r,follow_symlinks=False); assert [z.st_dev,z.st_ino]==ident
print(json.dumps(out,sort_keys=True))'''


def shell_test_not_exists(path: str) -> str:
    """Build and self-check one complete `[ ! -e PATH ]` shell command."""
    command = "[ ! -e " + shlex.quote(path) + " ]"
    if shlex.split(command) != ["[", "!", "-e", path, "]"]:
        raise RuntimeError("STEP393 generated shell test token mismatch")
    return command


def _load_file(name: str, path: Path, digest: str) -> Any:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError(f"STEP393 locked dependency mismatch: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"STEP393 cannot load dependency: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step393_worker_argv(argv: Sequence[bytes]) -> bool:
    if not argv or re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(os.fsdecode(argv[0])).name) is None:
        return False
    tokens = list(argv[1:])
    if tokens[:1] == [b"-u"]:
        tokens.pop(0)
    if len(tokens) not in (9, 10):
        return False
    entry_path = Path(os.fsdecode(tokens[0]))
    config_path = Path(os.fsdecode(tokens[1]))
    work = os.fsdecode(tokens[2])
    if work.startswith("--work-dir="):
        work_path = work.removeprefix("--work-dir=")
        tail = tokens[3:]
    elif tokens[2] == b"--work-dir" and len(tokens) == 10:
        work_path = os.fsdecode(tokens[3])
        tail = tokens[4:]
    else:
        return False
    root = Path(work_path).parent.parent
    return (
        root.is_absolute() and (ACTIVE_ROOT is None or root == ACTIVE_ROOT)
        and Path(work_path) == root / "run" / "work"
        and entry_path == root / "tools" / ENTRY_BASENAME
        and config_path == root / "tools" / CANONICAL_CONFIG_BASENAME
        and tail == [b"--gpus", b"8", b"--autoscale-lr", b"--max-iters", b"30",
                     b"--launcher=pytorch"]
    )


def _step393_docker_launcher_argv(argv: Sequence[bytes], port: int) -> bool:
    fixed = (b"timeout", b"--signal=TERM", b"--kill-after=30s", b"14400s", b"docker", b"exec")
    if tuple(argv[:6]) != fixed:
        return False
    try:
        container_index = argv.index(CONTAINER.encode(), 6)
    except ValueError:
        return False
    env = argv[6:container_index]
    if len(env) != 8 or any(env[index] != b"-e" for index in range(0, 8, 2)):
        return False
    variables = {}
    for item in env[1::2]:
        if b"=" not in item:
            return False
        key, value = item.split(b"=", 1)
        if key in variables:
            return False
        variables[key] = value
    gate_token = variables.get(b"STEP393_GATE_TOKEN_SHA256")
    if variables != {
        b"ASCEND_RT_VISIBLE_DEVICES": VISIBLE.encode(),
        b"TORCH_DEVICE_BACKEND_AUTOLOAD": b"0",
        b"PYTHONDONTWRITEBYTECODE": b"1",
        b"STEP393_GATE_TOKEN_SHA256": gate_token,
    } or gate_token is None or not re.fullmatch(rb"[0-9a-f]{64}", gate_token):
        return False
    tail = argv[container_index + 1:]
    if len(tail) != 10 or tuple(tail[:3]) != (b"bash", b"--noprofile", b"--norc"):
        return False
    runner, source, contract, shadow, output, port_arg, installed = tail[3:]
    root = str(Path(os.fsdecode(runner)).parent.parent)
    installed_path = Path(os.fsdecode(installed))
    return (
        Path(os.fsdecode(runner)).name == RUNNER_BASENAME
        and (ACTIVE_ROOT is None or Path(root) == ACTIVE_ROOT)
        and all(os.fsdecode(value).startswith("/") for value in (runner, source, contract, shadow, output, installed))
        and os.fsdecode(source) == root + "/source"
        and os.fsdecode(contract) == CONTRACT_DIR
        and os.fsdecode(shadow) == root + "/shadow_work/shadow"
        and os.fsdecode(output) == root + "/run"
        and installed_path.is_absolute()
        and (ACTIVE_INSTALLED is None or installed_path == ACTIVE_INSTALLED)
        and os.fsdecode(installed).endswith("/packages/vendors/customize")
        and port_arg == str(port).encode()
    )


def step393_launcher_argv(argv: Sequence[bytes], port: int) -> bool:
    if _step393_docker_launcher_argv(argv, port):
        return True
    tokens = tuple(os.fsdecode(token) for token in argv)
    if (len(tokens) < 8 or re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(tokens[0]).name) is None
            or tokens[2:4] != ("launch-held", "--gate") or tokens[5] != "--"):
        return False
    backend, gate = Path(tokens[1]), Path(tokens[4])
    root = backend.parent.parent
    return (backend == root / "tools" / BACKEND_BASENAME
            and gate == root / "run" / "host_start.gate"
            and _step393_docker_launcher_argv(tuple(os.fsencode(token) for token in tokens[6:]), port))


def step393_approved_argv(argv: Sequence[bytes], case_path: bytes, port: int) -> bool:
    if step393_launcher_argv(argv, port) or step393_worker_argv(argv):
        return True
    if tuple(argv[:2]) == (b"docker", b"exec") and step393_launcher_argv(
        (b"timeout", b"--signal=TERM", b"--kill-after=30s", b"14400s", *argv), port
    ):
        return True
    case = os.fsdecode(case_path)
    tokens = tuple(os.fsdecode(token) for token in argv)
    if not tokens:
        return False
    root = str(Path(case).parent.parent)
    if len(tokens) == 10 and Path(tokens[0]).name == "bash" and tokens[1:3] == ("--noprofile", "--norc"):
        return (tokens[3] == root + "/tools/" + RUNNER_BASENAME
                and tokens[4] == root + "/source" and tokens[5] == CONTRACT_DIR
                and tokens[6] == root + "/shadow_work/shadow"
                and tokens[7] == root + "/run" and tokens[8] == str(port)
                and tokens[9].startswith("/") and tokens[9].endswith("/packages/vendors/customize")
                and (ACTIVE_INSTALLED is None or Path(tokens[9]) == ACTIVE_INSTALLED))
    if Path(tokens[0]).name == "tee":
        return tokens[1:] in ((root + "/run/work/train.log",),
                              ("-a", root + "/run/work/train.log"))
    if (len(tokens) == 4 and Path(tokens[0]).name == "bash"
            and Path(tokens[1]).name == "ddp_train_30.sh"
            and tokens[1] == CONTRACT_DIR + "/test_harness/ddp_train_30.sh"
            and tokens[2] == case
            and tokens[3] == root + "/tools/" + CANONICAL_CONFIG_BASENAME):
        return True
    if (len(tokens) >= 8 and Path(tokens[0]).name.startswith("python")
            and tokens[1:4] == ("-m", "torch.distributed.launch", "--master_port")
            and tokens[4] == str(port) and tokens[5:8] == ("--nproc_per_node", "8", "--use_env")):
        return step393_worker_argv(tuple(os.fsencode(token) for token in (tokens[0], *tokens[8:])))
    return False


def load_guard() -> Any:
    guard = _load_file("_step393_approved_guard", GUARD_PATH, GUARD_SHA256)
    guard._base._worker_argv = step393_worker_argv
    guard._base._docker_launcher_argv = step393_launcher_argv
    guard._base._approved_group_member_argv = step393_approved_argv
    guard._base._argv_matches = lambda argv, case, port: step393_approved_argv(argv, case, port)
    return guard


def write_new_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{os.urandom(8).hex()}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, limit: int = 1024 * 1024) -> Any:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_size > limit:
        raise RuntimeError(f"STEP393 unsafe JSON: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise RuntimeError("STEP393 JSON exceeds limit")
            chunks.append(chunk)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    if not identity(before) == identity(opened) == identity(closed) == identity(after):
        raise RuntimeError("STEP393 JSON identity changed")
    return json.loads(b"".join(chunks))


def wait_file_set(directory: Path, prefix: str, process: subprocess.Popen[bytes], deadline: float) -> list[Path]:
    expected = {f"rank{rank}.json" for rank in range(8)}
    while True:
        names = {path.name for path in directory.glob("rank*.json")}
        if names == expected:
            return [directory / f"rank{rank}.json" for rank in range(8)]
        if not names.issubset(expected):
            raise RuntimeError(f"STEP393 unexpected {prefix} files")
        if process.poll() is not None:
            raise RuntimeError(f"STEP393 launcher exited before {prefix}: rc={process.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"STEP393 {prefix} timeout: {len(names)}/8")
        time.sleep(0.1)


def validate_ready(rows: list[Any], shadow_package: Path) -> None:
    required = {
        "schema", "rank", "local_rank", "world_size", "container_pid", "visible",
        "torch_version", "torch_npu_version", "npu_available", "device_count",
        "current_device", "startup_context_synchronized", "module_origin", "shadow_package",
        "instrumentation_requested", "fallback_not_observed", "task_queue_state",
        "task_queue_present", "task_queue_value_sha256",
    }
    for rank, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required:
            raise RuntimeError("STEP393 ready schema mismatch")
        exact = {
            "schema": "step393-rank-ready-v1", "rank": rank, "local_rank": rank,
            "world_size": 8, "visible": VISIBLE, "npu_available": True,
            "device_count": 8, "current_device": rank,
            "startup_context_synchronized": True,
            "module_origin": str(shadow_package / "__init__.py"),
            "shadow_package": str(shadow_package), "instrumentation_requested": False,
            "fallback_not_observed": True, "task_queue_state": "production-preserved",
        }
        if any(row.get(key) != value for key, value in exact.items()):
            raise RuntimeError("STEP393 ready value mismatch")
        if type(row.get("container_pid")) is not int or row["container_pid"] <= 1:
            raise RuntimeError("STEP393 ready PID mismatch")
        if (type(row.get("task_queue_present")) is not bool
                or (row["task_queue_present"] and re.fullmatch(
                    r"[0-9a-f]{64}", str(row.get("task_queue_value_sha256"))) is None)
                or (not row["task_queue_present"] and row.get("task_queue_value_sha256") is not None)):
            raise RuntimeError("STEP393 task queue preservation evidence mismatch")
        if not all(isinstance(row.get(key), str) and row[key] for key in ("torch_version", "torch_npu_version")):
            raise RuntimeError("STEP393 torch version evidence missing")


def launcher_command(root: Path, source: Path, contract: Path, shadow: Path,
                     output: Path, port: int, installed: Path, token_sha: str) -> list[str]:
    return [
        "timeout", "--signal=TERM", "--kill-after=30s", "14400s", "docker", "exec",
        "-e", f"ASCEND_RT_VISIBLE_DEVICES={VISIBLE}",
        "-e", "TORCH_DEVICE_BACKEND_AUTOLOAD=0",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", f"STEP393_GATE_TOKEN_SHA256={token_sha}",
        CONTAINER, "bash", "--noprofile", "--norc",
        str(root / "tools" / RUNNER_BASENAME), str(source), str(contract), str(shadow),
        str(output), str(port), str(installed),
    ]


def bootstrap_command(root: Path, docker_command: Sequence[str]) -> list[str]:
    return [sys.executable, str(root / "tools" / BACKEND_BASENAME), "launch-held",
            "--gate", str(root / "run" / "host_start.gate"), "--", *docker_command]


def open_bootstrap_pidfd(process: subprocess.Popen[bytes], pidfd_open: Any) -> int:
    """Acquire pidfd or wait for the still-held bootstrap to self-expire."""
    try:
        return pidfd_open(process.pid, 0)
    except BaseException as error:
        try:
            returncode = process.wait(timeout=70)
        except subprocess.TimeoutExpired as timeout:
            raise RuntimeError(
                "STEP393 held bootstrap did not self-expire after pidfd_open failure"
            ) from timeout
        raise RuntimeError(
            f"STEP393 pidfd_open failed before bootstrap release: rc={returncode}"
        ) from error


def _read_exact_gate(path: Path, expected: dict[str, Any]) -> bool:
    try:
        if read_json(path, 4096) != expected:
            raise RuntimeError(f"STEP393 gate payload mismatch: {path.name}")
        return True
    except FileNotFoundError:
        return False


def launch_held(args: argparse.Namespace) -> int:
    gate = Path(args.gate)
    if gate.name != "host_start.gate" or not gate.is_absolute():
        raise RuntimeError("STEP393 bootstrap gate path mismatch")
    finish_gate = gate.parent / "host_finish.gate"
    result_file = gate.parent / "bootstrap_result.json"
    deadline = time.monotonic() + 60
    signal.signal(signal.SIGTERM, lambda _signum, _frame: None)
    while True:
        if _read_exact_gate(finish_gate, {"schema": "step393-bootstrap-finish-v1"}):
            return 125
        if _read_exact_gate(gate, {"schema": "step393-bootstrap-start-v1"}):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("STEP393 bootstrap gate timeout")
        time.sleep(0.01)
    command = args.exec_argv[1:] if args.exec_argv[:1] == ["--"] else args.exec_argv
    if not command:
        raise RuntimeError("STEP393 bootstrap command missing")
    child = subprocess.Popen(command)
    returncode = child.wait()
    write_new_json(result_file, {
        "schema": "step393-bootstrap-result-v1", "child_returncode": returncode,
    })
    finish_deadline = time.monotonic() + 900
    while not _read_exact_gate(finish_gate, {"schema": "step393-bootstrap-finish-v1"}):
        if time.monotonic() >= finish_deadline:
            raise TimeoutError("STEP393 bootstrap finish gate timeout")
        time.sleep(0.05)
    return returncode


def cleanup_owned_seen(guard: Any, output: Path, ownership_sha: str, rank_sha: str | None,
                       token_sha: str, case_path: Path, port: int,
                       launcher_identity: Any, observed: dict[tuple[int, int], Any],
                       launcher_seen: bool, process: subprocess.Popen[bytes]) -> dict[str, Any]:
    ownership = guard.read_ownership_json(output / "launcher_ownership.json", ownership_sha)
    launcher_pid, launcher_start, launcher_pgid, manifest_port = guard.validate_ownership_manifest(ownership)
    if ((launcher_pid, launcher_start, launcher_pgid, manifest_port)
            != (launcher_identity.host_pid, launcher_identity.starttime,
                launcher_identity.pgid, port)):
        raise RuntimeError("STEP393 preserved launcher identity differs from ownership")
    errors: list[str] = []
    rank_cleanup = None
    launcher_cleanup = None
    stable = None
    port_free = False
    if rank_sha is not None:
        try:
            rank_manifest, identities = guard.read_rank_ownership_json(
                output / "rank_ownership.json", rank_sha,
                expected_launcher_sha256=ownership_sha, case_path=case_path, port=port)
            if rank_manifest["gate_token_sha256"] != token_sha:
                raise RuntimeError("STEP393 rank ownership gate token mismatch")
            guard.terminate_owned(identities, guard.owned_identity_alive, grace_seconds=5.0)
            rank_cleanup = {"schema": "step377-fixed-ranks-clean-v1", "rank_count": 8}
        except Exception as error:
            errors.append(f"rank_cleanup: {error}")
    else:
        try:
            if guard.approved_step377_rank_workers():
                raise RuntimeError("ownership_unestablished: residual rank lacks manifest")
        except Exception as error:
            errors.append(f"rank_cleanup: {error}")
    # The bootstrap deliberately execs the reviewed launcher, so argv changes
    # while PID/starttime/PGID remain fixed.  Preserve launcher_seen and the
    # core identity, then let the first cleanup snapshot bind the post-exec argv.
    def group_reader(manifest: dict[str, object], proc_root: Path):
        nonlocal launcher_seen
        members, launcher_seen = guard.authorized_group_snapshot(
            manifest, observed, launcher_seen, case_path, proc_root)
        return tuple(
            identity for identity in members
            if not (identity.host_pid == launcher_identity.host_pid
                    and identity.starttime == launcher_identity.starttime)
        )
    try:
        launcher_cleanup = guard.safe_group_cleanup(
            ownership, grace_seconds=5.0, case_path=case_path, group_reader=group_reader)
    except Exception as error:
        errors.append(f"launcher_cleanup: {error}")
    try:
        try:
            write_new_json(output / "host_finish.gate", {
                "schema": "step393-bootstrap-finish-v1",
            })
        except FileExistsError:
            if read_json(output / "host_finish.gate", 4096) != {
                    "schema": "step393-bootstrap-finish-v1"}:
                raise RuntimeError("STEP393 existing finish gate mismatch")
        process.wait(timeout=30)
    except Exception as error:
        errors.append(f"launcher_release: {error}")
    try:
        stable = guard.assert_stable_clear(guard.npu_smi_sample, case_path, port,
                                           launcher_identity.pgid)
    except Exception as error:
        errors.append(f"stable_clear: {error}")
    try:
        guard.assert_port_free(port)
        port_free = True
    except Exception as error:
        errors.append(f"port_free: {error}")
    if errors:
        raise RuntimeError("STEP393 cleanup domain errors: " + "; ".join(errors))
    return {"schema": "step393-cleanup-owned-v1", "rank_cleanup": rank_cleanup,
            "launcher_cleanup": launcher_cleanup, "stable_clear": stable,
            "port_free": port_free}


def parse_native_log(path: Path) -> dict[str, Any]:
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode):
        raise RuntimeError("STEP393 native log must be regular")
    text = path.read_text(encoding="utf-8", errors="replace")
    iterations = [int(value) for value in re.findall(r"\bIter\s*\[\s*(\d+)\s*/", text)]
    if iterations != list(range(1, 31)):
        raise RuntimeError(f"STEP393 native log iteration mismatch: {iterations}")
    lowered = text.lower()
    forbidden = [token for token in ("cpu fallback", "fallback to", "torch.linalg.qr", "profiling_mode") if token in lowered]
    if forbidden:
        raise RuntimeError(f"STEP393 fallback/profile evidence in native log: {forbidden}")
    return {
        "path": str(path), "type": "file", "size": status.st_size, "inode": status.st_ino,
        "device": status.st_dev, "sha256": sha256_file(path), "iterations": iterations,
        "created_in_new_run": True,
    }


def host_run(args: argparse.Namespace) -> int:
    global ACTIVE_ROOT, ACTIVE_INSTALLED
    guard = load_guard()
    root = Path(args.root).resolve(strict=True)
    ACTIVE_ROOT = root
    source = Path(args.source).resolve(strict=True)
    contract = Path(args.contract).resolve(strict=True)
    shadow = Path(args.shadow).resolve(strict=True)
    installed = Path(args.installed)
    if not installed.is_absolute() or installed == Path("/") or ".." in installed.parts:
        raise RuntimeError("STEP393 installed container path is unsafe")
    ACTIVE_INSTALLED = Path(os.path.normpath(str(installed)))
    output = root / "run"
    if output.exists():
        raise FileExistsError("STEP393 run directory already exists")
    output.mkdir(mode=0o700)
    for name in ("work", "evidence", "ready", "done", "failure", "gate_ack"):
        (output / name).mkdir(mode=0o700)
    token_sha = hashlib.sha256(os.urandom(32)).hexdigest()
    docker_command = launcher_command(root, source, contract, shadow, output, args.port, installed, token_sha)
    command = bootstrap_command(root, docker_command)
    if not step393_launcher_argv(tuple(os.fsencode(item) for item in command), args.port):
        raise RuntimeError("STEP393 launcher argv grammar rejected reviewed command")
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send = getattr(signal, "pidfd_send_signal", None)
    if pidfd_open is None or pidfd_send is None:
        raise RuntimeError("STEP393 pidfd lifecycle primitives unavailable")
    log_fd = os.open(output / "host_launcher.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    process: subprocess.Popen[bytes] | None = None
    spawn_pidfd: int | None = None
    ownership_sha: str | None = None
    rank_sha: str | None = None
    rank_manifest: dict[str, Any] | None = None
    primary: BaseException | None = None
    result: dict[str, Any] | None = None
    cleanup: dict[str, Any] | None = None
    launcher_identity: Any | None = None
    ownership_manifest: dict[str, Any] | None = None
    observed: dict[tuple[int, int], Any] = {}
    launcher_seen = False

    def observe_owned_group() -> tuple[Any, ...]:
        nonlocal launcher_seen
        if ownership_manifest is None:
            raise RuntimeError("STEP393 ownership manifest unavailable for observation")
        members, launcher_seen = guard.authorized_group_snapshot(
            ownership_manifest, observed, launcher_seen,
            root / "tools" / ENTRY_BASENAME, Path("/proc"),
        )
        return members
    try:
        process = subprocess.Popen(command, stdout=log_fd, stderr=subprocess.STDOUT, start_new_session=True)
        spawn_pidfd = open_bootstrap_pidfd(process, pidfd_open)
        identity = guard.read_process_identity(process.pid)
        if identity.pgid != process.pid or not step393_launcher_argv(identity.argv, args.port):
            raise RuntimeError("STEP393 spawned launcher identity mismatch")
        launcher_identity = identity
        ownership = {
            "schema": "step358-launcher-ownership-v1", "port": args.port,
            "launcher_host_pid": identity.host_pid, "launcher_starttime": identity.starttime,
            "launcher_pgid": identity.pgid,
        }
        ownership_manifest = ownership
        ownership_sha = write_new_json(output / "launcher_ownership.json", ownership)
        write_new_json(output / "host_start.gate", {"schema": "step393-bootstrap-start-v1"})
        observe_owned_group()
        deadline = time.monotonic() + 900
        ready_paths = wait_file_set(output / "ready", "ready", process, deadline)
        ready = [read_json(path, 65536) for path in ready_paths]
        validate_ready(ready, shadow / "mx_driving_cloud")
        observe_owned_group()
        binding = guard.stable_back8_binding(ready, guard.npu_smi_sample)
        observe_owned_group()
        rank_manifest = {
            "schema": "step377-rank-ownership-v1",
            "launcher_ownership_sha256": ownership_sha,
            "gate_token_sha256": token_sha,
            "case_path": str(root / "tools" / ENTRY_BASENAME), "port": args.port,
            "ranks": binding["bindings"],
        }
        guard.validate_rank_ownership_manifest(
            rank_manifest, expected_launcher_sha256=ownership_sha,
            case_path=root / "tools" / ENTRY_BASENAME, port=args.port,
        )
        rank_sha = write_new_json(output / "rank_ownership.json", rank_manifest)
        write_new_json(output / "start.gate", {
            "schema": "step393-host-gate-v1", "token_sha256": token_sha,
        })
        ack_paths = wait_file_set(output / "gate_ack", "gate ack", process, deadline)
        for rank, path in enumerate(ack_paths):
            if read_json(path, 65536) != {
                "schema": "step393-rank-gate-ack-v1", "rank": rank,
                "container_pid": ready[rank]["container_pid"], "token_sha256": token_sha,
            }:
                raise RuntimeError("STEP393 gate acknowledgement mismatch")
        observe_owned_group()
        bootstrap_result_path = output / "bootstrap_result.json"
        train_deadline = time.monotonic() + 14430
        next_observation = time.monotonic()
        bootstrap_result: Any | None = None
        while bootstrap_result is None:
            now = time.monotonic()
            if now >= train_deadline:
                raise TimeoutError("STEP393 launcher training timeout")
            if now >= next_observation:
                observe_owned_group()
                next_observation = now + 1.0
            try:
                bootstrap_result = read_json(bootstrap_result_path, 4096)
            except FileNotFoundError:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"STEP393 held bootstrap exited before result: rc={process.returncode}"
                    )
            time.sleep(0.1)
        if (not isinstance(bootstrap_result, dict)
                or set(bootstrap_result) != {"schema", "child_returncode"}
                or bootstrap_result.get("schema") != "step393-bootstrap-result-v1"
                or type(bootstrap_result.get("child_returncode")) is not int):
            raise RuntimeError("STEP393 bootstrap result schema mismatch")
        rc = bootstrap_result["child_returncode"]
        if rc != 0:
            raise RuntimeError(f"STEP393 launcher returned {rc}")
        done_paths = [output / "done" / f"rank{rank}.json" for rank in range(8)]
        if any(not path.is_file() for path in done_paths) or any(
            read_json(path, 65536) != {"schema": "step393-rank-done-v1", "rank": rank, "returncode": 0}
            for rank, path in enumerate(done_paths)
        ):
            raise RuntimeError("STEP393 rank completion mismatch")
        if any((output / "failure").iterdir()):
            raise RuntimeError("STEP393 rank failure evidence exists")
        loss = read_json(output / "loss_gate.json", 1024 * 1024)
        timing = read_json(output / "timing_window_report.json", 65536)
        native = parse_native_log(output / "work" / "train.log")
        forbidden_outputs = [
            str(path.relative_to(output)) for path in output.rglob("*")
            if any(token in path.name.lower() for token in ("profile", "capture", "dump"))
        ]
        if forbidden_outputs:
            raise RuntimeError(f"STEP393 capture/profile/dump output exists: {forbidden_outputs}")
        result = {
            "schema": "step393-e2e-shadow-configured-v1",
            "status": "E2E_SHADOW_CONFIGURED", "instrumentation_requested": False,
            "fallback_not_observed": True,
            "concrete_kernel_identity": "not_claimed_instrumentation_not_requested",
            "launcher_rc": 0, "rank_count": 8, "gate_ack_count": 8,
            "ready": ready, "binding": binding, "native_log": native,
            "loss_gate": loss, "timing": timing,
            "capture_profile_dump_count": 0,
            "launcher_ownership_sha256": ownership_sha,
            "rank_ownership_sha256": rank_sha,
            "rank_ownership": rank_manifest,
        }
        identities = guard.validate_rank_ownership_manifest(
            rank_manifest, expected_launcher_sha256=ownership_sha,
            case_path=root / "tools" / ENTRY_BASENAME, port=args.port,
        )
        if [not guard.owned_identity_alive(item) for item in identities] != [True] * 8:
            raise RuntimeError("STEP393 ranks remain alive before launcher release")
        for _sample in range(2):
            members = observe_owned_group()
            residual = [item for item in members if not (
                item.host_pid == launcher_identity.host_pid
                and item.starttime == launcher_identity.starttime
            )]
            if residual:
                raise RuntimeError("STEP393 group member remains before launcher release")
            time.sleep(0.05)
        write_new_json(output / "host_finish.gate", {
            "schema": "step393-bootstrap-finish-v1",
        })
        launcher_rc = process.wait(timeout=30)
        if launcher_rc != 0:
            raise RuntimeError(f"STEP393 held launcher returned {launcher_rc}")
    except BaseException as error:
        primary = error
    finally:
        os.close(log_fd)
        if process is not None and ownership_sha is not None:
            try:
                if (process.poll() == 0 and primary is None and result is not None
                        and rank_manifest is not None):
                    try:
                        identities = guard.validate_rank_ownership_manifest(
                            rank_manifest, expected_launcher_sha256=ownership_sha,
                            case_path=root / "tools" / ENTRY_BASENAME, port=args.port,
                        )
                        dead = [not guard.owned_identity_alive(identity) for identity in identities]
                        if dead != [True] * 8:
                            raise RuntimeError("STEP393 successful rank identities remain alive")
                        stable = guard.assert_stable_clear(
                            guard.npu_smi_sample, root / "tools" / ENTRY_BASENAME,
                            args.port, process.pid,
                        )
                        guard.assert_port_free(args.port)
                        cleanup = {"schema": "step393-success-postflight-v1", "rank_dead": dead,
                                   "launcher_poll": 0, "stable_clear": stable, "port_free": True}
                    except BaseException as success_postflight:
                        cleanup = cleanup_owned_seen(
                            guard, output, ownership_sha, rank_sha, token_sha,
                            root / "tools" / ENTRY_BASENAME, args.port, launcher_identity,
                            observed, launcher_seen, process)
                        raise success_postflight
                else:
                    if launcher_identity is None:
                        raise RuntimeError("STEP393 launcher identity evidence missing")
                    cleanup = cleanup_owned_seen(
                        guard, output, ownership_sha, rank_sha, token_sha,
                        root / "tools" / ENTRY_BASENAME, args.port, launcher_identity,
                        observed, launcher_seen, process)
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    setattr(primary, "cleanup_error", error)
        elif process is not None and spawn_pidfd is not None:
            # Authority comes from the pidfd opened immediately on our own
            # start_new_session child; no PID-based fallback is permitted.
            preownership_errors = []
            try:
                if process.poll() is None:
                    pidfd_send(spawn_pidfd, signal.SIGTERM, None, 0)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pidfd_send(spawn_pidfd, signal.SIGKILL, None, 0)
                        process.wait(timeout=30)
            except BaseException as error:
                preownership_errors.append(f"bootstrap_pidfd: {error}")
            try:
                if guard.approved_step377_rank_workers():
                    raise RuntimeError("STEP393 preownership rank appeared before bootstrap gate")
            except BaseException as error:
                preownership_errors.append(f"rank_clear: {error}")
            for label, action in (
                ("stable_clear", lambda: guard.assert_stable_clear(
                    guard.npu_smi_sample, root / "tools" / ENTRY_BASENAME, args.port,
                    process.pid)),
                ("port_free", lambda: guard.assert_port_free(args.port)),
            ):
                try:
                    action()
                except BaseException as error:
                    preownership_errors.append(f"{label}: {error}")
            if preownership_errors:
                error = RuntimeError("STEP393 preownership cleanup errors: " + "; ".join(preownership_errors))
                if primary is None:
                    primary = error
                else:
                    setattr(primary, "cleanup_error", error)
        if spawn_pidfd is not None:
            os.close(spawn_pidfd)
        if cleanup is not None:
            try:
                write_new_json(output / "cleanup_postflight.json", cleanup)
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    setattr(primary, "cleanup_write_error", error)
    if primary is not None:
        raise primary
    assert result is not None and cleanup is not None
    result["cleanup_postflight"] = cleanup
    write_new_json(output / "step393_result.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


def _critical_inventory(root: Path, *, include_all: bool) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"STEP393 inventory root unsafe: {root}")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"STEP393 inventory rejects symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        lower = relative.lower()
        if not include_all and not (
            "qr_v2" in lower or "qrv2" in lower or path.name == "binary_info_config.json"
            or path.name in {"__init__.py", "RECORD"}
        ):
            continue
        status = path.lstat()
        entries.append({"path": relative, "type": "file", "size": status.st_size,
                        "sha256": sha256_file(path)})
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {"root": str(root), "type": "directory", "file_count": len(entries),
            "entries": entries, "inventory_sha256": hashlib.sha256(payload).hexdigest()}


def container_snapshot(args: argparse.Namespace) -> int:
    evidence = read_json(Path(args.evidence), 1024 * 1024)
    attempt_entries = []
    rows = [evidence[key] for key in ("manifest", "receipt", "completion", "original_wheel")]
    for soc in ("ascend910_93", "ascend910b"):
        rows.extend((evidence["artifacts"][soc]["object"], evidence["artifacts"][soc]["json"]))
    for row in rows:
        path = Path(row["path"])
        status = path.lstat()
        if not stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode):
            raise RuntimeError("STEP393 attempt5 entry unsafe")
        actual = sha256_file(path)
        if actual != row["sha256"] or status.st_size != row["size"]:
            raise RuntimeError("STEP393 attempt5 entry changed")
        attempt_entries.append({"path": str(path), "type": "file", "size": status.st_size,
                                "sha256": actual})
    attempt_payload = json.dumps(attempt_entries, sort_keys=True, separators=(",", ":")).encode()
    shadow_root = Path(args.shadow).resolve(strict=True)
    shadow_manifest = read_json(Path(args.shadow_manifest), 1024 * 1024)
    if (shadow_manifest.get("candidate_identity")
            != "QrV2_qa_position_delta2_only_diagnostic_v1"):
        raise RuntimeError("STEP393 shadow candidate mismatch")
    artifacts = shadow_manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"ascend910_93", "ascend910b"}:
        raise RuntimeError("STEP393 shadow artifact SoC set mismatch")
    identity = shadow_manifest["candidate_identity"]
    kernel_root = shadow_root / "mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/kernel"
    for soc in ("ascend910_93", "ascend910b"):
        row = artifacts[soc]
        if not isinstance(row, dict):
            raise RuntimeError("STEP393 shadow artifact row mismatch")
        for kind in ("json", "object"):
            path_key = kind + "_path"
            sha_key = kind + "_sha256"
            path = Path(row[path_key]).resolve(strict=True)
            expected = kernel_root / soc / "qr_v2" / f"{identity}.{('json' if kind == 'json' else 'o')}"
            if path != expected or sha256_file(path) != row[sha_key]:
                raise RuntimeError("STEP393 shadow artifact route/SHA mismatch")
        config = kernel_root / "config" / soc / "qr_v2.json"
        binary = kernel_root / "config" / soc / "binary_info_config.json"
        config_value = read_json(config, 1024 * 1024)
        binary_value = read_json(binary, 1024 * 1024)
        expected_json = f"{soc}/qr_v2/{identity}.json"
        expected_object = f"{soc}/qr_v2/{identity}.o"
        rows_config = config_value.get("binList")
        rows_binary = binary_value.get("QrV2", {}).get("binaryList")
        if (not isinstance(rows_config, list) or len(rows_config) != 1
                or rows_config[0].get("binInfo") != {"jsonFilePath": expected_json}
                or not isinstance(rows_binary, list) or len(rows_binary) != 2
                or any(item.get("binPath") != expected_object for item in rows_binary)):
            raise RuntimeError("STEP393 shadow config route mismatch")
    shadow_inventory = _critical_inventory(shadow_root, include_all=False)
    shadow_inventory["manifest"] = {
        "path": str(Path(args.shadow_manifest)), "type": "file",
        "size": Path(args.shadow_manifest).stat().st_size,
        "sha256": sha256_file(Path(args.shadow_manifest)),
    }
    installed = _critical_inventory(Path(args.installed).resolve(strict=True), include_all=False)
    status_bytes = subprocess.run(
        ["git", "-C", args.source_repo, "status", "--porcelain=v2", "-z"],
        check=True, capture_output=True,
    ).stdout
    head = subprocess.run(
        ["git", "-C", args.source_repo, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    original = {
        "root": args.source_repo, "type": "git_worktree", "head": head,
        "status_bytes": len(status_bytes), "status_count": status_bytes.count(b"\0"),
        "status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "soap_blob": subprocess.run(
            ["git", "-C", args.source_repo, "rev-parse",
             "27b1d6d3f363619ad2faa244abe8fbc5a97faef6:projects/mmdet3d_plugin/optimizers/soap.py"],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
    }
    value = {
        "installed": installed, "original": original,
        "attempt5": {"root": str(Path(evidence["attempt"]["path"])),
                     "type": "fixed_file_set", "file_count": len(attempt_entries),
                     "entries": attempt_entries,
                     "inventory_sha256": hashlib.sha256(attempt_payload).hexdigest()},
        "shadow": shadow_inventory,
    }
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0


def host_boundary(args: argparse.Namespace) -> int:
    global ACTIVE_ROOT
    ACTIVE_ROOT = Path(args.root).resolve(strict=True)
    guard = load_guard()
    case = Path(args.root) / "tools" / ENTRY_BASENAME
    sample_a = guard.npu_smi_sample()
    guard.parse_back8_idle(sample_a)
    related = guard.stable_case_process_scan(case, args.port, IDLE_PGID)
    sample_b = guard.npu_smi_sample()
    guard.parse_back8_idle(sample_b)
    guard.assert_port_free(args.port)
    if related:
        raise RuntimeError("STEP393 related process boundary is not clear")
    print(json.dumps({
        "process": {"type": "process_set", "count": 0, "entries": []},
        "port": {"port": args.port, "free": True},
        "npu": {"scope": "back8", "process_count": 0, "entries": [],
                "sample_sha256": [hashlib.sha256(value.encode()).hexdigest()
                                  for value in (sample_a, sample_b)]},
    }, sort_keys=True))
    return 0


class RealBackend:
    """One two-hop session; all training artifacts remain remote and in place."""

    def __init__(self) -> None:
        self.legacy: Any | None = None
        self.remote_module: Any | None = None
        self.jump: Any | None = None
        self.target: Any | None = None
        self.sftp: Any | None = None
        self.remote_root: str | None = None
        self.runtime: dict[str, Any] | None = None
        self.last_run: dict[str, Any] | None = None

    def _host(self, script: str, timeout: int = 300) -> str:
        if self.target is None or self.legacy is None:
            raise RuntimeError("STEP393 remote session unavailable")
        out, _ = self.legacy.run_host_script(self.target, script, timeout=timeout)
        return out

    def open_once(self, plan: dict[str, Any]) -> Any:
        try:
            AUTHORITY_MAP.read_text(encoding="utf-8")
            PROJECT_MACHINE_MAP.read_text(encoding="utf-8")
            self.legacy = _load_file("_step393_step357", STEP357, STEP357_SHA256)
            if sha256_file(REMOTE_EXEC) != REMOTE_EXEC_SHA256:
                raise RuntimeError("STEP393 remote_exec SHA256 mismatch")
            self.remote_module = self.legacy.load_remote_module()
            info = self.remote_module.parse_machine_info()
            address = ipaddress.ip_address(str(info["target_host"]))
            if not address.is_private or str(address).split(".")[-1] != "42":
                raise RuntimeError("STEP393 target must be private and end in 42")
            self.remote_root = self.legacy.safe_remote_path(str(info["shared"]), plan["remote_directory"])
            self.jump, self.target = self.legacy.connect_target(self.remote_module, info)
            self.sftp = self.target.open_sftp()
            hostname, _ = self.legacy.run_host_script(self.target, "hostname", timeout=30)
            if hostname.strip() != EXPECTED_HOSTNAME:
                raise RuntimeError("STEP393 target hostname mismatch")
            self.runtime = self.legacy.container_probe(self.target)
            if self.runtime.get("container_name") != CONTAINER:
                raise RuntimeError("STEP393 container mismatch")
            return self
        except BaseException as primary:
            try:
                self.close(self)
            except BaseException as cleanup:
                setattr(primary, "cleanup_error", cleanup)
            raise

    def create_new_diag(self, _session: Any, _plan: dict[str, Any]) -> None:
        assert self.remote_root is not None
        root = shlex.quote(self.remote_root)
        self._host(f"set -eu; umask 077; [ ! -e {root} ]; mkdir -m 700 -- {root}; mkdir -m 700 -- {root}/tools", 30)

    def upload_locked(self, _session: Any, plan: dict[str, Any]) -> None:
        assert self.remote_root is not None and self.sftp is not None and self.legacy is not None
        files = plan["local_files"]
        before = json.loads(self._host(
            "python3 -c " + shlex.quote("import json,os,sys; s=os.stat(sys.argv[1],follow_symlinks=False); print(json.dumps([s.st_dev,s.st_ino]))")
            + " " + shlex.quote(self.remote_root + "/tools")
        ))
        for name, row in files.items():
            path = Path(row["local_path"])
            data = read_local_locked(path, row["sha256"])
            self.legacy.write_remote_new(self.sftp, self.remote_root + "/tools/" + name, data, mode=0o600)
        code = upload_readback_script()
        expected = {name: row["sha256"] for name, row in files.items()}
        value = json.loads(self._host("python3 -c " + shlex.quote(code) + " "
            + shlex.quote(self.remote_root + "/tools") + " " + shlex.quote(json.dumps(expected))
            + " " + shlex.quote(json.dumps(before))))
        if value != expected:
            raise RuntimeError("STEP393 upload readback mismatch")

    def archive_source(self, _session: Any, plan: dict[str, Any]) -> None:
        assert self.remote_root is not None
        source = self.remote_root + "/source"
        archive = self.remote_root + "/source.archive.tar"
        archive_manifest = self.remote_root + "/source_archive_manifest.json"
        script = (
            "set -eu; umask 077; " + shell_test_not_exists(source)
            + "; mkdir -m 700 -- " + shlex.quote(source)
            + "; " + shell_test_not_exists(archive) + "; git -C " + shlex.quote(plan["source_repo"])
            + " archive --format=tar -o " + shlex.quote(archive) + " " + shlex.quote(plan["source_commit"])
            + "; [ \"$(git get-tar-commit-id < " + shlex.quote(archive) + ")\" = "
            + shlex.quote(plan["source_commit"]) + " ]; tar -xf " + shlex.quote(archive)
            + " -C " + shlex.quote(source)
            + "; " + shell_test_not_exists(source + "/.git") + "; printf '%s\\n' "
            + shlex.quote(plan["source_commit"]) + " > " + shlex.quote(source + "/.step393_archived_commit")
            + "; python3 -c " + shlex.quote(
                "import hashlib,json,os,sys; p=sys.argv[1]; d=open(p,'rb').read(); "
                "v={'schema':'step393-source-archive-v1','commit':sys.argv[3],"
                "'tree':sys.argv[4],'archive':{'path':p,'type':'file','size':len(d),"
                "'sha256':hashlib.sha256(d).hexdigest()}}; fd=os.open(sys.argv[2],"
                "os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); "
                "os.write(fd,(json.dumps(v,sort_keys=True)+'\\n').encode()); os.fsync(fd); os.close(fd)"
            ) + " " + shlex.quote(archive) + " " + shlex.quote(archive_manifest) + " "
            + shlex.quote(plan["source_commit"]) + " \"$(git -C " + shlex.quote(plan["source_repo"])
            + " rev-parse " + shlex.quote(plan["source_commit"] + "^{tree}") + ")\""
        )
        self._host(script, 900)

    def verify_source(self, _session: Any, plan: dict[str, Any]) -> None:
        assert self.remote_root is not None
        code = "import ast,hashlib,json,subprocess,sys; from pathlib import Path; repo,src,commit,blob,rel,manifest=sys.argv[1:]; assert subprocess.check_output(['git','-C',repo,'rev-parse',commit+':'+rel],text=True).strip()==blob; p=Path(src)/rel; assert subprocess.check_output(['git','hash-object',str(p)],text=True).strip()==blob; m=json.load(open(manifest)); a=Path(m['archive']['path']); assert m['commit']==commit and m['tree']==subprocess.check_output(['git','-C',repo,'rev-parse',commit+'^{tree}'],text=True).strip() and m['archive']['size']==a.stat().st_size and m['archive']['sha256']==hashlib.sha256(a.read_bytes()).hexdigest(); t=ast.parse(p.read_text());\ndef dot(n):\n a=[]\n while isinstance(n,ast.Attribute): a.append(n.attr); n=n.value\n if isinstance(n,ast.Name): a.append(n.id)\n return '.'.join(reversed(a))\nlines=sorted(n.lineno for n in ast.walk(t) if isinstance(n,ast.Call) and dot(n.func)=='mx_driving_cloud.linalg.qr'); assert lines==[429,529]; print(json.dumps({'blob':blob,'call_lines':lines}))"
        value = json.loads(self._host("python3 -c " + shlex.quote(code) + " "
            + " ".join(shlex.quote(value) for value in (
                plan["source_repo"], self.remote_root + "/source", plan["source_commit"],
                plan["soap"]["blob"], plan["soap"]["relative"],
                self.remote_root + "/source_archive_manifest.json",
            )), 120))
        if value != {"blob": plan["soap"]["blob"], "call_lines": [429, 529]}:
            raise RuntimeError("STEP393 archived source contract mismatch")

    def prepare_shadow(self, _session: Any, plan: dict[str, Any]) -> None:
        assert self.remote_root is not None
        work = self.remote_root + "/shadow_work"
        self._host("mkdir -m 700 -- " + shlex.quote(work), 30)
        command = (
            "docker exec -e PYTHONDONTWRITEBYTECODE=1 " + CONTAINER + " python3 "
            + shlex.quote(self.remote_root + "/tools/step392_prepare_delta2_shadow.py")
            + " --attempt5-manifest " + shlex.quote(plan["attempt5_manifest"]["path"])
            + " --wheel " + shlex.quote(plan["original_wheel"]["path"])
            + " --approved-root " + shlex.quote(work)
            + " --output-manifest " + shlex.quote(work + "/shadow_manifest.json")
            + " --evidence " + shlex.quote(self.remote_root + "/tools/STEP392_attempt5_evidence.json")
        )
        self._host(command, 900)

    def snapshot(self, _session: Any, phase: str, plan: dict[str, Any]) -> dict[str, Any]:
        assert self.remote_root is not None and self.runtime is not None
        installed = self.runtime["installed_cloud_root"] + "/packages/vendors/customize"
        command = (
            "docker exec -e PYTHONDONTWRITEBYTECODE=1 " + CONTAINER + " python3 "
            + shlex.quote(self.remote_root + "/tools/step393_remote_backend.py") + " container-snapshot"
            + " --source-repo " + shlex.quote(plan["source_repo"])
            + " --evidence " + shlex.quote(self.remote_root + "/tools/STEP392_attempt5_evidence.json")
            + " --shadow " + shlex.quote(self.remote_root + "/shadow_work/shadow")
            + " --shadow-manifest " + shlex.quote(self.remote_root + "/shadow_work/shadow_manifest.json")
            + " --installed " + shlex.quote(installed)
        )
        protected = json.loads(self._host(command, 900))
        boundary = json.loads(self._host(
            "PYTHONDONTWRITEBYTECODE=1 python3 " + shlex.quote(self.remote_root + "/tools/step393_remote_backend.py")
            + " host-boundary --root " + shlex.quote(self.remote_root) + " --port " + str(plan["master_port"]), 120
        ))
        return {"schema": "step393-protected-snapshot-v2", "phase": phase, **protected, **boundary}

    def run_training_live_gated(self, _session: Any, plan: dict[str, Any]) -> dict[str, Any]:
        assert self.remote_root is not None and self.runtime is not None
        installed = self.runtime["installed_cloud_root"] + "/packages/vendors/customize"
        command = (
            "PYTHONDONTWRITEBYTECODE=1 python3 " + shlex.quote(self.remote_root + "/tools/step393_remote_backend.py")
            + " host-run --root " + shlex.quote(self.remote_root)
            + " --source " + shlex.quote(self.remote_root + "/source")
            + " --contract " + shlex.quote(plan["contract_dir"])
            + " --shadow " + shlex.quote(self.remote_root + "/shadow_work/shadow")
            + " --installed " + shlex.quote(installed)
            + " --port " + str(plan["master_port"])
        )
        self.last_run = json.loads(self._host(command, 15000))
        return self.last_run

    def run_loss_gate(self, _session: Any, _plan: dict[str, Any]) -> dict[str, Any]:
        if self.last_run is None:
            raise RuntimeError("STEP393 run result unavailable")
        return self.last_run["loss_gate"]

    def close(self, _session: Any) -> None:
        errors = []
        for resource in (self.sftp, self.target, self.jump):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError("STEP393 remote close failed") from errors[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("host-run")
    for name in ("root", "source", "contract", "shadow", "installed"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--port", required=True, type=int)
    snapshot = commands.add_parser("container-snapshot")
    for name in ("source-repo", "evidence", "shadow", "shadow-manifest", "installed"):
        snapshot.add_argument("--" + name, required=True)
    boundary = commands.add_parser("host-boundary")
    boundary.add_argument("--root", required=True)
    boundary.add_argument("--port", required=True, type=int)
    held = commands.add_parser("launch-held")
    held.add_argument("--gate", required=True)
    held.add_argument("exec_argv", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "host-run":
        return host_run(args)
    if args.command == "container-snapshot":
        return container_snapshot(args)
    if args.command == "launch-held":
        return launch_held(args)
    return host_boundary(args)


if __name__ == "__main__":
    raise SystemExit(main())
