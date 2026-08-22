#!/usr/bin/env python3
"""Disarmed STEP393 end-to-end delta2 shadow 30-step controller.

The executable transaction is deliberately double-gated.  STEP392 proved the
candidate in a standalone world8 case, but also exposed a race in the only
available owned-process cleanup primitive.  This controller therefore records
the complete E2E contract without permitting a remote connection until a
separate, reviewed per-PID/pidfd cleanup primitive is available.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".codex-tools"

# Two independent phase-transition gates.  Only a one-process reviewed wrapper
# may flip both in memory; source bytes remain disarmed.
E2E_READY = False
PROCESS_GUARD_READY = False
PROCESS_GUARD_BLOCKER = (
    "STEP393 reviewed process guard is wired but the one-shot E2E phase transition is not armed"
)

CONTAINER = "mapqr-leicheng"
SOURCE_REPO = "/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang"
SOURCE_COMMIT = "27b1d6d3f363619ad2faa244abe8fbc5a97faef6"
SOAP_REL = "projects/mmdet3d_plugin/optimizers/soap.py"
SOAP_BLOB = "77e412c4bece2ca95fd6dbf95732b89951924874"
SOAP_CALL_LINES = (429, 529)
CONTRACT_DIR = (
    "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/"
    "gpu_contract_alignment_f922c38_8npu_20260814T172611"
)
CONTRACT_FILES = {
    "test_harness/ddp_train_30.sh":
        "10ad92c723164d52b32734734c8b466f313200165ec1307cb7199e298bb1e0fc",
    "tools/train_spetr_gpu_seed0_runtime.py":
        "8c5b315b1741a1557293db1df1bd6c6699494970bc136c434b5b84af9aad65fa",
}
ATTEMPT5_DIR = (
    "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/"
    "step385_attempt5_qrv2_delta2_only_opc_build_20260822"
)
ATTEMPT5_MANIFEST = ATTEMPT5_DIR + "/work/release_manifest.json"
ATTEMPT5_MANIFEST_SHA256 = (
    "0221f5b64fe682d230f834554b3b8d977673f807c6a890c5279c835ebe173de8"
)
ORIGINAL_WHEEL = (
    ATTEMPT5_DIR
    + "/work/outer_original/"
    "mx_driving_cloud-26.0.7+CANN8.3.RC1.A3-cp311-cp311-linux_aarch64.whl"
)
ORIGINAL_WHEEL_SHA256 = (
    "23253f7fa2b9bfb1b6ff3c77df6620f6c559f68be154f6333246d73178eb5da9"
)
CANDIDATE_IDENTITY = "QrV2_qa_position_delta2_only_diagnostic_v1"
REMOTE_DIAG_NAME = "step393_attempt6_delta2_shadow_e2e30_back8_world8_20260822"
MASTER_PORT = 34393
VISIBLE_DEVICES = tuple(range(8, 16))
WORLD_SIZE = 8
MAX_ITERS = 30
LOSS_THRESHOLD = 0.02

RUNNER = TOOLS / "run_step393_delta2_shadow_30.sh"
SHADOW_PREPARER = TOOLS / "step392_prepare_delta2_shadow.py"
EVIDENCE = ROOT / "STEP392_attempt5_evidence.json"
LOSS_GATE = TOOLS / "step340_loss_gate.py"
GPU_ORACLE = TOOLS / "gpu_loss_800.json"
TRAINING_ENTRY = TOOLS / "step393_training_entry.py"
REMOTE_BACKEND = TOOLS / "step393_remote_backend.py"
PROCESS_GUARD = TOOLS / "step393_process_guard.py"
CANONICAL_CONFIG = TOOLS / "step393_canonical_aligned_gpu_contract_npu_runtime.py"
BASE_PROCESS_GUARD = TOOLS / "step377_process_guard.py"
REMOTE_EXEC = TOOLS / "remote_exec.py"
STEP357 = TOOLS / "step357_build_qrv2_release_remote.py"

LOCKED_LOCAL_SHA256 = {
    "step392_prepare_delta2_shadow.py":
        "93c27232ac9af85a10bb3c1f97ce525e67179de9012ec63412a39e72e5d0c4a6",
    "STEP392_attempt5_evidence.json":
        "90926484a28fbfe7e1e69f52d5154fe2edcfeed683ad9c0e02b06ca7c9ea3fc9",
    "step340_loss_gate.py":
        "b4e20111333f066183c5474d931b6248129065f4b80cfc9ce7177df5e44d9b7d",
    "gpu_loss_800.json":
        "67b36f3dbb36ff50b2a2bf68062d2e1589e2f55cb94207505fdd504e380a8851",
    "step393_training_entry.py":
        "fb0e48cfb9593bc70188b0a3b30ee5265eda205aa1f69fad407c6ba7e8a21f40",
    "step393_process_guard.py":
        "65a15e832d742f3cca2171126ba11e933599632e531e3b41ccdfbf5ffe2c95c0",
    "step393_canonical_aligned_gpu_contract_npu_runtime.py":
        "02aca0c7a33e05e972e268aecc1932bbf10f611fa523ec4696b94d58cd7f56a5",
    "step377_process_guard.py":
        "8f4886838c39f96e662ff2a5b3d17c79c9ee01d76bfe826f4b19fb63a66e8199",
}
LOCAL_DYNAMIC_SHA256 = {
    "run_step393_delta2_shadow_30.sh":
        "d5e82a674b4089a4bd506b5e4c583f8adabe7e33987b18f3373e7b94f7c343a4",
    "step393_remote_backend.py":
        "9b2fb36842725afe0fe9fd07a3aa12c5f90435ad0f81b380f89e5a13ba94bc98",
}
LOCAL_SUPPORT_SHA256 = {
    "remote_exec.py": "8dfcdda0630413db6cf3593756b81b6a633bc40fe1c761f8ea9a8c8a4e0ffaab",
    "step357_build_qrv2_release_remote.py":
        "bf111e2e7eee407e3af26f0ed4e1aab1f833f0e068e66e463664b115c1879d91",
}

FORBIDDEN_ACTIONS = (
    "package", "install", "build", "download", "git_clone", "pip_install",
    "modify_active_worktree", "profile", "capture", "dump", "debug",
    "per_qr_synchronize", "cpu_training", "gpu_training",
)
TRANSACTION_ACTIONS = (
    "read_current_host_mapping",
    "connect_two_hop_once",
    "require_exact_hostname_and_container",
    "create_exclusive_diagnostic_directory",
    "upload_locked_local_inventory",
    "git_archive_locked_commit_to_no_git_source",
    "verify_source_commit_blob_and_two_ast_call_anchors",
    "prepare_and_validate_step392_shadow",
    "snapshot_pre_installed_original_attempt5_shadow_process_port_npu",
    "run_native_30step_with_host_world8_npu_live_gate",
    "run_step340_loss_gate_iter_1_30_threshold_0_02_in_place",
    "report_timing_window_only",
    "snapshot_post_and_compare_closure",
    "retain_remote_logs_and_results_in_place",
)


class Backend(Protocol):
    """Reviewed backend boundary; no backend is instantiated while disarmed."""

    def open_once(self, plan: dict[str, Any]) -> Any: ...
    def create_new_diag(self, session: Any, plan: dict[str, Any]) -> None: ...
    def upload_locked(self, session: Any, plan: dict[str, Any]) -> None: ...
    def snapshot(self, session: Any, phase: str, plan: dict[str, Any]) -> dict[str, Any]: ...
    def archive_source(self, session: Any, plan: dict[str, Any]) -> None: ...
    def verify_source(self, session: Any, plan: dict[str, Any]) -> None: ...
    def prepare_shadow(self, session: Any, plan: dict[str, Any]) -> None: ...
    def run_training_live_gated(self, session: Any, plan: dict[str, Any]) -> dict[str, Any]: ...
    def run_loss_gate(self, session: Any, plan: dict[str, Any]) -> dict[str, Any]: ...
    def close(self, session: Any) -> None: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def build_plan() -> dict[str, Any]:
    """Return the immutable review plan without touching credentials or network."""
    armed = E2E_READY is True and PROCESS_GUARD_READY is True
    local_paths = {
        SHADOW_PREPARER.name: SHADOW_PREPARER, EVIDENCE.name: EVIDENCE,
        LOSS_GATE.name: LOSS_GATE, GPU_ORACLE.name: GPU_ORACLE,
        TRAINING_ENTRY.name: TRAINING_ENTRY, PROCESS_GUARD.name: PROCESS_GUARD,
        CANONICAL_CONFIG.name: CANONICAL_CONFIG,
        BASE_PROCESS_GUARD.name: BASE_PROCESS_GUARD, RUNNER.name: RUNNER,
        REMOTE_BACKEND.name: REMOTE_BACKEND,
    }
    local_hashes = {**LOCKED_LOCAL_SHA256, **LOCAL_DYNAMIC_SHA256}
    return {
        "schema": "step393-delta2-shadow-e2e30-plan-v1",
        "status": "GO_REVIEWED_ONCE" if armed else "NO_GO_PHASE_TRANSITION",
        "blocker": PROCESS_GUARD_BLOCKER,
        "container": CONTAINER,
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "soap": {"relative": SOAP_REL, "blob": SOAP_BLOB, "call_lines": list(SOAP_CALL_LINES)},
        "contract_dir": CONTRACT_DIR,
        "contract_files": dict(CONTRACT_FILES),
        "attempt5_manifest": {"path": ATTEMPT5_MANIFEST, "sha256": ATTEMPT5_MANIFEST_SHA256},
        "original_wheel": {"path": ORIGINAL_WHEEL, "sha256": ORIGINAL_WHEEL_SHA256},
        "candidate_identity": CANDIDATE_IDENTITY,
        "remote_directory": REMOTE_DIAG_NAME,
        "master_port": MASTER_PORT,
        "visible_devices": list(VISIBLE_DEVICES),
        "world_size": WORLD_SIZE,
        "max_iters": MAX_ITERS,
        "loss_gate": {"tool": LOSS_GATE.name, "oracle": GPU_ORACLE.name,
                      "start_iter": 1, "end_iter": 30, "threshold": LOSS_THRESHOLD},
        "shadow_first": True,
        "profile": False,
        "capture": False,
        "dump": False,
        "debug": False,
        "per_qr_synchronize": False,
        "download_remote_artifacts": False,
        "process_guard_approved": PROCESS_GUARD_READY,
        "local_files": {
            name: {"local_path": str(path), "sha256": local_hashes[name]}
            for name, path in local_paths.items()
        },
        "actions": list(TRANSACTION_ACTIONS),
        "forbidden": list(FORBIDDEN_ACTIONS),
    }


def validate_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("schema") != "step393-delta2-shadow-e2e30-plan-v1":
        raise RuntimeError("STEP393 plan schema mismatch")
    armed = E2E_READY is True and PROCESS_GUARD_READY is True
    exact = {
        "status": "GO_REVIEWED_ONCE" if armed else "NO_GO_PHASE_TRANSITION",
        "blocker": PROCESS_GUARD_BLOCKER,
        "container": CONTAINER, "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT, "contract_dir": CONTRACT_DIR,
        "candidate_identity": CANDIDATE_IDENTITY, "remote_directory": REMOTE_DIAG_NAME,
        "master_port": MASTER_PORT, "visible_devices": list(VISIBLE_DEVICES),
        "world_size": 8, "max_iters": 30, "shadow_first": True,
        "profile": False, "capture": False, "dump": False, "debug": False,
        "per_qr_synchronize": False, "download_remote_artifacts": False,
        "process_guard_approved": PROCESS_GUARD_READY,
    }
    if any(plan.get(key) != value for key, value in exact.items()):
        raise RuntimeError("STEP393 fixed plan field mismatch")
    if plan.get("soap") != {
        "relative": SOAP_REL, "blob": SOAP_BLOB, "call_lines": [429, 529]
    }:
        raise RuntimeError("STEP393 SOAP object/anchor mismatch")
    if plan.get("attempt5_manifest") != {
        "path": ATTEMPT5_MANIFEST, "sha256": ATTEMPT5_MANIFEST_SHA256
    } or plan.get("original_wheel") != {
        "path": ORIGINAL_WHEEL, "sha256": ORIGINAL_WHEEL_SHA256
    }:
        raise RuntimeError("STEP393 attempt5/original input mismatch")
    if plan.get("contract_files") != CONTRACT_FILES:
        raise RuntimeError("STEP393 STEP204 contract SHA mismatch")
    if plan.get("loss_gate") != {
        "tool": LOSS_GATE.name, "oracle": GPU_ORACLE.name,
        "start_iter": 1, "end_iter": 30, "threshold": 0.02,
    }:
        raise RuntimeError("STEP393 loss gate mismatch")
    if plan.get("actions") != list(TRANSACTION_ACTIONS):
        raise RuntimeError("STEP393 transaction order mismatch")
    if plan.get("forbidden") != list(FORBIDDEN_ACTIONS):
        raise RuntimeError("STEP393 forbidden action mismatch")
    local = plan.get("local_files")
    expected_hashes = {**LOCKED_LOCAL_SHA256, **LOCAL_DYNAMIC_SHA256}
    expected_paths = {
        SHADOW_PREPARER.name: SHADOW_PREPARER, EVIDENCE.name: EVIDENCE,
        LOSS_GATE.name: LOSS_GATE, GPU_ORACLE.name: GPU_ORACLE,
        TRAINING_ENTRY.name: TRAINING_ENTRY, PROCESS_GUARD.name: PROCESS_GUARD,
        CANONICAL_CONFIG.name: CANONICAL_CONFIG,
        BASE_PROCESS_GUARD.name: BASE_PROCESS_GUARD, RUNNER.name: RUNNER,
        REMOTE_BACKEND.name: REMOTE_BACKEND,
    }
    if not isinstance(local, dict) or set(local) != set(expected_paths):
        raise RuntimeError("STEP393 local upload inventory mismatch")
    for name, path in expected_paths.items():
        if local[name] != {"local_path": str(path), "sha256": expected_hashes[name]}:
            raise RuntimeError(f"STEP393 local upload contract mismatch: {name}")
    return plan


def local_preflight() -> dict[str, Any]:
    """Fail closed before reading machine maps, credentials, or loading SSH code."""
    plan = validate_plan(build_plan())
    if E2E_READY is not True:
        raise RuntimeError("STEP393 E2E controller is intentionally disarmed")
    if PROCESS_GUARD_READY is not True:
        raise RuntimeError(PROCESS_GUARD_BLOCKER)
    locked = {
        SHADOW_PREPARER.name: SHADOW_PREPARER,
        EVIDENCE.name: EVIDENCE,
        LOSS_GATE.name: LOSS_GATE,
        GPU_ORACLE.name: GPU_ORACLE,
        TRAINING_ENTRY.name: TRAINING_ENTRY,
        PROCESS_GUARD.name: PROCESS_GUARD,
        CANONICAL_CONFIG.name: CANONICAL_CONFIG,
        BASE_PROCESS_GUARD.name: BASE_PROCESS_GUARD,
        RUNNER.name: RUNNER,
        REMOTE_BACKEND.name: REMOTE_BACKEND,
    }
    expected_hashes = {**LOCKED_LOCAL_SHA256, **LOCAL_DYNAMIC_SHA256}
    if set(locked) != set(expected_hashes):
        raise RuntimeError("STEP393 local inventory mismatch")
    for name, path in locked.items():
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_hashes[name]:
            raise RuntimeError(f"STEP393 locked local file mismatch: {name}")
    for name, path in ((REMOTE_EXEC.name, REMOTE_EXEC), (STEP357.name, STEP357)):
        if path.is_symlink() or not path.is_file() or sha256_file(path) != LOCAL_SUPPORT_SHA256[name]:
            raise RuntimeError(f"STEP393 locked local support mismatch: {name}")
    return plan


def _validate_inventory(value: Any, label: str, *, exact_count: int | None = None) -> None:
    required = {"root", "type", "file_count", "entries", "inventory_sha256"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise RuntimeError(f"STEP393 {label} inventory schema mismatch")
    if (not isinstance(value["root"], str) or not value["root"].startswith("/")
            or value["type"] not in {"directory", "fixed_file_set"}
            or type(value["file_count"]) is not int or not isinstance(value["entries"], list)
            or value["file_count"] != len(value["entries"])
            or exact_count is not None and value["file_count"] != exact_count
            or not _hex(value["inventory_sha256"], 64)):
        raise RuntimeError(f"STEP393 {label} inventory value mismatch")
    for row in value["entries"]:
        if (not isinstance(row, dict) or set(row) != {"path", "type", "size", "sha256"}
                or not isinstance(row["path"], str) or not row["path"]
                or row["type"] != "file" or type(row["size"]) is not int or row["size"] < 0
                or not _hex(row["sha256"], 64)):
            raise RuntimeError(f"STEP393 {label} entry mismatch")
    payload = json.dumps(value["entries"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(payload).hexdigest() != value["inventory_sha256"]:
        raise RuntimeError(f"STEP393 {label} inventory digest mismatch")


def validate_snapshot(value: Any, *, phase: str) -> dict[str, Any]:
    keys = {"schema", "phase", "installed", "original", "attempt5", "shadow",
            "process", "port", "npu"}
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeError("STEP393 snapshot schema mismatch")
    if value.get("schema") != "step393-protected-snapshot-v2" or value.get("phase") != phase:
        raise RuntimeError("STEP393 snapshot phase mismatch")
    _validate_inventory(value["installed"], "installed")
    _validate_inventory(value["attempt5"], "attempt5", exact_count=8)
    _validate_inventory(value["shadow"], "shadow")
    manifest = value["shadow"].get("manifest")
    if (not isinstance(manifest, dict) or set(manifest) != {"path", "type", "size", "sha256"}
            or not manifest["path"].startswith("/") or manifest["type"] != "file"
            or type(manifest["size"]) is not int or not _hex(manifest["sha256"], 64)):
        raise RuntimeError("STEP393 shadow manifest mismatch")
    original = value["original"]
    if (not isinstance(original, dict) or set(original) != {
            "root", "type", "head", "status_bytes", "status_count", "status_sha256", "soap_blob"}
            or original["type"] != "git_worktree" or not original["root"].startswith("/")
            or not _hex(original["head"], 40) or type(original["status_bytes"]) is not int
            or type(original["status_count"]) is not int or not _hex(original["status_sha256"], 64)
            or original["soap_blob"] != SOAP_BLOB):
        raise RuntimeError("STEP393 original worktree snapshot mismatch")
    if value["process"] != {"type": "process_set", "count": 0, "entries": []}:
        raise RuntimeError("STEP393 process boundary is not clear")
    if value["port"] != {"port": MASTER_PORT, "free": True}:
        raise RuntimeError("STEP393 port boundary is not clear")
    npu = value["npu"]
    if (not isinstance(npu, dict) or npu.get("scope") != "back8"
            or npu.get("process_count") != 0 or npu.get("entries") != []
            or not isinstance(npu.get("sample_sha256"), list) or len(npu["sample_sha256"]) != 2
            or not all(_hex(item, 64) for item in npu["sample_sha256"])):
        raise RuntimeError("STEP393 NPU boundary is not clear")
    return value


def compare_closure(before: Any, after: Any) -> None:
    pre = validate_snapshot(before, phase="pre")
    post = validate_snapshot(after, phase="post")
    for name in ("installed", "original", "attempt5", "shadow"):
        if pre[name] != post[name]:
            raise RuntimeError(f"STEP393 {name} changed across transaction")


def validate_loss_gate(value: Any) -> dict[str, Any]:
    keys = {"status", "threshold", "iter_start", "iter_end", "expected_count",
            "pass_count", "fail_count", "failure_reason_counts", "first_failure",
            "max_relative_deviation", "max_relative_deviation_iter"}
    if (
        not isinstance(value, dict) or set(value) != keys or value.get("status") != "pass"
        or value.get("threshold") != 0.02 or value.get("iter_start") != 1
        or value.get("iter_end") != 30 or value.get("expected_count") != 30
        or value.get("pass_count") != 30 or value.get("fail_count") != 0
        or value.get("failure_reason_counts") != {} or value.get("first_failure") is not None
        or type(value.get("max_relative_deviation")) not in (int, float)
        or not math.isfinite(value["max_relative_deviation"])
        or not 0 <= value["max_relative_deviation"] <= 0.02
        or type(value.get("max_relative_deviation_iter")) is not int
        or value["max_relative_deviation_iter"] not in range(1, 31)
    ):
        raise RuntimeError("STEP393 GPU loss gate failed")
    return value


def _valid_worker_argv(argv: Any, run_root: Path) -> bool:
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
        return False
    tokens = list(argv)
    if not tokens or re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(tokens.pop(0)).name) is None:
        return False
    if tokens[:1] == ["-u"]:
        tokens.pop(0)
    if len(tokens) not in (9, 10):
        return False
    entry, config = Path(tokens[0]), Path(tokens[1])
    if tokens[2].startswith("--work-dir="):
        work = Path(tokens[2].removeprefix("--work-dir="))
        tail = tokens[3:]
    elif tokens[2] == "--work-dir" and len(tokens) == 10:
        work = Path(tokens[3])
        tail = tokens[4:]
    else:
        return False
    return (
        entry == run_root / "tools" / "step393_training_entry.py"
        and config == run_root / "tools" / CANONICAL_CONFIG.name
        and work == run_root / "run" / "work"
        and tail == ["--gpus", "8", "--autoscale-lr", "--max-iters", "30",
                     "--launcher=pytorch"]
    )


def validate_run(value: Any) -> dict[str, Any]:
    keys = {"schema", "status", "instrumentation_requested", "fallback_not_observed",
            "concrete_kernel_identity", "launcher_rc",
            "rank_count", "gate_ack_count", "ready", "binding", "native_log", "loss_gate",
            "timing", "capture_profile_dump_count", "launcher_ownership_sha256",
            "rank_ownership_sha256", "rank_ownership", "cleanup_postflight"}
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeError("STEP393 run schema mismatch")
    if (value["schema"] != "step393-e2e-shadow-configured-v1"
            or value["status"] != "E2E_SHADOW_CONFIGURED"
            or value["instrumentation_requested"] is not False
            or value["fallback_not_observed"] is not True
            or value["concrete_kernel_identity"] != "not_claimed_instrumentation_not_requested"
            or value["launcher_rc"] != 0 or value["rank_count"] != 8
            or value["gate_ack_count"] != 8 or value["capture_profile_dump_count"] != 0
            or not _hex(value["launcher_ownership_sha256"], 64)
            or not _hex(value["rank_ownership_sha256"], 64)):
        raise RuntimeError("STEP393 run fixed value mismatch")
    ready = value["ready"]
    ready_keys = {"schema", "rank", "local_rank", "world_size", "container_pid", "visible",
                  "torch_version", "torch_npu_version", "npu_available", "device_count",
                  "current_device", "startup_context_synchronized", "module_origin",
                  "shadow_package", "instrumentation_requested", "fallback_not_observed",
                  "task_queue_state", "task_queue_present", "task_queue_value_sha256"}
    if (not isinstance(ready, list) or len(ready) != 8
            or any(not isinstance(row, dict) or set(row) != ready_keys for row in ready)
            or {row.get("rank") for row in ready if isinstance(row, dict)} != set(range(8))
            or any(row.get("rank") != index or row.get("local_rank") != index
                   for index, row in enumerate(ready))
            or any(row.get("world_size") != 8 or row.get("visible") != "8,9,10,11,12,13,14,15"
                   or row.get("npu_available") is not True or row.get("device_count") != 8
                   or row.get("current_device") != row.get("local_rank")
                   or row.get("startup_context_synchronized") is not True
                   or row.get("instrumentation_requested") is not False
                   or row.get("fallback_not_observed") is not True
                   or row.get("task_queue_state") != "production-preserved" for row in ready)):
        raise RuntimeError("STEP393 run ready closure mismatch")
    task_queue_evidence = {(row["task_queue_present"], row["task_queue_value_sha256"])
                           for row in ready}
    if (len(task_queue_evidence) != 1
            or any(type(row["task_queue_present"]) is not bool
                   or (row["task_queue_present"] and not _hex(row["task_queue_value_sha256"], 64))
                   or (not row["task_queue_present"] and row["task_queue_value_sha256"] is not None)
                   for row in ready)):
        raise RuntimeError("STEP393 task queue preservation mismatch")
    binding = value["binding"]
    binding_keys = {"rank", "local_rank", "host_pid", "container_pid", "physical", "chip",
                    "device_id", "starttime", "pgid", "nspid", "argv"}
    if (not isinstance(binding, dict) or set(binding) != {"schema", "sample_sha256", "bindings"}
            or binding.get("schema") != "step377-back8-binding-v1"
            or not isinstance(binding.get("sample_sha256"), list)
            or len(binding["sample_sha256"]) != 2
            or not all(_hex(item, 64) for item in binding["sample_sha256"])
            or not isinstance(binding.get("bindings"), list) or len(binding["bindings"]) != 8
            or any(not isinstance(row, dict) or set(row) != binding_keys
                   or not isinstance(row["nspid"], list) or not isinstance(row["argv"], list)
                   for row in binding["bindings"])
            or {row.get("rank") for row in binding["bindings"]} != set(range(8))
            or {row.get("device_id") for row in binding["bindings"]} != set(range(8, 16))):
        raise RuntimeError("STEP393 rank/device binding mismatch")
    by_rank = {row["rank"]: row for row in binding["bindings"]}
    ownership = value["rank_ownership"]
    case_path = Path(str(ownership.get("case_path", ""))) if isinstance(ownership, dict) else Path()
    run_root = case_path.parent.parent
    integer_keys = binding_keys - {"nspid", "argv"}
    if (any(any(type(row[key]) is not int for key in integer_keys)
            or row["rank"] != row["local_rank"]
            or row["device_id"] != row["rank"] + 8
            or row["physical"] * 2 + row["chip"] != row["device_id"]
            or row["physical"] != 4 + row["rank"] // 2
            or row["chip"] != row["rank"] % 2
            or row["host_pid"] <= 1 or row["container_pid"] <= 1
            or row["starttime"] <= 0 or row["pgid"] <= 1
            or len(row["nspid"]) < 2
            or any(type(item) is not int or item <= 0 for item in row["nspid"])
            or row["nspid"][0] != row["host_pid"] or row["nspid"][-1] != row["container_pid"]
            or not _valid_worker_argv(row["argv"], run_root)
            for row in binding["bindings"])
            or len({row["host_pid"] for row in binding["bindings"]}) != 8
            or len({row["container_pid"] for row in binding["bindings"]}) != 8
            or len({(row["host_pid"], row["starttime"]) for row in binding["bindings"]}) != 8
            or len({row["pgid"] for row in binding["bindings"]}) != 1
            or any(by_rank[rank]["container_pid"] != ready[rank]["container_pid"]
                   for rank in range(8))):
        raise RuntimeError("STEP393 ready/binding strict identity mismatch")
    if (not isinstance(ownership, dict) or ownership.get("schema") != "step377-rank-ownership-v1"
            or ownership.get("launcher_ownership_sha256") != value["launcher_ownership_sha256"]
            or not _hex(ownership.get("gate_token_sha256"), 64)
            or not case_path.is_absolute() or case_path != run_root / "tools" / "step393_training_entry.py"
            or ownership.get("port") != MASTER_PORT or ownership.get("ranks") != binding["bindings"]):
        raise RuntimeError("STEP393 rank ownership cross-check mismatch")
    expected_shadow = run_root / "shadow_work" / "shadow" / "mx_driving_cloud"
    if any(Path(row["shadow_package"]) != expected_shadow
           or Path(row["module_origin"]) != expected_shadow / "__init__.py" for row in ready):
        raise RuntimeError("STEP393 ready shadow route mismatch")
    ownership_payload = (json.dumps(ownership, sort_keys=True, allow_nan=False) + "\n").encode()
    if hashlib.sha256(ownership_payload).hexdigest() != value["rank_ownership_sha256"]:
        raise RuntimeError("STEP393 rank ownership digest mismatch")
    native = value["native_log"]
    if (not isinstance(native, dict) or set(native) != {
            "path", "type", "size", "inode", "device", "sha256", "iterations",
            "created_in_new_run"}
            or native["type"] != "file" or native["size"] <= 0 or native["inode"] <= 0
            or Path(native["path"]) != run_root / "run" / "work" / "train.log"
            or native["created_in_new_run"] is not True
            or not _hex(native["sha256"], 64) or native["iterations"] != list(range(1, 31))):
        raise RuntimeError("STEP393 native log closure mismatch")
    validate_loss_gate(value["loss_gate"])
    timing = value["timing"]
    if (not isinstance(timing, dict) or timing.get("report_only") is not True
            or timing.get("iter_start") != 15 or timing.get("iter_end") != 29
            or timing.get("excluded") != [24] or timing.get("count") != 14):
        raise RuntimeError("STEP393 timing report mismatch")
    cleanup = value["cleanup_postflight"]
    stable = cleanup.get("stable_clear") if isinstance(cleanup, dict) else None
    if (not isinstance(cleanup, dict) or cleanup.get("schema") != "step393-success-postflight-v1"
            or cleanup.get("rank_dead") != [True] * 8 or cleanup.get("launcher_poll") != 0
            or cleanup.get("port_free") is not True or not isinstance(stable, dict)
            or stable.get("schema") != "step377-stable-clear-v1"
            or stable.get("back8_process_count") != 0 or stable.get("case_process_count") != 0
            or not isinstance(stable.get("sample_sha256"), list)
            or len(stable["sample_sha256"]) != 2
            or not all(_hex(item, 64) for item in stable["sample_sha256"])):
        raise RuntimeError("STEP393 success postflight mismatch")
    return value


def load_backend() -> Backend:
    spec = importlib.util.spec_from_file_location("_step393_real_backend", REMOTE_BACKEND)
    if spec is None or spec.loader is None:
        raise RuntimeError("STEP393 cannot load RealBackend")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RealBackend()


def execute(backend: Backend | None = None) -> dict[str, Any]:
    """Execute one remote transaction after both explicit source-review gates."""
    plan = local_preflight()
    if backend is None:
        backend = load_backend()
    session: Any | None = None
    primary: BaseException | None = None
    result: dict[str, Any] | None = None
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    try:
        session = backend.open_once(plan)
        backend.create_new_diag(session, plan)
        backend.upload_locked(session, plan)
        backend.archive_source(session, plan)
        backend.verify_source(session, plan)
        backend.prepare_shadow(session, plan)
        before = validate_snapshot(backend.snapshot(session, "pre", plan), phase="pre")
        run = validate_run(backend.run_training_live_gated(session, plan))
        loss = validate_loss_gate(backend.run_loss_gate(session, plan))
        after = validate_snapshot(backend.snapshot(session, "post", plan), phase="post")
        compare_closure(before, after)
        result = {
            "schema": "step393-delta2-shadow-e2e30-result-v1",
            "status": "E2E_SHADOW_CONFIGURED", "run": run, "loss_gate": loss,
            "closure": "pass", "remote_artifacts_downloaded": False,
        }
    except BaseException as error:
        primary = error
    finally:
        if session is not None and before is not None and after is None:
            try:
                after = validate_snapshot(backend.snapshot(session, "post", plan), phase="post")
                compare_closure(before, after)
            except BaseException as cleanup:
                if primary is None:
                    primary = cleanup
                else:
                    setattr(primary, "postflight_error", cleanup)
        if session is not None:
            try:
                backend.close(session)
            except BaseException as cleanup:
                if primary is None:
                    primary = cleanup
                else:
                    setattr(primary, "cleanup_error", cleanup)
    if primary is not None:
        raise primary
    if result is None:
        raise RuntimeError("STEP393 transaction produced no result")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(validate_plan(build_plan()), sort_keys=True))
        return 0
    execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
