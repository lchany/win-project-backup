#!/usr/bin/env python3
"""One focused offline contract group for the disarmed STEP393 E2E chain."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).resolve().parent
CONTROLLER_PATH = TOOLS / "step393_run_delta2_shadow_30.py"
BACKEND_PATH = TOOLS / "step393_remote_backend.py"
ENTRY_PATH = TOOLS / "step393_training_entry.py"
RUNNER_PATH = TOOLS / "run_step393_delta2_shadow_30.sh"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load("step393_controller", CONTROLLER_PATH)
B = load("step393_backend", BACKEND_PATH)
E = load("step393_entry", ENTRY_PATH)


def inventory(root: str, count: int) -> dict[str, object]:
    entries = [
        {"path": f"file{index}.bin", "type": "file", "size": index + 1,
         "sha256": f"{index + 1:064x}"}
        for index in range(count)
    ]
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {"root": root, "type": "fixed_file_set", "file_count": count,
            "entries": entries, "inventory_sha256": hashlib.sha256(payload).hexdigest()}


def snapshot(phase: str) -> dict[str, object]:
    shadow = inventory("/diag/shadow", 2)
    shadow["type"] = "directory"
    shadow["manifest"] = {"path": "/diag/shadow_manifest.json", "type": "file",
                          "size": 12, "sha256": "a" * 64}
    installed = inventory("/installed", 1)
    installed["type"] = "directory"
    return {
        "schema": "step393-protected-snapshot-v2", "phase": phase,
        "installed": installed,
        "original": {"root": C.SOURCE_REPO, "type": "git_worktree",
                     "head": C.SOURCE_COMMIT, "status_bytes": 7, "status_count": 1,
                     "status_sha256": "b" * 64, "soap_blob": C.SOAP_BLOB},
        "attempt5": inventory(C.ATTEMPT5_DIR, 8), "shadow": shadow,
        "process": {"type": "process_set", "count": 0, "entries": []},
        "port": {"port": C.MASTER_PORT, "free": True},
        "npu": {"scope": "back8", "process_count": 0, "entries": [],
                "sample_sha256": ["c" * 64, "d" * 64]},
    }


def loss_gate() -> dict[str, object]:
    return {"status": "pass", "threshold": 0.02, "iter_start": 1, "iter_end": 30,
            "expected_count": 30, "pass_count": 30, "fail_count": 0,
            "failure_reason_counts": {}, "first_failure": None,
            "max_relative_deviation": 0.01, "max_relative_deviation_iter": 7}


def run_result() -> dict[str, object]:
    ready = [
        {"schema": "step393-rank-ready-v1", "rank": rank, "local_rank": rank,
         "world_size": 8, "container_pid": 100 + rank,
         "visible": "8,9,10,11,12,13,14,15", "npu_available": True,
         "device_count": 8, "current_device": rank, "startup_context_synchronized": True,
         "torch_version": "2", "torch_npu_version": "2",
         "module_origin": "/diag/shadow_work/shadow/mx_driving_cloud/__init__.py",
         "shadow_package": "/diag/shadow_work/shadow/mx_driving_cloud",
         "instrumentation_requested": False, "fallback_not_observed": True,
         "task_queue_state": "production-preserved", "task_queue_present": False,
         "task_queue_value_sha256": None}
        for rank in range(8)
    ]
    bindings = [{"rank": rank, "local_rank": rank, "host_pid": 1000 + rank,
                 "container_pid": 100 + rank, "physical": 4 + rank // 2,
                 "chip": rank % 2, "device_id": 8 + rank, "starttime": 500 + rank,
                 "pgid": 900, "nspid": [1000 + rank, 100 + rank],
                 "argv": ["python3", "-u", "/diag/tools/step393_training_entry.py",
                          "/diag/tools/step393_canonical_aligned_gpu_contract_npu_runtime.py",
                          "--work-dir", "/diag/run/work", "--gpus", "8",
                          "--autoscale-lr", "--max-iters", "30", "--launcher=pytorch"]}
                for rank in range(8)]
    ownership = {"schema": "step377-rank-ownership-v1",
                 "launcher_ownership_sha256": "f" * 64, "gate_token_sha256": "2" * 64,
                 "case_path": "/diag/tools/step393_training_entry.py",
                 "port": C.MASTER_PORT, "ranks": bindings}
    ownership_sha = hashlib.sha256(
        (json.dumps(ownership, sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()
    return {
        "schema": "step393-e2e-shadow-configured-v1", "status": "E2E_SHADOW_CONFIGURED",
        "instrumentation_requested": False, "fallback_not_observed": True,
        "concrete_kernel_identity": "not_claimed_instrumentation_not_requested",
        "launcher_rc": 0, "rank_count": 8, "gate_ack_count": 8,
        "ready": ready,
        "binding": {"schema": "step377-back8-binding-v1",
                    "sample_sha256": ["5" * 64, "6" * 64], "bindings": bindings},
        "native_log": {"path": "/diag/run/work/train.log", "type": "file",
                       "size": 100, "inode": 9, "device": 3, "sha256": "e" * 64,
                       "iterations": list(range(1, 31)), "created_in_new_run": True},
        "loss_gate": loss_gate(),
        "timing": {"report_only": True, "iter_start": 15, "iter_end": 29,
                   "excluded": [24], "count": 14},
        "capture_profile_dump_count": 0, "launcher_ownership_sha256": "f" * 64,
        "rank_ownership_sha256": ownership_sha, "rank_ownership": ownership,
        "cleanup_postflight": {"schema": "step393-success-postflight-v1",
                               "rank_dead": [True] * 8, "launcher_poll": 0,
                               "stable_clear": {"schema": "step377-stable-clear-v1",
                                                "back8_process_count": 0,
                                                "case_process_count": 0,
                                                "sample_sha256": ["3" * 64, "4" * 64]},
                               "port_free": True},
    }


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def open_once(self, _plan): self.calls.append("open"); return self
    def create_new_diag(self, _session, _plan): self.calls.append("create")
    def upload_locked(self, _session, _plan): self.calls.append("upload")
    def archive_source(self, _session, _plan): self.calls.append("archive")
    def verify_source(self, _session, _plan): self.calls.append("verify")
    def prepare_shadow(self, _session, _plan): self.calls.append("shadow")
    def snapshot(self, _session, phase, _plan): self.calls.append(phase); return snapshot(phase)
    def run_training_live_gated(self, _session, _plan): self.calls.append("run"); return run_result()
    def run_loss_gate(self, _session, _plan): self.calls.append("loss"); return loss_gate()
    def close(self, _session): self.calls.append("close")


class Step393FocusedContract(unittest.TestCase):
    def test_final_contract_and_negative_table(self) -> None:
        plan = C.validate_plan(C.build_plan())
        self.assertEqual(plan["status"], "NO_GO_PHASE_TRANSITION")
        self.assertEqual(plan["remote_directory"],
                         "step393_attempt6_delta2_shadow_e2e30_back8_world8_20260822")
        self.assertFalse(C.E2E_READY)
        self.assertFalse(C.PROCESS_GUARD_READY)
        with self.assertRaisesRegex(RuntimeError, "intentionally disarmed"):
            C.local_preflight()

        for name, row in plan["local_files"].items():
            self.assertEqual(C.sha256_file(Path(row["local_path"])), row["sha256"], name)
        self.assertEqual(
            B.GUARD_SHA256,
            C.LOCKED_LOCAL_SHA256["step393_process_guard.py"],
        )
        for name, path in ((C.REMOTE_EXEC.name, C.REMOTE_EXEC), (C.STEP357.name, C.STEP357)):
            self.assertEqual(C.sha256_file(path), C.LOCAL_SUPPORT_SHA256[name])

        runner = RUNNER_PATH.read_text(encoding="utf-8")
        training_entry = ENTRY_PATH.read_text(encoding="utf-8")
        self.assertIn("torch.empty(1", training_entry)
        self.assertIn("torch.npu.synchronize()", training_entry)
        for token in (
            "expected_commit=27b1d6d3f363619ad2faa244abe8fbc5a97faef6",
            "expected_soap_blob=77e412c4bece2ca95fd6dbf95732b89951924874",
            "lines == [429, 529]", "MAX_ITERS=30", "GPUS=8",
            "ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15",
            "torch.npu.device_count() == 8", "shadow_module",
            "ASCEND_CUSTOM_OPP_PATH=\"$shadow_opp:$installed_custom_opp\"",
            "step340_loss_gate.py", "--threshold 0.02", "--start-iter 1 --end-iter 30",
            "LD_PRELOAD PYTHONSTARTUP PYTHONHOME", "PYTHONNOUSERSITE=1", "PYTHONSAFEPATH=1",
            "production-preserved",
            "step393_canonical_aligned_gpu_contract_npu_runtime.py",
            "canonical_config_resolved", "archived_base",
        ):
            self.assertIn(token, runner)
        self.assertNotIn("HCCL_DEBUG TASK_QUEUE_ENABLE", runner)
        for forbidden in ("pkill", "killpg", "kill -- -", "pip install", "apt install",
                          "git clone", "curl ", "wget ", "torch.cuda",
                          "torch.npu.synchronize()", "SOAP_QR_DUMP_DIR=", "PROFILING_MODE="):
            self.assertNotIn(forbidden, runner)
        syntax = subprocess.run(["bash", "-n", str(RUNNER_PATH)], check=False,
                                capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        token_sha = "2" * 64
        B.ACTIVE_INSTALLED = Path("/opt/cloud/packages/vendors/customize")
        command = B.launcher_command(Path("/diag"), Path("/diag/source"), Path(B.CONTRACT_DIR),
                                     Path("/diag/shadow_work/shadow"), Path("/diag/run"),
                                     C.MASTER_PORT, Path("/opt/cloud/packages/vendors/customize"), token_sha)
        encoded = tuple(item.encode() for item in command)
        self.assertTrue(B.step393_launcher_argv(encoded, C.MASTER_PORT))
        bootstrap = tuple(item.encode() for item in B.bootstrap_command(Path("/diag"), command))
        self.assertTrue(B.step393_launcher_argv(bootstrap, C.MASTER_PORT))
        bad_bootstrap = list(bootstrap)
        bad_bootstrap[4] = b"/diag/run/wrong.gate"
        self.assertFalse(B.step393_launcher_argv(tuple(bad_bootstrap), C.MASTER_PORT))
        held = mock.Mock(pid=123)
        held.wait.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "pidfd_open failed before bootstrap release"):
            B.open_bootstrap_pidfd(held, mock.Mock(side_effect=OSError("pidfd failed")))
        held.wait.assert_called_once_with(timeout=70)
        worker = tuple(item.encode() for item in (
            "python3", "-u", "/diag/tools/step393_training_entry.py",
            "/diag/tools/step393_canonical_aligned_gpu_contract_npu_runtime.py",
            "--work-dir", "/diag/run/work",
            "--gpus", "8", "--autoscale-lr", "--max-iters", "30", "--launcher=pytorch"))
        self.assertTrue(B.step393_worker_argv(worker))
        grammar_negatives = [
            (*encoded[:-1], b"/tmp/not-installed"),
            (*encoded[:-1], b"/other/packages/vendors/customize"),
            tuple(b"mapqr" if item == B.CONTAINER.encode() else item for item in encoded),
            tuple(b"7" if item == b"8" else item for item in worker),
            (*worker, b"--unexpected"),
            (b"python3", b"/tmp/step392_delta2_worker.py", b"--launcher=pytorch"),
        ]
        for index, bad in enumerate(grammar_negatives):
            with self.subTest(grammar_negative=index):
                self.assertFalse(B.step393_launcher_argv(bad, C.MASTER_PORT)
                                 or B.step393_worker_argv(bad))
        compile(B.upload_readback_script(), "<step393-upload-readback>", "exec")
        unusual = "/tmp/example with spaces;and-semicolon"
        self.assertEqual(shlex.split(B.shell_test_not_exists(unusual)),
                         ["[", "!", "-e", unusual, "]"])

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "new.json"
            E.write_new_json(target, {"ok": True})
            with self.assertRaises(FileExistsError):
                E.write_new_json(target, {"ok": False})

        plan_negatives = {
            "container": "mapqr", "source_commit": "0" * 40,
            "soap": {"relative": C.SOAP_REL, "blob": C.SOAP_BLOB, "call_lines": [429]},
            "world_size": 7, "max_iters": 29, "shadow_first": False, "profile": True,
            "capture": True, "dump": True, "debug": True, "per_qr_synchronize": True,
            "download_remote_artifacts": True, "actions": list(reversed(C.TRANSACTION_ACTIONS)),
        }
        for key, bad_value in plan_negatives.items():
            with self.subTest(plan_negative=key):
                bad = copy.deepcopy(plan)
                bad[key] = bad_value
                with self.assertRaises(RuntimeError):
                    C.validate_plan(bad)

        pre = snapshot("pre")
        post = snapshot("post")
        C.compare_closure(pre, post)
        bad_snapshot = copy.deepcopy(post)
        bad_snapshot["attempt5"]["entries"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "inventory digest"):
            C.validate_snapshot(bad_snapshot, phase="post")
        changed = copy.deepcopy(post)
        changed["installed"] = inventory("/installed", 2)
        changed["installed"]["type"] = "directory"
        with self.assertRaisesRegex(RuntimeError, "installed changed"):
            C.compare_closure(pre, changed)

        valid_run = run_result()
        C.validate_run(valid_run)
        for key, bad_value in (
            ("status", "CANDIDATE_AIC_CONFIRMED"),
            ("concrete_kernel_identity", C.CANDIDATE_IDENTITY),
            ("rank_count", 7), ("capture_profile_dump_count", 1),
        ):
            with self.subTest(run_negative=key):
                bad = copy.deepcopy(valid_run)
                bad[key] = bad_value
                with self.assertRaises(RuntimeError):
                    C.validate_run(bad)
        bad_binding = copy.deepcopy(valid_run)
        bad_binding["binding"]["bindings"][0]["physical"] = -1
        bad_binding["binding"]["bindings"][0]["chip"] = 10
        bad_binding["rank_ownership"]["ranks"][0]["physical"] = -1
        bad_binding["rank_ownership"]["ranks"][0]["chip"] = 10
        with self.assertRaisesRegex(RuntimeError, "strict identity"):
            C.validate_run(bad_binding)

        fake = FakeBackend()
        with mock.patch.object(C, "E2E_READY", True), mock.patch.object(C, "PROCESS_GUARD_READY", True):
            self.assertEqual(C.local_preflight()["status"], "GO_REVIEWED_ONCE")
            result = C.execute(fake)
        self.assertEqual(result["status"], "E2E_SHADOW_CONFIGURED")
        self.assertEqual(fake.calls, ["open", "create", "upload", "archive", "verify", "shadow",
                                      "pre", "run", "loss", "post", "close"])

        dry = subprocess.run([sys.executable, str(CONTROLLER_PATH), "--dry-run"], check=True,
                             capture_output=True, text=True)
        self.assertEqual(json.loads(dry.stdout)["status"], "NO_GO_PHASE_TRANSITION")


if __name__ == "__main__":
    unittest.main()
