#!/usr/bin/env python3
"""Focused tests for the STEP384 delta2-only diagnostic build adapter."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import build_step384_qrv2_delta2_only_diagnostic as adapter
import test_step376_delta1_probe_build_wiring as step376_tests


class Step384BuildAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ready = mock.patch.object(adapter, "BUILD_READY", True)
        self.ready.start()
        adapter._activate()

    def tearDown(self) -> None:
        self.ready.stop()

    def test_identity_sha_reverse_v4_and_disarmed_state_are_locked(self) -> None:
        base = adapter._load_base()
        adapter._validate_active_wiring(base)
        self.assertEqual(adapter.BIN_NAME, "QrV2_qa_position_delta2_only_diagnostic_v1")
        self.assertEqual(base.EXPECTED_CANDIDATE_SHA256, "e352ac31f97980bc5c92caac663414782b78fc3004ff36709b6be8000353b003")
        self.assertEqual(adapter.diagnostic_patcher.EXPECTED_V4_CANDIDATE_SHA256, "2213dbae5027c179f09c8e3b43be51587ba22e3eb2a7546311048efc3844614b")
        self.assertTrue(adapter.BUILD_READY)
        self.assertFalse(hasattr(adapter, "REMOTE_READY"))

    def test_false_guard_is_first_and_performs_zero_io_or_loading(self) -> None:
        with mock.patch.object(adapter, "BUILD_READY", False), mock.patch.object(
            adapter, "_activate", side_effect=AssertionError("loaded")
        ), mock.patch.object(adapter, "_marker_path", side_effect=AssertionError("I/O")):
            calls = (
                lambda: adapter.prepare_release(Path("outer"), Path("work"), Path("root")),
                lambda: adapter.build_release(Path("work"), Path("opc"), Path("contract"), Path("installed"), Path("root")),
                lambda: adapter.main(["prepare"]),
            )
            for call in calls:
                with self.assertRaisesRegex(RuntimeError, "BUILD_READY is false"):
                    call()

    def test_source_default_build_ready_is_false_without_test_patch(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import build_step384_qrv2_delta2_only_diagnostic as a; assert a.BUILD_READY is False"],
            cwd=Path(__file__).resolve().parent,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dependency_sha_drift_is_rejected_before_dynamic_import(self) -> None:
        with mock.patch.object(adapter, "_sha256_file", return_value="0" * 64), mock.patch.object(
            adapter, "_load_module", side_effect=AssertionError("pre-import rejection failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "SHA drift"):
                adapter._activate()

    def test_tool_closure_locks_step384_patcher(self) -> None:
        hashes = adapter._tool_hashes()
        self.assertEqual(set(hashes), {"diagnostic_adapter_sha256", "audited_adapter_sha256", "base_builder_sha256", "step384_patcher_sha256", "v4_patcher_sha256"})
        self.assertEqual(hashes["step384_patcher_sha256"], "2bdaf51e3b08388ca5fcb156e0602312b4f1de3dfc533da6e2d7778d10d3820c")
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_private_audited_adapter_does_not_mutate_step376_import(self) -> None:
        import build_qrv2_diagnostic_probe as step376

        self.assertIsNot(adapter.audited_adapter, step376)
        self.assertEqual(step376.BIN_NAME, "QrV2_vtv_direct_qa_legacy_probe_v6")
        self.assertEqual(adapter.audited_adapter.BIN_NAME, adapter.BIN_NAME)

    def test_base_package_and_all_are_poisoned(self) -> None:
        base = adapter._load_base()
        for call in (
            lambda: base.package_release(Path("unused")),
            lambda: base.parse_args(["package", "unused"]),
            lambda: base.main(["all", "unused", "unused"]),
            lambda: adapter.package_release(Path("unused")),
        ):
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                call()
        for command in ("package", "all"):
            with self.subTest(command=command), mock.patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    adapter.parse_args([command])

    def test_approved_root_and_manifest_exclusive_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "work").mkdir()
            with self.assertRaises(FileExistsError):
                adapter._approved_workdir(root, root / "work", require_new=True)
            with self.assertRaises(ValueError):
                adapter._approved_workdir(root, root / "escape", require_new=True)

    def test_prepare_manifest_has_double_flags_and_no_release_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            base = adapter._load_base()
            manifest = {
                "status": "prepared",
                "policy": {},
                "candidate": {"source_sha256": adapter.diagnostic_patcher.EXPECTED_CANDIDATE_SHA256},
                "original": {"source_sha256": adapter.diagnostic_patcher.EXPECTED_SOURCE_SHA256},
            }
            decorated = adapter.audited_adapter._decorate_manifest(manifest, status="prepared")
            self.assertEqual(decorated["status"], "prepared")
            for layer in ("policy", "candidate"):
                for key, value in adapter._diagnostic_flags().items():
                    self.assertEqual(decorated[layer][key], value)
            self.assertEqual(decorated["package"], {"status": adapter.FORBIDDEN_PACKAGE_STATUS})
            work.mkdir()
            with (work / adapter.MANIFEST_NAME).open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(decorated))
            with self.assertRaises(FileExistsError):
                (work / adapter.MANIFEST_NAME).open("x", encoding="utf-8")
            adapter._assert_no_release_outputs(work, decorated)

    def test_nonzero_build_keeps_prepared_manifest_and_complete_residue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            prepared = {"status": "prepared", "sentinel": "not-consumable"}
            manifest_path = work / adapter.MANIFEST_NAME
            manifest_path.write_text(json.dumps(prepared), encoding="utf-8")
            residue = work / "build" / "soc" / "opc.log"
            residue.parent.mkdir(parents=True)
            residue.write_text("failure evidence", encoding="utf-8")
            base = types.SimpleNamespace()
            with mock.patch.object(adapter.audited_adapter, "_approved_workdir", return_value=(root, work)), mock.patch.object(adapter.audited_adapter, "_read_manifest", return_value=prepared), mock.patch.object(adapter.audited_adapter, "_validate_manifest"), mock.patch.object(adapter.audited_adapter, "_validate_active_wiring"), mock.patch.object(base, "build_release", side_effect=RuntimeError("opc rc=7"), create=True), mock.patch.object(base, "write_json_atomic", create=True), mock.patch.object(base, "_guard_tools", create=True):
                with self.assertRaisesRegex(RuntimeError, "rc=7"):
                    adapter.build_release(work, root / "opc", root / "contract", root / "installed", root, _base=base)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), prepared)
            self.assertEqual(residue.read_text(encoding="utf-8"), "failure evidence")
            self.assertNotEqual(prepared["status"], adapter.DIAGNOSTIC_BUILT_STATUS)
            marker = json.loads((work / adapter.ATTEMPT_MARKER_NAME).read_text(encoding="utf-8"))
            self.assertEqual(marker["schema"], adapter.ATTEMPT_MARKER_SCHEMA)
            self.assertEqual(marker["status"], "in_progress_nonconsumable")
            self.assertEqual(marker["attempt_identity"], adapter.ATTEMPT_IDENTITY)

    def test_marker_creation_failure_prevents_activation_and_opc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            with mock.patch.object(adapter, "_create_attempt_marker", side_effect=OSError("marker full")), mock.patch.object(adapter, "_activate", side_effect=AssertionError("activated")):
                with self.assertRaisesRegex(OSError, "marker full"):
                    adapter.build_release(work, root / "opc", root / "contract", root / "installed", root)

    def test_retry_after_failure_rejects_before_loading_or_opc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            (work / adapter.ATTEMPT_MARKER_NAME).write_text("{}", encoding="utf-8")
            with mock.patch.object(adapter, "_activate", side_effect=AssertionError("loaded")):
                with self.assertRaisesRegex(RuntimeError, "prior non-consumable"):
                    adapter.build_release(work, root / "opc", root / "contract", root / "installed", root)

    def test_crash_leaves_in_progress_marker_and_retry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            with mock.patch.object(adapter.audited_adapter, "build_release", side_effect=KeyboardInterrupt("crash")):
                with self.assertRaises(KeyboardInterrupt):
                    adapter.build_release(work, root / "opc", root / "contract", root / "installed", root)
            marker = json.loads((work / adapter.ATTEMPT_MARKER_NAME).read_text(encoding="utf-8"))
            self.assertEqual(marker["status"], "in_progress_nonconsumable")
            with mock.patch.object(adapter, "_activate", side_effect=AssertionError("retry activated")):
                with self.assertRaisesRegex(RuntimeError, "prior non-consumable"):
                    adapter.build_release(work, root / "opc", root / "contract", root / "installed", root)

    def test_canonical_success_alias_failure_marks_attempt_and_preserves_residue(self) -> None:
        helper = step376_tests.Step376DiagnosticAdapterTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            step376_tests, "adapter", adapter
        ):
            root = Path(directory)
            work, prepared = helper._prepare(root)
            base = adapter._load_base()
            opc, ascend_opp, installed, contract = helper._runtime_fixture(root, work, base)
            original_write = base.write_new_inside

            def fail_alias_copy(isolated_root: Path, output: Path, payload: bytes) -> None:
                if work / "build" / base.ALIAS_SOC_KEY in output.parents:
                    raise RuntimeError("alias copy failure")
                original_write(isolated_root, output, payload)

            with mock.patch.dict(
                os.environ,
                {"ASCEND_OPP_PATH": str(ascend_opp), "PYTHONPATH": "poisoned"},
                clear=False,
            ), mock.patch.object(base, "write_new_inside", side_effect=fail_alias_copy):
                with self.assertRaisesRegex(RuntimeError, "alias copy failure"):
                    adapter.build_release(work, opc, contract, installed, root, _base=base)
            canonical_dir = work / "build" / base.CANONICAL_SOC_KEY
            self.assertTrue((canonical_dir / "opc.log").is_file())
            self.assertTrue(any((canonical_dir / "output").glob("*.o")))
            self.assertEqual(json.loads((work / adapter.MANIFEST_NAME).read_text(encoding="utf-8")), prepared)
            self.assertTrue((work / adapter.ATTEMPT_MARKER_NAME).is_file())

    def test_seal_closure_failure_marks_attempt_without_consumable_manifest(self) -> None:
        helper = step376_tests.Step376DiagnosticAdapterTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            step376_tests, "adapter", adapter
        ):
            root = Path(directory)
            work, prepared = helper._prepare(root)
            base = adapter._load_base()
            opc, ascend_opp, installed, contract = helper._runtime_fixture(root, work, base)
            with mock.patch.dict(
                os.environ,
                {"ASCEND_OPP_PATH": str(ascend_opp), "PYTHONPATH": "poisoned"},
                clear=False,
            ), mock.patch.object(
                adapter.audited_adapter,
                "_validate_built_artifact_closure",
                side_effect=RuntimeError("closure failed before seal"),
            ):
                with self.assertRaisesRegex(RuntimeError, "closure failed"):
                    adapter.build_release(work, opc, contract, installed, root, _base=base)
            self.assertEqual(json.loads((work / adapter.MANIFEST_NAME).read_text(encoding="utf-8")), prepared)
            self.assertTrue((work / adapter.ATTEMPT_MARKER_NAME).is_file())

    def test_prepare_and_build_seal_both_soc_artifacts_as_diagnostic(self) -> None:
        helper = step376_tests.Step376DiagnosticAdapterTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            step376_tests, "adapter", adapter
        ):
            base, work, manifest = helper._successful_build(Path(directory))
            self.assertEqual(manifest["status"], adapter.DIAGNOSTIC_BUILT_STATUS)
            marker = json.loads((work / adapter.ATTEMPT_MARKER_NAME).read_text(encoding="utf-8"))
            self.assertEqual(marker["status"], "completed_consumable")
            adapter._validate_manifest(
                base, work, manifest, expected_status=adapter.DIAGNOSTIC_BUILT_STATUS
            )
            self.assertEqual(set(manifest["artifacts"]), set(base.SOCS))
            for layer in ("policy", "candidate"):
                self.assertTrue(manifest[layer]["diagnostic_only"])
                self.assertTrue(manifest[layer]["package_forbidden"])
                self.assertFalse(manifest[layer]["release_candidate"])
            self.assertEqual(manifest["package"], {"status": adapter.FORBIDDEN_PACKAGE_STATUS})
            self.assertFalse((work / "release").exists())

    def test_rename_failure_leaves_in_progress_and_nonconsumable(self) -> None:
        helper = step376_tests.Step376DiagnosticAdapterTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            step376_tests, "adapter", adapter
        ):
            root = Path(directory)
            work, _ = helper._prepare(root)
            base = adapter._load_base()
            opc, ascend_opp, installed, contract = helper._runtime_fixture(root, work, base)
            with mock.patch.dict(
                os.environ,
                {"ASCEND_OPP_PATH": str(ascend_opp), "PYTHONPATH": "poisoned"},
                clear=False,
            ), mock.patch.object(adapter.os, "rename", side_effect=OSError("rename failed")):
                with self.assertRaisesRegex(OSError, "rename failed"):
                    adapter.build_release(work, opc, contract, installed, root, _base=base)
            marker = json.loads((work / adapter.ATTEMPT_MARKER_NAME).read_text(encoding="utf-8"))
            self.assertEqual(marker["status"], "in_progress_nonconsumable")
            manifest = json.loads((work / adapter.MANIFEST_NAME).read_text(encoding="utf-8"))
            with self.assertRaisesRegex(RuntimeError, "not completed"):
                adapter._validate_manifest(base, work, manifest, expected_status=adapter.DIAGNOSTIC_BUILT_STATUS)

    def test_rename_is_unique_commit_and_post_commit_close_is_swallowed(self) -> None:
        helper = step376_tests.Step376DiagnosticAdapterTests(methodName="runTest")
        committed = False
        real_rename = os.rename
        real_close = os.close
        real_fsync = os.fsync

        def commit_rename(*args, **kwargs):
            nonlocal committed
            result = real_rename(*args, **kwargs)
            committed = True
            return result

        def guarded_close(fd: int) -> None:
            if committed:
                raise OSError("post-commit close failure")
            real_close(fd)

        def guarded_fsync(fd: int) -> None:
            if committed:
                raise AssertionError("fsync occurred after commit")
            real_fsync(fd)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            step376_tests, "adapter", adapter
        ):
            with mock.patch.object(adapter.os, "rename", side_effect=commit_rename), mock.patch.object(
                adapter.os, "close", side_effect=guarded_close
            ), mock.patch.object(adapter.os, "fsync", side_effect=guarded_fsync):
                base, work, manifest = helper._successful_build(Path(directory))
            self.assertTrue(committed)
            adapter._validate_manifest(base, work, manifest, expected_status=adapter.DIAGNOSTIC_BUILT_STATUS)

    def test_committed_build_recovers_after_caller_interrupt(self) -> None:
        helper = step376_tests.Step376DiagnosticAdapterTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            step376_tests, "adapter", adapter
        ):
            base, work, manifest = helper._successful_build(Path(directory))
            try:
                raise KeyboardInterrupt("caller interrupted after return")
            except KeyboardInterrupt:
                recovered = json.loads((work / adapter.MANIFEST_NAME).read_text(encoding="utf-8"))
            adapter._validate_manifest(base, work, recovered, expected_status=adapter.DIAGNOSTIC_BUILT_STATUS)
            self.assertEqual(recovered, manifest)

    def test_marker_copy_replacement_and_manifest_tamper_are_rejected(self) -> None:
        helper = step376_tests.Step376DiagnosticAdapterTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            step376_tests, "adapter", adapter
        ):
            base, work, manifest = helper._successful_build(Path(directory))
            marker_path = work / adapter.ATTEMPT_MARKER_NAME
            copied = marker_path.read_bytes()
            marker_path.unlink()
            marker_path.write_bytes(copied)
            with self.assertRaisesRegex(RuntimeError, "replacement detected"):
                adapter._validate_manifest(base, work, manifest, expected_status=adapter.DIAGNOSTIC_BUILT_STATUS)
            marker_path.unlink()
            marker_path.symlink_to(work / adapter.MANIFEST_NAME)
            with self.assertRaises(OSError):
                adapter._validate_manifest(base, work, manifest, expected_status=adapter.DIAGNOSTIC_BUILT_STATUS)
            marker_path.unlink()
            marker_path.write_bytes(b"{malformed")
            with self.assertRaisesRegex(RuntimeError, "malformed"):
                adapter._validate_manifest(base, work, manifest, expected_status=adapter.DIAGNOSTIC_BUILT_STATUS)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            step376_tests, "adapter", adapter
        ):
            base, work, manifest = helper._successful_build(Path(directory))
            manifest_path = work / adapter.MANIFEST_NAME
            manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(RuntimeError, "manifest SHA drift"):
                adapter._validate_manifest(base, work, manifest, expected_status=adapter.DIAGNOSTIC_BUILT_STATUS)

    def test_artifact_closure_requires_both_socs_and_rejects_packages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            object_path = root / "kernel.o"
            json_path = root / "kernel.json"
            log_path = root / "build" / "canonical" / "opc.log"
            log_path.parent.mkdir(parents=True)
            object_path.write_bytes(b"object")
            json_path.write_bytes(b"json")
            log_path.write_bytes(b"log")
            base = types.SimpleNamespace(
                SOCS=("canonical", "alias"),
                CANONICAL_SOC_KEY="canonical",
                ALIAS_SOC_KEY="alias",
                _assert_no_symlinks=lambda _root: None,
                _validate_artifacts=lambda _build: (
                    object_path,
                    json_path,
                    {"kernelName": "k", "binFileName": "k", "_audited_concrete_entries": []},
                ),
                sha256_file=adapter._sha256_file,
            )
            canonical = {
                "status": "built_structure_valid",
                "object_path": str(object_path),
                "object_size": object_path.stat().st_size,
                "object_sha256": adapter._sha256_file(object_path),
                "json_path": str(json_path),
                "json_size": json_path.stat().st_size,
                "json_sha256": adapter._sha256_file(json_path),
                "opc_log_path": str(log_path),
                "opc_log_size": log_path.stat().st_size,
                "opc_log_sha256": adapter._sha256_file(log_path),
                "kernel_name": "k",
                "bin_file_name": "k",
                "concrete_entries": [],
            }
            with self.assertRaisesRegex(RuntimeError, "alias OPC log closure failed"):
                adapter._validate_built_artifact_closure(base, root, {"artifacts": {"canonical": canonical}}, enrich=False)
            (root / "new.whl").write_bytes(b"forbidden")
            with self.assertRaisesRegex(RuntimeError, "wheel forbidden"):
                adapter._assert_no_release_outputs(root, {"paths": {}})


if __name__ == "__main__":
    unittest.main()
