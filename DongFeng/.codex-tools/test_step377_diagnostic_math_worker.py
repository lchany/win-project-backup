#!/usr/bin/env python3
"""Offline tests for the STEP377 diagnostic math-worker adapter."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import step377_diagnostic_math_worker as adapter


class Step377DiagnosticWorkerTests(unittest.TestCase):
    @staticmethod
    def _fake_worker(referenced: dict[str, int]):
        mappings = {index: name for index, name in enumerate(referenced, 1)}
        references = {index: count for index, count in enumerate(referenced.values(), 1)}
        legacy = types.SimpleNamespace(
            collect_runtime_identity=lambda _root: (mappings, references, [], [])
        )
        worker = types.SimpleNamespace(
            CANDIDATE_AIC=adapter.V5_AIC,
            CANDIDATE_AIV=adapter.V5_AIV,
            legacy=legacy,
        )

        def verify(_root, *, expected_aic_references):
            if referenced.get(worker.CANDIDATE_AIC, 0) != expected_aic_references:
                raise RuntimeError("base identity mismatch")
            allowed = {worker.CANDIDATE_AIC, worker.CANDIDATE_AIV}
            if any(name not in allowed for name in referenced):
                raise RuntimeError("base forbidden identity")
            return {"pass": True, "raw_profile_retained": True}

        worker.verify_profile = verify
        worker._finalize_call = lambda *_args, **_kwargs: {
            "input_unmodified": True, "shape_pass": True, "finite_pass": True,
            "reconstruction": {"violation_count": 0},
            "orthogonality": {"violation_count": 0},
            "full_rank_projection": {"required": True, "pass": True},
            "lower_triangle_exact_zero": True,
        }
        worker._normalize_json_diagnostic = lambda _value: (_value, 0)
        return worker

    def test_dependencies_are_locked_and_loading_does_not_import_torch(self) -> None:
        before = set(sys.modules)
        worker = adapter.load_worker()
        self.assertTrue(callable(worker.main))
        self.assertNotIn("torch", set(sys.modules) - before)
        self.assertNotIn("torch_npu", set(sys.modules) - before)
        for path in (adapter.WORKER_PATH, adapter.COLD_CASE_PATH, adapter.ORACLE_PATH):
            self.assertEqual(adapter.sha256_file(path), adapter.EXPECTED_SHA256[path.name])
        cold = adapter._load_module("_step377_test_cold", adapter.COLD_CASE_PATH)
        self.assertTrue(callable(cold.wait_release))
        self.assertNotIn("_step377_test_cold", sys.modules)

        with tempfile.TemporaryDirectory() as directory:
            dataclass_module = Path(directory) / "dataclass_module.py"
            dataclass_module.write_text(
                "from __future__ import annotations\n"
                "from dataclasses import dataclass\n"
                "@dataclass\nclass Value:\n    field: int\n",
                encoding="utf-8",
            )
            loaded = adapter._load_module("_step377_test_dataclass", dataclass_module)
            self.assertEqual(loaded.Value(3).field, 3)
            self.assertNotIn("_step377_test_dataclass", sys.modules)

    def test_isolated_exec_failure_restores_previous_module_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.py"
            path.write_text("raise RuntimeError('exec failed')\n", encoding="utf-8")
            previous = types.ModuleType("previous")
            sys.modules["_step377_broken"] = previous
            try:
                with self.assertRaisesRegex(RuntimeError, "exec failed"):
                    adapter._load_module("_step377_broken", path)
                self.assertIs(sys.modules["_step377_broken"], previous)
            finally:
                sys.modules.pop("_step377_broken", None)

    def test_exact_diagnostic_aic_reference_passes_with_explicit_zero_counts(self) -> None:
        worker = self._fake_worker({adapter.DIAGNOSTIC_AIC: 1})
        restore = adapter.install_worker_identity(worker)
        try:
            result = worker.verify_profile(Path("profile"), expected_aic_references=1)
            self.assertEqual(result["diagnostic_aic_task_reference_count"], 1)
            for key in (
                "diagnostic_aiv_task_reference_count", "original_task_reference_count",
                "v4_task_reference_count", "v5_task_reference_count",
                "unknown_qrv2_task_reference_count",
            ):
                self.assertEqual(result[key], 0)
            self.assertTrue(result["raw_profile_retained"])
        finally:
            restore()
        self.assertEqual(worker.CANDIDATE_AIC, adapter.V5_AIC)

    def test_zero_two_aiv_old_v4_v5_and_unknown_references_fail(self) -> None:
        cases = (
            {},
            {adapter.DIAGNOSTIC_AIC: 2},
            {adapter.DIAGNOSTIC_AIC: 1, adapter.DIAGNOSTIC_AIV: 1},
            {adapter.DIAGNOSTIC_AIC: 1, adapter.ORIGINAL_AIC: 1},
            {adapter.DIAGNOSTIC_AIC: 1, adapter.ORIGINAL_AIV: 1},
            {adapter.DIAGNOSTIC_AIC: 1, adapter.V4_AIC: 1},
            {adapter.DIAGNOSTIC_AIC: 1, adapter.V4_AIV: 1},
            {adapter.DIAGNOSTIC_AIC: 1, adapter.V5_AIC: 1},
            {adapter.DIAGNOSTIC_AIC: 1, adapter.V5_AIV: 1},
            {adapter.DIAGNOSTIC_AIC: 1, "QrV2_unknown_0_mix_aic": 1},
        )
        for referenced in cases:
            with self.subTest(referenced=referenced):
                worker = self._fake_worker(referenced)
                restore = adapter.install_worker_identity(worker)
                try:
                    with self.assertRaises(RuntimeError):
                        worker.verify_profile(Path("profile"), expected_aic_references=1)
                finally:
                    restore()

    def test_expected_reference_argument_must_be_one(self) -> None:
        worker = self._fake_worker({adapter.DIAGNOSTIC_AIC: 1})
        restore = adapter.install_worker_identity(worker)
        try:
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                worker.verify_profile(Path("profile"), expected_aic_references=2)
        finally:
            restore()

    def test_success_call_is_enriched_with_exact_predicate_summary(self) -> None:
        worker = self._fake_worker({adapter.DIAGNOSTIC_AIC: 1})
        original = worker._finalize_call
        restore = adapter.install_worker_identity(worker)
        try:
            result = worker._finalize_call()
            self.assertEqual(set(result["predicate_status"].values()), {"pass"})
            self.assertEqual(result["failed_predicates"], [])
            self.assertEqual(result["not_evaluated_predicates"], [])
            self.assertTrue(result["diagnostic_scalars_finite"])
            self.assertEqual(result["diagnostic_nonfinite_scalar_count"], 0)
            self.assertEqual(result["reconstruction_violation_count"], 0)
            self.assertEqual(result["orthogonality_violation_count"], 0)
            self.assertTrue(result["projection_pass"])
        finally:
            restore()
        self.assertIs(worker._finalize_call, original)

    def test_wait_maps_only_to_diagnostic_gate_and_never_creates_release_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observed = []
            def create_gate(path, timeout_seconds=120):
                observed.append((path, timeout_seconds))
                path.write_text(json.dumps({
                    "schema": adapter.GATE_SCHEMA, "token": "1" * 32,
                }))

            cold = types.SimpleNamespace(wait_release=create_gate)
            original = cold.wait_release
            restore = adapter.install_diagnostic_wait(cold)
            legacy = root / adapter.LEGACY_START_NAME
            (root / adapter.GATE_ACK_DIR).mkdir()
            with mock.patch.dict(adapter.os.environ, {"LOCAL_RANK": "3"}):
                cold.wait_release(legacy, timeout_seconds=7)
            self.assertEqual(observed, [(root / adapter.DIAGNOSTIC_START_NAME, 7)])
            ack = json.loads((root / adapter.GATE_ACK_DIR / "rank3.json").read_text())
            self.assertEqual(ack["token_sha256"], adapter.hashlib.sha256(("1" * 32).encode()).hexdigest())
            self.assertFalse(legacy.exists())
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                cold.wait_release(root / "wrong")
            restore()
            self.assertIs(cold.wait_release, original)

    def test_duplicate_hashes_for_same_referenced_identity_are_rejected(self) -> None:
        worker = self._fake_worker({adapter.DIAGNOSTIC_AIC: 1})
        worker.legacy.collect_runtime_identity = lambda _root: (
            {11: adapter.DIAGNOSTIC_AIC, 22: adapter.DIAGNOSTIC_AIC},
            {11: 1, 22: 1},
            [],
            [],
        )
        restore = adapter.install_worker_identity(worker)
        try:
            with self.assertRaisesRegex(RuntimeError, "multiple hashes"):
                worker.verify_profile(Path("profile"), expected_aic_references=1)
        finally:
            restore()

    def test_diagnostic_gate_rejects_preexisting_symlink_and_replacement(self) -> None:
        for mutation in ("preexisting", "symlink", "replacement"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                legacy = root / adapter.LEGACY_START_NAME
                diagnostic = root / adapter.DIAGNOSTIC_START_NAME
                target = root / "target"
                target.touch()
                if mutation == "preexisting":
                    diagnostic.touch()
                elif mutation == "symlink":
                    diagnostic.symlink_to(target)

                def create(path, timeout_seconds=120):
                    path.write_text(json.dumps({
                        "schema": adapter.GATE_SCHEMA, "token": "1" * 32,
                    }))

                cold = types.SimpleNamespace(wait_release=create)
                restore = adapter.install_diagnostic_wait(cold)
                try:
                    if mutation != "replacement":
                        with self.assertRaises(RuntimeError):
                            cold.wait_release(legacy)
                    else:
                        real_open = adapter.os.open

                        def replace_before_open(path, *args, **kwargs):
                            if Path(path) == diagnostic:
                                path = Path(path)
                                path.unlink()
                                path.write_bytes(b"replacement")
                            return real_open(path, *args, **kwargs)

                        with mock.patch.object(adapter.os, "open", side_effect=replace_before_open):
                            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                                cold.wait_release(legacy)
                finally:
                    restore()
                self.assertFalse(legacy.exists())

    def test_gate_wrong_token_and_replace_restore_are_rejected(self) -> None:
        for mutation in ("wrong_token", "replace_restore"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                legacy = root / adapter.LEGACY_START_NAME
                diagnostic = root / adapter.DIAGNOSTIC_START_NAME
                (root / adapter.GATE_ACK_DIR).mkdir()

                def create(path, timeout_seconds=120):
                    token = "x" * 32 if mutation == "wrong_token" else "1" * 32
                    path.write_text(json.dumps({"schema": adapter.GATE_SCHEMA, "token": token}))

                cold = types.SimpleNamespace(wait_release=create)
                restore = adapter.install_diagnostic_wait(cold)
                try:
                    if mutation == "wrong_token":
                        with mock.patch.dict(adapter.os.environ, {"LOCAL_RANK": "0"}), self.assertRaises(RuntimeError):
                            cold.wait_release(legacy)
                    else:
                        real_open = adapter.os.open
                        original = None

                        def replace_restore(path, *args, **kwargs):
                            nonlocal original
                            if Path(path) == diagnostic and original is None:
                                original = diagnostic.read_bytes()
                                diagnostic.unlink()
                                diagnostic.write_bytes(original)
                            return real_open(path, *args, **kwargs)

                        with mock.patch.dict(adapter.os.environ, {"LOCAL_RANK": "0"}), mock.patch.object(
                            adapter.os, "open", side_effect=replace_restore
                        ), self.assertRaisesRegex(RuntimeError, "identity changed"):
                            cold.wait_release(legacy)
                finally:
                    restore()

    def test_cli_forces_exactly_one_first_profiled_only_and_rejects_user_flag(self) -> None:
        args = argparse.Namespace(
            input_dir=Path("input"), output_dir=Path("output"),
            shadow_root=Path("shadow"), installed_custom_opp=Path("installed"),
        )
        argv = adapter._underlying_argv(args)
        self.assertEqual(argv.count("--first-profiled-only"), 1)
        with self.assertRaises(SystemExit):
            adapter.parse_args([
                "--input-dir", "i", "--output-dir", "o", "--shadow-root", "s",
                "--installed-custom-opp", "c", "--first-profiled-only",
            ])

    def test_run_restores_argv_modules_and_injections_after_failure(self) -> None:
        args = argparse.Namespace(
            input_dir=Path("input"), output_dir=Path("output"),
            shadow_root=Path("shadow"), installed_custom_opp=Path("installed"),
        )
        cold = types.SimpleNamespace(wait_release=lambda *_a, **_k: None)
        worker = self._fake_worker({adapter.DIAGNOSTIC_AIC: 1})
        original_verify = worker.verify_profile
        worker.main = mock.Mock(side_effect=RuntimeError("worker failed"))
        previous_argv = sys.argv
        previous_cold = sys.modules.get("step343_qrv2_cold_case")
        with mock.patch.object(adapter, "_guard_dependencies"), mock.patch.object(
            adapter, "_load_module", return_value=cold
        ), mock.patch.object(adapter, "load_worker", return_value=worker):
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                adapter.run(args)
        self.assertIs(sys.argv, previous_argv)
        self.assertIs(worker.verify_profile, original_verify)
        self.assertEqual(worker.CANDIDATE_AIC, adapter.V5_AIC)
        self.assertIs(sys.modules.get("step343_qrv2_cold_case"), previous_cold)

    def test_partial_install_failure_restores_wait_and_module_binding(self) -> None:
        args = argparse.Namespace(
            input_dir=Path("input"), output_dir=Path("output"),
            shadow_root=Path("shadow"), installed_custom_opp=Path("installed"),
        )
        original_wait = lambda *_args, **_kwargs: None
        cold = types.SimpleNamespace(wait_release=original_wait)
        worker = self._fake_worker({adapter.DIAGNOSTIC_AIC: 1})
        previous = sys.modules.get("step343_qrv2_cold_case")
        with mock.patch.object(adapter, "_guard_dependencies"), mock.patch.object(
            adapter, "_load_module", return_value=cold
        ), mock.patch.object(adapter, "load_worker", return_value=worker), mock.patch.object(
            adapter, "install_worker_identity", side_effect=RuntimeError("install failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "install failed"):
                adapter.run(args)
        self.assertIs(cold.wait_release, original_wait)
        self.assertIs(sys.modules.get("step343_qrv2_cold_case"), previous)

    def test_restore_failure_does_not_replace_primary_or_skip_other_restores(self) -> None:
        args = argparse.Namespace(
            input_dir=Path("input"), output_dir=Path("output"),
            shadow_root=Path("shadow"), installed_custom_opp=Path("installed"),
        )
        original_wait = lambda *_args, **_kwargs: None
        cold = types.SimpleNamespace(wait_release=original_wait)
        worker = self._fake_worker({adapter.DIAGNOSTIC_AIC: 1})
        primary = RuntimeError("primary worker failure")
        worker.main = mock.Mock(side_effect=primary)

        def install_with_bad_restore(_worker):
            return lambda: (_ for _ in ()).throw(OSError("restore failed"))

        previous = sys.modules.get("step343_qrv2_cold_case")
        with mock.patch.object(adapter, "_guard_dependencies"), mock.patch.object(
            adapter, "_load_module", return_value=cold
        ), mock.patch.object(adapter, "load_worker", return_value=worker), mock.patch.object(
            adapter, "install_worker_identity", side_effect=install_with_bad_restore
        ):
            with self.assertRaises(RuntimeError) as raised:
                adapter.run(args)
        self.assertIs(raised.exception, primary)
        self.assertEqual(primary.args, ("primary worker failure",))
        self.assertEqual(str(primary), "primary worker failure")
        self.assertEqual(len(primary.cleanup_errors), 1)
        self.assertIs(cold.wait_release, original_wait)
        self.assertIs(sys.modules.get("step343_qrv2_cold_case"), previous)


if __name__ == "__main__":
    unittest.main()
