#!/usr/bin/env python3
"""Offline tests for the STEP377 diagnostic host adapter."""

from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import step377_diagnostic_host_case as adapter
import step377_diagnostic_math_worker as worker_adapter


def wrapper_contract():
    return {
        "gate": "PASS", "source_sha256": adapter.WRAPPER_SOURCE_SHA256,
        "threshold": 80, "block_tiling": 64,
    }


class Step377HostTests(unittest.TestCase):
    @staticmethod
    def _host_surface(host):
        host.terminate_group = lambda *_args: None
        host.snapshot_owned_npu_processes = lambda *_args: {}
        host.legacy = types.SimpleNamespace(
            preflight=lambda *_args: 0,
            cleanup_owned_and_postflight=lambda *_args: 0,
            npu_smi=lambda: "unused",
        )
        return host
    @staticmethod
    def _fixture(root: Path) -> tuple[Path, Path, dict[int, str]]:
        inputs = root / "inputs"
        output = root / "output"
        inputs.mkdir()
        output.mkdir()
        hashes = {}
        for rank in adapter.RANKS:
            path = inputs / f"rank{rank}_step10_ind0_192x192_BAD.pt"
            path.write_bytes(f"input-{rank}".encode())
            hashes[rank] = adapter.sha256_file(path)
            call = {
                "case_id": f"step260_rank{rank}_profiled", "shape": [192, 192],
                "dtype": "torch.float32", "eligible_mx_branch": True,
                "input_sha256": str(rank) * 64 if rank else "0" * 64,
                "mx_qr_call_delta": 1,
                "mx_qr_input": {"shape": [192, 192], "dtype": "torch.float32", "contiguous": True},
                "expected_padded_shape": [192, 192], "wrapper_branch": "mx_fixed",
                "public_qr_mode": "complete", "cpu_fp32_projection_control_max": 0.0,
                "input_unmodified": True,
                "contract_pass": True, "shape_pass": True, "input_finite": True,
                "q_finite": True, "r_finite": True,
                "nonfinite_count": {"input": 0, "q": 0, "r": 0}, "finite_pass": True,
                "reconstruction": {"violation_count": 0, "max_abs": 0.0, "max_bound": 0.0, "max_scaled": 0.0},
                "orthogonality": {"violation_count": 0, "max_abs": 0.0, "max_bound": 0.0, "max_scaled": 0.0},
                "lower_triangle_exact_zero": True, "lower_triangle_required": True,
                "fp64": {
                    "candidate_reconstruction_relative_fro": 0.0,
                    "candidate_orthogonality_relative_fro": 0.0,
                    "reference_reconstruction_relative_fro": 0.0,
                    "reference_orthogonality_relative_fro": 0.0,
                    "numerical_rank": 192, "rank_threshold": 0.0,
                },
                "full_rank_projection": {
                    "required": True,
                    "candidate_to_reference": {"relative_fro": 0.0, "relative_max": 0.0},
                    "reference_to_candidate": {"relative_fro": 0.0, "relative_max": 0.0},
                    "control_max": 0.0, "tolerance": 0.0, "pass": True,
                },
                "predicate_status": {
                    "input_unmodified": "pass", "shape": "pass", "finite": "pass",
                    "reconstruction": "pass", "orthogonality": "pass",
                    "lower_triangle_exact_zero": "pass", "projection": "pass",
                },
                "failed_predicates": [], "not_evaluated_predicates": [],
                "diagnostic_scalars_finite": True,
                "diagnostic_nonfinite_scalar_count": 0,
                "reconstruction_violation_count": 0,
                "orthogonality_violation_count": 0, "projection_pass": True,
                "elapsed_ms": 1.0,
            }
            done = {
                "rank": rank, "local_rank": rank, "world_size": 8,
                "input_file_sha256": hashes[rank], "call_count": 1,
                "eligible_call_count": 1, "mx_qr_call_count": 1,
                "eligible_fallback_count": 0, "all_contract_pass": True,
                "profiler_identity_pass": True, "first_profiled_only": True,
                "state_diagnostic_only": False,
                "calls": [call],
            }
            identity = {
                "pass": True, "diagnostic_identity": "QrV2_vtv_direct_qa_legacy_probe_v6",
                "diagnostic_aic_task_reference_count": 1,
                "diagnostic_aiv_task_reference_count": 0,
                "original_task_reference_count": 0, "v4_task_reference_count": 0,
                "v5_task_reference_count": 0, "unknown_qrv2_task_reference_count": 0,
                "raw_profile_retained": True,
            }
            (output / "done").mkdir(exist_ok=True)
            (output / "done" / f"rank{rank}.json").write_text(json.dumps(done))
            (output / f"profiler_identity_rank{rank}.json").write_text(json.dumps(identity))
            profile = output / f"profile_rank{rank}"
            profile.mkdir()
            (profile / "raw.bin").write_bytes(b"raw")
        return inputs, output, hashes

    def test_dependencies_are_sha_locked(self) -> None:
        adapter._guard_dependencies()
        self.assertEqual(
            adapter.WRAPPER_SOURCE_SHA256,
            "2e2171c4931e4796ecb1ec1a85d01846f25b3054e82b94fe7abc976e7cc02ee3",
        )
        for path in (adapter.HOST_PATH, adapter.WORLD8_PATH, adapter.WORKER_ADAPTER):
            self.assertEqual(adapter.sha256_file(path), adapter.EXPECTED_SHA256[path.name])

    def test_complete_world8_outputs_pass_and_profiles_remain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs, output, hashes = self._fixture(Path(directory))
            result = adapter.validate_outputs(output, hashes)
            self.assertEqual(len(result["ranks"]), 8)
            self.assertEqual(len(result["raw_profiles"]), 8)
            self.assertTrue(all((output / f"profile_rank{rank}/raw.bin").is_file() for rank in adapter.RANKS))

    def test_rank_call_math_identity_and_profile_mutations_fail(self) -> None:
        mutations = (
            lambda done, identity: done.update({"call_count": 2}),
            lambda done, identity: done["calls"][0].update({"input_unmodified": False}),
            lambda done, identity: done["calls"][0].update({"finite_pass": False}),
            lambda done, identity: done["calls"][0].update({"q_finite": False}),
            lambda done, identity: done["calls"][0].update({"nonfinite_count": {"input": 0, "q": 1, "r": 0}}),
            lambda done, identity: done["calls"][0].update({"mx_qr_input": {"shape": [256, 256], "dtype": "torch.float32", "contiguous": True}}),
            lambda done, identity: done["calls"][0]["reconstruction"].update({"violation_count": 1}),
            lambda done, identity: done["calls"][0]["orthogonality"].update({"violation_count": 1}),
            lambda done, identity: done["calls"][0].update({"lower_triangle_exact_zero": False}),
            lambda done, identity: done["calls"][0]["fp64"].update({"candidate_reconstruction_relative_fro": math.inf}),
            lambda done, identity: done["calls"][0]["full_rank_projection"].update({"pass": False}),
            lambda done, identity: done["calls"][0]["predicate_status"].update({"finite": "not_evaluated"}),
            lambda done, identity: done["calls"][0].update({"failed_predicates": ["finite"]}),
            lambda done, identity: done["calls"][0].update({"diagnostic_nonfinite_scalar_count": 1}),
            lambda done, identity: done["calls"][0].update({"reconstruction_violation_count": 1}),
            lambda done, identity: done["calls"][0].update({"elapsed_ms": math.inf}),
            lambda done, identity: identity.update({"v5_task_reference_count": 1}),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                _inputs, output, hashes = self._fixture(Path(directory))
                done_path = output / "done/rank0.json"
                identity_path = output / "profiler_identity_rank0.json"
                done = json.loads(done_path.read_text())
                identity = json.loads(identity_path.read_text())
                mutate(done, identity)
                done_path.write_text(json.dumps(done))
                identity_path.write_text(json.dumps(identity))
                with self.assertRaises(RuntimeError):
                    adapter.validate_outputs(output, hashes)

    def test_real_done_and_call_schemas_are_exact_and_types_are_strict(self) -> None:
        mutations = (
            lambda done: done.update({"extra": 1}),
            lambda done: done["calls"][0].update({"extra": 1}),
            lambda done: done.update({"call_count": True}),
            lambda done: done["calls"][0].update({"mx_qr_call_delta": True}),
            lambda done: done["calls"][0]["reconstruction"].update({"extra": 0.0}),
            lambda done: done["calls"][0]["fp64"].pop("rank_threshold"),
            lambda done: done["calls"][0]["full_rank_projection"]["candidate_to_reference"].update({"extra": 0.0}),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                _inputs, output, hashes = self._fixture(Path(directory))
                path = output / "done/rank0.json"
                done = json.loads(path.read_text())
                mutate(done)
                path.write_text(json.dumps(done))
                with self.assertRaises(RuntimeError):
                    adapter.validate_outputs(output, hashes)

    def test_missing_empty_and_symlink_profile_fail(self) -> None:
        for mutation in ("missing", "empty", "symlink"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _inputs, output, hashes = self._fixture(root)
                profile = output / "profile_rank0"
                (profile / "raw.bin").unlink()
                if mutation == "missing":
                    profile.rmdir()
                elif mutation == "symlink":
                    profile.rmdir()
                    profile.symlink_to(output / "profile_rank1", target_is_directory=True)
                with self.assertRaises(RuntimeError):
                    adapter.validate_outputs(output, hashes)

    def test_gate_is_o_excl_regular_and_never_uses_legacy_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / adapter.GATE_NAME
            report = adapter._write_gate(gate)
            self.assertTrue(gate.is_file())
            self.assertIn("token_sha256", report)
            self.assertFalse((root / "release_after_npu_smi").exists())
            with self.assertRaises(FileExistsError):
                adapter._write_gate(gate)

    def test_gate_fsync_failure_never_publishes_or_leaves_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); gate = root / adapter.GATE_NAME
            prepared, payload = adapter._prepare_gate(gate)
            with mock.patch.object(adapter.os, "fsync", side_effect=OSError("fsync failed")):
                with self.assertRaisesRegex(OSError, "fsync failed"):
                    adapter._publish_gate(gate, prepared, payload)
            self.assertFalse(gate.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_gate_postlink_dir_fsync_failure_is_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / adapter.GATE_NAME
            prepared, payload = adapter._prepare_gate(gate)
            with mock.patch.object(adapter.os, "fsync", side_effect=(None, OSError("dir fsync failed"))):
                result = adapter._publish_gate(gate, prepared, payload)
            self.assertTrue(gate.is_file())
            self.assertTrue(result["published"])
            self.assertFalse(result["dir_fsync_ok"])
            self.assertIn("dir fsync failed", result["dir_fsync_error"])

    def test_rank_json_precommit_and_postlink_fsync_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "rank_ownership.json"
            with mock.patch.object(adapter.os, "fsync", side_effect=OSError("file fsync failed")):
                with self.assertRaisesRegex(OSError, "file fsync failed"):
                    adapter._write_new_json(path, {"schema": "x"}, committed_error_ok=True)
            self.assertFalse(path.exists())
            self.assertEqual(list(root.iterdir()), [])
            with mock.patch.object(adapter.os, "fsync", side_effect=(None, OSError("dir fsync failed"))):
                result = adapter._write_new_json(path, {"schema": "x"}, committed_error_ok=True)
            self.assertTrue(path.is_file())
            self.assertTrue(result["published"])
            self.assertFalse(result["dir_fsync_ok"])
            self.assertEqual(json.loads(path.read_text()), {"schema": "x"})

    def test_gate_revalidation_rejects_content_tamper_and_inode_replacement(self) -> None:
        for mutation in ("tamper", "replace"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                gate = Path(directory) / adapter.GATE_NAME
                contract = adapter._write_gate(gate)
                if mutation == "tamper":
                    gate.write_text(json.dumps({
                        "schema": "step377-diagnostic-start-v1", "token": "0" * 32
                    }))
                else:
                    gate.unlink()
                    gate.write_text(json.dumps({
                        "schema": "step377-diagnostic-start-v1", "token": "0" * 32
                    }))
                with self.assertRaises(RuntimeError):
                    adapter._verify_gate(gate, contract)

    def test_ready_file_rejects_symlink_and_schema_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text("{}")
            link = root / "ready.json"
            link.symlink_to(target)
            with self.assertRaises(RuntimeError):
                adapter._read_bounded_regular_json(link, "ready")
        row = {key: None for key in adapter.READY_KEYS}
        row["unexpected"] = True
        with self.assertRaisesRegex(RuntimeError, "schema"):
            adapter._validate_ready_row(0, row)

    def test_diagnostic_wait_uses_world8_binding_and_no_release_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _inputs, output, _hashes = self._fixture(root)
            (output / "ready").mkdir()
            (output / "failure").mkdir()
            pairs = [(physical, chip) for physical in range(4, 8) for chip in range(2)]
            for rank, (_physical, _chip) in enumerate(pairs):
                row = {
                    "rank": rank, "local_rank": rank, "world_size": 8,
                    "visible": adapter.VISIBLE, "gate_pass": True,
                    "shadow_gate": True, "npu_available": True, "device_count": 8,
                    "opp_first_shadow": True,
                    "custom_opp_role_sequence": ["shadow", "base"],
                    "wrapper_contract": wrapper_contract(), "container_pid": 100 + rank,
                    "module_file_sha256": {
                        "cloud_init": "1" * 64,
                        "cloud_extension": "2" * 64,
                        "cloud_linalg": "3" * 64,
                    },
                }
                (output / "ready" / f"rank{rank}.json").write_text(json.dumps(row))
            physical = [(*pair, 100 + rank) for rank, pair in enumerate(pairs)]
            statuses = []
            legacy = types.SimpleNamespace(
                BACK8_PAIRS=set(pairs), BACK8_DEVICE_IDS=set(range(8, 16)),
                npu_smi=lambda: "smi",
                parse_back8=lambda *_args: (_ for _ in ()).throw(AssertionError("legacy parser called")),
                container_pid=lambda *_args: (_ for _ in ()).throw(AssertionError("legacy pid mapper called")),
                validate_rank_device_mapping=lambda *_args: (_ for _ in ()).throw(AssertionError("legacy mapping called")),
                atomic_json=lambda path, value: (
                    statuses.append(copy.deepcopy(value)),
                    path.write_text(json.dumps(value)),
                )[-1],
            )
            host = types.SimpleNamespace(legacy=legacy)
            def poll():
                gate = output / adapter.GATE_NAME
                ack_dir = output / adapter.GATE_ACK_DIR
                if gate.is_file() and not list(ack_dir.glob("rank*.json")):
                    contract = adapter._write_gate  # keep fixture tied to public contract shape
                    del contract
                    payload = json.loads(gate.read_text())
                    gate_status = gate.stat()
                    token_sha = adapter.hashlib.sha256(payload["token"].encode()).hexdigest()
                    for rank in adapter.RANKS:
                        (ack_dir / f"rank{rank}.json").write_text(json.dumps({
                            "schema": "step377-diagnostic-gate-ack-v1", "rank": rank,
                            "gate_device": gate_status.st_dev, "gate_inode": gate_status.st_ino,
                            "token_sha256": token_sha,
                        }))
                return None
            process = types.SimpleNamespace(poll=poll)
            (output / "launcher_ownership.json").write_text(json.dumps({
                "schema": "step358-launcher-ownership-v1", "port": 29950,
                "launcher_host_pid": 50, "launcher_starttime": 500, "launcher_pgid": 50,
            }))
            host._step377_port = 29950
            def stable_binding(rows, _sample):
                self.assertFalse((output / adapter.GATE_NAME).exists())
                return {"bindings": [
                    {"rank": row["rank"], "local_rank": row["local_rank"], "device_id": 8 + row["local_rank"]}
                    for row in rows
                ]}
            def reread_rank(path, *_args, **_kwargs):
                self.assertTrue(path.is_file())
                self.assertFalse((output / adapter.GATE_NAME).exists())
                return ({}, ())
            guard = types.SimpleNamespace(stable_back8_binding=stable_binding,
                                          read_rank_ownership_json=reread_rank)
            drifting = types.SimpleNamespace(
                stable_back8_binding=lambda *_args: (_ for _ in ()).throw(
                    RuntimeError("mapping changed between samples")
                )
            )
            with self.assertRaisesRegex(RuntimeError, "mapping changed"):
                adapter.diagnostic_wait_for_results(host, drifting, output, process, 1)
            result = adapter.diagnostic_wait_for_results(host, guard, output, process, 1)
            self.assertEqual(result["status"], "DIAGNOSTIC_RANKS_DONE")
            self.assertTrue((output / adapter.GATE_NAME).is_file())
            self.assertFalse((output / "release_after_npu_smi").exists())
            self.assertTrue(all("release_created" not in status for status in statuses))
            self.assertEqual(result["gate_ack_count"], 8)
            self.assertEqual(result["module_file_sha256"], {
                "cloud_init": "1" * 64, "cloud_extension": "2" * 64,
                "cloud_linalg": "3" * 64,
            })

    def test_gate_ack_wrong_token_and_replace_restore_are_rejected(self) -> None:
        for mutation in ("wrong_token", "replacement_inode"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / adapter.GATE_ACK_DIR).mkdir()
                gate = adapter._write_gate(root / adapter.GATE_NAME)
                for rank in adapter.RANKS:
                    row = {
                        "schema": "step377-diagnostic-gate-ack-v1", "rank": rank,
                        "gate_device": gate["device"], "gate_inode": gate["inode"],
                        "token_sha256": gate["token_sha256"],
                    }
                    (root / adapter.GATE_ACK_DIR / f"rank{rank}.json").write_text(json.dumps(row))
                adapter._validate_gate_acks(root, gate)
                path = root / adapter.GATE_ACK_DIR / "rank0.json"
                row = json.loads(path.read_text())
                if mutation == "wrong_token":
                    row["token_sha256"] = "0" * 64
                else:
                    row["gate_inode"] += 1
                path.unlink()
                path.write_text(json.dumps(row))
                with self.assertRaises(RuntimeError):
                    adapter._validate_gate_acks(root, gate)

    def test_ready_nested_schema_and_cross_rank_module_hashes_are_strict(self) -> None:
        row = {
            "rank": 0, "local_rank": 0, "world_size": 8, "visible": adapter.VISIBLE,
            "gate_pass": True, "shadow_gate": True, "npu_available": True,
            "device_count": 8, "opp_first_shadow": True,
            "custom_opp_role_sequence": ["shadow", "base"], "container_pid": 1,
            "wrapper_contract": {**wrapper_contract(), "extra": True},
            "module_file_sha256": {"cloud_init": "1" * 64, "cloud_extension": "2" * 64, "cloud_linalg": "3" * 64},
        }
        with self.assertRaises(RuntimeError):
            adapter._validate_ready_row(0, row)
        valid = copy.deepcopy(row)
        valid["wrapper_contract"].pop("extra")
        mutations = []
        for key in wrapper_contract():
            missing = copy.deepcopy(valid); missing["wrapper_contract"].pop(key)
            mutations.append(missing)
        for key, value in (
            ("gate", "pass"), ("source_sha256", "0" * 64),
            ("threshold", 81), ("threshold", True),
            ("block_tiling", 32), ("block_tiling", True),
        ):
            changed = copy.deepcopy(valid); changed["wrapper_contract"][key] = value
            mutations.append(changed)
        for changed in mutations:
            with self.subTest(wrapper=changed["wrapper_contract"]), self.assertRaises(RuntimeError):
                adapter._validate_ready_row(0, changed)
        rows = []
        for rank in adapter.RANKS:
            item = copy.deepcopy(valid)
            item.update({"rank": rank, "local_rank": rank, "container_pid": rank + 1})
            adapter._validate_ready_row(rank, item)
            rows.append(item)
        rows[7]["module_file_sha256"]["cloud_init"] = "9" * 64
        with self.assertRaisesRegex(RuntimeError, "differ"):
            adapter._validate_ready_consistency(rows)

    def test_run_forces_worker_and_first_only_restores_wait_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, output, _hashes = self._fixture(root)
            shadow = root / "shadow"; shadow.mkdir()
            installed = root / "installed"; installed.mkdir()
            original_wait = lambda *_args: None
            observed = {}

            def host_run(args):
                observed.update(vars(args))
                (output / "controller_status.json").write_text(json.dumps({
                    "schema": "step377-diagnostic-world8-v1",
                    "status": "DIAGNOSTIC_RANKS_DONE", "diagnostic_only": True,
                    "gate_ack_count": 8,
                    "module_file_sha256": {
                        "cloud_init": "1" * 64, "cloud_extension": "2" * 64,
                        "cloud_linalg": "3" * 64,
                    },
                    "gate": {"token_sha256": "4" * 64},
                    "launcher_ownership_sha256": "5" * 64,
                    "rank_ownership_sha256": "6" * 64,
                }))
                return 0

            host = self._host_surface(types.SimpleNamespace(wait_for_results=original_wait, run=host_run))
            args = argparse.Namespace(
                port=1234, output_dir=output, input_dir=inputs,
                shadow_root=shadow, installed_custom_opp=installed,
            )
            with mock.patch.object(adapter, "_guard_dependencies"), mock.patch.object(
                adapter, "load_host", return_value=host
            ):
                self.assertEqual(adapter.run(args), 0)
            self.assertEqual(observed["worker"], adapter.WORKER_ADAPTER)
            self.assertFalse(observed["first_profiled_only"])
            underlying = worker_adapter._underlying_argv(argparse.Namespace(
                input_dir=inputs, output_dir=output, shadow_root=shadow,
                installed_custom_opp=installed,
            ))
            self.assertEqual(underlying.count("--first-profiled-only"), 1)
            self.assertFalse(observed["state_diagnostic_only"])
            self.assertIs(host.wait_for_results, original_wait)
            summary = json.loads((output / adapter.SUMMARY_NAME).read_text())
            self.assertTrue(summary["diagnostic_only"])
            self.assertFalse(summary["release_candidate"])
            self.assertTrue(summary["raw_profiles_retained"])
            self.assertEqual(summary["module_file_sha256"]["cloud_init"], "1" * 64)
            self.assertEqual(summary["gate_token_sha256"], "4" * 64)

    def test_input_change_and_release_output_block_summary(self) -> None:
        for mutation in ("input", "release"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs, output, _ = self._fixture(root)
                shadow = root / "shadow"; shadow.mkdir()
                installed = root / "installed"; installed.mkdir()
                host = self._host_surface(types.SimpleNamespace(wait_for_results=lambda *_a: None))

                def host_run(_args):
                    if mutation == "input":
                        (inputs / "rank0_step10_ind0_192x192_BAD.pt").write_bytes(b"changed")
                    else:
                        (output / "release_after_npu_smi").touch()
                    return 0

                host.run = host_run
                args = argparse.Namespace(port=1, output_dir=output, input_dir=inputs, shadow_root=shadow, installed_custom_opp=installed)
                with mock.patch.object(adapter, "_guard_dependencies"), mock.patch.object(adapter, "load_host", return_value=host):
                    with self.assertRaises(RuntimeError):
                        adapter.run(args)
                self.assertFalse((output / adapter.SUMMARY_NAME).exists())

    def test_host_failure_code_is_preserved_and_wait_restored_for_cleanup_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, output, _ = self._fixture(root)
            shadow = root / "shadow"; shadow.mkdir()
            installed = root / "installed"; installed.mkdir()
            original_wait = lambda *_args: None
            host = self._host_surface(types.SimpleNamespace(wait_for_results=original_wait, run=lambda _args: 122))
            args = argparse.Namespace(port=1, output_dir=output, input_dir=inputs, shadow_root=shadow, installed_custom_opp=installed)
            with mock.patch.object(adapter, "_guard_dependencies"), mock.patch.object(adapter, "load_host", return_value=host):
                self.assertEqual(adapter.run(args), 122)
            self.assertIs(host.wait_for_results, original_wait)
            self.assertFalse((output / adapter.SUMMARY_NAME).exists())

    def test_nonzero_host_status_still_runs_post_input_hash_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, output, _ = self._fixture(root)
            shadow = root / "shadow"; shadow.mkdir()
            installed = root / "installed"; installed.mkdir()
            host = self._host_surface(types.SimpleNamespace(wait_for_results=lambda *_args: None))

            def failed_run(_args):
                (inputs / "rank7_step10_ind0_192x192_BAD.pt").write_bytes(b"drift")
                return 122

            host.run = failed_run
            args = argparse.Namespace(port=1, output_dir=output, input_dir=inputs, shadow_root=shadow, installed_custom_opp=installed)
            with mock.patch.object(adapter, "_guard_dependencies"), mock.patch.object(adapter, "load_host", return_value=host):
                with self.assertRaisesRegex(RuntimeError, "inputs changed"):
                    adapter.run(args)
            self.assertFalse((output / adapter.SUMMARY_NAME).exists())

    def test_restore_failure_does_not_replace_host_primary_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, output, _ = self._fixture(root)
            shadow = root / "shadow"; shadow.mkdir()
            installed = root / "installed"; installed.mkdir()
            primary = RuntimeError("host primary")
            original_wait = lambda *_args: None

            class Host:
                def __init__(self):
                    object.__setattr__(self, "wait_for_results", original_wait)
                    object.__setattr__(self, "injected", False)

                def __setattr__(self, name, value):
                    if name == "wait_for_results" and self.injected and value is original_wait:
                        raise OSError("restore failed")
                    object.__setattr__(self, name, value)
                    if name == "wait_for_results" and value is not original_wait:
                        object.__setattr__(self, "injected", True)

                def run(self, _args):
                    raise primary

            host = Host()
            self._host_surface(host)
            args = argparse.Namespace(port=1, output_dir=output, input_dir=inputs, shadow_root=shadow, installed_custom_opp=installed)
            with mock.patch.object(adapter, "_guard_dependencies"), mock.patch.object(adapter, "load_host", return_value=host):
                with self.assertRaises(RuntimeError) as raised:
                    adapter.run(args)
            self.assertIs(raised.exception, primary)
            self.assertEqual(primary.args, ("host primary",))
            self.assertEqual(str(primary), "host primary")
            self.assertIsInstance(primary.cleanup_error, OSError)

    def test_patched_cleanup_pidfd_unavailable_and_legacy_cleanup_never_called(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); inputs, output, _ = self._fixture(root)
            shadow = root / "shadow"; shadow.mkdir(); installed = root / "installed"; installed.mkdir()
            ownership = {"schema": "step358-launcher-ownership-v1", "port": 1,
                         "launcher_host_pid": 50, "launcher_starttime": 500, "launcher_pgid": 50}
            (output / "launcher_ownership.json").write_text(json.dumps(ownership))
            legacy_calls = []
            host = self._host_surface(types.SimpleNamespace(wait_for_results=lambda *_a: None))
            host.legacy.preflight = lambda *_a: legacy_calls.append("legacy_preflight")
            host.legacy.cleanup_owned_and_postflight = lambda *_a: legacy_calls.append("legacy_cleanup")
            def host_run(args):
                host.legacy.preflight(output, args.port)
                host.terminate_group(object())
                return 122
            host.run = host_run
            fake_guard = types.SimpleNamespace(
                assert_port_free=lambda _p: None, assert_stable_clear=lambda *_a: {},
                read_ownership_json=lambda _p, _sha: ownership,
                validate_ownership_manifest=lambda _o: (50, 500, 50, 1),
                safe_group_cleanup=lambda _o, **_k: (_ for _ in ()).throw(RuntimeError("pidfd signaling unavailable")),
            )
            args = argparse.Namespace(port=1, output_dir=output, input_dir=inputs,
                                      shadow_root=shadow, installed_custom_opp=installed)
            with mock.patch.object(adapter, "_guard_dependencies"), \
                 mock.patch.object(adapter, "load_host", return_value=host), \
                 mock.patch.object(adapter, "load_process_guard", return_value=fake_guard):
                with self.assertRaisesRegex(RuntimeError, "pidfd signaling unavailable"):
                    adapter.run(args)
            self.assertEqual(legacy_calls, [])

    def test_full_patched_pre_ownership_terminate_post_chain_and_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); inputs, output, _ = self._fixture(root)
            shadow = root / "shadow"; shadow.mkdir(); installed = root / "installed"; installed.mkdir()
            ownership = {"schema": "step358-launcher-ownership-v1", "port": 1,
                         "launcher_host_pid": 50, "launcher_starttime": 500, "launcher_pgid": 50}
            (output / "launcher_ownership.json").write_text(json.dumps(ownership))
            calls = []; legacy_calls = []
            host = self._host_surface(types.SimpleNamespace(wait_for_results=lambda *_a: None))
            host.legacy.preflight = lambda *_a: legacy_calls.append("legacy_pre")
            host.legacy.cleanup_owned_and_postflight = lambda *_a: legacy_calls.append("legacy_post")
            originals = (host.wait_for_results, host.terminate_group, host.snapshot_owned_npu_processes,
                         host.legacy.preflight, host.legacy.cleanup_owned_and_postflight)
            def host_run(args):
                host.legacy.preflight(output, args.port)
                host.terminate_group(object())
                host.legacy.cleanup_owned_and_postflight(output, args.port)
                return 122
            host.run = host_run
            fake_guard = types.SimpleNamespace(
                assert_port_free=lambda _p: calls.append("port"),
                assert_stable_clear=lambda *_a: calls.append("clear") or {},
                read_ownership_json=lambda _p, _sha: calls.append("ownership") or ownership,
                validate_ownership_manifest=lambda _o: (50, 500, 50, 1),
                approved_step377_rank_workers=lambda: (),
                safe_group_cleanup=lambda _o, **_k: calls.append("cleanup") or {},
            )
            args = argparse.Namespace(port=1, output_dir=output, input_dir=inputs,
                                      shadow_root=shadow, installed_custom_opp=installed)
            with mock.patch.object(adapter, "_guard_dependencies"), \
                 mock.patch.object(adapter, "load_host", return_value=host), \
                 mock.patch.object(adapter, "load_process_guard", return_value=fake_guard):
                self.assertEqual(adapter.run(args), 122)
            self.assertEqual(legacy_calls, [])
            self.assertEqual(calls, ["clear", "port", "ownership", "cleanup", "clear", "port",
                                     "ownership", "cleanup", "clear", "port"])
            self.assertIs(host.wait_for_results, originals[0])
            self.assertIs(host.terminate_group, originals[1])
            self.assertIs(host.snapshot_owned_npu_processes, originals[2])
            self.assertIs(host.legacy.preflight, originals[3])
            self.assertIs(host.legacy.cleanup_owned_and_postflight, originals[4])

    def test_pre_rank_ownership_live_worker_fails_without_signal_and_runs_other_domains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); inputs, output, _ = self._fixture(root)
            shadow = root / "shadow"; shadow.mkdir(); installed = root / "installed"; installed.mkdir()
            ownership = {"schema": "step358-launcher-ownership-v1", "port": 1,
                         "launcher_host_pid": 50, "launcher_starttime": 500, "launcher_pgid": 50}
            (output / "launcher_ownership.json").write_text(json.dumps(ownership))
            calls = []; signals = []
            host = self._host_surface(types.SimpleNamespace(wait_for_results=lambda *_a: None))
            def host_run(args):
                host.terminate_group(object())
                return 122
            host.run = host_run
            fake_guard = types.SimpleNamespace(
                assert_port_free=lambda _p: calls.append("port"),
                assert_stable_clear=lambda *_a: calls.append("clear") or {},
                read_ownership_json=lambda *_a: ownership,
                validate_ownership_manifest=lambda _o: (50, 500, 50, 1),
                approved_step377_rank_workers=lambda: (object(),),
                terminate_owned=lambda *_a, **_k: signals.append("signal"),
                safe_group_cleanup=lambda _o, **_k: calls.append("launcher") or {},
            )
            args = argparse.Namespace(port=1, output_dir=output, input_dir=inputs,
                                      shadow_root=shadow, installed_custom_opp=installed)
            with mock.patch.object(adapter, "_guard_dependencies"), \
                 mock.patch.object(adapter, "load_host", return_value=host), \
                 mock.patch.object(adapter, "load_process_guard", return_value=fake_guard), \
                 self.assertRaisesRegex(RuntimeError, "ownership_unestablished"):
                adapter.run(args)
            self.assertEqual(calls, ["launcher", "clear", "port"])
            self.assertEqual(signals, [])


if __name__ == "__main__":
    unittest.main()
