#!/usr/bin/env python3
"""Offline tests for the disarmed STEP377 remote controller skeleton."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import subprocess
import tempfile
import types
import unittest
import zipfile
from unittest import mock

import step377_run_diagnostic_remote as controller
import step377_process_guard as guard


def ownership_evidence(manifest):
    return {"manifest": manifest, "sha256": "a" * 64, "device": 1, "inode": 2,
            "size": 3, "mtime_ns": 4, "ctime_ns": 5}


def guarded_cleanup_result(with_rank=True):
    return {
        "schema": "step377-cleanup-owned-v1", "port_free": True,
        "launcher_cleanup": {"schema": "step377-owned-group-clean-v1", "member_count": 0,
                    "consecutive_empty_group_scans": 2, "external_stable_clear_required": True},
        "rank_cleanup": ({"schema": "step377-fixed-ranks-clean-v1", "rank_count": 8} if with_rank else None),
        "stable_clear": {"schema": "step377-stable-clear-v1", "back8_process_count": 0,
                         "case_process_count": 0, "sample_sha256": ["a" * 64, "b" * 64]},
    }


def rank_ownership_evidence(launcher_sha="a" * 64):
    ranks = []
    for rank in range(8):
        ranks.append({"rank": rank, "local_rank": rank, "host_pid": 100 + rank,
                      "container_pid": 10 + rank, "physical": 4 + rank // 2,
                      "chip": rank % 2, "device_id": 8 + rank, "starttime": 1000 + rank,
                      "pgid": 200 + rank, "nspid": [100 + rank, 10 + rank],
                      "argv": ["python", "worker"]})
    return ownership_evidence({"schema": "step377-rank-ownership-v1",
        "launcher_ownership_sha256": launcher_sha, "gate_token_sha256": "d" * 64,
        "case_path": "/safe/diagnostics/step377/step377_diagnostic_host_case.py",
        "port": controller.PORT, "ranks": ranks})


class Step377RemoteControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        # No unit test may construct the production SSH backend. Tests that
        # exercise execute() must inject the in-memory Backend below.
        patcher = mock.patch.object(
            controller, "load_backend",
            side_effect=AssertionError("production remote backend is forbidden in offline tests"),
        )
        self.production_backend = patcher.start()
        self.addCleanup(patcher.stop)

    def _python(self, script, *arguments):
        return subprocess.run(
            ["python3", "-c", script, *(str(item) for item in arguments)],
            capture_output=True, text=True, check=False,
        )

    def test_embedded_safe_reader_real_python_fixture(self) -> None:
        body = "import sys; print(json.dumps(safe_json(sys.argv[1],32),sort_keys=True))"
        script = controller.embedded_script(body)
        with tempfile.TemporaryDirectory() as directory:
            root = controller.Path(directory)
            good = root / "good.json"
            good.write_text('{"a":1}')
            result = subprocess.run(
                ["python3", "-c", script, str(good)], capture_output=True,
                text=True, check=False,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(json.loads(result.stdout), {"a": 1})
            link = root / "link.json"
            link.symlink_to(good)
            rejected = subprocess.run(
                ["python3", "-c", script, str(link)], capture_output=True,
                text=True, check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            large = root / "large.json"
            large.write_text(json.dumps({"payload": "x" * 64}))
            rejected = subprocess.run(
                ["python3", "-c", script, str(large)], capture_output=True,
                text=True, check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_safe_file_deterministically_rejects_replace_after_open(self) -> None:
        body = r'''import os,sys
from pathlib import Path
p=Path(sys.argv[1]); replacement=Path(sys.argv[2])
safe_file(p,None,after_open=lambda:os.replace(replacement,p))
print('unexpected')'''
        with tempfile.TemporaryDirectory() as directory:
            root = controller.Path(directory)
            original = root / "original"; replacement = root / "replacement"
            original.write_bytes(b"before"); replacement.write_bytes(b"after")
            result = self._python(controller.embedded_script(body), original, replacement)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("identity changed", result.stderr)

    def test_upload_embedded_script_real_fixture_accepts_exact_and_rejects_extra(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = controller.Path(directory)
            (root / "inputs").mkdir(); (root / "run").mkdir()
            (root / "tool.py").write_bytes(b"tool")
            (root / "inputs/rank.pt").write_bytes(b"input")
            expected = {
                "tool.py": hashlib.sha256(b"tool").hexdigest(),
                "inputs/rank.pt": hashlib.sha256(b"input").hexdigest(),
            }
            accepted = self._python(controller.upload_embedded_script(), root, json.dumps(expected))
            self.assertEqual((accepted.returncode, json.loads(accepted.stdout)), (0, {"count": 2}))
            (root / "extra").write_text("unexpected")
            self.assertNotEqual(self._python(controller.upload_embedded_script(), root, json.dumps(expected)).returncode, 0)

    def test_upload_embedded_script_rejects_deterministic_inputs_directory_replace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = controller.Path(directory); (root / "inputs").mkdir(); (root / "run").mkdir()
            (root / "tool.py").write_bytes(b"tool"); (root / "inputs/rank.pt").write_bytes(b"input")
            replacement = controller.Path(directory + "-replacement")
            try:
                replacement.mkdir(); (replacement / "rank.pt").write_bytes(b"input")
                expected = {"tool.py": hashlib.sha256(b"tool").hexdigest(), "inputs/rank.pt": hashlib.sha256(b"input").hexdigest()}
                result = self._python(controller.upload_embedded_script(), root, json.dumps(expected), "race-inputs", replacement)
                self.assertNotEqual(result.returncode, 0)
            finally:
                if replacement.exists():
                    for child in replacement.iterdir(): child.unlink()
                    replacement.rmdir()

    def test_artifact_embedded_script_real_fixture_accepts_hashes_and_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = controller.Path(directory); manifest = root / "manifest.json"; wheel = root / "original.whl"
            manifest.write_bytes(b"manifest"); wheel.write_bytes(b"wheel")
            args = (manifest, hashlib.sha256(b"manifest").hexdigest(), wheel, hashlib.sha256(b"wheel").hexdigest())
            accepted = self._python(controller.artifact_embedded_script(), *args)
            self.assertEqual(set(json.loads(accepted.stdout)), {"manifest", "wheel"})
            link = root / "link.whl"; link.symlink_to(wheel)
            self.assertNotEqual(self._python(controller.artifact_embedded_script(), manifest, args[1], link, args[3]).returncode, 0)

    def test_local_regular_reader_rejects_replace_during_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = controller.Path(directory); path = root / "input"; replacement = root / "new"
            path.write_bytes(b"before"); replacement.write_bytes(b"after")
            real_read = controller.os.read; replaced = False
            def racing_read(fd, size):
                nonlocal replaced
                if not replaced:
                    replaced = True; controller.os.replace(replacement, path)
                return real_read(fd, size)
            with mock.patch.object(controller.os, "read", side_effect=racing_read), self.assertRaisesRegex(RuntimeError, "identity changed"):
                controller.read_local_regular(path)

    def test_guard_rejects_replaced_and_same_size_rewritten_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = controller.Path(directory); ownership = root / "ownership.json"
            value = {"schema": "step358-launcher-ownership-v1", "port": controller.PORT,
                     "launcher_host_pid": 50, "launcher_starttime": 500, "launcher_pgid": 50}
            original = json.dumps(value, sort_keys=True).encode(); ownership.write_bytes(original)
            expected = hashlib.sha256(original).hexdigest()
            replacement = root / "replacement.json"
            replacement.write_bytes(original.replace(b"500", b"501"))
            controller.os.replace(replacement, ownership)
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                guard.read_ownership_json(ownership, expected)
            ownership.write_bytes(original)
            rewritten = original.replace(b"500", b"502")
            self.assertEqual(len(rewritten), len(original))
            with ownership.open("r+b") as stream:
                stream.write(rewritten); stream.flush(); controller.os.fsync(stream.fileno())
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                guard.read_ownership_json(ownership, expected)

    def test_rank_ownership_evidence_tamper_missing_and_distinct_pgids(self) -> None:
        evidence = rank_ownership_evidence()
        accepted = controller._validate_rank_ownership_evidence(
            evidence, "a" * 64,
            "/safe/diagnostics/step377/step377_diagnostic_host_case.py",
        )
        self.assertEqual({row["pgid"] for row in accepted["manifest"]["ranks"]}, set(range(200, 208)))
        for mutation in (
            lambda row: row["manifest"]["ranks"].pop(),
            lambda row: row["manifest"]["ranks"][0].update({"host_pid": row["manifest"]["ranks"][1]["host_pid"]}),
            lambda row: row["manifest"].update({"launcher_ownership_sha256": "0" * 64}),
        ):
            candidate = copy.deepcopy(evidence); mutation(candidate)
            with self.assertRaises(RuntimeError):
                controller._validate_rank_ownership_evidence(candidate, "a" * 64,
                    "/safe/diagnostics/step377/step377_diagnostic_host_case.py")

    def test_snapshot_embedded_scripts_real_fixtures_positive_and_negative(self) -> None:
        artifact_code, installed_code, process_code = controller.snapshot_embedded_scripts()
        with tempfile.TemporaryDirectory() as directory:
            root = controller.Path(directory); artifact = root / "artifact"; artifact.write_bytes(b"locked")
            row = {"wheel": {"path": str(artifact)}}
            accepted = self._python(artifact_code, json.dumps(row))
            self.assertEqual(json.loads(accepted.stdout)["wheel"]["sha256"], hashlib.sha256(b"locked").hexdigest())
            link = root / "artifact-link"; link.symlink_to(artifact)
            self.assertNotEqual(self._python(artifact_code, json.dumps({"wheel": {"path": str(link)}})).returncode, 0)

            installed = root / "installed"; kernel = installed / "op_impl/ai_core/tbe/kernel"
            dynamic = installed / "op_impl/ai_core/tbe/customize_impl/dynamic"
            dynamic.mkdir(parents=True)
            (dynamic / "qr_v2.cpp").write_bytes(b"source-cpp")
            (dynamic / "qr_v2.py").write_bytes(b"source-py")
            cache = dynamic / "__pycache__"; cache.mkdir()
            (cache / "qr_v2.cpython-311.pyc").write_bytes(b"runtime-pyc")
            include = installed / "op_api/include"; include.mkdir(parents=True)
            (include / "aclnn_qr_v2.h").write_bytes(b"api-header")
            ops_config = installed / "op_impl/ai_core/tbe/config"
            for soc in ("ascend910_93", "ascend910b"):
                config_dir = ops_config / soc; config_dir.mkdir(parents=True)
                (config_dir / f"aic-{soc}-ops-info.json").write_text(
                    json.dumps({"QrV2": {"opFile": {"value": "qr_v2"}}})
                )
            op_info = installed / "op_impl/ai_core/tbe/op_info_cfg/ai_core"
            op_info.mkdir(parents=True)
            (op_info / "npu_supported_ops.json").write_text(json.dumps({"QrV2": {}}))
            for soc in ("ascend910_93", "ascend910b"):
                qdir = kernel / soc / "qr_v2"; config = kernel / "config" / soc
                qdir.mkdir(parents=True); config.mkdir(parents=True)
                name = "QrV2_fixture"
                (qdir / f"{name}.json").write_text('{}'); (qdir / f"{name}.o").write_bytes(b"object")
                (config / "qr_v2.json").write_text(json.dumps({"binList": [{"binInfo": {"jsonFilePath": f"{soc}/qr_v2/{name}.json"}}]}))
                (config / "binary_info_config.json").write_text(json.dumps({"QrV2": {"binaryList": [{"binPath": f"{soc}/qr_v2/{name}.o"}, {"binPath": f"{soc}/qr_v2/{name}.o"}]}}))
            installed_result = self._python(installed_code, installed)
            self.assertEqual(installed_result.returncode, 0, installed_result.stderr)
            inventory = json.loads(installed_result.stdout)
            self.assertEqual(len(inventory["entries"]), 36)
            unknown_pyc = cache / "qr_v2.cpython-312.pyc"
            unknown_pyc.write_bytes(b"wrong-abi")
            self.assertNotEqual(self._python(installed_code, installed).returncode, 0)
            unknown_pyc.unlink()
            unknown_header = include / "aclnn_qr_v2_extra.h"
            unknown_header.write_bytes(b"unexpected")
            self.assertNotEqual(self._python(installed_code, installed).returncode, 0)
            unknown_header.unlink()
            unknown_source = dynamic / "qr_v2_extra.cpp"
            unknown_source.write_bytes(b"unexpected")
            self.assertNotEqual(self._python(installed_code, installed).returncode, 0)
            unknown_source.unlink()
            required_source = dynamic / "qr_v2.cpp"
            required_source.unlink()
            self.assertNotEqual(self._python(installed_code, installed).returncode, 0)
            required_source.write_bytes(b"source-cpp")
            semantic = installed / "opaque.json"
            semantic.write_text(json.dumps({"route": "ascend9999/operators/QrV2.bin"}))
            self.assertNotEqual(self._python(installed_code, installed).returncode, 0)
            semantic.unlink()
            raced = self._python(installed_code, installed, "race-add")
            self.assertNotEqual(raced.returncode, 0)
            (installed / "race-added").unlink()
            extra = kernel / "ascend9999/qr_v2"; extra.mkdir(parents=True)
            self.assertNotEqual(self._python(installed_code, installed).returncode, 0)
            extra.rmdir(); extra.parent.rmdir()
            qr = kernel / "config/ascend910b/qr_v2.json"; qr.unlink(); qr.symlink_to(artifact)
            self.assertNotEqual(self._python(installed_code, installed).returncode, 0)

            token = "step377-fixture-process-token"
            sleeper = subprocess.Popen(["python3", "-c", "import time; time.sleep(30)", token])
            try:
                found = json.loads(self._python(process_code, token, "unlikely-port-377").stdout)
                self.assertIn(sleeper.pid, found)
            finally:
                sleeper.terminate(); sleeper.wait(timeout=5)
            absent = json.loads(self._python(process_code, "step377-definitely-absent-token", "unlikely-port-378").stdout)
            self.assertEqual(absent, [])

    def test_shadow_embedded_script_real_fixture_accepts_contract_and_rejects_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = controller.Path(directory); shadow = root / "shadow"; package = shadow / "mx_driving_cloud"; package.mkdir(parents=True)
            record_file = shadow / "mx_driving_cloud-1.0.dist-info/RECORD"; record_file.parent.mkdir(); record_file.write_bytes(b"record")
            artifacts = {}
            source_artifacts = {}
            for soc in ("ascend910_93", "ascend910b"):
                artifact_dir = shadow / "mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/kernel" / soc / "qr_v2"
                artifact_dir.mkdir(parents=True)
                json_path = artifact_dir / "QrV2_vtv_direct_qa_legacy_probe_v6.json"; object_path = artifact_dir / "QrV2_vtv_direct_qa_legacy_probe_v6.o"
                json_path.write_bytes(soc.encode()); object_path.write_bytes((soc + "-object").encode())
                source_json = root / f"source-{soc}.json"; source_object = root / f"source-{soc}.o"
                source_json.write_bytes(soc.encode()); source_object.write_bytes((soc + "-object").encode())
                source_artifacts[soc] = {"json_path": str(source_json), "json_sha256": hashlib.sha256(source_json.read_bytes()).hexdigest(), "object_path": str(source_object), "object_sha256": hashlib.sha256(source_object.read_bytes()).hexdigest()}
                config = shadow / "mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/kernel/config" / soc
                config.mkdir(parents=True); qr_config = config / "qr_v2.json"; binary_config = config / "binary_info_config.json"
                identity = "QrV2_vtv_direct_qa_legacy_probe_v6"
                qr_config.write_text(json.dumps({"binList": [{"binInfo": {"jsonFilePath": f"{soc}/qr_v2/{identity}.json"}}]}))
                binary_config.write_text(json.dumps({"QrV2": {"binaryList": [{"binPath": f"{soc}/qr_v2/{identity}.o"}, {"binPath": f"{soc}/qr_v2/{identity}.o"}]}}))
                artifacts[soc] = {
                    "json_path": str(json_path), "json_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
                    "object_path": str(object_path), "object_sha256": hashlib.sha256(object_path.read_bytes()).hexdigest(),
                    "config_sha256": hashlib.sha256(qr_config.read_bytes()).hexdigest(),
                    "binary_info_config_sha256": hashlib.sha256(binary_config.read_bytes()).hexdigest(),
                }
            digest_a, digest_b = "a" * 64, "b" * 64
            manifest = {
                "schema": "step377-diagnostic-shadow-v1", "status": "diagnostic_shadow_unvalidated",
                "diagnostic_only": True, "package_forbidden": True, "source_overlay": False, "record_unchanged": True,
                "attempt3_manifest": {"sha256": digest_a}, "original_wheel": {"sha256": digest_b},
                "shadow_root": str(shadow), "package_root": str(package),
                "candidate_identity": "QrV2_vtv_direct_qa_legacy_probe_v6", "record": {"relative_path": "mx_driving_cloud-1.0.dist-info/RECORD", "sha256": hashlib.sha256(b"record").hexdigest()},
                "attempt3_artifact_inputs": source_artifacts, "artifacts": artifacts,
            }
            source_manifest = root / "release_manifest.json"
            source_manifest.write_text(json.dumps({"artifacts": source_artifacts}))
            digest_a = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
            manifest["attempt3_manifest"]["sha256"] = digest_a
            manifest["attempt3_manifest"]["path"] = str(source_manifest.resolve())
            wheel = root.parent / (root.name + "-original.whl")
            self.addCleanup(lambda: wheel.exists() and wheel.unlink())
            with zipfile.ZipFile(wheel, "w") as archive:
                for path in shadow.rglob("*"):
                    if path.is_file(): archive.write(path, path.relative_to(shadow).as_posix())
            digest_b = hashlib.sha256(wheel.read_bytes()).hexdigest()
            manifest["original_wheel"] = {"path": str(wheel.resolve()), "sha256": digest_b}
            (root / "shadow_manifest.json").write_text(json.dumps(manifest))
            args = (root, manifest["status"], digest_a, digest_b, source_manifest, wheel)
            accepted = self._python(controller.shadow_embedded_script(), *args)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertEqual(json.loads(accepted.stdout), {"status": manifest["status"]})
            wrong_path_args = (*args[:-1], source_manifest)
            self.assertNotEqual(self._python(controller.shadow_embedded_script(), *wrong_path_args).returncode, 0)
            original_wheel_sha = manifest["original_wheel"]["sha256"]
            manifest["original_wheel"]["sha256"] = "0" * 64
            (root / "shadow_manifest.json").write_text(json.dumps(manifest))
            bad_whole_args = (root, manifest["status"], digest_a, "0" * 64, source_manifest, wheel)
            self.assertNotEqual(self._python(controller.shadow_embedded_script(), *bad_whole_args).returncode, 0)
            manifest["original_wheel"]["sha256"] = original_wheel_sha
            (root / "shadow_manifest.json").write_text(json.dumps(manifest))
            config_path = shadow / "mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend910b/qr_v2.json"
            config_value = json.loads(config_path.read_text()); config_value["unexpected"] = True
            config_path.write_text(json.dumps(config_value))
            manifest["artifacts"]["ascend910b"]["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
            (root / "shadow_manifest.json").write_text(json.dumps(manifest))
            self.assertNotEqual(self._python(controller.shadow_embedded_script(), *args).returncode, 0)
            del config_value["unexpected"]; config_path.write_text(json.dumps(config_value))
            manifest["artifacts"]["ascend910b"]["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
            (root / "shadow_manifest.json").write_text(json.dumps(manifest))
            fifo = shadow / "special-fifo"; controller.os.mkfifo(fifo)
            self.assertNotEqual(self._python(controller.shadow_embedded_script(), *args).returncode, 0)
            fifo.unlink()
            raced_args = (*args, "race-add")
            self.assertNotEqual(self._python(controller.shadow_embedded_script(), *raced_args).returncode, 0)
            (shadow / "race-added").unlink()
            extra_mutable = shadow / "mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/qr_v2/extra.o"
            extra_mutable.write_bytes(b"extra")
            self.assertNotEqual(self._python(controller.shadow_embedded_script(), *args).returncode, 0)
            extra_mutable.unlink()
            empty = shadow / "empty-directory"; empty.mkdir()
            self.assertNotEqual(self._python(controller.shadow_embedded_script(), *args).returncode, 0)
            empty.rmdir()
            link = shadow / "tree-link"; link.symlink_to(record_file)
            self.assertNotEqual(self._python(controller.shadow_embedded_script(), *args).returncode, 0)
            link.unlink()
            original_source_sha = manifest["attempt3_artifact_inputs"]["ascend910b"]["json_sha256"]
            manifest["attempt3_artifact_inputs"]["ascend910b"]["json_sha256"] = "0" * 64
            (root / "shadow_manifest.json").write_text(json.dumps(manifest))
            self.assertNotEqual(self._python(controller.shadow_embedded_script(), *args).returncode, 0)
            manifest["attempt3_artifact_inputs"]["ascend910b"]["json_sha256"] = original_source_sha
            (root / "shadow_manifest.json").write_text(json.dumps(manifest))
            with zipfile.ZipFile(wheel, "a") as archive:
                archive.writestr("other-1.0.dist-info/RECORD", b"other")
            digest_b = hashlib.sha256(wheel.read_bytes()).hexdigest()
            manifest["original_wheel"]["sha256"] = digest_b
            (root / "shadow_manifest.json").write_text(json.dumps(manifest))
            args = (root, manifest["status"], digest_a, digest_b, source_manifest, wheel)
            self.assertNotEqual(self._python(controller.shadow_embedded_script(), *args).returncode, 0)
            (root / "forbidden.whl").write_bytes(b"wheel")
            self.assertNotEqual(self._python(controller.shadow_embedded_script(), *args).returncode, 0)

    def test_summary_forbidden_and_ownership_embedded_scripts_real_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = controller.Path(directory); summary = root / "summary.json"; summary.write_text('{"ok":true}')
            self.assertEqual(json.loads(self._python(controller.summary_embedded_script(), summary).stdout), {"ok": True})
            summary_link = root / "summary-link.json"; summary_link.symlink_to(summary)
            self.assertNotEqual(self._python(controller.summary_embedded_script(), summary_link).returncode, 0)

            self.assertEqual(json.loads(self._python(controller.forbidden_embedded_script(), root).stdout), {"bad": []})
            forbidden = root / "release"; forbidden.mkdir()
            self.assertNotEqual(self._python(controller.forbidden_embedded_script(), root).returncode, 0)
            forbidden.rmdir()

            ownership = root / "ownership.json"
            value = {"schema": "step358-launcher-ownership-v1", "port": controller.PORT,
                     "launcher_host_pid": 10, "launcher_starttime": 20, "launcher_pgid": 10}
            ownership.write_text(json.dumps(value))
            required = controller.ownership_embedded_script(optional=False)
            self.assertEqual(json.loads(self._python(required, ownership, controller.PORT).stdout)["manifest"], value)
            value["launcher_host_pid"] = 1; ownership.write_text(json.dumps(value))
            self.assertNotEqual(self._python(required, ownership, controller.PORT).returncode, 0)
            missing = root / "missing.json"
            self.assertIsNone(json.loads(self._python(controller.ownership_embedded_script(optional=True), missing, controller.PORT).stdout))

    def test_disarmed_before_mapping_helper_or_files(self) -> None:
        with mock.patch.object(controller, "NPU_READY", False), mock.patch.object(
            controller.Path, "read_text"
        ) as mapping, mock.patch.object(
            controller, "sha256_file"
        ) as digest, self.assertRaisesRegex(RuntimeError, "disarmed"):
            controller.execute()
        mapping.assert_not_called()
        digest.assert_not_called()

    def test_exact_inventory_and_diagnostic_plan_constants(self) -> None:
        self.assertFalse(controller.NPU_READY)
        self.assertEqual(len(controller.FILES), 18)
        self.assertIn(controller.PROCESS_GUARD, controller.FILES)
        self.assertEqual({path.name for path in controller.FILES}, set(controller.EXPECTED_SHA256))
        self.assertEqual(controller.CONTAINER, "mapqr-leicheng")
        self.assertEqual(
            controller.REMOTE_DIAG_NAME,
            "step377_attempt10_qrv2_vtv_direct_qa_legacy_probe_v6_shadow_world8_20260822",
        )
        self.assertNotIn(controller.REMOTE_DIAG_NAME, {
            "step377_qrv2_vtv_direct_qa_legacy_probe_v6_shadow_world8_20260822",
            "step377_attempt1_qrv2_vtv_direct_qa_legacy_probe_v6_shadow_world8_20260822",
            "step377_attempt2_qrv2_vtv_direct_qa_legacy_probe_v6_shadow_world8_20260822",
            "step377_attempt3_qrv2_vtv_direct_qa_legacy_probe_v6_shadow_world8_20260822",
            "step377_attempt4_qrv2_vtv_direct_qa_legacy_probe_v6_shadow_world8_20260822",
            "step377_attempt5_qrv2_vtv_direct_qa_legacy_probe_v6_shadow_world8_20260822",
            "step377_attempt6_qrv2_vtv_direct_qa_legacy_probe_v6_shadow_world8_20260822",
            "step377_attempt7_qrv2_vtv_direct_qa_legacy_probe_v6_shadow_world8_20260822",
            "step377_attempt8_qrv2_vtv_direct_qa_legacy_probe_v6_shadow_world8_20260822",
        })
        self.assertEqual(
            controller.EXPECTED_SHA256["step377_diagnostic_host_case.py"],
            "91dd54cf26183861d2e389944b7232337b10e28a1544a66584a826dd1d7bc704",
        )
        self.assertEqual(
            controller.PROCESS_GUARD_SHA256,
            "7b4dcb578fd5227f51cf54b2acaa0591840261794b3296eeafa5731e76ad27c5",
        )
        self.assertEqual(
            controller.EXPECTED_SHA256["step377_process_guard.py"],
            controller.PROCESS_GUARD_SHA256,
        )
        self.assertEqual(controller.DRY_RUN_ACTIONS.count("world8_back8_once"), 1)
        self.assertEqual(set(controller.FORBIDDEN_ACTIONS), {
            "package", "wheel_write", "release", "install", "modify_installed",
            "train", "download_remote_artifacts",
        })
        self.assertEqual(controller.ATTEMPT3_MANIFEST, controller.ATTEMPT3_DIR + "/work/release_manifest.json")
        self.assertEqual(
            controller.ATTEMPT3_MANIFEST_SHA256,
            "18f7434836014f012f9308bfdf95f2f4b9f9a846cf3eb99942e0e22cfda8c6a1",
        )
        self.assertEqual(
            controller.IMMUTABLE_ORIGINAL_WHEEL,
            controller.ATTEMPT3_DIR
            + "/work/outer_original/mx_driving_cloud-26.0.7+CANN8.3.RC1.A3-cp311-cp311-linux_aarch64.whl",
        )
        self.assertEqual(
            controller.IMMUTABLE_ORIGINAL_WHEEL_SHA256,
            "23253f7fa2b9bfb1b6ff3c77df6620f6c559f68be154f6333246d73178eb5da9",
        )

    def test_patched_ready_still_rejects_unarmed_artifacts_before_files(self) -> None:
        with mock.patch.object(controller, "NPU_READY", True), mock.patch.object(
            controller, "ATTEMPT3_MANIFEST_SHA256", None
        ), mock.patch.object(
            controller, "sha256_file"
        ) as digest, self.assertRaisesRegex(RuntimeError, "artifact contract"):
            controller.local_preflight()
        digest.assert_not_called()
        self.production_backend.assert_not_called()

    @staticmethod
    def _summary():
        return {
            "schema": "step377-diagnostic-host-summary-v1",
            "status": "diagnostic_world8_pass", "diagnostic_only": True,
            "release_candidate": False, "rank_count": 8,
            "raw_profiles_retained": True,
            "input_sha256": dict(controller.EXPECTED_INPUT_SHA256),
            "module_file_sha256": {
                "cloud_init": "a" * 64, "cloud_extension": "b" * 64,
                "cloud_linalg": "c" * 64,
            },
            "gate_token_sha256": "d" * 64,
            "launcher_ownership_sha256": "a" * 64,
            "rank_ownership_sha256": "a" * 64,
            "ranks": [
                {"rank": rank, "call_count": 1, "identity_pass": True}
                for rank in range(8)
            ],
            "raw_profiles": [{"rank": rank, "file_count": 1, "total_bytes": 1} for rank in range(8)],
        }

    def test_strict_summary_accepts_only_diagnostic_world8(self) -> None:
        value = self._summary()
        self.assertIs(controller.validate_summary(value), value)
        mutations = (
            lambda row: row.update({"extra": True}),
            lambda row: row.update({"status": "PASS"}),
            lambda row: row.update({"diagnostic_only": False}),
            lambda row: row.update({"release_candidate": True}),
            lambda row: row.update({"rank_count": True}),
            lambda row: row.update({"raw_profiles_retained": False}),
            lambda row: row["input_sha256"].pop("7"),
            lambda row: row["module_file_sha256"].update({"cloud_init": "X" * 64}),
            lambda row: row.update({"gate_token_sha256": "0"}),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                row = copy.deepcopy(value)
                mutation(row)
                with self.assertRaises(RuntimeError):
                    controller.validate_summary(row)

    def test_main_and_dry_run_are_fail_closed(self) -> None:
        for argv in ([], ["--dry-run"]):
            with self.subTest(argv=argv), mock.patch.object(
                controller, "NPU_READY", False
            ), self.assertRaisesRegex(RuntimeError, "disarmed"):
                controller.main(argv)
        self.production_backend.assert_not_called()

    def test_armed_dry_run_is_local_and_succeeds(self) -> None:
        with mock.patch.object(controller, "NPU_READY", True), mock.patch.object(
            controller, "execute"
        ) as execute:
            self.assertEqual(controller.main(["--dry-run"]), 0)
        execute.assert_not_called()
        self.production_backend.assert_not_called()

    class Backend:
        def __init__(self, summary, fail_at=None):
            self.summary = summary
            self.fail_at = fail_at
            self.calls = []
            self.host_started = False
            self.target = types.SimpleNamespace(close=self._target_close)
            self.jump = types.SimpleNamespace(close=self._jump_close)

        def _call(self, name, *args, **kwargs):
            self.calls.append((name, args, kwargs))
            if self.fail_at == name:
                raise TimeoutError(name)

        def parse_machine_info(self, path): self._call("mapping", path); return {"host": "redacted"}
        def connect_target(self, info): self._call("connect", info); return self.jump, self.target
        def require_hostname(self, target): self._call("hostname", target)
        def require_container(self, target, name): self._call("container", target, name)
        def exclusive_directory(self, target, name, mode): self._call("exclusive", target, name, mode=mode)
        def upload_new(self, *args): self._call("upload", *args)
        def verify_uploads(self, *args): self._call("verify_uploads", *args)
        def validate_readonly_artifacts(self, *args): self._call("artifacts", *args); return {"locked": True}
        def snapshot(self, *args):
            self._call("snapshot", *args)
            installed_root = "/opt/ascend/cloud/packages/vendors/customize"
            return {"schema": "step377-protected-snapshot-v1", "artifacts": {},
                    "runtime": {"installed_cloud_root": "/opt/ascend/cloud"},
                    "installed_root": installed_root,
                    "installed_qrv2": {"root": installed_root}, "related_processes": [],
                    "back8": {"rows": [], "device_ids": [], "host_pids": []}}
        def prepare_shadow(self, *args, **kwargs): self._call("shadow", *args, **kwargs)
        def validate_shadow(self, *args, **kwargs): self._call("validate_shadow", *args, **kwargs)
        def pre_host_closure(self, *args, **kwargs): self._call("pre_host_closure", *args, **kwargs)
        def run_host_once(self, *args, **kwargs):
            self.host_started = True
            self._call("host", *args, **kwargs)
            return ownership_evidence({"schema": "step358-launcher-ownership-v1", "port": controller.PORT, "launcher_host_pid": 10, "launcher_starttime": 20, "launcher_pgid": 10})
        def read_ownership(self, *args):
            self._call("read_ownership", *args)
            return ownership_evidence({"schema": "step358-launcher-ownership-v1", "port": controller.PORT, "launcher_host_pid": 10, "launcher_starttime": 20, "launcher_pgid": 10}) if self.host_started else None
        def read_rank_ownership(self, *args): self._call("read_rank_ownership", *args); return rank_ownership_evidence()
        def read_summary(self, *args): self._call("summary", *args); return self.summary
        def scan_forbidden_outputs(self, *args): self._call("forbidden", *args)
        def cleanup_owned(self, *args): self._call("cleanup", *args); return guarded_cleanup_result()
        def _target_close(self): self._call("target_close")
        def _jump_close(self): self._call("jump_close")

    def test_mock_transaction_success_uploads_exactly_once_and_closes(self) -> None:
        backend = self.Backend(self._summary())
        plan = {"diagnostic_only": True}
        with mock.patch.object(controller, "dry_run_plan", return_value=plan):
            result = controller.execute(backend)
        self.assertEqual(result["status"], "diagnostic_world8_pass")
        self.assertEqual(result["cleanup_postflight"], {
            "schema": "step377-controller-cleanup-postflight-v1",
            "rank_evidence_present": True, "rank_cleanup_count": 8,
            "rank_identities_dead": True, "launcher_member_count": 0,
            "stable_clear_samples": 2, "port_free": True,
        })
        names = [row[0] for row in backend.calls]
        self.assertEqual(names.count("upload"), len(controller.FILES))
        self.assertEqual(names.count("host"), 1)
        self.assertEqual(names.count("pre_host_closure"), 1)
        self.assertEqual(names.count("snapshot"), 2)
        host_call = next(row for row in backend.calls if row[0] == "host")
        self.assertEqual(
            host_call[2]["installed_custom_opp"],
            "/opt/ascend/cloud/packages/vendors/customize",
        )
        self.assertLess(names.index("read_rank_ownership"), names.index("summary"))
        self.assertEqual(names[-5:], ["cleanup", "forbidden", "snapshot", "target_close", "jump_close"])
        uploads = [row[1][2] for row in backend.calls if row[0] == "upload"]
        self.assertEqual({name.rsplit("/", 1)[-1] for name in uploads}, {path.name for path in controller.FILES})

    def test_mock_timeout_preserves_primary_cleans_and_closes(self) -> None:
        backend = self.Backend(self._summary(), fail_at="host")
        with mock.patch.object(controller, "dry_run_plan", return_value={"diagnostic_only": True}):
            with self.assertRaises(TimeoutError) as raised:
                controller.execute(backend)
        self.assertEqual(raised.exception.args, ("host",))
        names = [row[0] for row in backend.calls]
        self.assertEqual(names[-5:], ["cleanup", "forbidden", "snapshot", "target_close", "jump_close"])
        self.assertIn("forbidden", names)
        self.assertEqual(names.count("snapshot"), 2)

    def test_published_gate_durability_failure_uses_rank_evidence_cleanup(self) -> None:
        backend = self.Backend(self._summary())
        def published_failure(*args, **kwargs):
            backend.host_started = True
            backend._call("host", *args, **kwargs)
            error = RuntimeError("published artifact durability errors: gate_dir_fsync")
            error.published_artifact_errors = ("gate_dir_fsync: OSError: injected",)
            raise error
        backend.run_host_once = published_failure
        with mock.patch.object(controller, "dry_run_plan", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "published artifact durability errors") as raised:
                controller.execute(backend)
        self.assertEqual(raised.exception.published_artifact_errors,
                         ("gate_dir_fsync: OSError: injected",))
        names = [row[0] for row in backend.calls]
        self.assertIn("read_ownership", names)
        self.assertIn("read_rank_ownership", names)
        cleanup = next(row for row in backend.calls if row[0] == "cleanup")
        rank_evidence = cleanup[1][-1]
        self.assertIsNotNone(rank_evidence)
        self.assertEqual(rank_evidence["manifest"]["gate_token_sha256"], "d" * 64)
        self.assertNotIn("ownership_unestablished", str(raised.exception))

    def test_close_failures_do_not_replace_transaction_primary(self) -> None:
        backend = self.Backend(self._summary(), fail_at="shadow")
        backend.target.close = lambda: (_ for _ in ()).throw(OSError("close"))
        with mock.patch.object(controller, "dry_run_plan", return_value={}):
            with self.assertRaises(TimeoutError) as raised:
                controller.execute(backend)
        self.assertEqual(raised.exception.args, ("shadow",))
        self.assertEqual(len(raised.exception.cleanup_errors), 1)
        self.assertIn("jump_close", [row[0] for row in backend.calls])

    def test_missing_ownership_never_invokes_cleanup(self) -> None:
        backend = self.Backend(self._summary(), fail_at="host")
        backend.read_ownership = lambda *args: (backend._call("read_ownership", *args) or None)
        with mock.patch.object(controller, "dry_run_plan", return_value={}):
            with self.assertRaises(TimeoutError):
                controller.execute(backend)
        names = [row[0] for row in backend.calls]
        self.assertIn("read_ownership", names)
        self.assertNotIn("cleanup", names)

    def test_missing_rank_evidence_still_runs_launcher_cleanup_fail_closed(self) -> None:
        backend = self.Backend(self._summary())
        backend.read_rank_ownership = lambda *args: (backend._call("read_rank_ownership", *args) or None)
        with mock.patch.object(controller, "dry_run_plan", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "summary ownership closure"):
                controller.execute(backend)
        cleanup_calls = [row for row in backend.calls if row[0] == "cleanup"]
        self.assertEqual(len(cleanup_calls), 1)
        self.assertIsNone(cleanup_calls[0][1][-1])

    def test_installed_root_contract_and_post_snapshot_drift_fail_closed(self) -> None:
        valid = {
            "schema": "step377-protected-snapshot-v1", "artifacts": {},
            "runtime": {"installed_cloud_root": "/opt/ascend/cloud"},
            "installed_root": "/opt/ascend/cloud/packages/vendors/customize",
            "installed_qrv2": {"root": "/opt/ascend/cloud/packages/vendors/customize"},
            "related_processes": [],
            "back8": {"rows": [], "device_ids": [], "host_pids": []},
        }
        self.assertIs(controller._validate_snapshot(valid), valid)
        for cloud_root in (None, "relative/cloud", "/opt/../cloud", "/opt//cloud", "/opt/cloud/"):
            changed = copy.deepcopy(valid)
            changed["runtime"]["installed_cloud_root"] = cloud_root
            with self.subTest(cloud_root=cloud_root), self.assertRaises(RuntimeError):
                controller._validate_snapshot(changed)
        for key, value in (
            ("installed_root", "/different/packages/vendors/customize"),
            ("installed_qrv2", {"root": "/different/packages/vendors/customize"}),
        ):
            changed = copy.deepcopy(valid); changed[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                controller._validate_snapshot(changed)

        backend = self.Backend(self._summary())
        snapshots = [valid, {**copy.deepcopy(valid),
                             "runtime": {"installed_cloud_root": "/opt/ascend/cloud-v2"},
                             "installed_root": "/opt/ascend/cloud-v2/packages/vendors/customize",
                             "installed_qrv2": {"root": "/opt/ascend/cloud-v2/packages/vendors/customize"}}]
        backend.snapshot = lambda *args: (backend._call("snapshot", *args) or snapshots.pop(0))
        with mock.patch.object(controller, "dry_run_plan", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "installed/runtime/artifact closure changed"):
                controller.execute(backend)
        self.assertIn("cleanup", [row[0] for row in backend.calls])

    def test_guard_wiring_has_no_legacy_parser_or_cleanup(self) -> None:
        snapshot_source = inspect.getsource(controller.RealBackend.snapshot)
        cleanup_source = inspect.getsource(controller.RealBackend.cleanup_owned)
        self.assertNotIn("load_step343_parser", snapshot_source)
        self.assertNotIn("step343_world8_controller", cleanup_source)
        self.assertNotIn("killpg", cleanup_source)
        self.assertNotIn("--cleanup-owned", cleanup_source)
        self.assertIn("require_stable_clear", snapshot_source)
        self.assertIn("cleanup-owned", cleanup_source)

    def test_pidfd_unavailable_guard_failure_is_fail_closed(self) -> None:
        legacy = types.SimpleNamespace(run_host_script=mock.Mock(side_effect=RuntimeError("pidfd signaling unavailable")))
        backend = controller.RealBackend(legacy)
        backend.remote_root = "/safe/diagnostics/step377"
        ownership = ownership_evidence({"schema": "step358-launcher-ownership-v1", "port": controller.PORT,
                     "launcher_host_pid": 2, "launcher_starttime": 2, "launcher_pgid": 2})
        with self.assertRaisesRegex(RuntimeError, "pidfd signaling unavailable"):
            backend.cleanup_owned(object(), controller.PORT, ownership)

    def test_real_backend_is_thin_step357_adapter_with_explicit_methods(self) -> None:
        required = {
            "parse_machine_info", "connect_target", "require_hostname",
            "require_container", "exclusive_directory", "upload_new",
            "verify_uploads",
            "validate_readonly_artifacts", "snapshot", "prepare_shadow",
            "validate_shadow", "pre_host_closure", "require_stable_clear", "run_host_once", "read_summary",
            "read_ownership", "read_rank_ownership", "scan_forbidden_outputs", "cleanup_owned",
        }
        self.assertTrue(required.issubset(set(dir(controller.RealBackend))))
        calls = []
        legacy = types.SimpleNamespace(
            EXPECTED_HOSTNAME="expected",
            run_host_script=lambda target, script, timeout: (calls.append((script, timeout)) or ("expected\n", "")),
            container_probe=lambda target: calls.append(("probe", target)),
        )
        backend = controller.RealBackend(legacy)
        target = object()
        backend.require_hostname(target)
        backend.require_container(target, controller.CONTAINER)
        self.assertEqual(calls[0], ("hostname", 30))
        self.assertEqual(calls[1], ("probe", target))

    def test_real_backend_sftp_open_failure_closes_both_clients(self) -> None:
        closed = []
        target = types.SimpleNamespace(
            open_sftp=lambda: (_ for _ in ()).throw(OSError("sftp")),
            close=lambda: closed.append("target"),
        )
        jump = types.SimpleNamespace(close=lambda: closed.append("jump"))
        legacy = types.SimpleNamespace(connect_target=lambda module, info: (jump, target))
        backend = controller.RealBackend(legacy)
        backend.remote_module = object()
        with self.assertRaisesRegex(OSError, "sftp"):
            backend.connect_target({})
        self.assertEqual(closed, ["target", "jump"])

    def test_real_backend_generates_real_shadow_host_and_cleanup_commands(self) -> None:
        commands = []
        legacy = types.SimpleNamespace(
            run=lambda target, command, timeout: (commands.append(("run", command, timeout)) or ("{}", "")),
            run_host_script=lambda target, script, timeout: (commands.append(("host", script, timeout)) or ("", "")),
        )
        backend = controller.RealBackend(legacy)
        backend.remote_root = "/safe/diagnostics/step377"
        backend.prepare_shadow(object(), {}, timeout=900)
        with mock.patch.object(controller, "IMMUTABLE_ORIGINAL_WHEEL", "/immutable/original.whl"):
            backend.prepare_shadow(object(), {}, timeout=900)
        shadow_command = commands[-1][1]
        self.assertIn("step377_prepare_diagnostic_shadow.py", shadow_command)
        self.assertIn("--attempt3-manifest", shadow_command)
        self.assertIn("--output-manifest", shadow_command)
        with self.assertRaises(RuntimeError):
            backend.run_host_once(object(), {}, devices=list(range(8)), world_size=8, port=1, timeout=1)
        installed_root = "/opt/ascend/cloud/packages/vendors/customize"
        with mock.patch.object(
            backend, "_run_json",
            return_value=ownership_evidence({
                "schema": "step358-launcher-ownership-v1", "port": controller.PORT,
                "launcher_host_pid": 2, "launcher_starttime": 2, "launcher_pgid": 2,
            }),
        ):
            backend.run_host_once(
                object(), {}, devices=list(range(8, 16)), world_size=8,
                port=controller.PORT, timeout=1800, installed_custom_opp=installed_root,
            )
        host_command = commands[-1][1]
        self.assertIn("--installed-custom-opp " + installed_root, host_command)
        self.assertNotIn("/latest/opp", host_command)
        for invalid in ("relative", "/opt/../customize", "/opt//customize"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(RuntimeError, "path contract"):
                backend.run_host_once(
                    object(), {}, devices=list(range(8, 16)), world_size=8,
                    port=controller.PORT, timeout=1800, installed_custom_opp=invalid,
                )
        stable = {"schema": "step377-snapshot-idle-v1", "port_free": True,
                  "stable_clear": guarded_cleanup_result()["stable_clear"]}
        legacy.run_host_script = lambda target, script, timeout: (commands.append(("host", script, timeout)) or (json.dumps(stable), ""))
        backend.require_stable_clear(object())
        idle_command = commands[-1][1]
        self.assertIn("snapshot-idle", idle_command)
        self.assertIn("--case-path /safe/diagnostics/step377/step377_diagnostic_host_case.py", idle_command)
        self.assertIn("--port " + str(controller.PORT), idle_command)
        self.assertNotIn("--proc-root", idle_command)
        guarded = guarded_cleanup_result(with_rank=False)
        legacy.run_host_script = lambda target, script, timeout: (commands.append(("host", script, timeout)) or (json.dumps(guarded), ""))
        backend.cleanup_owned(object(), controller.PORT, ownership_evidence({"schema": "step358-launcher-ownership-v1", "port": controller.PORT, "launcher_host_pid": 2, "launcher_starttime": 2, "launcher_pgid": 2}))
        wrapper = commands[-1][1]
        self.assertIn("step377_process_guard.py", wrapper)
        self.assertIn("cleanup-owned", wrapper)
        self.assertIn("--expected-ownership-sha256 " + "a" * 64, wrapper)
        self.assertIn("--case-path /safe/diagnostics/step377/step377_diagnostic_host_case.py", wrapper)
        self.assertIn("--port " + str(controller.PORT), wrapper)
        self.assertNotIn("--proc-root", wrapper)
        self.assertNotIn("killpg", wrapper)
        self.assertNotIn("step343_world8_controller", wrapper)
        guarded_rank = guarded_cleanup_result(with_rank=True)
        legacy.run_host_script = lambda target, script, timeout: (commands.append(("host", script, timeout)) or (json.dumps(guarded_rank), ""))
        backend.cleanup_owned(object(), controller.PORT,
                              ownership_evidence({"schema": "step358-launcher-ownership-v1", "port": controller.PORT, "launcher_host_pid": 2, "launcher_starttime": 2, "launcher_pgid": 2}),
                              rank_ownership_evidence())
        rank_command = commands[-1][1]
        self.assertIn("--rank-ownership /safe/diagnostics/step377/run/rank_ownership.json", rank_command)
        self.assertIn("--expected-rank-ownership-sha256 " + "a" * 64, rank_command)
        self.assertIn("--expected-gate-token-sha256 " + "d" * 64, rank_command)


if __name__ == "__main__":
    unittest.main()
