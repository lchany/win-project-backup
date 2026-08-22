#!/usr/bin/env python3
"""Offline wiring tests for the disabled STEP376 remote build controller."""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import step376_build_qrv2_delta1_probe_remote as controller


class Step376RemoteWiringTests(unittest.TestCase):
    @staticmethod
    def _valid_summary() -> dict:
        flags = {
            "artifact_class": "diagnostic_probe",
            "diagnostic_only": True,
            "release_candidate": False,
            "package_forbidden": True,
        }
        entries = [
            controller.CANDIDATE_IDENTITY + "_0_mix_aic",
            controller.CANDIDATE_IDENTITY + "_0_mix_aiv",
        ]
        artifacts = {
            soc: {
                "object_sha256": "1" * 64,
                "json_sha256": "2" * 64,
                "opc_log_sha256": str(index + 3) * 64,
                "concrete_entries": entries,
            }
            for index, soc in enumerate(controller.SOCS)
        }
        return {
            "status": "diagnostic_built_unvalidated",
            "policy": flags,
            "candidate": {
                "identity": controller.CANDIDATE_IDENTITY,
                "source_sha256": controller.CANDIDATE_SHA256,
                "reverse_v4_sha256": controller.REVERSE_V4_SHA256,
                **flags,
            },
            "package": {"status": "forbidden_diagnostic_probe"},
            "tools": dict(controller.EXPECTED_TOOL_SHA256),
            "artifacts": artifacts,
            "alias_bytes_equal": True,
            "installed_inventory_closed": True,
            "runtime_inventory_closed": True,
            "release_outputs_absent": True,
        }

    @staticmethod
    def _summary_filesystem_fixture(root: Path) -> tuple[Path, list[str]]:
        work = root / "work"
        work.mkdir()
        wheel = work / "wheel_original" / "original.whl"
        wheel.parent.mkdir()
        wheel.write_bytes(b"original-wheel")
        flags = {
            "artifact_class": "diagnostic_probe",
            "diagnostic_only": True,
            "release_candidate": False,
            "package_forbidden": True,
        }
        artifacts = {}
        entries = sorted(
            (
                controller.CANDIDATE_IDENTITY + "_0_mix_aic",
                controller.CANDIDATE_IDENTITY + "_0_mix_aiv",
            )
        )
        for index, soc in enumerate(controller.SOCS):
            output = work / "build" / soc / "output"
            output.mkdir(parents=True)
            obj = output / (controller.CANDIDATE_IDENTITY + ".o")
            meta = output / (controller.CANDIDATE_IDENTITY + ".json")
            log = output.parent / "opc.log"
            obj.write_bytes(b"same-object")
            meta.write_bytes(b"same-json")
            log.write_bytes(f"log-{index}".encode())
            artifacts[soc] = {
                "object_path": str(obj),
                "object_size": obj.stat().st_size,
                "object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
                "json_path": str(meta),
                "json_size": meta.stat().st_size,
                "json_sha256": hashlib.sha256(meta.read_bytes()).hexdigest(),
                "opc_log_path": str(log),
                "opc_log_size": log.stat().st_size,
                "opc_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                "kernel_name": controller.CANDIDATE_IDENTITY,
                "bin_file_name": controller.CANDIDATE_IDENTITY,
                "concrete_entries": entries,
            }
        manifest = {
            "status": "diagnostic_built_unvalidated",
            "policy": dict(flags),
            "candidate": {
                "identity": controller.CANDIDATE_IDENTITY,
                "bin_name": controller.CANDIDATE_IDENTITY,
                "source_sha256": controller.CANDIDATE_SHA256,
                "reverse_v4_sha256": controller.REVERSE_V4_SHA256,
                "structure_assertions": {
                    "candidate_identity": controller.CANDIDATE_IDENTITY,
                    "reverse_v4_sha256": controller.REVERSE_V4_SHA256,
                },
                **flags,
            },
            "package": {"status": "forbidden_diagnostic_probe"},
            "tools": dict(controller.EXPECTED_TOOL_SHA256),
            "build_runtime": {
                "installed_inventory_closed": True,
                "runtime_inventory_closed": True,
            },
            "paths": {"extracted_wheel": str(wheel)},
            "immutable_guards": {
                "extracted_original_wheel": {
                    "path": str(wheel),
                    "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                }
            },
            "artifacts": artifacts,
        }
        manifest_path = work / "release_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        arguments = [
            str(manifest_path),
            controller.CANDIDATE_IDENTITY,
            controller.CANDIDATE_SHA256,
            controller.REVERSE_V4_SHA256,
            json.dumps(controller.EXPECTED_TOOL_SHA256, sort_keys=True),
        ]
        return manifest_path, arguments

    def test_disabled_build_fails_before_loading_helper_or_mapping(self) -> None:
        self.assertFalse(controller.BUILD_READY)
        self.assertEqual(
            controller.DIAG_NAME,
            "step376_attempt3_qrv2_vtv_direct_qa_legacy_probe_v6_build_20260822",
        )
        historical_names = {
            "step376_qrv2_vtv_direct_qa_legacy_probe_v6_build_20260822",
            "step376_retry2_qrv2_vtv_direct_qa_legacy_probe_v6_build_20260822",
        }
        self.assertNotIn(controller.DIAG_NAME, historical_names)
        with mock.patch.object(
            controller.importlib.util, "spec_from_file_location"
        ) as load_spec, self.assertRaisesRegex(RuntimeError, "intentionally disabled"):
            controller._dry_run_payload()
        load_spec.assert_not_called()

    def test_exclusive_directory_script_is_valid_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "probe dir;$(false)'特殊"
            script = controller._exclusive_directory_script(str(target))
            syntax = subprocess.run(
                ["bash", "-n"], input=script, text=True, capture_output=True
            )
            self.assertEqual(syntax.returncode, 0, syntax.stderr)

            created = subprocess.run(
                ["bash", "-c", script], text=True, capture_output=True
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            self.assertTrue(target.is_dir())

            rejected = subprocess.run(
                ["bash", "-c", script], text=True, capture_output=True
            )
            self.assertEqual(rejected.returncode, 73)
            self.assertTrue(target.is_dir())

    def test_exclusive_directory_script_rejects_dangling_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dangling"
            target.symlink_to(Path(directory) / "missing-target")
            result = subprocess.run(
                ["bash", "-c", controller._exclusive_directory_script(str(target))],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(target.is_symlink())
            self.assertFalse(target.exists())

    def test_exclusive_directory_script_allows_only_one_concurrent_creator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "contended"
            script = controller._exclusive_directory_script(str(target))
            processes = [
                subprocess.Popen(
                    ["bash", "-c", script],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for _ in range(8)
            ]
            results = [process.communicate(timeout=5) + (process.returncode,) for process in processes]
            self.assertEqual(sum(returncode == 0 for _out, _err, returncode in results), 1)
            self.assertTrue(target.is_dir())

    def test_locked_helper_remote_exec_and_upload_hashes_match(self) -> None:
        expected = {
            controller.LEGACY_PATH: controller.LEGACY_SHA256,
            controller.REMOTE_EXEC_PATH: controller.REMOTE_EXEC_SHA256,
            **{
                path: controller.EXPECTED_INPUTS[path.name]
                for path in controller.input_files()
            },
        }
        for path, digest in expected.items():
            with self.subTest(path=path.name):
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
                self.assertEqual(controller.sha256_file(path), digest)

    def test_legacy_load_uses_exact_five_inputs_and_poisoned_execute(self) -> None:
        with mock.patch.object(controller, "BUILD_READY", True):
            legacy = controller.load_legacy()
        self.assertEqual(
            tuple(path.name for path in legacy.input_files()),
            tuple(path.name for path in controller.input_files()),
        )
        self.assertEqual(len(legacy.input_files()), 5)
        self.assertEqual(legacy.EXPECTED_INPUTS, controller.EXPECTED_INPUTS)
        names = {path.name for path in legacy.input_files()}
        self.assertNotIn("step372_patch_qr_v2_matmul_position_v5.py", names)
        self.assertFalse(any(name.endswith(".whl") for name in names))
        with self.assertRaisesRegex(RuntimeError, "legacy.execute is forbidden"):
            legacy.execute()

    def test_hash_drift_and_extra_v5_input_fail_closed(self) -> None:
        with mock.patch.object(controller, "BUILD_READY", True), mock.patch.object(
            controller, "REMOTE_EXEC_SHA256", "0" * 64
        ):
            with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                controller.load_legacy()

        v5 = controller.TOOLS / "step372_patch_qr_v2_matmul_position_v5.py"
        with mock.patch.object(controller, "BUILD_READY", True), mock.patch.object(
            controller,
            "input_files",
            return_value=controller.input_files() + (v5,),
        ):
            with self.assertRaisesRegex(RuntimeError, "fixed contract"):
                controller.load_legacy()

    def test_dry_run_action_and_forbidden_sets_are_exact(self) -> None:
        with mock.patch.object(controller, "BUILD_READY", True):
            legacy = controller.load_legacy()
        legacy.load_remote_module = lambda: object()
        legacy.local_preflight = lambda _module: {"target_host": "10.0.0.42"}
        payload = controller._dry_run_payload(legacy)
        self.assertEqual(payload["actions"], list(controller.DRY_RUN_ACTIONS))
        self.assertEqual(payload["forbidden"], list(controller.FORBIDDEN_ACTIONS))
        self.assertEqual(payload["diagnostics_name"], controller.DIAG_NAME)
        self.assertEqual(payload["target_suffix"], "42")
        self.assertEqual(payload["input_sha256"], controller.EXPECTED_INPUTS)

    def test_inherited_remote_path_helper_rejects_escape(self) -> None:
        with mock.patch.object(controller, "BUILD_READY", True):
            legacy = controller.load_legacy()
        self.assertEqual(
            legacy.safe_remote_path("/shared", controller.DIAG_NAME),
            "/shared/diagnostics/" + controller.DIAG_NAME,
        )
        with self.assertRaisesRegex(RuntimeError, "unsafe remote diagnostics path"):
            legacy.safe_remote_path("/shared", "../escape")
        with self.assertRaisesRegex(RuntimeError, "unsafe remote diagnostics path"):
            legacy.safe_remote_path("relative", controller.DIAG_NAME)

    def test_container_contract_uses_only_adapter_prepare_and_build(self) -> None:
        legacy = types.SimpleNamespace(OPC="/opt/opc")
        remote_diag = "/shared/diagnostics/" + controller.DIAG_NAME
        contract = {
            "ascend_opp": "/opt/opp",
            "installed_cloud_root": "/installed/cloud",
            "opc": {"path": "/resolved/toolkit/bin/opc"},
        }
        script = controller._container_script(legacy, contract, remote_diag)
        self.assertEqual(script.count("python3 build_qrv2_diagnostic_probe.py"), 2)
        self.assertIn(" prepare ", script)
        self.assertIn(" build ", script)
        self.assertEqual(script.count("--approved-root " + remote_diag), 2)
        self.assertEqual(script.count("--workdir " + remote_diag + "/work"), 2)
        self.assertNotIn(" package ", script)
        self.assertNotIn("step372_patch", script)
        self.assertNotIn("torchrun", script)
        self.assertIn("export PYTHONDONTWRITEBYTECODE=1", script)
        self.assertIn("--opc /resolved/toolkit/bin/opc", script)
        self.assertNotIn(legacy.OPC, script)
        command_tokens = [
            shlex.split(line)
            for line in script.splitlines()
            if line.strip().startswith("python3 ")
        ]
        self.assertEqual([tokens[2] for tokens in command_tokens], ["prepare", "build"])
        executed_tokens = {token for tokens in command_tokens for token in tokens}
        for forbidden in controller.FORBIDDEN_ACTIONS:
            self.assertNotIn(forbidden, executed_tokens)

    def test_contract_opc_path_validation_rejects_bad_shapes(self) -> None:
        invalid = (
            {},
            {"opc": None},
            {"opc": {}},
            {"opc": {"path": None}},
            {"opc": {"path": 7}},
            {"opc": {"path": ""}},
            {"opc": {"path": "relative/opc"}},
            {"opc": {"path": "/resolved/opc\x00suffix"}},
        )
        for contract in invalid:
            with self.subTest(contract=contract), self.assertRaises(RuntimeError):
                controller._validated_contract_opc_path(contract)

    def test_contract_realpath_is_shared_by_build_and_snapshots_and_alias_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "resolved tool 'one'"
            second = root / "resolved tool two"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            alias = root / "opc-alias"
            alias.symlink_to(first)
            locked = str(first.resolve(strict=True))
            legacy = types.SimpleNamespace(
                CONTAINER="mapqr-leicheng",
                OPC=str(alias),
            )
            contract = {
                "ascend_opp": "/opt/opp",
                "installed_cloud_root": "/installed",
                "opc": {"path": locked},
            }
            script_before = controller._container_script(
                legacy, contract, "/shared/diagnostics/probe"
            )
            alias.unlink()
            alias.symlink_to(second)
            script_after = controller._container_script(
                legacy, contract, "/shared/diagnostics/probe"
            )
            self.assertEqual(script_before, script_after)
            build_tokens = shlex.split(
                next(line for line in script_after.splitlines() if " build " in line)
            )
            self.assertEqual(build_tokens[build_tokens.index("--opc") + 1], locked)
            self.assertNotIn(str(alias), script_after)

            commands = []
            snapshot = {
                "snapshot_schema": 1,
                "runtime": {},
                "installed": {},
                "related_build_processes": [],
            }
            legacy.run = lambda _target, command, *, timeout: (
                commands.append(command) or json.dumps(snapshot),
                "",
            )
            controller._run_snapshot(
                legacy,
                object(),
                "/shared/diagnostics/probe",
                "/shared/diagnostics/probe/container_contract.json",
                contract,
            )
            snapshot_tokens = shlex.split(commands[0])
            self.assertIn(locked, snapshot_tokens)
            self.assertNotIn(str(alias), commands[0])

    def test_base_regular_file_gate_still_rejects_missing_and_symlink(self) -> None:
        import build_qrv2_release as base

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing-opc"
            with self.assertRaises(FileNotFoundError):
                base._regular_file_inventory(missing, label="OPC executable")
            target = root / "real-opc"
            target.write_bytes(b"opc")
            alias = root / "opc-alias"
            alias.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "must not be a symlink"):
                base._regular_file_inventory(alias, label="OPC executable")

    def test_embedded_production_code_has_no_assert_and_optimize_cannot_bypass(self) -> None:
        for label, source in (
            ("upload", controller._upload_gate_code()),
            ("snapshot", controller._snapshot_code()),
            ("summary", controller._summary_code()),
        ):
            with self.subTest(label=label):
                tree = ast.parse(source)
                self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))
                compile(source, f"<{label}>", "exec", optimize=2)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {}
            for index in range(6):
                path = root / f"input-{index}"
                path.write_bytes(str(index).encode())
                expected[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "extra-directory").mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    "-O",
                    "-c",
                    controller._upload_gate_code(),
                    str(root),
                    json.dumps(expected, sort_keys=True),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("upload entry count mismatch", result.stderr)

    def test_python_commands_disable_bytecode_and_isolated_run_creates_no_cache(self) -> None:
        snapshot_commands = []
        snapshot = {
            "snapshot_schema": 1,
            "runtime": {},
            "installed": {},
            "related_build_processes": [],
        }
        legacy = types.SimpleNamespace(
            CONTAINER="mapqr-leicheng",
            OPC="/opt/opc",
            run=lambda _target, command, *, timeout: (
                snapshot_commands.append(command) or json.dumps(snapshot),
                "",
            ),
        )
        controller._run_snapshot(
            legacy,
            object(),
            "/shared/diagnostics/probe",
            "/shared/diagnostics/probe/container_contract.json",
            {
                "installed_cloud_root": "/installed",
                "opc": {"path": "/resolved/toolkit/bin/opc"},
            },
        )
        self.assertEqual(len(snapshot_commands), 1)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", snapshot_commands[0])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "local_probe.py").write_text("VALUE = 1\n", encoding="utf-8")
            result = subprocess.run(
                [
                    "env",
                    "PYTHONDONTWRITEBYTECODE=1",
                    sys.executable,
                    "-c",
                    "import local_probe; assert local_probe.VALUE == 1",
                ],
                cwd=root,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "0"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "__pycache__").exists())
            self.assertEqual(list(root.rglob("*.pyc")), [])

    def test_upload_gate_rejects_extra_directory_and_fifo(self) -> None:
        for kind in ("directory", "fifo"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                expected = {}
                for index in range(6):
                    path = root / f"input-{index}"
                    path.write_bytes(str(index).encode())
                    expected[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                extra = root / "extra"
                if kind == "directory":
                    extra.mkdir()
                else:
                    os.mkfifo(extra)
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        controller._upload_gate_code(),
                        str(root),
                        json.dumps(expected, sort_keys=True),
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("upload entry count mismatch", result.stderr)

    def test_summary_code_contains_filesystem_and_runtime_hard_gates(self) -> None:
        source = controller._summary_code()
        for required in (
            "diagnostic_built_unvalidated",
            "forbidden_diagnostic_probe",
            "installed_inventory_closed",
            "runtime_inventory_closed",
            "object_bytes",
            "json_bytes",
            "opc_log_size",
            "concrete_entries",
            "work/'release'",
            "suffix.lower()=='.zip'",
            "suffix.lower()=='.whl'",
            "relative_to(build)",
            "relative_to(output)",
            "allowed wheel differs from immutable guard",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)

    def test_optimized_summary_rejects_artifact_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, arguments = self._summary_filesystem_fixture(root)
            valid = subprocess.run(
                [sys.executable, "-O", "-c", controller._summary_code(), *arguments],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(valid.returncode, 0, valid.stderr)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            outside = root / "outside.o"
            outside.write_bytes(b"same-object")
            canonical = manifest["artifacts"][controller.SOCS[0]]
            canonical["object_path"] = str(outside)
            canonical["object_size"] = outside.stat().st_size
            canonical["object_sha256"] = hashlib.sha256(
                outside.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            rejected = subprocess.run(
                [sys.executable, "-O", "-c", controller._summary_code(), *arguments],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("artifact path escaped SoC build root", rejected.stderr)

    def test_summary_accepts_only_the_complete_diagnostic_contract(self) -> None:
        controller._validate_summary(self._valid_summary())
        mutations = (
            ("status", lambda value: value.update({"status": "built"})),
            ("policy", lambda value: value["policy"].update({"diagnostic_only": False})),
            ("candidate", lambda value: value["candidate"].update({"source_sha256": "0" * 64})),
            ("package", lambda value: value.update({"package": {"status": "pending"}})),
            ("tools", lambda value: value["tools"].update({"v4_patcher_sha256": "0" * 64})),
            ("concrete", lambda value: value["artifacts"][controller.SOCS[0]].update({"concrete_entries": []})),
            ("alias_sha", lambda value: value["artifacts"][controller.SOCS[1]].update({"object_sha256": "9" * 64})),
            ("opc_log", lambda value: value["artifacts"][controller.SOCS[0]].update({"opc_log_sha256": "short"})),
            ("installed", lambda value: value.update({"installed_inventory_closed": False})),
            ("runtime", lambda value: value.update({"runtime_inventory_closed": False})),
            ("release", lambda value: value.update({"release_outputs_absent": False})),
            ("extra_top", lambda value: value.update({"unexpected": True})),
            ("artifact_list", lambda value: value.update({"artifacts": []})),
            ("upper_hex", lambda value: value["artifacts"][controller.SOCS[0]].update({"opc_log_sha256": "A" * 64})),
            ("entry_tuple", lambda value: value["artifacts"][controller.SOCS[0]].update({"concrete_entries": tuple(value["artifacts"][controller.SOCS[0]]["concrete_entries"])})),
            ("bool_as_int", lambda value: value.update({"alias_bytes_equal": 1})),
            ("policy_bool_as_int", lambda value: value["policy"].update({"diagnostic_only": 1})),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                summary = copy.deepcopy(self._valid_summary())
                mutate(summary)
                with self.assertRaises(RuntimeError):
                    controller._validate_summary(summary)

    def test_controller_source_never_calls_legacy_execute(self) -> None:
        source = Path(controller.__file__).read_text(encoding="utf-8")
        self.assertNotIn("legacy.execute(", source)
        tree = ast.parse(source)
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))

    def test_successful_execute_has_exact_transaction_and_closes_all_resources(self) -> None:
        class Channel:
            def __init__(self) -> None:
                self.timeout = None

            def settimeout(self, timeout) -> None:
                self.timeout = timeout

        class Sftp:
            def __init__(self) -> None:
                self.closed = False
                self.channel = Channel()

            def get_channel(self):
                return self.channel

            def close(self) -> None:
                self.closed = True

        class Client:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        jump = Client()
        target = Client()
        sftp = Sftp()
        target.open_sftp = lambda: sftp
        uploads = []
        commands = []
        host_scripts = []
        snapshot = {
            "snapshot_schema": 1,
            "runtime": {"opc": "closed"},
            "installed": {"qr_v2": "unchanged"},
            "related_build_processes": [],
        }

        def run(_target, command, *, timeout):
            commands.append((command, timeout))
            if command == "hostname":
                return "expected-host\n", ""
            if "uploaded_input_gate=PASS" in command:
                return "uploaded_input_gate=PASS\n", ""
            if "snapshot_schema" in command:
                return json.dumps(snapshot), ""
            if "timeout --signal=TERM" in command:
                return "", ""
            if "diagnostic_built_unvalidated" in command:
                return json.dumps(self._valid_summary()), ""
            raise AssertionError(f"unexpected command: {command}")

        legacy = types.SimpleNamespace(
            CONTAINER="mapqr-leicheng",
            OPC="/opt/opc",
            EXPECTED_HOSTNAME="expected-host",
            load_remote_module=lambda: object(),
            local_preflight=lambda _module: {"shared": "/shared"},
            connect_target=lambda _module, _info: (jump, target),
            run=run,
            container_probe=lambda _target: {
                "schema_version": 1,
                "container_name": "mapqr-leicheng",
                "inspect_container_id": "a" * 64,
                "inspect_hostname": "container-host",
                "opc": {"path": "/opt/opc", "sha256": "1" * 64},
                "cann_version_files": [{"path": "/version", "sha256": "2" * 64}],
                "ascend_opp": "/opt/opp",
                "installed_cloud_root": "/installed/cloud",
            },
            safe_remote_path=lambda _shared, name: "/shared/diagnostics/" + name,
            run_host_script=lambda _target, script, *, timeout: host_scripts.append(
                (script, timeout)
            ),
            write_remote_new=lambda _sftp, path, payload: uploads.append(
                (path, payload)
            ),
        )
        with mock.patch.object(controller, "load_legacy", return_value=legacy):
            result = controller.execute()

        self.assertEqual(result["remote_diagnostics_name"], controller.DIAG_NAME)
        self.assertTrue(result["uploaded_gate"])
        self.assertEqual(len(host_scripts), 1)
        self.assertIn("[ ! -e", host_scripts[0][0])
        self.assertIn("mkdir -m 700", host_scripts[0][0])
        self.assertEqual(len(uploads), 6)
        self.assertEqual(
            {Path(path).name for path, _payload in uploads},
            set(controller.EXPECTED_INPUTS) | {"container_contract.json"},
        )
        build_commands = [command for command, _ in commands if "timeout --signal=TERM" in command]
        self.assertEqual(len(build_commands), 1)
        self.assertIn("--kill-after=30s 600s", build_commands[0])
        self.assertEqual(
            sum("snapshot_schema" in command for command, _ in commands), 2
        )
        self.assertTrue(all(timeout > 0 for _command, timeout in commands))
        for command, _timeout in commands:
            if "python3" in command or "timeout --signal=TERM" in command:
                self.assertIn("PYTHONDONTWRITEBYTECODE=1", command)
        self.assertEqual(sftp.channel.timeout, controller.REMOTE_SHORT_TIMEOUT)
        self.assertTrue(sftp.closed)
        self.assertTrue(target.closed)
        self.assertTrue(jump.closed)

    def test_execute_directory_creation_failure_is_not_retried_or_replaced(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        jump = Client()
        target = Client()
        target.open_sftp = mock.Mock(side_effect=AssertionError("upload must not start"))
        creation_error = OSError(17, "exclusive directory already exists")
        original_args = creation_error.args
        original_text = str(creation_error)
        create = mock.Mock(side_effect=creation_error)
        write_remote_new = mock.Mock()

        def run(_target, command, *, timeout):
            self.assertEqual(command, "hostname")
            return "expected-host\n", ""

        legacy = types.SimpleNamespace(
            EXPECTED_HOSTNAME="expected-host",
            load_remote_module=lambda: object(),
            local_preflight=lambda _module: {"shared": "/shared"},
            connect_target=lambda _module, _info: (jump, target),
            run=run,
            container_probe=lambda _target: {"schema_version": 1},
            safe_remote_path=lambda _shared, name: "/shared/diagnostics/" + name,
            run_host_script=create,
            write_remote_new=write_remote_new,
        )
        with mock.patch.object(controller, "load_legacy", return_value=legacy):
            with self.assertRaises(OSError) as raised:
                controller.execute()

        self.assertIs(raised.exception, creation_error)
        self.assertEqual(raised.exception.args, original_args)
        self.assertEqual(str(raised.exception), original_text)
        create.assert_called_once()
        target.open_sftp.assert_not_called()
        write_remote_new.assert_not_called()
        self.assertTrue(target.closed)
        self.assertTrue(jump.closed)

    def test_timeout_keeps_primary_and_attaches_postflight_mismatch(self) -> None:
        primary = RuntimeError("bounded build timed out")
        original_args = primary.args
        original_text = str(primary)
        snapshots = [
            {
                "snapshot_schema": 1,
                "runtime": {"state": "before"},
                "installed": {"state": "before"},
                "related_build_processes": [],
            },
            {
                "snapshot_schema": 1,
                "runtime": {"state": "after"},
                "installed": {"state": "before"},
                "related_build_processes": [],
            },
        ]

        def run(_target, command, *, timeout):
            if "snapshot_schema" in command:
                return json.dumps(snapshots.pop(0)), ""
            if "timeout --signal=TERM" in command:
                raise primary
            raise AssertionError(command)

        legacy = types.SimpleNamespace(
            CONTAINER="mapqr-leicheng", OPC="/opt/opc", run=run
        )
        contract = {
            "installed_cloud_root": "/installed",
            "ascend_opp": "/opp",
            "opc": {"path": "/resolved/toolkit/bin/opc"},
        }
        with self.assertRaises(RuntimeError) as caught:
            controller._run_build_transaction(
                legacy, object(), contract, "/shared/diagnostics/probe"
            )
        self.assertIs(caught.exception, primary)
        self.assertEqual(primary.args, original_args)
        self.assertEqual(str(primary), original_text)
        self.assertIsInstance(primary.cleanup_error, RuntimeError)
        self.assertFalse(snapshots)

    def test_postflight_process_or_inventory_mismatch_fails(self) -> None:
        cases = (
            (
                "process",
                {"runtime": {"x": 1}, "installed": {"x": 1}, "related_build_processes": [9]},
                "process postflight",
            ),
            (
                "installed",
                {"runtime": {"x": 1}, "installed": {"x": 2}, "related_build_processes": []},
                "installed QrV2 snapshot changed",
            ),
        )
        for label, after_fields, message in cases:
            with self.subTest(label=label):
                before = {
                    "snapshot_schema": 1,
                    "runtime": {"x": 1},
                    "installed": {"x": 1},
                    "related_build_processes": [],
                }
                after = {"snapshot_schema": 1, **after_fields}
                snapshots = [before, after]

                def run(_target, command, *, timeout):
                    if "snapshot_schema" in command:
                        return json.dumps(snapshots.pop(0)), ""
                    return "", ""

                legacy = types.SimpleNamespace(
                    CONTAINER="mapqr-leicheng", OPC="/opt/opc", run=run
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    controller._run_build_transaction(
                        legacy,
                        object(),
                        {
                            "installed_cloud_root": "/installed",
                            "ascend_opp": "/opp",
                            "opc": {"path": "/resolved/toolkit/bin/opc"},
                        },
                        "/shared/diagnostics/probe",
                    )

    def test_multiple_close_errors_do_not_replace_primary_and_all_are_attempted(self) -> None:
        calls = []

        class Broken:
            def __init__(self, label):
                self.label = label

            def close(self):
                calls.append(self.label)
                raise OSError(self.label + " close failed")

        primary = RuntimeError("build failed")
        args = primary.args
        text = str(primary)
        result = controller._close_resources_preserving(
            primary,
            (
                ("SFTP close", Broken("sftp")),
                ("target close", Broken("target")),
                ("jump close", Broken("jump")),
            ),
        )
        self.assertIs(result, primary)
        self.assertEqual(calls, ["sftp", "target", "jump"])
        self.assertEqual(primary.args, args)
        self.assertEqual(str(primary), text)
        self.assertEqual(len(primary.cleanup_errors), 3)

    def test_hostile_cleanup_formatting_and_primary_attributes_cannot_replace_primary(self) -> None:
        class UnprintableCleanup(OSError):
            def __str__(self) -> str:
                raise RuntimeError("cleanup formatting trap")

        primary = RuntimeError("primary remains")
        original_args = primary.args
        original_text = str(primary)
        controller._append_cleanup_error(
            primary, UnprintableCleanup(), "postflight"
        )
        evidence = controller._failure_evidence(primary)
        self.assertEqual(evidence["primary"]["message"], original_text)
        self.assertIn("<unprintable:", evidence["cleanup_errors"][0]["message"])
        self.assertEqual(primary.args, original_args)
        self.assertEqual(str(primary), original_text)

        class HostilePrimary(RuntimeError):
            def __getattribute__(self, name):
                if name in {"cleanup_error", "cleanup_errors", "add_note"}:
                    raise RuntimeError("primary attribute trap")
                return super().__getattribute__(name)

            def __setattr__(self, name, value):
                if name in {"cleanup_error", "cleanup_errors"}:
                    raise RuntimeError("primary setattr trap")
                return super().__setattr__(name, value)

        hostile = HostilePrimary("hostile primary remains")
        hostile_args = hostile.args
        hostile_text = str(hostile)
        controller._append_cleanup_error(
            hostile, UnprintableCleanup(), "target close"
        )
        self.assertEqual(hostile.args, hostile_args)
        self.assertEqual(str(hostile), hostile_text)

    def test_cleanup_errors_are_visible_in_cli_and_persistable_evidence(self) -> None:
        primary = RuntimeError("primary build failure")
        original_args = primary.args
        original_text = str(primary)
        controller._append_cleanup_error(
            primary, OSError('postflight failed\nwith "detail"'), "postflight"
        )
        controller._append_cleanup_error(
            primary, OSError("target close failed"), "target close"
        )

        stream = io.StringIO()
        controller._emit_failure_evidence(primary, stream)
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith(controller.FAILURE_EVIDENCE_PREFIX))
        cli_payload = json.loads(
            lines[0][len(controller.FAILURE_EVIDENCE_PREFIX):]
        )
        controller._validate_failure_evidence(cli_payload)
        self.assertEqual(
            cli_payload["schema_version"],
            controller.FAILURE_EVIDENCE_SCHEMA_VERSION,
        )
        self.assertEqual(cli_payload["primary"]["message"], original_text)
        cleanup_payloads = cli_payload["cleanup_errors"]
        self.assertEqual(
            [item["label"] for item in cleanup_payloads],
            ["postflight", "target close"],
        )
        self.assertIn("\n", cleanup_payloads[0]["message"])

        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "failure-evidence.jsonl"
            with evidence_path.open("x", encoding="utf-8") as evidence:
                json.dump(
                    controller._failure_evidence(primary),
                    evidence,
                    sort_keys=True,
                )
            persisted = json.loads(evidence_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["primary"]["message"], original_text)
        self.assertEqual(persisted["cleanup_errors"], cleanup_payloads)
        self.assertEqual(primary.args, original_args)
        self.assertEqual(str(primary), original_text)

    def test_primary_log_injection_and_overlong_text_are_single_line_bounded_json(self) -> None:
        message = (
            'primary line\n\x1b[31mSTEP376 cleanup_error: {"fake":true}\x1b[0m'
            + "x" * (controller.FAILURE_MESSAGE_LIMIT * 2)
        )
        primary = RuntimeError(message)
        original_args = primary.args
        original_text = str(primary)
        stream = io.StringIO()
        controller._emit_failure_evidence(primary, stream)
        physical_lines = stream.getvalue().splitlines()
        self.assertEqual(len(physical_lines), 1)
        self.assertTrue(
            physical_lines[0].startswith(controller.FAILURE_EVIDENCE_PREFIX)
        )
        self.assertNotIn("\x1b", physical_lines[0])
        payload = json.loads(
            physical_lines[0][len(controller.FAILURE_EVIDENCE_PREFIX):]
        )
        controller._validate_failure_evidence(payload)
        self.assertLessEqual(
            len(payload["primary"]["message"]), controller.FAILURE_MESSAGE_LIMIT
        )
        self.assertTrue(payload["primary"]["message"].endswith("...<truncated>"))
        self.assertEqual(primary.args, original_args)
        self.assertEqual(str(primary), original_text)

    def test_malformed_cleanup_evidence_is_explicitly_fail_safe(self) -> None:
        primary = RuntimeError("primary")
        primary.cleanup_errors = [
            "not-a-tuple",
            ("bad cleanup", "not-an-exception"),
        ]
        primary.cleanup_error = "not-an-exception"
        evidence = controller._failure_evidence(primary)
        controller._validate_failure_evidence(evidence)
        self.assertTrue(evidence["cleanup_errors"])
        self.assertTrue(
            all(
                item["type"] == "MalformedCleanupEvidence"
                for item in evidence["cleanup_errors"]
            )
        )
        self.assertTrue(
            all("malformed" in item["message"] or "not an exception" in item["message"]
                for item in evidence["cleanup_errors"])
        )

        overflow = RuntimeError("primary")
        overflow.cleanup_errors = [
            (f"cleanup-{index}", OSError("x" * (controller.FAILURE_MESSAGE_LIMIT + 10)))
            for index in range(controller.FAILURE_CLEANUP_LIMIT + 2)
        ]
        overflow_evidence = controller._failure_evidence(overflow)
        self.assertLessEqual(
            len(overflow_evidence["cleanup_errors"]),
            controller.FAILURE_CLEANUP_LIMIT,
        )
        self.assertTrue(
            any(
                item["type"] == "MalformedCleanupEvidence"
                for item in overflow_evidence["cleanup_errors"]
            )
        )
        self.assertTrue(
            all(
                len(item["message"]) <= controller.FAILURE_MESSAGE_LIMIT
                for item in overflow_evidence["cleanup_errors"]
            )
        )

    def test_failure_evidence_validator_rejects_malformed_schema_and_limits(self) -> None:
        valid = controller._failure_evidence(RuntimeError("primary"))
        mutations = (
            lambda value: value.update({"schema_version": True}),
            lambda value: value.update({"extra": True}),
            lambda value: value.update({"cleanup_errors": {}}),
            lambda value: value.update(
                {
                    "cleanup_errors": [
                        {"label": "x", "type": "OSError", "message": "x"}
                    ]
                    * (controller.FAILURE_CLEANUP_LIMIT + 1)
                }
            ),
            lambda value: value["primary"].update(
                {"message": "x" * (controller.FAILURE_MESSAGE_LIMIT + 1)}
            ),
        )
        for mutate in mutations:
            evidence = copy.deepcopy(valid)
            mutate(evidence)
            with self.assertRaises(RuntimeError):
                controller._validate_failure_evidence(evidence)

    def test_connection_clients_close_in_finally(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        jump = Client()
        target = Client()
        legacy = types.SimpleNamespace(
            EXPECTED_HOSTNAME="expected-host",
            load_remote_module=lambda: object(),
            local_preflight=lambda _module: {"shared": "/shared"},
            connect_target=lambda _module, _info: (jump, target),
            run=lambda _client, _command, **_kwargs: ("wrong-host\n", ""),
        )
        with mock.patch.object(controller, "load_legacy", return_value=legacy):
            with self.assertRaisesRegex(RuntimeError, "hostname mismatch"):
                controller.execute()
        self.assertTrue(target.closed)
        self.assertTrue(jump.closed)

    def test_jump_closes_even_when_target_close_raises(self) -> None:
        class Jump:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        class Target:
            def close(self) -> None:
                raise OSError("target close failed")

        jump = Jump()
        legacy = types.SimpleNamespace(
            EXPECTED_HOSTNAME="expected-host",
            load_remote_module=lambda: object(),
            local_preflight=lambda _module: {"shared": "/shared"},
            connect_target=lambda _module, _info: (jump, Target()),
            run=lambda _client, _command, **_kwargs: ("wrong-host\n", ""),
        )
        with mock.patch.object(controller, "load_legacy", return_value=legacy):
            with self.assertRaisesRegex(RuntimeError, "hostname mismatch") as caught:
                controller.execute()
        self.assertIsInstance(caught.exception.cleanup_error, OSError)
        self.assertTrue(jump.closed)


if __name__ == "__main__":
    unittest.main()
