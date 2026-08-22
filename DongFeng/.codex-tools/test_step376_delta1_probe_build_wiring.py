#!/usr/bin/env python3
"""Focused tests for the STEP376 delta1-only diagnostic build adapter."""

from __future__ import annotations

import copy
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import build_qrv2_diagnostic_probe as adapter


TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
OUTER_ZIP = PROJECT / (
    "cann-driving-cloud-ops-26.0.7-a3-cann8.3.rc1-torch2.7.1-py3.11-aarch64.zip"
)
EXPECTED_ADAPTER_SHA256 = "fc65fecc58cefb86f64b6e71d64a21e5e4bc1416b42f1cd696aff6bbdedc299e"


class Step376DiagnosticAdapterTests(unittest.TestCase):
    def _prepare(self, root: Path) -> tuple[Path, dict]:
        workdir = root / "work"
        return workdir, adapter.prepare_release(OUTER_ZIP, workdir, root)

    def _runtime_fixture(
        self, root: Path, workdir: Path, base: types.ModuleType
    ) -> tuple[Path, Path, Path, Path]:
        opc = root / "fake-opc.py"
        opc.write_text(
            """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

values = {}
for argument in sys.argv[1:]:
    if argument.startswith('--') and '=' in argument:
        key, value = argument[2:].split('=', 1)
        values[key] = value
output = Path(values['output'])
debug = Path(values['debug_dir'])
name = values['bin_filename']
(output / (name + '.o')).write_bytes(
    ('object-' + values['soc_version'] + '\\0' + name + '_0_mix_aic\\0' + name + '_0_mix_aiv\\0').encode('ascii')
)
(output / (name + '.json')).write_text(json.dumps({
    'binFileName': name,
    'kernelName': name,
    'kernelList': [{'kernelName': name + '_0'}],
    'supportInfo': {
        'opMode': 'dynamic',
        'simplifiedKeyMode': 0,
        'simplifiedKey': [
            'QrV2/d=0,p=0/0,2/0,2/0,2',
            'QrV2/d=1,p=0/0,2/0,2/0,2',
        ],
        'inputs': [{'shape': [-2], 'ori_shape': [-2]}],
        'outputs': [
            {'shape': [-2], 'ori_shape': [-2]},
            {'shape': [-2], 'ori_shape': [-2]},
        ],
    },
}), encoding='utf-8')
(debug / 'pythonpath.txt').write_text(os.environ['PYTHONPATH'], encoding='utf-8')
""",
            encoding="utf-8",
        )
        opc.chmod(0o755)

        ascend_opp = root / "ascend-opp"
        platform_adapter = (
            ascend_opp
            / "built-in/op_impl/ai_core/tbe/impl/util/platform_adapter.py"
        )
        platform_adapter.parent.mkdir(parents=True)
        platform_adapter.write_text("# test fixture\n", encoding="utf-8")
        cann_version = root / "version.cfg"
        cann_version.write_text("Version=8.3.RC1\nPackage=CANN\n", encoding="utf-8")

        installed_cloud = root / "installed" / "mx_driving_cloud"
        shutil.copytree(workdir / "wheel_original" / "mx_driving_cloud", installed_cloud)
        contract = {
            "schema_version": base.CONTAINER_CONTRACT_SCHEMA,
            "container_name": base.EXPECTED_CONTAINER,
            "inspect_container_id": "a" * 64,
            "inspect_hostname": socket.gethostname(),
            "opc": {
                "path": str(opc.resolve()),
                "sha256": base.sha256_file(opc),
            },
            "cann_version_files": [
                {
                    "path": str(cann_version.resolve()),
                    "sha256": base.sha256_file(cann_version),
                }
            ],
        }
        contract_path = root / "container-contract.json"
        contract_path.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
        return opc, ascend_opp, installed_cloud, contract_path

    def _successful_build(
        self, root: Path
    ) -> tuple[types.ModuleType, Path, dict]:
        workdir, _ = self._prepare(root)
        base = adapter._load_base()
        opc, ascend_opp, installed, contract = self._runtime_fixture(
            root, workdir, base
        )
        with mock.patch.dict(
            os.environ,
            {"ASCEND_OPP_PATH": str(ascend_opp), "PYTHONPATH": "poisoned"},
            clear=False,
        ):
            manifest = adapter.build_release(
                workdir, opc, contract, installed, root, _base=base
            )
        return base, workdir, manifest

    def test_adapter_loads_from_isolated_directory_without_v5_patcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "build_qrv2_diagnostic_probe.py",
                "build_qrv2_release.py",
                "step375_patch_qr_v2_vtv_direct_qa_legacy_probe_v6.py",
                "step338_patch_qr_v2_lifetime.py",
            ):
                shutil.copy2(TOOLS / name, root / name)
            self.assertFalse((root / "step372_patch_qr_v2_matmul_position_v5.py").exists())
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import build_qrv2_diagnostic_probe as a; "
                        "b=a._load_base(); "
                        "assert b.candidate_patcher is a.diagnostic_patcher; "
                        "assert b.BIN_NAME == a.BIN_NAME"
                    ),
                ],
                cwd=root,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_active_wiring_and_four_tool_hashes_are_exact(self) -> None:
        base = adapter._load_base()
        adapter._validate_active_wiring(base)
        self.assertIs(base.candidate_patcher, adapter.diagnostic_patcher)
        self.assertIs(base.build_candidate, adapter.diagnostic_patcher.build_candidate)
        self.assertEqual(base.BIN_NAME, "QrV2_vtv_direct_qa_legacy_probe_v6")
        self.assertEqual(
            base.EXPECTED_CANDIDATE_SHA256,
            "ef5db14e09170806acb7c5227fd619f3f5ffdc7d31f36e49058cc88987fce180",
        )
        hashes = adapter._tool_hashes()
        self.assertEqual(
            set(hashes),
            {
                "diagnostic_adapter_sha256",
                "base_builder_sha256",
                "step375_patcher_sha256",
                "v4_patcher_sha256",
            },
        )
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))
        self.assertEqual(hashes["diagnostic_adapter_sha256"], EXPECTED_ADAPTER_SHA256)
        self.assertEqual(
            hashes["base_builder_sha256"],
            "d6f2aea68574422ba6e28f220c5581adf8b25019494882dea156a80b1b513e90",
        )
        self.assertEqual(
            hashes["step375_patcher_sha256"],
            "98a655f89ac5efedd760067fdda595d9b5fe376b1e51fdc1b12d59c727711768",
        )
        self.assertEqual(
            hashes["v4_patcher_sha256"],
            "d4b260919440e6298c9caea5ca3a7c8bb7426a30ed84cc68d934250ee0ae9fe2",
        )

    def test_isolated_loader_restores_existing_v5_module_binding(self) -> None:
        sentinel = types.ModuleType(adapter.BASE_PATCHER_IMPORT)
        with mock.patch.dict(
            sys.modules, {adapter.BASE_PATCHER_IMPORT: sentinel}, clear=False
        ):
            base = adapter._load_base()
            self.assertIs(base.candidate_patcher, adapter.diagnostic_patcher)
            self.assertIs(sys.modules[adapter.BASE_PATCHER_IMPORT], sentinel)

    def test_loader_restores_module_binding_when_exec_module_fails(self) -> None:
        sentinel = types.ModuleType(adapter.BASE_PATCHER_IMPORT)
        with mock.patch.dict(
            sys.modules, {adapter.BASE_PATCHER_IMPORT: sentinel}, clear=False
        ), mock.patch(
            "importlib.machinery.SourceFileLoader.exec_module",
            side_effect=RuntimeError("exec failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "exec failed"):
                adapter._load_base()
            self.assertIs(sys.modules[adapter.BASE_PATCHER_IMPORT], sentinel)
            self.assertFalse(
                any(name.startswith("_step376_qrv2_release_base_") for name in sys.modules)
            )

    def test_active_wiring_rejects_v5_swap_bin_and_function_drift(self) -> None:
        cases = (
            (
                "v5_swap",
                lambda base: setattr(
                    base,
                    "candidate_patcher",
                    types.SimpleNamespace(
                        __name__="step372_patch_qr_v2_matmul_position_v5"
                    ),
                ),
                "not STEP375",
            ),
            (
                "bin",
                lambda base: setattr(base, "BIN_NAME", "QrV2_matmul_position_fix_v5"),
                "bin identity drift",
            ),
            (
                "function",
                lambda base: setattr(base, "build_candidate", lambda source: source),
                "function drift",
            ),
            (
                "release_api",
                lambda base: setattr(base, "package_release", lambda *_args: None),
                "release API was not poisoned",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label):
                base = adapter._load_base()
                mutate(base)
                with self.assertRaisesRegex(RuntimeError, message):
                    adapter._validate_active_wiring(base)

    def test_base_release_and_cli_entrypoints_are_poisoned(self) -> None:
        base = adapter._load_base()
        calls = (
            lambda: base.package_release(Path("unused")),
            lambda: base.parse_args(["package", "unused"]),
            lambda: base.main(["package", "unused"]),
            lambda: base.main(["all", "unused", "unused"]),
        )
        for call in calls:
            with self.assertRaisesRegex(RuntimeError, "APIs are forbidden"):
                call()

    def test_approved_root_rejects_escape_symlink_missing_and_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "approved"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "exactly approved_root/work"):
                adapter.prepare_release(OUTER_ZIP, root / "escape", root)

            (root / "work").mkdir()
            with self.assertRaisesRegex(FileExistsError, "reuse is forbidden"):
                adapter.prepare_release(OUTER_ZIP, root / "work", root)

            missing = parent / "missing"
            with self.assertRaises(FileNotFoundError):
                adapter.prepare_release(OUTER_ZIP, missing / "work", missing)

            target = parent / "target"
            target.mkdir()
            link = parent / "root-link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                adapter.prepare_release(OUTER_ZIP, link / "work", link)

    def test_prepare_exception_restores_base_monkey_patches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = adapter._load_base()
            original_write = base.write_json_new
            original_guard = base._guard_tools
            with mock.patch.object(
                base, "prepare_release", side_effect=RuntimeError("prepare failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "prepare failed"):
                    adapter.prepare_release(
                        OUTER_ZIP, root / "work", root, _base=base
                    )
            self.assertIs(base.write_json_new, original_write)
            self.assertIs(base._guard_tools, original_guard)

    def test_prepare_writes_double_flags_and_forbidden_package_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = adapter._load_base()
            original_write = base.write_json_new
            original_guard = base._guard_tools
            workdir = root / "work"
            manifest = adapter.prepare_release(
                OUTER_ZIP, workdir, root, _base=base
            )
            self.assertIs(base.write_json_new, original_write)
            self.assertIs(base._guard_tools, original_guard)
            persisted = json.loads(
                (workdir / adapter.MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest, persisted)
            self.assertEqual(persisted["status"], "prepared")
            for layer in ("policy", "candidate"):
                for key, value in adapter._diagnostic_flags().items():
                    self.assertEqual(persisted[layer][key], value)
            self.assertEqual(persisted["candidate"]["identity"], adapter.BIN_NAME)
            self.assertEqual(persisted["candidate"]["bin_name"], adapter.BIN_NAME)
            self.assertEqual(
                persisted["package"], {"status": adapter.FORBIDDEN_PACKAGE_STATUS}
            )
            self.assertEqual(persisted["tools"], adapter._tool_hashes())

    def test_build_atomically_seals_without_persisting_built(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workdir, _ = self._prepare(root)
            base = adapter._load_base()
            opc, ascend_opp, installed, contract = self._runtime_fixture(
                root, workdir, base
            )
            writes = []
            real_atomic = base.write_json_atomic

            def recording_atomic(root_path, path, value):
                writes.append(copy.deepcopy(value))
                real_atomic(root_path, path, value)

            base.write_json_atomic = recording_atomic
            original_guard = base._guard_tools
            with mock.patch.dict(
                os.environ,
                {"ASCEND_OPP_PATH": str(ascend_opp), "PYTHONPATH": "poisoned"},
                clear=False,
            ):
                manifest = adapter.build_release(
                    workdir, opc, contract, installed, root, _base=base
                )

            self.assertIs(base.write_json_atomic, recording_atomic)
            self.assertIs(base._guard_tools, original_guard)

            self.assertEqual(len(writes), 1)
            self.assertEqual(writes[0]["status"], adapter.DIAGNOSTIC_BUILT_STATUS)
            self.assertNotEqual(writes[0]["status"], "built")
            self.assertEqual(manifest, writes[0])
            self.assertEqual(
                manifest["package"], {"status": adapter.FORBIDDEN_PACKAGE_STATUS}
            )
            persisted = json.loads(
                (workdir / adapter.MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(persisted, manifest)
            self.assertFalse((workdir / "release").exists())
            for soc_key in base.SOCS:
                artifact = manifest["artifacts"][soc_key]
                for key in ("object_size", "json_size"):
                    self.assertGreater(artifact[key], 0)
                self.assertGreaterEqual(artifact["opc_log_size"], 0)
                for key in ("object_path", "json_path", "opc_log_path"):
                    self.assertTrue(Path(artifact[key]).is_file())

            canonical = manifest["artifacts"][base.CANONICAL_SOC_KEY]
            tamper_cases = (
                ("object", Path(canonical["object_path"])),
                ("json", Path(canonical["json_path"])),
                ("opc_log", Path(canonical["opc_log_path"])),
            )
            for label, path in tamper_cases:
                with self.subTest(tamper=label):
                    original = path.read_bytes()
                    path.write_bytes(original + b"\n")
                    with self.assertRaisesRegex(RuntimeError, "closure"):
                        adapter._validate_manifest(
                            base,
                            workdir,
                            manifest,
                            expected_status=adapter.DIAGNOSTIC_BUILT_STATUS,
                        )
                    self.assertEqual(
                        json.loads(
                            (workdir / adapter.MANIFEST_NAME).read_text(
                                encoding="utf-8"
                            )
                        )["status"],
                        adapter.DIAGNOSTIC_BUILT_STATUS,
                    )
                    path.write_bytes(original)
                    adapter._validate_manifest(
                        base,
                        workdir,
                        manifest,
                        expected_status=adapter.DIAGNOSTIC_BUILT_STATUS,
                    )

    def test_build_revalidates_manifest_before_running(self) -> None:
        mutations = (
            (
                "flag",
                lambda manifest: manifest["candidate"].update(
                    {"diagnostic_only": False}
                ),
                "candidate.diagnostic_only",
            ),
            (
                "tool_sha",
                lambda manifest: manifest["tools"].update(
                    {"step375_patcher_sha256": "0" * 64}
                ),
                "tool SHA guard",
            ),
            (
                "candidate_sha",
                lambda manifest: manifest["candidate"].update(
                    {"source_sha256": "0" * 64}
                ),
                "candidate SHA manifest drift",
            ),
        )
        for label, mutate, expected in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                workdir, manifest = self._prepare(root)
                mutate(manifest)
                (workdir / adapter.MANIFEST_NAME).write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, expected):
                    adapter.build_release(
                        workdir,
                        root / "missing-opc",
                        root / "missing-contract",
                        root / "missing-install",
                        root,
                    )

    def test_alias_closure_rejects_individually_closed_soc_byte_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, workdir, manifest = self._successful_build(root)
            divergent = copy.deepcopy(manifest)
            alias = divergent["artifacts"][base.ALIAS_SOC_KEY]
            alias_object = Path(alias["object_path"])
            original = alias_object.read_bytes()
            alias_object.write_bytes(original + b"alias-divergence")
            alias["object_size"] = alias_object.stat().st_size
            alias["object_sha256"] = base.sha256_file(alias_object)

            with self.assertRaisesRegex(
                RuntimeError, "SoC alias object SHA closure failed"
            ):
                adapter._validate_built_artifact_closure(
                    base, workdir, divergent, enrich=False
                )

    def test_preseal_race_tamper_is_rejected_and_manifest_stays_prepared(self) -> None:
        for artifact_kind in ("object", "json"):
            with self.subTest(artifact=artifact_kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                workdir, _ = self._prepare(root)
                base = adapter._load_base()
                opc, ascend_opp, installed, contract = self._runtime_fixture(
                    root, workdir, base
                )
                real_guard = adapter._guard_tools
                reached_preseal = False

                def tamper_at_preseal(manifest):
                    nonlocal reached_preseal
                    real_guard(manifest)
                    if manifest.get("status") != adapter.DIAGNOSTIC_BUILT_STATUS:
                        return
                    reached_preseal = True
                    canonical = manifest["artifacts"][base.CANONICAL_SOC_KEY]
                    path = Path(canonical[f"{artifact_kind}_path"])
                    suffix = b" " if artifact_kind == "json" else b"preseal-race"
                    path.write_bytes(path.read_bytes() + suffix)

                with mock.patch.dict(
                    os.environ,
                    {"ASCEND_OPP_PATH": str(ascend_opp), "PYTHONPATH": "poisoned"},
                    clear=False,
                ), mock.patch.object(
                    adapter, "_guard_tools", side_effect=tamper_at_preseal
                ):
                    with self.assertRaisesRegex(RuntimeError, "closure"):
                        adapter.build_release(
                            workdir, opc, contract, installed, root, _base=base
                        )

                self.assertTrue(reached_preseal)
                persisted = json.loads(
                    (workdir / adapter.MANIFEST_NAME).read_text(encoding="utf-8")
                )
                self.assertEqual(persisted["status"], "prepared")
                self.assertNotEqual(
                    persisted["status"], adapter.DIAGNOSTIC_BUILT_STATUS
                )

    def test_interrupted_build_leaves_prepared_not_built(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workdir, _ = self._prepare(root)
            base = adapter._load_base()
            opc, ascend_opp, installed, contract = self._runtime_fixture(
                root, workdir, base
            )
            original_atomic = base.write_json_atomic
            original_guard = base._guard_tools
            with mock.patch.dict(
                os.environ,
                {"ASCEND_OPP_PATH": str(ascend_opp), "PYTHONPATH": "poisoned"},
                clear=False,
            ), mock.patch.object(
                base.subprocess, "run", side_effect=InterruptedError("simulated")
            ):
                with self.assertRaises(InterruptedError):
                    adapter.build_release(
                        workdir, opc, contract, installed, root, _base=base
                    )
            self.assertIs(base.write_json_atomic, original_atomic)
            self.assertIs(base._guard_tools, original_guard)
            persisted = json.loads(
                (workdir / adapter.MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["status"], "prepared")
            self.assertNotIn('"status": "built"', json.dumps(persisted, sort_keys=True))

    def test_package_entrypoints_are_permanently_disabled(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "permanently forbidden"):
            adapter.package_release(Path("unused"))
        for command in ("package", "all"):
            with self.subTest(command=command), mock.patch(
                "sys.stderr", new_callable=io.StringIO
            ):
                with self.assertRaises(SystemExit):
                    adapter.parse_args([command])
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                adapter.parse_args(
                    [
                        "prepare",
                        "--outer-zip",
                        str(OUTER_ZIP),
                        "--workdir",
                        "work",
                    ]
                )

    def test_release_directory_wheel_zip_and_symlinks_are_rejected(self) -> None:
        cases = ("release", "new.whl", "new.zip", "wheel-link.whl")
        for label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                allowed = root / "original.whl"
                allowed.write_bytes(b"original")
                manifest = {"paths": {"extracted_wheel": str(allowed)}}
                if label == "release":
                    (root / "release").mkdir()
                elif label == "wheel-link.whl":
                    (root / label).symlink_to(allowed)
                else:
                    (root / label).write_bytes(b"forbidden")
                with self.assertRaisesRegex(RuntimeError, "forbidden|release directory"):
                    adapter._assert_no_release_outputs(root, manifest)

    def test_only_original_extracted_wheel_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.whl"
            original.write_bytes(b"original")
            adapter._assert_no_release_outputs(
                root, {"paths": {"extracted_wheel": str(original)}}
            )


if __name__ == "__main__":
    unittest.main()
