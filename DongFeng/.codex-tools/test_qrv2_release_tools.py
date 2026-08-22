#!/usr/bin/env python3
"""Static/offline tests for the QrV2 release patch and packaging tools."""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import build_qrv2_release as release  # noqa: E402
import step338_patch_qr_v2_lifetime as patcher  # noqa: E402
import step358_qrv2_release_math_worker as worker  # noqa: E402
import step358_prepare_release_shadow as shadow  # noqa: E402
import step372_patch_qr_v2_matmul_position_v5 as v5_patcher  # noqa: E402


class QrV2PatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (TOOLS / "qr_v2.cpp").read_bytes()

    def test_candidate_matches_audited_hash_and_structure(self) -> None:
        candidate = patcher.build_candidate(self.source)
        report = patcher.verify_candidate_structure(self.source, candidate)
        self.assertEqual(report["source_sha256"], patcher.EXPECTED_SOURCE_SHA256)
        self.assertEqual(report["candidate_sha256"], patcher.EXPECTED_CANDIDATE_SHA256)
        self.assertEqual(report["source_free_tensor_calls"], 13)
        self.assertEqual(report["candidate_free_tensor_calls"], 10)
        self.assertTrue(report["process_release_outside_core0"])
        self.assertEqual(report["alpha_buffer_bytes"], 64)
        self.assertEqual(report["alpha_buffer_fp32_elements"], 16)
        self.assertEqual(report["alpha_duplicate_calls"], 2)
        self.assertEqual(
            report["v1_equivalent_sha256"],
            patcher.EXPECTED_V1_CANDIDATE_SHA256,
        )
        self.assertTrue(report["v2_delta_alpha_only"])
        self.assertEqual(
            report["v2_candidate_sha256"],
            patcher.EXPECTED_V2_CANDIDATE_SHA256,
        )
        self.assertTrue(report["v3_delta_mte3_mte2_only"])
        self.assertEqual(report["v3_mte3_mte2_sequence_count"], 1)
        self.assertTrue(report["v3_mte3_mte2_after_workspace_copy"])
        self.assertTrue(report["v3_mte3_mte2_before_matmul_a"])
        self.assertEqual(
            report["v3_candidate_sha256"],
            patcher.EXPECTED_V3_CANDIDATE_SHA256,
        )
        self.assertTrue(report["v4_delta_sync_and_ownership_only"])
        self.assertTrue(report["v4_per_core_scratch_offset"])
        self.assertEqual(report["v4_update_a_mte3_mte2_sequence_count"], 1)
        self.assertEqual(report["v4_sync_all_calls_added"], 1)
        self.assertTrue(report["process_sync_after_q_writeback"])
        self.assertTrue(report["process_sync_before_release"])
        self.assertTrue(report["process_sync_outside_core0"])
        self.assertTrue(report["base_tiling_slot_references_unchanged"])

        text = candidate.decode("utf-8")
        self.assertEqual(text.count("pipe->InitBuffer(alphaBuf, ALPHA_BUF_BYTES);"), 1)
        self.assertEqual(
            text.count(
                "Duplicate(alphaLocal, static_cast<DTYPE_A>(0), "
                "ALPHA_BUF_FP32_ELEMENTS);"
            ),
            2,
        )
        self.assertNotIn(
            "Duplicate(alphaLocal, static_cast<DTYPE_A>(0), 2 * UB_ALIGN_SIZE);",
            text,
        )
        self.assertEqual(text.count(patcher.V3_MTE3_MTE2_SEQUENCE), 1)
        self.assertEqual(text.count(patcher.V4_UPDATE_A_MTE3_MTE2_SEQUENCE), 1)
        self.assertEqual(text.count("uint64_t calcQScratchOffset = "), 1)
        self.assertEqual(text.count("workspaceInGm[calcQScratchOffset]"), 2)
        v3_equivalent = text.replace(
            patcher.V4_CALC_Q_SCRATCH_BLOCK,
            patcher.V3_CALC_Q_SCRATCH_BLOCK,
            1,
        )
        v3_equivalent = v3_equivalent.replace(
            patcher.V4_UPDATE_A_MTE3_MTE2_SEQUENCE,
            "",
            1,
        )
        v3_equivalent = v3_equivalent.replace(patcher.V4_PROCESS_SYNC, "", 1)
        self.assertEqual(
            patcher.sha256_bytes(v3_equivalent.encode("utf-8")),
            patcher.EXPECTED_V3_CANDIDATE_SHA256,
        )

    def test_structure_gate_rejects_extra_or_misplaced_v3_event_sequence(self) -> None:
        candidate = patcher.build_candidate(self.source)
        extra = candidate.replace(
            patcher.V3_MTE3_MTE2_SEQUENCE.encode("utf-8"),
            (patcher.V3_MTE3_MTE2_SEQUENCE * 2).encode("utf-8"),
            1,
        )
        with mock.patch.object(
            patcher,
            "EXPECTED_CANDIDATE_SHA256",
            patcher.sha256_bytes(extra),
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly one CalcQ MTE3_MTE2"):
                patcher.verify_candidate_structure(self.source, extra)

        text = candidate.decode("utf-8")
        without = text.replace(patcher.V3_MTE3_MTE2_SEQUENCE, "", 1)
        update_copy = "        DataCopy(workspaceInGm[offsetW], aLocal, this->blockElement);\n"
        self.assertEqual(without.count(update_copy), 1)
        misplaced = without.replace(
            update_copy,
            update_copy + patcher.V3_MTE3_MTE2_SEQUENCE,
            1,
        ).encode("utf-8")
        with mock.patch.object(
            patcher,
            "EXPECTED_CANDIDATE_SHA256",
            patcher.sha256_bytes(misplaced),
        ):
            with self.assertRaisesRegex(RuntimeError, "inside CalcQForLARFB"):
                patcher.verify_candidate_structure(self.source, misplaced)

    def test_structure_gate_rejects_old_alpha_duplicate_element_count(self) -> None:
        candidate = patcher.build_candidate(self.source)
        mutated = candidate.replace(
            b"ALPHA_BUF_FP32_ELEMENTS);",
            b"2 * UB_ALIGN_SIZE);",
            1,
        )
        with mock.patch.object(
            patcher,
            "EXPECTED_CANDIDATE_SHA256",
            patcher.sha256_bytes(mutated),
        ):
            with self.assertRaisesRegex(RuntimeError, "alphaBuf Duplicate"):
                patcher.verify_candidate_structure(self.source, mutated)

    def test_structure_gate_rejects_alpha_count_expression_inflation(self) -> None:
        candidate = patcher.build_candidate(self.source)
        mutated = candidate.replace(
            b"ALPHA_BUF_FP32_ELEMENTS);",
            b"ALPHA_BUF_FP32_ELEMENTS + 48);",
            1,
        )
        with mock.patch.object(
            patcher,
            "EXPECTED_CANDIDATE_SHA256",
            patcher.sha256_bytes(mutated),
        ):
            with self.assertRaisesRegex(RuntimeError, "alphaBuf Duplicate"):
                patcher.verify_candidate_structure(self.source, mutated)

    def test_structure_gate_rejects_extra_sync_or_base_tiling_slot(self) -> None:
        candidate = patcher.build_candidate(self.source)
        mutations = (
            (b"            SyncAll();\n", b"            SyncAll();\n            SyncAll();\n", "SyncAll"),
            (
                b"        for (auto i = 1; i <= blockp; ++i) {\n",
                b"        baseTilingInfos[0] = InitBaseTiling(0, 0);\n"
                b"        for (auto i = 1; i <= blockp; ++i) {\n",
                "baseTilingInfos",
            ),
        )
        for old, new, error in mutations:
            with self.subTest(error=error):
                mutated = candidate.replace(old, new, 1)
                self.assertNotEqual(mutated, candidate)
                with mock.patch.object(
                    patcher,
                    "EXPECTED_CANDIDATE_SHA256",
                    patcher.sha256_bytes(mutated),
                ):
                    with self.assertRaisesRegex(RuntimeError, error):
                        patcher.verify_candidate_structure(self.source, mutated)

    def test_structure_gate_rejects_shared_calc_q_scratch(self) -> None:
        candidate = patcher.build_candidate(self.source)
        mutated = candidate.replace(
            b"workspaceInGm[calcQScratchOffset]",
            b"workspaceInGm[this->m * this->blockSize]",
        )
        self.assertEqual(mutated.count(b"workspaceInGm[this->m * this->blockSize]"), 2)
        with mock.patch.object(
            patcher,
            "EXPECTED_CANDIDATE_SHA256",
            patcher.sha256_bytes(mutated),
        ):
            with self.assertRaisesRegex(RuntimeError, "shared scratch"):
                patcher.verify_candidate_structure(self.source, mutated)

    def test_structure_gate_rejects_missing_update_a_event(self) -> None:
        candidate = patcher.build_candidate(self.source)
        mutated = candidate.replace(
            patcher.V4_UPDATE_A_MTE3_MTE2_SEQUENCE.encode("utf-8"),
            b"",
            1,
        )
        with mock.patch.object(
            patcher,
            "EXPECTED_CANDIDATE_SHA256",
            patcher.sha256_bytes(mutated),
        ):
            with self.assertRaisesRegex(RuntimeError, "independent UpdateA"):
                patcher.verify_candidate_structure(self.source, mutated)

    def test_structure_gate_rejects_process_sync_inside_core0(self) -> None:
        candidate = patcher.build_candidate(self.source)
        text = candidate.decode("utf-8").replace(patcher.V4_PROCESS_SYNC, "", 1)
        target = "                WaitFlag<HardEvent::MTE3_V>(0);\n            }\n"
        self.assertEqual(text.count(target), 1)
        mutated = text.replace(
            target,
            "                WaitFlag<HardEvent::MTE3_V>(0);\n"
            "                SyncAll();\n"
            "            }\n",
            1,
        ).encode("utf-8")
        with mock.patch.object(
            patcher,
            "EXPECTED_CANDIDATE_SHA256",
            patcher.sha256_bytes(mutated),
        ):
            with self.assertRaisesRegex(RuntimeError, "outside the core0"):
                patcher.verify_candidate_structure(self.source, mutated)

    def test_structure_gate_rejects_process_sync_after_release(self) -> None:
        candidate = patcher.build_candidate(self.source)
        text = candidate.decode("utf-8").replace(patcher.V4_PROCESS_SYNC, "", 1)
        target = (
            "            tTQue.FreeTensor<DTYPE_A>(tLocal);\n"
            "            vTQue.FreeTensor<DTYPE_A>(vLocal);\n"
            "            aTQue.FreeTensor<DTYPE_A>(aLocal);\n"
        )
        self.assertEqual(text.count(target), 1)
        mutated = text.replace(target, target + patcher.V4_PROCESS_SYNC, 1).encode("utf-8")
        with mock.patch.object(
            patcher,
            "EXPECTED_CANDIDATE_SHA256",
            patcher.sha256_bytes(mutated),
        ):
            with self.assertRaisesRegex(RuntimeError, "SyncAll -> release"):
                patcher.verify_candidate_structure(self.source, mutated)

    def test_rejects_wrong_source(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "source SHA-256 mismatch"):
            patcher.build_candidate(self.source + b"\n")

    def test_rejects_symlink_source_argument(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.cpp"
            source.write_bytes(self.source)
            link = root / "source-link.cpp"
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                patcher.main([str(link), "--check"])

    def test_never_overwrites_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.cpp"
            output = root / "candidate.cpp"
            source.write_bytes(self.source)
            output.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                patcher.main([str(source), str(output)])
            self.assertEqual(output.read_bytes(), b"keep")


class QrV2ReleaseBuilderTests(unittest.TestCase):
    @staticmethod
    def _valid_artifact_metadata() -> dict:
        name = release.BIN_NAME
        unknown_rank = {"shape": [-2], "ori_shape": [-2]}
        return {
            "binFileName": name,
            "kernelName": name,
            "supportInfo": {
                "opMode": "dynamic",
                "simplifiedKeyMode": 0,
                "simplifiedKey": list(release.QRV2_SIMPLIFIED_KEYS),
                "inputs": [dict(unknown_rank)],
                "outputs": [dict(unknown_rank), dict(unknown_rank)],
            },
        }

    @staticmethod
    def _shadow_package_fixture(root: Path) -> Path:
        package = root / "mx_driving_cloud"
        source = package / shadow.CANDIDATE_SOURCE_REL
        source.parent.mkdir(parents=True)
        source.write_bytes(
            v5_patcher.build_candidate((TOOLS / "qr_v2.cpp").read_bytes())
        )
        kernel_root = package / "packages/vendors/customize/op_impl/ai_core/tbe/kernel"
        for soc in shadow.SOCS:
            config_root = kernel_root / "config" / soc
            kernel_dir = kernel_root / soc / "qr_v2"
            config_root.mkdir(parents=True)
            kernel_dir.mkdir(parents=True)
            (config_root / "qr_v2.json").write_text(
                json.dumps(
                    {
                        "binList": [
                            {
                                "binInfo": {
                                    "jsonFilePath": (
                                        f"{soc}/qr_v2/{shadow.EXPECTED_KERNEL}.json"
                                    )
                                }
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (config_root / shadow.BINARY_INFO_CONFIG_NAME).write_text(
                json.dumps({"QrV2": shadow._expected_binary_info(soc)}),
                encoding="utf-8",
            )
            metadata = QrV2ReleaseBuilderTests._valid_artifact_metadata()
            (kernel_dir / f"{shadow.EXPECTED_KERNEL}.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            (kernel_dir / f"{shadow.EXPECTED_KERNEL}.o").write_bytes(
                (
                    f"\0{shadow.EXPECTED_AIC}\0{shadow.EXPECTED_AIV}\0"
                ).encode("ascii")
            )
        return package

    def test_v5_identity_and_reverse_v4_contract_are_unique(self) -> None:
        source = (TOOLS / "qr_v2.cpp").read_bytes()
        candidate = v5_patcher.build_candidate(source)
        report = v5_patcher.verify_candidate_structure(source, candidate)
        self.assertEqual(release.BIN_NAME, v5_patcher.CANDIDATE_IDENTITY)
        self.assertEqual(shadow.EXPECTED_KERNEL, release.BIN_NAME)
        self.assertEqual(worker.CANDIDATE_AIC, f"{release.BIN_NAME}_0_mix_aic")
        self.assertEqual(worker.CANDIDATE_AIV, f"{release.BIN_NAME}_0_mix_aiv")
        self.assertEqual(release.EXPECTED_CANDIDATE_SHA256, report["candidate_sha256"])
        self.assertEqual(
            report["reverse_v4_sha256"], v5_patcher.EXPECTED_V4_CANDIDATE_SHA256
        )
        self.assertNotEqual(release.BIN_NAME, "QrV2_lifetime_alpha_sync_fix_v4")

    def test_shadow_accepts_exact_dynamic_v5_registration_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = self._shadow_package_fixture(Path(directory))
            report = shadow._candidate_artifacts(package)
            self.assertEqual(set(report), set(shadow.SOCS))
            for soc in shadow.SOCS:
                self.assertIn("binary_info_config_sha256", report[soc])

    def test_shadow_rejects_non_v5_candidate_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = self._shadow_package_fixture(Path(directory))
            source = package / shadow.CANDIDATE_SOURCE_REL
            source.write_bytes(
                patcher.build_candidate((TOOLS / "qr_v2.cpp").read_bytes())
            )
            with self.assertRaisesRegex(RuntimeError, "candidate source SHA-256 mismatch"):
                shadow._candidate_artifacts(package)

    def test_shadow_rejects_static_fixed_or_missing_registration_contracts(self) -> None:
        mutations = (
            (
                "static_op_mode",
                lambda metadata: metadata["supportInfo"].update({"opMode": "static"}),
                None,
                "opMode must be dynamic",
            ),
            (
                "fixed_shape",
                lambda metadata: metadata["supportInfo"]["inputs"][0].update(
                    {"shape": [192, 192]}
                ),
                None,
                "shape must be dynamic unknown-rank",
            ),
            (
                "wrong_simplified_key_mode",
                lambda metadata: metadata["supportInfo"].update(
                    {"simplifiedKeyMode": 1}
                ),
                None,
                "simplifiedKeyMode mismatch",
            ),
            (
                "missing_binary_info",
                lambda metadata: None,
                shadow.BINARY_INFO_CONFIG_NAME,
                "artifact missing",
            ),
        )
        for label, mutate, missing_name, expected_error in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                package = self._shadow_package_fixture(Path(directory))
                kernel_root = (
                    package / "packages/vendors/customize/op_impl/ai_core/tbe/kernel"
                )
                if missing_name is not None:
                    (kernel_root / "config" / shadow.SOCS[0] / missing_name).unlink()
                else:
                    metadata_path = (
                        kernel_root
                        / shadow.SOCS[0]
                        / "qr_v2"
                        / f"{shadow.EXPECTED_KERNEL}.json"
                    )
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    mutate(metadata)
                    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, expected_error):
                    shadow._candidate_artifacts(package)

    def test_shadow_rejects_wrong_binary_info_candidate_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = self._shadow_package_fixture(Path(directory))
            binary_info = (
                package
                / "packages/vendors/customize/op_impl/ai_core/tbe/kernel/config"
                / shadow.SOCS[0]
                / shadow.BINARY_INFO_CONFIG_NAME
            )
            payload = json.loads(binary_info.read_text(encoding="utf-8"))
            payload["QrV2"]["binaryList"][1]["binPath"] = (
                f"{shadow.SOCS[0]}/qr_v2/QrV2_stale_v1.o"
            )
            binary_info.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "binary-info contract mismatch"):
                shadow._candidate_artifacts(package)

    def test_shadow_rejects_config_and_kernel_list_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = self._shadow_package_fixture(Path(directory))
            kernel_root = (
                package / "packages/vendors/customize/op_impl/ai_core/tbe/kernel"
            )
            config = kernel_root / "config" / shadow.SOCS[0] / "qr_v2.json"
            payload = json.loads(config.read_text(encoding="utf-8"))
            payload["binList"][0]["binInfo"]["staleField"] = True
            config.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "config binInfo schema mismatch"):
                shadow._candidate_artifacts(package)

        with tempfile.TemporaryDirectory() as directory:
            package = self._shadow_package_fixture(Path(directory))
            kernel_root = (
                package / "packages/vendors/customize/op_impl/ai_core/tbe/kernel"
            )
            metadata_path = (
                kernel_root
                / shadow.SOCS[0]
                / "qr_v2"
                / f"{shadow.EXPECTED_KERNEL}.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["kernelList"] = [{"kernelName": shadow.EXPECTED_KERNEL}]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "kernelList identity mismatch"):
                shadow._candidate_artifacts(package)

    def test_dynamic_descriptor_uses_unknown_rank_for_input_and_outputs(self) -> None:
        descriptor = release.input_descriptor()
        self.assertEqual(set(descriptor), {"op_type", "op_list"})
        self.assertEqual(descriptor["op_type"], "QrV2")
        self.assertEqual(len(descriptor["op_list"]), 1)
        operation = descriptor["op_list"][0]
        self.assertEqual(len(operation["inputs"]), 1)
        self.assertEqual(len(operation["outputs"]), 2)
        self.assertEqual(operation["attrs"], [])
        self.assertEqual(operation["bin_filename"], release.BIN_NAME)
        tensors = [*operation["inputs"], *operation["outputs"]]
        expected_tensor = {
            "shape": [-2],
            "format": "ND",
            "dtype": "float32",
        }
        self.assertEqual(tensors, [expected_tensor, expected_tensor, expected_tensor])
        self.assertTrue(all("range" not in tensor for tensor in tensors))
        self.assertTrue(all("ori_shape" not in tensor for tensor in tensors))
        self.assertTrue(all("ori_format" not in tensor for tensor in tensors))
        self.assertNotIn("192", json.dumps(descriptor))

        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            for soc_key in release.SOCS:
                command = release.opc_command(workdir, soc_key, "/opt/opc")
                self.assertEqual(command.count("--op_mode=dynamic"), 1)
                self.assertEqual(command.count("--simplified_key_mode=0"), 1)
                self.assertFalse(any(argument.startswith("--tiling_key") for argument in command))

    def test_artifact_accepts_cann83_top_level_schema_with_exact_binary_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            output = build / "output"
            output.mkdir()
            name = release.BIN_NAME
            (output / f"{name}.o").write_bytes(
                f"{name}_0_mix_aic\0{name}_0_mix_aiv\0".encode("ascii")
            )
            metadata = self._valid_artifact_metadata()
            for tensor in [
                *metadata["supportInfo"]["inputs"],
                *metadata["supportInfo"]["outputs"],
            ]:
                del tensor["ori_shape"]
            (output / f"{name}.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            _object, _metadata, parsed = release._validate_artifacts(build)
            self.assertEqual(
                parsed["_audited_concrete_entries"],
                [f"{name}_0_mix_aic", f"{name}_0_mix_aiv"],
            )

    def test_artifact_rejects_wrong_or_extra_concrete_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            output = build / "output"
            output.mkdir()
            name = release.BIN_NAME
            (output / f"{name}.o").write_bytes(
                f"{name}_0_mix_aic\0QrV2_unexpected_0_mix_aiv\0".encode("ascii")
            )
            (output / f"{name}.json").write_text(
                json.dumps(self._valid_artifact_metadata()), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "concrete-entry contract"):
                release._validate_artifacts(build)

    def test_artifact_rejects_prefixed_or_suffixed_symbol_tokens(self) -> None:
        for bad_aic, bad_aiv in (
            (
                f"X{release.BIN_NAME}_0_mix_aic",
                f"{release.BIN_NAME}_0_mix_aiv",
            ),
            (
                f"{release.BIN_NAME}_0_mix_aic",
                f"{release.BIN_NAME}_0_mix_aiv_extra",
            ),
        ):
            with self.subTest(bad_aic=bad_aic, bad_aiv=bad_aiv):
                with tempfile.TemporaryDirectory() as directory:
                    build = Path(directory)
                    output = build / "output"
                    output.mkdir()
                    name = release.BIN_NAME
                    (output / f"{name}.o").write_bytes(
                        f"\0{bad_aic}\0{bad_aiv}\0".encode("ascii")
                    )
                    (output / f"{name}.json").write_text(
                        json.dumps(self._valid_artifact_metadata()),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(RuntimeError, "concrete-entry contract"):
                        release._validate_artifacts(build)

    def test_artifact_rejects_invalid_dynamic_metadata_contracts(self) -> None:
        mutations = (
            (
                "fixed_shape",
                lambda metadata: metadata["supportInfo"]["inputs"][0].update(
                    {"shape": [192, 192], "ori_shape": [192, 192]}
                ),
                "must be unknown-rank",
            ),
            (
                "missing_simplified_key",
                lambda metadata: metadata["supportInfo"].pop("simplifiedKey"),
                "simplifiedKey contract",
            ),
            (
                "wrong_present_ori_shape",
                lambda metadata: metadata["supportInfo"]["outputs"][1].update(
                    {"ori_shape": [192, 192]}
                ),
                "ori_shape must be unknown-rank when present",
            ),
            (
                "wrong_kernel_list_name",
                lambda metadata: metadata.update(
                    {"kernelList": [{"kernelName": release.BIN_NAME}]}
                ),
                "kernelList name contract",
            ),
            (
                "duplicate_kernel_list_entry",
                lambda metadata: metadata.update(
                    {
                        "kernelList": [
                            {"kernelName": f"{release.BIN_NAME}_0"},
                            {"kernelName": f"{release.BIN_NAME}_0"},
                        ]
                    }
                ),
                "kernelList schema contract",
            ),
        )
        for label, mutate, expected_error in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                build = Path(directory)
                output = build / "output"
                output.mkdir()
                name = release.BIN_NAME
                (output / f"{name}.o").write_bytes(
                    f"{name}_0_mix_aic\0{name}_0_mix_aiv\0".encode("ascii")
                )
                metadata = self._valid_artifact_metadata()
                mutate(metadata)
                (output / f"{name}.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, expected_error):
                    release._validate_artifacts(build)

    outer_zip = PROJECT / (
        "cann-driving-cloud-ops-26.0.7-a3-cann8.3.rc1-torch2.7.1-py3.11-aarch64.zip"
    )

    def _runtime_fixture(
        self, root: Path, workdir: Path, *, contract_overrides: dict | None = None
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
mutate = os.environ.get('QRV2_TEST_MUTATE')
if mutate:
    Path(mutate).write_text('mutated-by-fake-opc\\n', encoding='utf-8')
""",
            encoding="utf-8",
        )
        opc.chmod(0o755)

        ascend_opp = root / "ascend-opp"
        opp_tbe = ascend_opp / "built-in/op_impl/ai_core/tbe"
        platform_adapter = opp_tbe / "impl/util/platform_adapter.py"
        platform_adapter.parent.mkdir(parents=True)
        platform_adapter.write_text("# test fixture\n", encoding="utf-8")
        cann_version = root / "version.cfg"
        cann_version.write_text("Version=8.3.RC1\nPackage=CANN\n", encoding="utf-8")

        installed_cloud = root / "installed" / "mx_driving_cloud"
        shutil.copytree(workdir / "wheel_original" / "mx_driving_cloud", installed_cloud)
        contract = {
            "schema_version": release.CONTAINER_CONTRACT_SCHEMA,
            "container_name": release.EXPECTED_CONTAINER,
            "inspect_container_id": "a" * 64,
            "inspect_hostname": socket.gethostname(),
            "opc": {
                "path": str(opc.resolve()),
                "sha256": release.sha256_file(opc),
            },
            "cann_version_files": [
                {
                    "path": str(cann_version.resolve()),
                    "sha256": release.sha256_file(cann_version),
                }
            ],
        }
        if contract_overrides:
            contract.update(contract_overrides)
        contract_path = root / "container-contract.json"
        contract_path.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
        return opc, ascend_opp, installed_cloud, contract_path

    def _fake_build(self, root: Path, workdir: Path) -> dict:
        opc, ascend_opp, installed_cloud, contract = self._runtime_fixture(root, workdir)
        with mock.patch.dict(
            os.environ,
            {"ASCEND_OPP_PATH": str(ascend_opp), "PYTHONPATH": "poisoned-parent-path"},
            clear=False,
        ):
            return release.build_release(workdir, opc, contract, installed_cloud)

    def test_realpath_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "isolated"
            outside = parent / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "escape").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "realpath escapes isolated root"):
                release._assert_output_parent(root, root / "escape" / "artifact.o")

    def test_outer_archive_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outer_link = root / "vendor.zip"
            outer_link.symlink_to(self.outer_zip)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                release.prepare_release(outer_link, root / "release-work")

    def test_container_contract_rejects_name_id_hostname_and_opc_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workdir = root / "release-work"
            release.prepare_release(self.outer_zip, workdir)
            opc, _, _, contract_path = self._runtime_fixture(root, workdir)
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["container_name"] = "mapqr"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exact-name gate failed"):
                release._validate_container_contract(contract_path, opc)
            contract["container_name"] = release.EXPECTED_CONTAINER
            contract["inspect_container_id"] = "short-id"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "64 lowercase hex"):
                release._validate_container_contract(contract_path, opc)
            contract["inspect_container_id"] = "a" * 64
            contract["inspect_hostname"] = "not-the-current-container"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "container hostname mismatch"):
                release._validate_container_contract(contract_path, opc)
            contract["inspect_hostname"] = socket.gethostname()
            contract["opc"]["sha256"] = "0" * 64
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "OPC SHA-256 mismatch"):
                release._validate_container_contract(contract_path, opc)

    def test_build_closes_inventory_and_controls_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workdir = root / "release-work"
            release.prepare_release(self.outer_zip, workdir)
            manifest = self._fake_build(root, workdir)
            self.assertTrue(manifest["build_runtime"]["installed_inventory_closed"])
            self.assertEqual(manifest["build_runtime"]["opc"]["sha256"], release.sha256_file(root / "fake-opc.py"))
            self.assertEqual(
                manifest["build_runtime"]["cann_version_files"][0]["text_summary"],
                ["Version=8.3.RC1", "Package=CANN"],
            )
            self.assertEqual(
                manifest["build_runtime"]["installed_qrv2_before"],
                manifest["build_runtime"]["installed_qrv2_after"],
            )
            expected_pythonpath = manifest["build_runtime"]["controlled_pythonpath"]
            self.assertNotIn("poisoned-parent-path", expected_pythonpath)
            observed = (
                workdir / "build" / release.CANONICAL_SOC_KEY / "debug/pythonpath.txt"
            ).read_text(encoding="utf-8")
            self.assertEqual(observed, expected_pythonpath)
            canonical = manifest["artifacts"][release.CANONICAL_SOC_KEY]
            alias = manifest["artifacts"][release.ALIAS_SOC_KEY]
            self.assertEqual(alias["artifact_mode"], "dav2201_alias_copy")
            self.assertEqual(alias["object_sha256"], canonical["object_sha256"])
            self.assertEqual(alias["json_sha256"], canonical["json_sha256"])

    def test_build_detects_cann_version_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workdir = root / "release-work"
            release.prepare_release(self.outer_zip, workdir)
            opc, ascend_opp, installed_cloud, contract = self._runtime_fixture(root, workdir)
            with mock.patch.dict(
                os.environ,
                {
                    "ASCEND_OPP_PATH": str(ascend_opp),
                    "QRV2_TEST_MUTATE": str(root / "version.cfg"),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "CANN version file 0 SHA-256 mismatch"):
                    release.build_release(workdir, opc, contract, installed_cloud)

    def test_installed_inventory_detects_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workdir = root / "release-work"
            release.prepare_release(self.outer_zip, workdir)
            _, _, installed_cloud, _ = self._runtime_fixture(root, workdir)
            before = release.installed_qrv2_inventory(installed_cloud)
            wrapper = installed_cloud / "packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.py"
            wrapper.write_bytes(wrapper.read_bytes() + b"\n# changed\n")
            after = release.installed_qrv2_inventory(installed_cloud)
            with self.assertRaisesRegex(RuntimeError, "inventory changed"):
                release._assert_inventory_unchanged(before, after, label="installed QrV2")

    def test_installed_inventory_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workdir = root / "release-work"
            release.prepare_release(self.outer_zip, workdir)
            _, _, installed_cloud, _ = self._runtime_fixture(root, workdir)
            wrapper = installed_cloud / "packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.py"
            real_wrapper = wrapper.with_name("qr_v2.real.py")
            wrapper.rename(real_wrapper)
            wrapper.symlink_to(real_wrapper.name)
            with self.assertRaisesRegex(RuntimeError, "rejects symlink"):
                release.installed_qrv2_inventory(installed_cloud)

    def test_prepare_nested_archive_and_manifest(self) -> None:
        original_sha = release.sha256_file(self.outer_zip)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workdir = root / "release-work"
            manifest = release.prepare_release(self.outer_zip, workdir)
            self.assertEqual(manifest["status"], "prepared")
            self.assertEqual(set(manifest["build_inputs"]), set(release.SOCS))
            self.assertEqual(
                manifest["candidate"]["source_sha256"], release.EXPECTED_CANDIDATE_SHA256
            )
            self.assertEqual(manifest["candidate"]["identity"], release.BIN_NAME)
            self.assertEqual(
                manifest["candidate"]["v4_candidate_sha256"],
                v5_patcher.EXPECTED_V4_CANDIDATE_SHA256,
            )
            self.assertEqual(
                manifest["candidate"]["reverse_v4_sha256"],
                v5_patcher.EXPECTED_V4_CANDIDATE_SHA256,
            )
            self.assertEqual(manifest["tools"], release._release_tool_hashes())
            self.assertEqual(
                manifest["tools"]["patcher_dependency_sha256"],
                release.sha256_file(Path(v5_patcher.release_v4.__file__).resolve()),
            )
            for soc_key, soc_version in release.SOCS.items():
                command = manifest["opc_commands"][soc_key]
                if soc_key == release.CANONICAL_SOC_KEY:
                    self.assertIn(f"--soc_version={soc_version}", command)
                else:
                    self.assertEqual(
                        command,
                        ["alias-copy", release.CANONICAL_SOC_KEY, release.ALIAS_SOC_KEY],
                    )
                self.assertEqual(
                    release.sha256_file(workdir / "build" / soc_key / "qr_v2.cpp"),
                    release.EXPECTED_CANDIDATE_SHA256,
                )
            self.assertEqual(release.sha256_file(self.outer_zip), original_sha)

    def test_tool_guard_rejects_v5_patcher_or_v4_dependency_hash_drift(self) -> None:
        for key in ("patcher_sha256", "patcher_dependency_sha256"):
            with self.subTest(key=key):
                manifest = {"tools": release._release_tool_hashes()}
                manifest["tools"][key] = "0" * 64
                with self.assertRaisesRegex(RuntimeError, "release tool hash guard failed"):
                    release._guard_tools(manifest)

    def test_offline_repack_updates_record_without_touching_original(self) -> None:
        original_sha = release.sha256_file(self.outer_zip)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workdir = root / "release-work"
            release.prepare_release(self.outer_zip, workdir)
            self._fake_build(root, workdir)
            packaged = release.package_release(workdir)
            new_wheel = Path(packaged["package"]["wheel_path"])
            new_outer = Path(packaged["package"]["outer_zip_path"])
            self.assertTrue(
                new_outer.name.endswith("-qrv2-matmul-position-fix-v5.zip"),
                new_outer.name,
            )
            release.verify_wheel_record(new_wheel)
            with zipfile.ZipFile(new_wheel) as wheel:
                self.assertEqual(
                    release.sha256_bytes(wheel.read(release.SOURCE_REL)),
                    release.EXPECTED_CANDIDATE_SHA256,
                )
                for soc_key in release.SOCS:
                    config_path = (
                        f"{release.KERNEL_ROOT_REL}/config/{soc_key}/qr_v2.json"
                    )
                    config = json.loads(wheel.read(config_path))
                    self.assertEqual(
                        config["binList"][0]["binInfo"]["jsonFilePath"],
                        f"{soc_key}/qr_v2/{release.BIN_NAME}.json",
                    )
                    binary_info_path = (
                        f"{release.KERNEL_ROOT_REL}/config/{soc_key}/"
                        f"{release.BINARY_INFO_CONFIG_NAME}"
                    )
                    binary_info_bytes = wheel.read(binary_info_path)
                    binary_info = json.loads(binary_info_bytes)
                    self.assertEqual(
                        binary_info["QrV2"],
                        release._qrv2_binary_info_entry(soc_key, release.BIN_NAME),
                    )
                    self.assertEqual(
                        packaged["package"]["packaged_files"][soc_key][
                            "binary_info_config"
                        ],
                        {
                            "path": binary_info_path,
                            "sha256": release.sha256_bytes(binary_info_bytes),
                        },
                    )
            with zipfile.ZipFile(new_outer) as outer:
                self.assertEqual(
                    release.sha256_bytes(outer.read(release.WHEEL_NAME)),
                    release.sha256_file(new_wheel),
                )
            self.assertEqual(release.sha256_file(self.outer_zip), original_sha)

    def test_package_rejects_alias_artifact_even_if_manifest_sha_is_synchronized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workdir = root / "release-work"
            release.prepare_release(self.outer_zip, workdir)
            manifest = self._fake_build(root, workdir)
            alias_object = Path(
                manifest["artifacts"][release.ALIAS_SOC_KEY]["object_path"]
            )
            alias_object.write_bytes(alias_object.read_bytes() + b"\0alias-tamper\0")
            manifest["artifacts"][release.ALIAS_SOC_KEY]["object_sha256"] = (
                release.sha256_file(alias_object)
            )
            manifest["artifacts"][release.ALIAS_SOC_KEY]["object_size"] = (
                alias_object.stat().st_size
            )
            (workdir / "release_manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "alias object is not byte-identical"):
                release.package_release(workdir)

    def test_binary_info_rewrite_rejects_residual_extra_entry_and_wrong_soc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workdir = root / "release-work"
            release.prepare_release(self.outer_zip, workdir)
            soc_key = release.CANONICAL_SOC_KEY
            binary_info_path = (
                workdir
                / "wheel_original"
                / release.KERNEL_ROOT_REL
                / "config"
                / soc_key
                / release.BINARY_INFO_CONFIG_NAME
            )
            original = binary_info_path.read_bytes()
            candidate = release._updated_binary_info_config(original, soc_key)
            release._validate_binary_info_config_delta(original, candidate, soc_key)

            old_path = f"{soc_key}/qr_v2/{release.ORIGINAL_BIN_NAME}.o"
            new_path = f"{soc_key}/qr_v2/{release.BIN_NAME}.o"
            residual_object = json.loads(candidate)
            residual_object["QrV2"]["binaryList"][0]["binPath"] = old_path
            with self.assertRaisesRegex(RuntimeError, "candidate QrV2 binary-info"):
                release._validate_binary_info_config_delta(
                    original, json.dumps(residual_object).encode("utf-8"), soc_key
                )

            extra_entry_object = json.loads(candidate)
            extra_entry = dict(extra_entry_object["QrV2"]["binaryList"][0])
            extra_entry["binPath"] = new_path
            extra_entry_object["QrV2"]["binaryList"].append(extra_entry)
            with self.assertRaisesRegex(RuntimeError, "candidate QrV2 binary-info"):
                release._validate_binary_info_config_delta(
                    original, json.dumps(extra_entry_object).encode("utf-8"), soc_key
                )

            with self.assertRaisesRegex(RuntimeError, "original QrV2 binary-info"):
                release._updated_binary_info_config(original, release.ALIAS_SOC_KEY)

    def test_package_rejects_tampered_binary_info_manifest_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workdir = root / "release-work"
            release.prepare_release(self.outer_zip, workdir)
            manifest = self._fake_build(root, workdir)
            manifest["original"]["binary_info_config_sha256"][
                release.ALIAS_SOC_KEY
            ] = "0" * 64
            (workdir / "release_manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                RuntimeError, "original binary-info manifest SHA-256 mismatch"
            ):
                release.package_release(workdir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
