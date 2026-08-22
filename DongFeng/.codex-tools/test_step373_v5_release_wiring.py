#!/usr/bin/env python3
"""Static fail-closed tests for the QrV2 v5 build/release wiring."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_qrv2_release as release  # noqa: E402
import step373_build_qrv2_v5_release_remote as controller  # noqa: E402
import step372_patch_qr_v2_matmul_position_v5 as v5_patcher  # noqa: E402


class V5ReleaseWiringTests(unittest.TestCase):
    def test_build_controller_is_armed_only_with_reviewed_hashes(self) -> None:
        self.assertTrue(controller.BUILD_READY)
        legacy = controller.load_legacy()
        self.assertEqual(legacy.REMOTE_DIAG_NAME, controller.DIAG_NAME)

    def test_build_controller_rejects_any_locked_hash_drift(self) -> None:
        for attribute in (
            "BUILDER_SHA256",
            "LEGACY_SHA256",
            "V5_PATCHER_SHA256",
            "V4_PATCHER_SHA256",
        ):
            with self.subTest(attribute=attribute), mock.patch.object(
                controller, attribute, "0" * 64
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                    controller.load_legacy()

    def test_locked_local_hashes_match_exact_files(self) -> None:
        expected = {
            controller.LEGACY_PATH: controller.LEGACY_SHA256,
            controller.BUILDER: controller.BUILDER_SHA256,
            controller.V5_PATCHER: controller.V5_PATCHER_SHA256,
            controller.V4_PATCHER: controller.V4_PATCHER_SHA256,
        }
        for path, digest in expected.items():
            with self.subTest(path=path.name):
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
                self.assertEqual(controller.sha256_file(path), digest)

    def test_v5_build_upload_inventory_is_exact_and_transitive(self) -> None:
        with mock.patch.object(controller, "BUILD_READY", True):
            legacy = controller.load_legacy()
        inputs = legacy.input_files()
        self.assertEqual(
            tuple(path.name for path in inputs),
            (
                legacy.OUTER_ZIP.name,
                release.__file__.rsplit("/", 1)[-1],
                v5_patcher.__file__.rsplit("/", 1)[-1],
                v5_patcher.release_v4.__file__.rsplit("/", 1)[-1],
            ),
        )
        self.assertEqual({path.name for path in inputs}, set(legacy.EXPECTED_INPUTS))
        for path in inputs:
            self.assertEqual(legacy.sha256_file(path), legacy.EXPECTED_INPUTS[path.name])

    def test_candidate_identity_and_reverse_v4_are_closed(self) -> None:
        source = (TOOLS / "qr_v2.cpp").read_bytes()
        candidate = release.build_candidate(source)
        report = release.verify_candidate_structure(source, candidate)
        self.assertEqual(release.BIN_NAME, controller.EXPECTED_KERNEL)
        self.assertEqual(release.BIN_NAME, v5_patcher.CANDIDATE_IDENTITY)
        self.assertEqual(report["candidate_sha256"], controller.EXPECTED_SOURCE_SHA256)
        self.assertEqual(
            report["reverse_v4_sha256"], v5_patcher.EXPECTED_V4_CANDIDATE_SHA256
        )
        self.assertTrue(report["delta_exactly_two_position_fixes"])

    def test_v4_release_artifact_is_not_an_active_build_input(self) -> None:
        active_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                controller.BUILDER,
                controller.V5_PATCHER,
                TOOLS / "step358_prepare_release_shadow.py",
                TOOLS / "step358_qrv2_release_math_worker.py",
            )
        )
        self.assertNotIn("4c158915bd5ae3fad4834a4f88028702d2d6fb534d69da45cd06f0b536f8dead", active_sources)
        self.assertNotIn("step370_qrv2_lifetime_alpha_sync_v4_release", active_sources)


if __name__ == "__main__":
    unittest.main()
