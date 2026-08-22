#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import step372_patch_qr_v2_matmul_position_v5 as patcher  # noqa: E402


SOURCE = TOOLS / "qr_v2.cpp"


class MatmulPositionV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_bytes()

    def test_exact_candidate_and_reverse_v4_pass(self) -> None:
        candidate = patcher.build_candidate(self.source)
        report = patcher.verify_candidate_structure(self.source, candidate)
        self.assertEqual(report["candidate_identity"], "QrV2_matmul_position_fix_v5")
        self.assertEqual(report["qa_positions"], ["VECIN", "GM", "VECIN"])
        self.assertEqual(report["reverse_v4_sha256"], patcher.EXPECTED_V4_CANDIDATE_SHA256)
        self.assertTrue(report["delta_exactly_two_position_fixes"])

    def test_source_change_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "source SHA-256 mismatch"):
            patcher.build_candidate(self.source + b"\n")

    def test_candidate_change_fails_closed(self) -> None:
        candidate = patcher.build_candidate(self.source)
        with self.assertRaisesRegex(RuntimeError, "candidate bytes differ"):
            patcher.verify_candidate_structure(self.source, candidate + b"\n")

    def test_wrong_locked_candidate_hash_fails_closed(self) -> None:
        with mock.patch.object(patcher, "EXPECTED_CANDIDATE_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "candidate SHA-256 mismatch"):
                patcher.build_candidate(self.source)

    def test_vtv_gm_tensor_a_is_rejected(self) -> None:
        candidate = patcher.build_candidate(self.source)
        damaged = candidate.replace(
            patcher.V5_DIRECT_VTV_A.encode(),
            b"        this->vtvMatmulObj.SetTensorA(workspaceInGm[0]);\n",
            1,
        )
        with mock.patch.object(patcher, "build_unverified", return_value=damaged), mock.patch.object(
            patcher, "EXPECTED_CANDIDATE_SHA256", patcher.sha256_bytes(damaged)
        ):
            with self.assertRaisesRegex(RuntimeError, "direct vLocal|receives GM"):
                patcher.verify_candidate_structure(self.source, damaged)

    def test_qa_old_position_declaration_is_rejected(self) -> None:
        candidate = patcher.build_candidate(self.source)
        damaged = candidate.replace(
            patcher.V5_QA_DECLARATION.encode(),
            patcher.V4_QA_DECLARATION.encode(),
            1,
        )
        with mock.patch.object(patcher, "build_unverified", return_value=damaged), mock.patch.object(
            patcher, "EXPECTED_CANDIDATE_SHA256", patcher.sha256_bytes(damaged)
        ):
            with self.assertRaisesRegex(RuntimeError, "qa position declaration"):
                patcher.verify_candidate_structure(self.source, damaged)

    def test_extra_change_breaks_reverse_v4_gate(self) -> None:
        candidate = patcher.build_candidate(self.source)
        damaged = candidate.replace(b"float32_t scalar = 1.0;", b"float32_t scalar = 2.0;", 1)
        with mock.patch.object(patcher, "build_unverified", return_value=damaged), mock.patch.object(
            patcher, "EXPECTED_CANDIDATE_SHA256", patcher.sha256_bytes(damaged)
        ):
            with self.assertRaisesRegex(RuntimeError, "delta exceeds"):
                patcher.verify_candidate_structure(self.source, damaged)


if __name__ == "__main__":
    unittest.main()
