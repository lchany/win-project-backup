#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import step375_patch_qr_v2_vtv_direct_qa_legacy_probe_v6 as patcher  # noqa: E402


SOURCE = TOOLS / "qr_v2.cpp"


class VtvDirectQaLegacyProbeV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_bytes()

    def test_exact_delta1_only_candidate_and_reverse_v4(self) -> None:
        candidate = patcher.build_candidate(self.source)
        report = patcher.verify_candidate_structure(self.source, candidate)
        self.assertEqual(
            report["candidate_identity"], "QrV2_vtv_direct_qa_legacy_probe_v6"
        )
        self.assertEqual(report["candidate_sha256"], patcher.EXPECTED_CANDIDATE_SHA256)
        self.assertEqual(
            report["reverse_v4_sha256"], patcher.EXPECTED_V4_CANDIDATE_SHA256
        )
        self.assertTrue(report["delta1_direct_vlocal"])
        self.assertFalse(report["delta2_qa_position_change"])
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["release_candidate"])

    def test_source_drift_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "source SHA-256 mismatch"):
            patcher.build_candidate(self.source + b"\n")

    def test_candidate_drift_fails_closed(self) -> None:
        candidate = patcher.build_candidate(self.source)
        with self.assertRaisesRegex(RuntimeError, "candidate bytes differ"):
            patcher.verify_candidate_structure(self.source, candidate + b"\n")

    def test_accidental_delta2_is_rejected(self) -> None:
        candidate = patcher.build_candidate(self.source)
        damaged = candidate.replace(
            patcher.V4_QA_DECLARATION.encode(),
            patcher.V5_QA_DECLARATION.encode(),
            1,
        )
        with mock.patch.object(
            patcher, "build_unverified", return_value=damaged
        ), mock.patch.object(
            patcher, "EXPECTED_CANDIDATE_SHA256", patcher.sha256_bytes(damaged)
        ):
            with self.assertRaisesRegex(RuntimeError, "legacy qa declaration"):
                patcher.verify_candidate_structure(self.source, damaged)

    def test_missing_delta1_is_rejected(self) -> None:
        candidate = patcher.build_candidate(self.source)
        damaged = candidate.replace(
            patcher.DIRECT_VTV_A.encode(),
            patcher.release_v4.V4_CALC_Q_SCRATCH_BLOCK.encode(),
            1,
        )
        with mock.patch.object(
            patcher, "build_unverified", return_value=damaged
        ), mock.patch.object(
            patcher, "EXPECTED_CANDIDATE_SHA256", patcher.sha256_bytes(damaged)
        ):
            with self.assertRaisesRegex(RuntimeError, "delta1 direct vLocal"):
                patcher.verify_candidate_structure(self.source, damaged)

    def test_four_cell_candidate_matrix_sha_closes(self) -> None:
        matrix = patcher.candidate_matrix(self.source)
        hashes = {name: patcher.sha256_bytes(payload) for name, payload in matrix.items()}
        self.assertEqual(
            hashes,
            {
                "v4": patcher.EXPECTED_V4_CANDIDATE_SHA256,
                "delta1_only": patcher.EXPECTED_CANDIDATE_SHA256,
                "delta2_only": patcher.EXPECTED_DELTA2_ONLY_SHA256,
                "delta1_and_delta2_v5": patcher.EXPECTED_V5_SHA256,
            },
        )
        self.assertEqual(len(set(hashes.values())), 4)

    def test_third_change_breaks_reverse_v4_gate(self) -> None:
        candidate = patcher.build_candidate(self.source)
        damaged = candidate.replace(
            b"float32_t scalar = 1.0;", b"float32_t scalar = 2.0;", 1
        )
        with mock.patch.object(
            patcher, "build_unverified", return_value=damaged
        ), mock.patch.object(
            patcher, "EXPECTED_CANDIDATE_SHA256", patcher.sha256_bytes(damaged)
        ):
            with self.assertRaisesRegex(RuntimeError, "delta exceeds delta1"):
                patcher.verify_candidate_structure(self.source, damaged)

    def test_main_never_overwrites_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.cpp"
            output = root / "candidate.cpp"
            source.write_bytes(self.source)
            output.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                patcher.write_new_file(output, b"replace")
            self.assertEqual(output.read_bytes(), b"keep")
            with self.assertRaisesRegex(ValueError, "output must be a new path"):
                patcher.main([str(source), str(output)])
            self.assertEqual(output.read_bytes(), b"keep")

    def test_source_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_source = root / "real-source.cpp"
            source_link = root / "source-link.cpp"
            real_source.write_bytes(self.source)
            source_link.symlink_to(real_source.name)
            with self.assertRaisesRegex(ValueError, "source must not be a symlink"):
                patcher.main([str(source_link), "--check"])

    def test_dangling_output_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.cpp"
            output = root / "candidate.cpp"
            source.write_bytes(self.source)
            output.symlink_to("missing-target.cpp")
            with self.assertRaisesRegex(ValueError, "output must be a new path"):
                patcher.main([str(source), str(output)])
            self.assertTrue(output.is_symlink())

    def test_output_equal_to_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.cpp"
            source.write_bytes(self.source)
            with self.assertRaisesRegex(ValueError, "output must be a new path"):
                patcher.main([str(source), str(source)])
            self.assertEqual(source.read_bytes(), self.source)

    def test_check_rejects_output_argument(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.cpp"
            output = root / "candidate.cpp"
            source.write_bytes(self.source)
            with self.assertRaisesRegex(ValueError, "output must be omitted"):
                patcher.main([str(source), str(output), "--check"])
            self.assertFalse(output.exists())

    def test_output_is_required_without_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.cpp"
            source.write_bytes(self.source)
            with self.assertRaisesRegex(ValueError, "output is required"):
                patcher.main([str(source)])

    def test_main_creates_new_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.cpp"
            output = root / "candidate.cpp"
            source.write_bytes(self.source)
            self.assertEqual(patcher.main([str(source), str(output)]), 0)
            self.assertEqual(output.read_bytes(), patcher.build_candidate(self.source))

    def test_write_failure_removes_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.cpp"
            with mock.patch.object(
                patcher.os, "fsync", side_effect=RuntimeError("fsync failure")
            ):
                with self.assertRaisesRegex(RuntimeError, "fsync failure"):
                    patcher.write_new_file(output, b"partial")
            self.assertFalse(output.exists())

    def test_write_and_cleanup_failure_preserve_primary_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.cpp"
            primary = RuntimeError("fsync failure")
            original_args = primary.args
            original_text = str(primary)
            with mock.patch.object(
                patcher.os, "fsync", side_effect=primary
            ), mock.patch.object(
                Path, "unlink", side_effect=RuntimeError("unlink failure")
            ):
                with self.assertRaises(RuntimeError) as caught:
                    patcher.write_new_file(output, b"partial")
            self.assertIs(caught.exception, primary)
            self.assertEqual(caught.exception.args, original_args)
            self.assertEqual(str(caught.exception), original_text)
            self.assertIn("candidate cleanup failed", caught.exception.cleanup_error)
            self.assertIn("unlink failure", caught.exception.cleanup_error)
            self.assertTrue(output.exists())

    def test_cleanup_error_add_note_is_best_effort(self) -> None:
        class NoteCapableError(RuntimeError):
            def __init__(self, message: str) -> None:
                super().__init__(message)
                self.notes: list[str] = []

            def add_note(self, note: str) -> None:
                self.notes.append(note)

        primary = NoteCapableError("primary")
        original_args = primary.args
        original_text = str(primary)
        patcher._append_cleanup_error(primary, RuntimeError("cleanup"))
        self.assertEqual(primary.args, original_args)
        self.assertEqual(str(primary), original_text)
        self.assertEqual(primary.notes, [primary.cleanup_error])

    def test_cleanup_error_add_note_failure_does_not_replace_primary(self) -> None:
        class BrokenNoteError(RuntimeError):
            def add_note(self, note: str) -> None:
                raise RuntimeError(f"note failure: {note}")

        primary = BrokenNoteError("primary")
        original_args = primary.args
        original_text = str(primary)
        patcher._append_cleanup_error(primary, RuntimeError("cleanup"))
        self.assertEqual(primary.args, original_args)
        self.assertEqual(str(primary), original_text)
        self.assertIn("candidate cleanup failed", primary.cleanup_error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
