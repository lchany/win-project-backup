#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import step384_patch_qr_v2_delta2_only_diagnostic as patcher  # noqa: E402


SOURCE = TOOLS / "qr_v2.cpp"


class Delta2OnlyDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_bytes()

    def test_exact_delta2_only_candidate_and_reverse_v4(self) -> None:
        candidate = patcher.build_candidate(self.source)
        report = patcher.verify_candidate_structure(self.source, candidate)
        self.assertEqual(report["candidate_identity"], patcher.CANDIDATE_IDENTITY)
        self.assertEqual(report["candidate_sha256"], patcher.EXPECTED_CANDIDATE_SHA256)
        self.assertEqual(report["reverse_v4_sha256"], patcher.EXPECTED_V4_CANDIDATE_SHA256)
        self.assertTrue(report["delta2_qa_position_change"])
        self.assertFalse(report["delta1_direct_vlocal"])
        self.assertTrue(report["calc_q_per_core_gm_scratch_retained"])
        self.assertTrue(report["calc_q_mte3_mte2_retained"])
        self.assertTrue(report["diagnostic_only"])
        self.assertEqual(
            report["diagnostic_question"],
            "with delta1 absent and v4 per-core GM scratch restored, whether delta2-only "
            "completes normally and changes v4 finite behavior",
        )
        self.assertFalse(report["release_candidate"])
        self.assertTrue(report["package_forbidden"])

    def test_source_and_candidate_drift_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "source SHA-256 mismatch"):
            patcher.build_candidate(self.source + b"\n")
        candidate = patcher.build_candidate(self.source)
        with self.assertRaisesRegex(RuntimeError, "candidate bytes differ"):
            patcher.verify_candidate_structure(self.source, candidate + b"\n")

    def test_direct_vlocal_is_rejected(self) -> None:
        candidate = patcher.build_candidate(self.source)
        damaged = candidate.replace(
            patcher.release_v4.V4_CALC_Q_SCRATCH_BLOCK.encode(),
            patcher.DIRECT_VLOCAL.encode(),
            1,
        )
        with mock.patch.object(patcher, "build_unverified", return_value=damaged), mock.patch.object(
            patcher, "EXPECTED_CANDIDATE_SHA256", patcher.sha256_bytes(damaged)
        ):
            with self.assertRaisesRegex(RuntimeError, "per-core GM scratch"):
                patcher.verify_candidate_structure(self.source, damaged)

    def test_third_change_breaks_reverse_v4_gate(self) -> None:
        candidate = patcher.build_candidate(self.source)
        damaged = candidate.replace(b"float32_t scalar = 1.0;", b"float32_t scalar = 2.0;", 1)
        with mock.patch.object(patcher, "build_unverified", return_value=damaged), mock.patch.object(
            patcher, "EXPECTED_CANDIDATE_SHA256", patcher.sha256_bytes(damaged)
        ):
            with self.assertRaisesRegex(RuntimeError, "delta exceeds delta2"):
                patcher.verify_candidate_structure(self.source, damaged)

    def test_check_mode_and_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.cpp"
            output = root / "candidate.cpp"
            source.write_bytes(self.source)
            self.assertEqual(patcher.main([str(source), "--check"]), 0)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "output must be omitted"):
                patcher.main([str(source), str(output), "--check"])
            with self.assertRaisesRegex(ValueError, "output is required"):
                patcher.main([str(source)])

    def test_o_excl_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.cpp"
            output.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                patcher.write_new_file(output, b"replace")
            self.assertEqual(output.read_bytes(), b"keep")

    def test_main_creates_new_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.cpp"
            output = root / "candidate.cpp"
            source.write_bytes(self.source)
            self.assertEqual(patcher.main([str(source), str(output)]), 0)
            self.assertEqual(output.read_bytes(), patcher.build_candidate(self.source))

    def test_symlink_inputs_and_outputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_source = root / "real.cpp"
            source_link = root / "source.cpp"
            output_link = root / "candidate.cpp"
            real_source.write_bytes(self.source)
            source_link.symlink_to(real_source.name)
            output_link.symlink_to("missing.cpp")
            with self.assertRaisesRegex(ValueError, "source must not be a symlink"):
                patcher.main([str(source_link), "--check"])
            with self.assertRaisesRegex(ValueError, "output must be a new path"):
                patcher.main([str(real_source), str(output_link)])

    def test_write_failure_removes_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.cpp"
            with mock.patch.object(patcher.os, "fsync", side_effect=RuntimeError("fsync failure")):
                with self.assertRaisesRegex(RuntimeError, "fsync failure"):
                    patcher.write_new_file(output, b"partial")
            self.assertFalse(output.exists())

    def test_write_failure_does_not_remove_racing_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.cpp"

            def replace_then_fail(_descriptor: int) -> None:
                output.unlink()
                output.write_bytes(b"replacement")
                raise RuntimeError("fsync failure")

            with mock.patch.object(patcher.os, "fsync", side_effect=replace_then_fail):
                with self.assertRaisesRegex(RuntimeError, "fsync failure"):
                    patcher.write_new_file(output, b"partial")
            self.assertEqual(output.read_bytes(), b"replacement")

    def test_cleanup_failure_preserves_primary_and_records_detail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.cpp"
            real_unlink = patcher.os.unlink

            def fail_candidate_unlink(path: object, *args: object, **kwargs: object) -> None:
                if path == output.name and kwargs.get("dir_fd") is not None:
                    raise PermissionError("cleanup denied")
                real_unlink(path, *args, **kwargs)

            with mock.patch.object(patcher.os, "fsync", side_effect=RuntimeError("fsync failure")), \
                    mock.patch.object(patcher.os, "unlink", side_effect=fail_candidate_unlink):
                with self.assertRaisesRegex(RuntimeError, "fsync failure") as raised:
                    patcher.write_new_file(output, b"partial")
            self.assertIn("PermissionError: cleanup denied", raised.exception.cleanup_error)
            self.assertTrue(output.exists())

    def test_parent_directory_descriptor_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.cpp"
            real_open = patcher.os.open
            opened: list[int] = []

            def capture_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
                descriptor = real_open(path, flags, *args, **kwargs)
                if path == output.parent:
                    opened.append(descriptor)
                return descriptor

            with mock.patch.object(patcher.os, "open", side_effect=capture_open):
                patcher.write_new_file(output, b"candidate")
            self.assertEqual(len(opened), 1)
            with self.assertRaises(OSError):
                os.fstat(opened[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
