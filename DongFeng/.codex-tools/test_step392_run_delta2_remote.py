#!/usr/bin/env python3
"""Focused offline controller-contract tests for STEP392."""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import step392_run_delta2_remote as controller


def summary() -> dict:
    zeros = {"concrete_aiv_hits": 0, "original_hits": 0, "v4_hits": 0, "v5_hits": 0, "legacy_v6_hits": 0, "unknown_hits": 0}
    return {
        "schema": "step392-diagnostic-host-summary-v1", "status": "diagnostic_world8_pass",
        "diagnostic_only": True, "release_candidate": False, "rank_count": 8,
        "raw_profiles_retained": True, "input_sha256": dict(controller.EXPECTED_INPUT_SHA256),
        "module_file_sha256": {"cloud_init": "a"*64, "cloud_extension": "b"*64, "cloud_linalg": "c"*64},
        "gate_token_sha256": "d"*64, "launcher_ownership_sha256": "e"*64, "rank_ownership_sha256": "f"*64,
        "candidate_identity": "QrV2_qa_position_delta2_only_diagnostic_v1", "concrete_aic_hit_count": 8,
        "concrete_aiv_hit_count": 0, "forbidden_identity_hit_count": 0,
        "math_gate": {"input_unmodified": True, "input_finite": True, "q_finite": True, "r_finite": True, "reconstruction": True, "orthogonality": True, "lower_triangle_exact_zero": True, "cpu_qr_projection": True},
        "ranks": [{"rank": rank, "call_count": 1, "identity_pass": True, "concrete_aic_hits": 1, **zeros} for rank in range(8)],
        "raw_profiles": [{"rank": rank, "file_count": 1, "total_bytes": 1} for rank in range(8)],
    }


class Step392RemoteContractTests(unittest.TestCase):
    def test_disarmed_attempt5_world8_summary_and_forbidden_scan(self) -> None:
        self.assertFalse(controller.NPU_READY)
        self.assertEqual(controller.CONTAINER, "mapqr-leicheng")
        self.assertEqual(controller.ATTEMPT5_MANIFEST_SHA256, "0221f5b64fe682d230f834554b3b8d977673f807c6a890c5279c835ebe173de8")
        self.assertEqual(controller.ATTEMPT5_RECEIPT_SHA256, "2d5845c7c2dd74b5d50c7689e3c89f0c1144e472e804f34d8bbd421770d26f9c")
        self.assertEqual(controller.ATTEMPT5_COMPLETION_SHA256, "d37f0bf9754fe9e67a5428b133d396792f9237975b2e120258095730c4ba8dda")
        self.assertEqual(controller.DRY_RUN_ACTIONS.count("world8_back8_once"), 1)
        self.assertEqual(set(controller.FORBIDDEN_ACTIONS), {"package", "wheel_write", "release", "install", "modify_installed", "train", "training", "30step", "build", "download_remote_artifacts"})
        with mock.patch.object(controller.Path, "read_text") as mapping, self.assertRaisesRegex(RuntimeError, "disarmed"):
            controller.execute()
        mapping.assert_not_called()
        value = summary(); self.assertIs(controller.validate_summary(value), value)
        mutations = (
            lambda row: row.update({"concrete_aic_hit_count": 7}),
            lambda row: row.update({"concrete_aiv_hit_count": 1}),
            lambda row: row["ranks"][0].update({"legacy_v6_hits": 1}),
            lambda row: row["math_gate"].update({"orthogonality": False}),
        )
        for mutate in mutations:
            row=copy.deepcopy(value); mutate(row)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError): controller.validate_summary(row)
        script=controller.forbidden_embedded_script()
        with tempfile.TemporaryDirectory() as directory:
            ok=subprocess.run(["python3","-c",script,directory,json.dumps(list(controller.FORBIDDEN_ACTIONS))],capture_output=True,text=True)
            self.assertEqual(ok.returncode,0,ok.stderr)
            Path(directory,"training").mkdir()
            bad=subprocess.run(["python3","-c",script,directory,json.dumps(list(controller.FORBIDDEN_ACTIONS))],capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)


if __name__ == "__main__":
    unittest.main()
