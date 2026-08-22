#!/usr/bin/env python3
"""Focused offline identity/math tests for the STEP392 worker layer."""

from __future__ import annotations

import types
import unittest
from pathlib import Path

import step392_delta2_math_worker as worker_adapter


def fake_worker(referenced: dict[str, int]):
    mappings = {index: name for index, name in enumerate(referenced, 1)}
    references = {index: count for index, count in enumerate(referenced.values(), 1)}
    worker = types.SimpleNamespace(
        CANDIDATE_AIC=worker_adapter.V5_AIC,
        CANDIDATE_AIV=worker_adapter.V5_AIV,
        legacy=types.SimpleNamespace(collect_runtime_identity=lambda _root: (mappings, references, [], [])),
    )
    worker.verify_profile = lambda _root, expected_aic_references=1: {"pass": True}
    worker._finalize_call = lambda *_a, **_k: {
        "input_unmodified": True, "shape_pass": True, "finite_pass": True,
        "reconstruction": {"violation_count": 0},
        "orthogonality": {"violation_count": 0},
        "full_rank_projection": {"required": True, "pass": True},
        "lower_triangle_exact_zero": True,
    }
    worker._normalize_json_diagnostic = lambda value: (value, 0)
    return worker


class Step392WorkerContractTests(unittest.TestCase):
    def test_delta2_exact_aic_math_and_negative_identity_table(self) -> None:
        worker = fake_worker({worker_adapter.DIAGNOSTIC_AIC: 1})
        restore = worker_adapter.install_worker_identity(worker)
        try:
            identity = worker.verify_profile(Path("profile"), expected_aic_references=1)
            self.assertEqual(identity["diagnostic_aic_task_reference_count"], 1)
            for key in ("diagnostic_aiv_task_reference_count", "original_task_reference_count", "v4_task_reference_count", "v5_task_reference_count", "legacy_v6_task_reference_count", "unknown_qrv2_task_reference_count"):
                self.assertEqual(identity[key], 0)
            call = worker._finalize_call()
            self.assertEqual(set(call["predicate_status"].values()), {"pass"})
            self.assertTrue(call["projection_pass"])
        finally:
            restore()
        bad = ({}, {worker_adapter.DIAGNOSTIC_AIC: 2}, {worker_adapter.DIAGNOSTIC_AIC: 1, worker_adapter.DIAGNOSTIC_AIV: 1}, {worker_adapter.DIAGNOSTIC_AIC: 1, worker_adapter.LEGACY_V6_AIC: 1}, {worker_adapter.DIAGNOSTIC_AIC: 1, worker_adapter.V4_AIC: 1}, {worker_adapter.DIAGNOSTIC_AIC: 1, worker_adapter.V5_AIC: 1}, {worker_adapter.DIAGNOSTIC_AIC: 1, worker_adapter.ORIGINAL_AIC: 1})
        for referenced in bad:
            candidate = fake_worker(referenced)
            restore = worker_adapter.install_worker_identity(candidate)
            try:
                with self.subTest(referenced=referenced), self.assertRaises(RuntimeError):
                    candidate.verify_profile(Path("profile"), expected_aic_references=1)
            finally:
                restore()


if __name__ == "__main__":
    unittest.main()
