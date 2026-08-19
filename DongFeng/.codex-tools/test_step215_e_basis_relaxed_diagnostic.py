#!/usr/bin/env python3
"""Local CPU-free unit tests for STEP-215-E basis-relaxed policy wiring."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("step215_e_soap_two_cycle_gate.py")
SPEC = importlib.util.spec_from_file_location("step215_e_gate_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load STEP-215-E harness")
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class BasisRelaxedPolicyTest(unittest.TestCase):
    def required_cli(self) -> list[str]:
        return [
            "--repo", "repo",
            "--config", "config",
            "--checkpoint", "checkpoint",
            "--adapter", "adapter",
            "--output-dir", "output",
            "--expected-soap-sha256", "0" * 64,
            "--q-orthogonality-limit", "1e-5",
        ]

    def test_cli_defaults_to_strict_raw_q(self) -> None:
        args = GATE.parse_args(self.required_cli())
        self.assertFalse(args.basis_relaxed_diagnostic)

    def test_cli_requires_explicit_flag(self) -> None:
        args = GATE.parse_args(self.required_cli() + ["--basis-relaxed-diagnostic"])
        self.assertTrue(args.basis_relaxed_diagnostic)

    def test_q_orthogonality_calibration_is_explicit_and_capped(self) -> None:
        self.assertEqual(GATE.configure_q_orthogonality_limit(1.0e-5), 1.0e-5)
        self.assertEqual(GATE.configure_q_orthogonality_limit(2.0e-5), 2.0e-5)
        with self.assertRaisesRegex(RuntimeError, "exceeds calibrated hard maximum"):
            GATE.configure_q_orthogonality_limit(2.0001e-5)
        with self.assertRaisesRegex(RuntimeError, "at least 1e-5"):
            GATE.configure_q_orthogonality_limit(9.999e-6)
        GATE.configure_q_orthogonality_limit(1.0e-5)

    def test_relaxation_is_only_for_two_cross_implementation_cycles(self) -> None:
        expected = {
            "cycle1-baselineA-candidate-adaptive",
            "cycle2-baselineA-candidate-adaptive",
        }
        self.assertEqual(set(GATE.BASIS_RELAXED_COMPARISONS), expected)
        for label in expected:
            self.assertTrue(GATE.basis_relaxed_for_comparison(label, True))
            self.assertFalse(GATE.basis_relaxed_for_comparison(label, False))
        for label in (
            "initial-baselineA-candidate",
            "resume-candidate-adaptive",
            "candidate:cycle1-save-load",
            "candidate:cycle2-continuous-resume",
        ):
            self.assertFalse(GATE.basis_relaxed_for_comparison(label, True))

    def test_q_relaxation_does_not_relax_non_q_tensor_or_global_limits(self) -> None:
        GATE.enforce_distance_limits(
            "relaxed-q-only",
            q_worst_nrmse=0.9,
            other_worst_relative_l2=9.0e-5,
            other_worst_path="optimizer_state/state_dict/state/0/GG/0",
            other_global_relative_l2=8.0e-5,
            q_limit=GATE.Q_LIMIT,
            other_limit=GATE.OTHER_LIMIT,
            ignore_q_distance=True,
        )
        with self.assertRaisesRegex(RuntimeError, "non-Q tensor"):
            GATE.enforce_distance_limits(
                "per-tensor-still-hard",
                q_worst_nrmse=0.9,
                other_worst_relative_l2=1.01e-4,
                other_worst_path="parameters/0",
                other_global_relative_l2=1.0e-6,
                q_limit=GATE.Q_LIMIT,
                other_limit=GATE.OTHER_LIMIT,
                ignore_q_distance=True,
            )
        with self.assertRaisesRegex(RuntimeError, "other relative-L2"):
            GATE.enforce_distance_limits(
                "global-still-hard",
                q_worst_nrmse=0.9,
                other_worst_relative_l2=1.0e-6,
                other_worst_path="parameters/0",
                other_global_relative_l2=1.01e-4,
                q_limit=GATE.Q_LIMIT,
                other_limit=GATE.OTHER_LIMIT,
                ignore_q_distance=True,
            )

    def test_strict_mode_rejects_raw_q_distance(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Q NRMSE"):
            GATE.enforce_distance_limits(
                "strict",
                q_worst_nrmse=GATE.Q_LIMIT * 2.0,
                other_worst_relative_l2=0.0,
                other_worst_path="",
                other_global_relative_l2=0.0,
                q_limit=GATE.Q_LIMIT,
                other_limit=GATE.OTHER_LIMIT,
                ignore_q_distance=False,
            )

    def test_shell_entrypoints_are_default_strict_and_explicit_opt_in(self) -> None:
        runner = MODULE_PATH.with_name("step215_e_run_inside_container.sh").read_text(encoding="utf-8")
        host = MODULE_PATH.with_name("step215_e_host_launch_contract.sh").read_text(encoding="utf-8")
        self.assertIn("BASIS_RELAXED_DIAGNOSTIC:-0", runner)
        self.assertIn("basis_args+=(--basis-relaxed-diagnostic)", runner)
        self.assertIn('if [ "$#" -eq 9 ]', host)
        self.assertIn('if [ "$9" != --basis-relaxed-diagnostic ]', host)
        self.assertIn('-e BASIS_RELAXED_DIAGNOSTIC="$basis_relaxed_diagnostic"', host)
        self.assertIn('--q-orthogonality-limit "$q_orthogonality_limit"', runner)
        self.assertIn('--q-orthogonality-limit=2e-5', host)
        self.assertIn('if [ "${10}" != --q-orthogonality-limit=2e-5 ]', host)
        self.assertIn('-e Q_ORTHOGONALITY_LIMIT="$q_orthogonality_limit"', host)


if __name__ == "__main__":
    unittest.main()
