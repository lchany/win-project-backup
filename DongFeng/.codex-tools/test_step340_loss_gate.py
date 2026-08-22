#!/usr/bin/env python3
"""Unit tests for step340_loss_gate.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("step340_loss_gate.py")
SPEC = importlib.util.spec_from_file_location("step340_loss_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class LossGateTests(unittest.TestCase):
    def run_cli(self, gpu: str, npu: str, *extra: str) -> tuple[int, dict]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu_path = root / "gpu.json"
            npu_path = root / "npu.log"
            gpu_path.write_text(gpu, encoding="utf-8")
            npu_path.write_text(npu, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--gpu", str(gpu_path), "--npu", str(npu_path), *extra],
                check=False,
                capture_output=True,
                text=True,
            )
            return result.returncode, json.loads(result.stdout)

    def test_pass_uses_actual_expected_count(self) -> None:
        code, summary = self.run_cli(
            '{"1":{"loss":100},"2":{"loss":-50}}',
            "Iter [1/2] loss: 102\nIter [2/2] loss: -49\n",
        )
        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["expected_count"], 2)
        self.assertEqual(summary["pass_count"], 2)
        self.assertAlmostEqual(summary["max_relative_deviation"], 0.02)

    def test_threshold_failure_has_nonzero_exit_and_first_failure(self) -> None:
        code, summary = self.run_cli(
            '{"1":{"loss":100},"2":{"loss":100}}',
            "Iter [1/2] loss: 103\nIter [2/2] loss: 100\n",
        )
        self.assertEqual(code, 1)
        self.assertEqual(summary["first_failure"]["iter"], 1)
        self.assertEqual(summary["first_failure"]["reasons"], ["threshold_exceeded"])
        self.assertEqual(summary["pass_count"], 1)

    def test_missing_and_duplicate_iterations_fail(self) -> None:
        code, summary = self.run_cli(
            '[{"iter":1,"loss":10},{"iter":1,"loss":10},{"iter":2,"loss":10}]',
            "Iter [1/3] loss: 10\nIter [3/3] loss: 10\n",
        )
        self.assertEqual(code, 1)
        self.assertIn("gpu_duplicate", summary["first_failure"]["reasons"])
        self.assertEqual(summary["expected_count"], 3)
        self.assertEqual(summary["fail_count"], 3)

    def test_zero_gpu_and_zero_npu_pass_with_zero_relative_deviation(self) -> None:
        code, summary = self.run_cli(
            '{"1":{"loss":0}}',
            "Iter [1/1] loss: 0\n",
        )
        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["pass_count"], 1)
        self.assertEqual(summary["fail_count"], 0)
        self.assertEqual(summary["max_relative_deviation"], 0.0)
        self.assertEqual(summary["max_relative_deviation_iter"], 1)

    def test_nan_inf_and_zero_gpu_nonzero_npu_fail(self) -> None:
        gpu = '{"1":{"loss":0},"2":{"loss":"nan"},"3":{"loss":10}}'
        npu = "Iter [1/3] loss: 1\nIter [2/3] loss: 2\nIter [3/3] loss: inf\n"
        code, summary = self.run_cli(gpu, npu)
        self.assertEqual(code, 1)
        self.assertEqual(summary["fail_count"], 3)
        self.assertEqual(summary["first_failure"]["reasons"], ["gpu_zero_npu_nonzero"])
        self.assertEqual(summary["failure_reason_counts"]["gpu_zero_npu_nonzero"], 1)
        self.assertEqual(summary["failure_reason_counts"]["gpu_non_finite_loss"], 1)
        self.assertEqual(summary["failure_reason_counts"]["npu_non_finite_loss"], 1)

    def test_explicit_range_allows_longer_gpu_reference(self) -> None:
        code, summary = self.run_cli(
            '{"1":{"loss":10},"2":{"loss":10},"3":{"loss":999}}',
            "Iter [1/2] loss: 10\nIter [2/2] loss: 10\n",
            "--start-iter", "1", "--end-iter", "2",
        )
        self.assertEqual(code, 0)
        self.assertEqual(summary["expected_count"], 2)

    def test_gpu_log_format_can_be_selected_explicitly(self) -> None:
        code, summary = self.run_cli(
            "Iter [1/2] loss: 10\nIter [2/2] loss: 20\n",
            "Iter [1/2] loss: 10.1\nIter [2/2] loss: 19.8\n",
            "--gpu-format", "log",
        )
        self.assertEqual(code, 0)
        self.assertEqual(summary["pass_count"], 2)

    def test_optional_sub_loss_summary(self) -> None:
        code, summary = self.run_cli(
            '{"1":{"loss":10,"frame_loss":2},"2":{"loss":10,"frame_loss":0}}',
            "Iter [1/2] loss: 10, frame_loss: 3\nIter [2/2] loss: 10, frame_loss: 1\n",
            "--sub-losses",
        )
        self.assertEqual(code, 0)
        metric = summary["sub_losses"]["frame_loss"]
        self.assertEqual(metric["compared_count"], 1)
        self.assertEqual(metric["zero_denominator_count"], 1)
        self.assertAlmostEqual(metric["max_relative_deviation"], 0.5)

    def test_unreadable_or_unparseable_input_is_error(self) -> None:
        code, summary = self.run_cli("not json", "Iter [1/1] loss: 1\n")
        self.assertEqual(code, 2)
        self.assertEqual(summary["status"], "error")


if __name__ == "__main__":
    unittest.main()
