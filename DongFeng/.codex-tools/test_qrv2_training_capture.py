from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import qrv2_training_capture as capture
from patch_soap_qrv2_training_capture import patch_source


class CaptureEnvironment:
    def __init__(self, root: Path, backend: str) -> None:
        self.values = {
            "QR_CAPTURE_DIR": str(root),
            "QR_CAPTURE_BACKEND": backend,
            "QR_CAPTURE_TARGET_STEP": "10",
            "QR_CAPTURE_TARGET_FACTOR": "0",
            "QR_CAPTURE_TARGET_SHAPE": "3x3",
            "QR_CAPTURE_MAX_PER_RANK": "1",
            "QR_CAPTURE_RUN_ID": "unit-test",
            "QR_CAPTURE_SOURCE_COMMIT": "0" * 40,
            "QR_CAPTURE_SOAP_SHA256": "1" * 64,
            "QR_CAPTURE_CONFIG_SHA256": "2" * 64,
            "QR_CAPTURE_CHECKPOINT_SHA256": "3" * 64,
            "QR_CAPTURE_SEED": "0",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "8",
            "ASCEND_RT_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15",
        }
        self.patch = mock.patch.dict(os.environ, self.values, clear=False)

    def __enter__(self) -> None:
        self.patch.start()
        capture._reset_for_tests()

    def __exit__(self, *args: object) -> None:
        self.patch.stop()


class TrainingCaptureTests(unittest.TestCase):
    def test_mx_input_exists_before_backend_and_nonfinite_output_is_saved(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with CaptureEnvironment(root, "mx"):
                def bad_mx(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    self.assertEqual(len(list(root.glob("*_input.pt"))), 1)
                    q = torch.eye(3)
                    q[:, 2] = torch.nan
                    r = torch.eye(3)
                    r[2, 2] = torch.nan
                    return q, r

                q, r = capture.qr(
                    torch.eye(3), mx_qr=bad_mx, optimizer_step=10,
                    factor_index=0, call_site="unit",
                )
                self.assertTrue(torch.isnan(q).any())
                self.assertTrue(torch.isnan(r).any())
            complete = json.loads(next(root.glob("*_complete.json")).read_text())
            self.assertEqual(complete["runtime"]["backend"], "mx")
            self.assertEqual(complete["q"]["nan"], 3)
            self.assertEqual(complete["r"]["nan"], 1)
            self.assertRegex(complete["input_file_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(len(list(root.glob("*_input.pt"))), 1)
            self.assertEqual(len(list(root.glob("*_output.pt"))), 1)

    def test_cpu_backend_uses_official_cpu_qr_and_returns_training_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            called = False
            with CaptureEnvironment(root, "cpu"):
                def forbidden_mx(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    nonlocal called
                    called = True
                    raise AssertionError("MX backend must not run")

                value = torch.randn(3, 3)
                q, r = capture.qr(
                    value, mx_qr=forbidden_mx, optimizer_step=10,
                    factor_index=0, call_site="unit",
                )
            self.assertFalse(called)
            self.assertEqual(q.device, value.device)
            self.assertEqual(r.device, value.device)
            self.assertTrue(torch.allclose(q @ r, value, rtol=1e-5, atol=1e-6))
            complete = json.loads(next(root.glob("*_complete.json")).read_text())
            self.assertEqual(complete["runtime"]["backend"], "cpu")
            self.assertEqual(complete["q"]["nan"], 0)
            self.assertEqual(complete["r"]["nan"], 0)

    def test_backend_exception_preserves_input_and_failure_record(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with CaptureEnvironment(root, "mx"):
                def failing(_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    raise RuntimeError("operator failure")

                with self.assertRaisesRegex(RuntimeError, "operator failure"):
                    capture.qr(
                        torch.eye(3), mx_qr=failing, optimizer_step=10,
                        factor_index=0, call_site="unit",
                    )
            self.assertEqual(len(list(root.glob("*_input.pt"))), 1)
            self.assertEqual(len(list(root.glob("*_failed.json"))), 1)
            self.assertEqual(len(list(root.glob("*_output.pt"))), 0)

    def test_non_target_call_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with CaptureEnvironment(root, "mx"):
                q, r = capture.qr(
                    torch.eye(3), mx_qr=torch.linalg.qr, optimizer_step=9,
                    factor_index=0, call_site="unit",
                )
            self.assertTrue(torch.equal(q @ r, torch.eye(3)))
            self.assertEqual(list(root.iterdir()), [])

    def test_wrong_visible_devices_fail_before_backend(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            called = False
            with CaptureEnvironment(root, "mx"):
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

                def forbidden(_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    nonlocal called
                    called = True
                    return torch.eye(3), torch.eye(3)

                with self.assertRaisesRegex(RuntimeError, "exact visible devices"):
                    capture.qr(
                        torch.eye(3), mx_qr=forbidden, optimizer_step=10,
                        factor_index=0, call_site="unit",
                    )
            self.assertFalse(called)

    def test_missing_source_identity_fails_before_backend(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with CaptureEnvironment(root, "mx"):
                del os.environ["QR_CAPTURE_SOURCE_COMMIT"]
                with self.assertRaisesRegex(RuntimeError, "QR_CAPTURE_SOURCE_COMMIT"):
                    capture.qr(
                        torch.eye(3), mx_qr=torch.linalg.qr, optimizer_step=10,
                        factor_index=0, call_site="unit",
                    )


class SoapPatchTests(unittest.TestCase):
    def test_exact_two_calls_are_wrapped_with_step_and_factor_context(self) -> None:
        source = """import torch
import mx_driving_cloud
class SOAP:
    def sync(self, state, precond_list):
        for ind, power_iter in enumerate(precond_list):
            Q, _ = mx_driving_cloud.linalg.qr(power_iter)
    def plan(self, state, precond_list):
        ind = 0
        plan = []
        plan.append({
                \"original_dtype\": precond_list[ind].dtype,
            })
        return plan
    def finish(self, entry):
        if entry is not None:
            Q, _ = mx_driving_cloud.linalg.qr(entry[\"power_iter\"])
"""
        patched = patch_source(source)
        self.assertEqual(patched.count("qrv2_training_capture.qr("), 2)
        self.assertIn("optimizer_step=int(state['step'])", patched)
        self.assertIn('optimizer_step=entry["optimizer_step"]', patched)
        compile(patched, "soap.py", "exec")

    def test_patch_refuses_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly two"):
            patch_source("import mx_driving_cloud\n")


if __name__ == "__main__":
    unittest.main()
