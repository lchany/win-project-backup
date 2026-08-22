#!/usr/bin/env python3
"""Small CPU-only contract tests for STEP358 result serialization."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import build_qrv2_release as release
import qrv2_release_oracle as oracle
import step358_run_release_math_remote as controller
import step358_prepare_release_shadow as prepare
import step358_qrv2_release_math_worker as worker
import step358_host_case as host

worker.oracle = oracle


class Step358ResultContractTests(unittest.TestCase):
    @staticmethod
    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def set_nested(payload, path: tuple, value) -> None:
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    def summary_fixture(self, root: Path):
        done = []
        identities = []
        for rank, input_sha in enumerate(
            controller.EXPECTED_INPUT_FILE_SHA256_BY_RANK
        ):
            call = {
                "shape": [192, 192],
                "dtype": "torch.float32",
                "eligible_mx_branch": True,
                "wrapper_branch": "mx_fixed",
                "expected_padded_shape": [192, 192],
                "mx_qr_call_delta": 1,
                "mx_qr_input": {
                    "shape": [192, 192],
                    "dtype": "torch.float32",
                    "contiguous": True,
                },
                "input_unmodified": True,
                "shape_pass": True,
                "finite_pass": True,
                "reconstruction": {"violation_count": 0},
                "orthogonality": {"violation_count": 0},
            }
            done.append(
                {
                    "rank": rank,
                    "local_rank": rank,
                    "world_size": 8,
                    "input_file_sha256": input_sha,
                    "all_contract_pass": True,
                    "profiler_identity_pass": True,
                    "first_profiled_only": True,
                    "state_diagnostic_only": True,
                    "call_count": 1,
                    "eligible_call_count": 1,
                    "mx_qr_call_count": 1,
                    "eligible_fallback_count": 0,
                    "calls": [call],
                }
            )
            identities.append(
                {
                    "pass": True,
                    "candidate_aic_reference_count": 1,
                    "expected_aic_reference_count": 1,
                    "raw_profile_retained": True,
                }
            )
        controller_status = {
            "status": "PASS",
            "physical_device_ids": list(range(8, 16)),
        }
        self.persist_summary_fixture(root, done, identities, controller_status)
        return done, identities, controller_status

    def persist_summary_fixture(
        self,
        root: Path,
        done: list[dict],
        identities: list[dict],
        controller_status: dict,
    ) -> None:
        for rank, row in enumerate(done):
            self.write_json(root / "done" / f"rank{rank}.json", row)
        for rank, row in enumerate(identities):
            self.write_json(root / f"profiler_identity_rank{rank}.json", row)
        self.write_json(root / "controller_status.json", controller_status)
        self.write_json(root / "postflight_status.json", {"status": "PASS"})
        self.write_json(root / "finally_cleanup_status.json", {"status": "PASS"})

    def execute_summary(self, root: Path) -> dict:
        output = io.StringIO()
        argv = [
            "summary",
            str(root),
            json.dumps(controller.EXPECTED_INPUT_FILE_SHA256_BY_RANK),
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
            exec(controller.SUMMARY_CODE, {})
        return json.loads(output.getvalue())

    def host_args(self, root: Path) -> SimpleNamespace:
        shadow_root = root / "shadow"
        (shadow_root / "mx_driving_cloud/packages/vendors/customize").mkdir(
            parents=True
        )
        input_dir = root / "inputs"
        input_dir.mkdir()
        worker_path = root / "worker.py"
        worker_path.write_text("# test worker\n", encoding="utf-8")
        output_dir = root / "output"
        output_dir.mkdir()
        return SimpleNamespace(
            shadow_root=shadow_root,
            installed_custom_opp=Path("/opt/test-installed-customize"),
            input_dir=input_dir,
            worker=worker_path,
            output_dir=output_dir,
            port=34358,
            first_profiled_only=True,
            state_diagnostic_only=True,
        )

    def test_launcher_ownership_failures_always_run_cleanup_postflight(self) -> None:
        for failure_point in ("identity", "process_group", "ownership_write"):
            with self.subTest(failure_point=failure_point), tempfile.TemporaryDirectory(
                prefix="qrv2_launcher_cleanup_"
            ) as raw:
                args = self.host_args(Path(raw))
                process = mock.MagicMock(pid=43210, returncode=0)
                process.wait.return_value = 0
                cleanup = mock.Mock(return_value=0)
                identity_effect = (
                    RuntimeError("injected launcher identity failure")
                    if failure_point == "identity"
                    else None
                )
                atomic_effect = (
                    RuntimeError("injected ownership write failure")
                    if failure_point == "ownership_write"
                    else None
                )
                process_group_effect = (
                    RuntimeError("injected process group failure")
                    if failure_point == "process_group"
                    else None
                )
                with (
                    mock.patch.object(host.legacy, "preflight"),
                    mock.patch.object(host.subprocess, "Popen", return_value=process),
                    mock.patch.object(
                        host.legacy,
                        "process_starttime",
                        side_effect=identity_effect,
                        return_value=123456,
                    ),
                    mock.patch.object(
                        host.os,
                        "getpgid",
                        side_effect=process_group_effect,
                        return_value=43210,
                    ),
                    mock.patch.object(
                        host.legacy,
                        "atomic_json",
                        side_effect=atomic_effect,
                    ),
                    mock.patch.object(
                        host,
                        "snapshot_owned_npu_processes",
                        side_effect=RuntimeError("injected missing ownership snapshot"),
                    ),
                    mock.patch.object(host, "terminate_group") as terminate,
                    mock.patch.object(
                        host.legacy,
                        "cleanup_owned_and_postflight",
                        cleanup,
                    ),
                ):
                    self.assertEqual(host.run(args), 122)

                terminate.assert_called_once_with(process)
                cleanup.assert_called_once_with(args.output_dir, args.port)
                error = (args.output_dir / "controller_error.txt").read_text(
                    encoding="utf-8"
                )
                expected_primary = {
                    "identity": "injected launcher identity failure",
                    "process_group": "injected process group failure",
                    "ownership_write": "injected ownership write failure",
                }[failure_point]
                self.assertIn(expected_primary, error)
                self.assertIn("injected missing ownership snapshot", error)
                self.assertEqual(
                    (args.output_dir / "postflight_rc.txt").read_text(
                        encoding="utf-8"
                    ),
                    "0\n",
                )

    def test_cleanup_failures_do_not_replace_primary_ownership_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qrv2_launcher_error_priority_") as raw:
            args = self.host_args(Path(raw))
            process = mock.MagicMock(pid=43210, returncode=0)
            process.wait.return_value = 0
            with (
                mock.patch.object(host.legacy, "preflight"),
                mock.patch.object(host.subprocess, "Popen", return_value=process),
                mock.patch.object(host.legacy, "process_starttime", return_value=123456),
                mock.patch.object(host.os, "getpgid", return_value=43210),
                mock.patch.object(
                    host.legacy,
                    "atomic_json",
                    side_effect=PermissionError("ownership denied"),
                ),
                mock.patch.object(
                    host,
                    "snapshot_owned_npu_processes",
                    side_effect=RuntimeError("snapshot unavailable"),
                ),
                mock.patch.object(
                    host,
                    "terminate_group",
                    side_effect=RuntimeError("terminate failed"),
                ),
                mock.patch.object(
                    host.legacy,
                    "cleanup_owned_and_postflight",
                    side_effect=RuntimeError("postflight failed"),
                ) as cleanup,
            ):
                self.assertEqual(host.run(args), 122)

            cleanup.assert_called_once_with(args.output_dir, args.port)
            error = (args.output_dir / "controller_error.txt").read_text(
                encoding="utf-8"
            )
            self.assertTrue(
                error.startswith("PermissionError: ownership denied"), error
            )
            self.assertIn("termination failed: RuntimeError: terminate failed", error)
            postflight_error = (args.output_dir / "postflight_error.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("postflight failed", postflight_error)
            self.assertEqual(
                (args.output_dir / "postflight_rc.txt").read_text(encoding="utf-8"),
                "123\n",
            )

    def test_first_profiled_only_is_forwarded_and_summary_closes_to_one_call(self) -> None:
        tools = Path(__file__).resolve().parent
        worker_source = (tools / "step358_qrv2_release_math_worker.py").read_text(
            encoding="utf-8"
        )
        host_source = (tools / "step358_host_case.py").read_text(encoding="utf-8")
        controller_source = (
            tools / "step358_run_release_math_remote.py"
        ).read_text(encoding="utf-8")

        self.assertIn('parser.add_argument("--first-profiled-only"', worker_source)
        self.assertIn("if not args.first_profiled_only:", worker_source)
        self.assertIn("if rank == 0 and not args.first_profiled_only:", worker_source)
        self.assertIn('"first_profiled_only": args.first_profiled_only', worker_source)
        self.assertIn('parser.add_argument("--first-profiled-only"', host_source)
        self.assertIn(
            '(" --first-profiled-only" if args.first_profiled_only else "")',
            host_source,
        )
        self.assertIn(
            '" --state-diagnostic-only --first-profiled-only"', controller_source
        )
        self.assertIn("row['first_profiled_only'] is True", controller_source)
        self.assertIn("row['state_diagnostic_only'] is True", controller_source)
        self.assertIn("row['call_count']==1", controller_source)
        self.assertIn("row['eligible_call_count']==1", controller_source)
        self.assertIn("row['mx_qr_call_count']==1", controller_source)
        self.assertIn("row['input_unmodified'] is True", controller_source)
        self.assertIn("row['shape_pass'] is True", controller_source)
        self.assertIn("row['finite_pass'] is True", controller_source)
        self.assertIn("row['reconstruction']['violation_count']==0", controller_source)
        self.assertIn("row['orthogonality']['violation_count']==0", controller_source)
        compile(controller.SUMMARY_CODE, "<STEP366_SUMMARY_CODE>", "exec")
        self.assertEqual(
            controller.EXPECTED_RELEASE_KERNEL,
            "QrV2_matmul_position_fix_v5",
        )
        self.assertTrue(controller.V5_RELEASE_READY)
        self.assertEqual(
            controller.REMOTE_DIAG_NAME,
            "step374_qrv2_matmul_position_v5_first192_20260821",
        )
        self.assertIn(
            "step373_qrv2_matmul_position_v5_release_20260821",
            controller.REMOTE_WHEEL,
        )
        self.assertEqual(
            controller.REMOTE_WHEEL_SHA256,
            "f20c3db839b669ef6919b2a40df80b475676a9c0149910e0c73eb65064b8c11b",
        )
        self.assertEqual(controller.PORT, 34359)
        self.assertEqual(
            controller.EXPECTED_SHA256["step358_host_case.py"],
            controller.sha256_file(tools / "step358_host_case.py"),
        )

    def test_summary_executes_pass_with_exact_eight_rank_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qrv2_summary_pass_") as raw:
            root = Path(raw)
            self.summary_fixture(root)
            result = self.execute_summary(root)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["physical_device_ids"], list(range(8, 16)))
        self.assertEqual(result["shape_by_rank"], [[192, 192]] * 8)
        self.assertEqual(result["dtype_by_rank"], ["torch.float32"] * 8)
        self.assertEqual(result["eligible_mx_branch_by_rank"], [True] * 8)
        self.assertEqual(result["wrapper_branch_by_rank"], ["mx_fixed"] * 8)
        self.assertEqual(
            result["input_file_sha256_by_rank"],
            controller.EXPECTED_INPUT_FILE_SHA256_BY_RANK,
        )
        self.assertEqual(result["raw_profile_retained_by_rank"], [True] * 8)
        self.assertTrue(result["raw_profiles_retained"])

    def test_summary_rejects_each_exact_case_identity_drift(self) -> None:
        mutations = (
            ("shape", "done", (0, "calls", 0, "shape"), [193, 193]),
            ("dtype", "done", (0, "calls", 0, "dtype"), "torch.float16"),
            (
                "eligible",
                "done",
                (0, "calls", 0, "eligible_mx_branch"),
                False,
            ),
            (
                "wrapper",
                "done",
                (0, "calls", 0, "wrapper_branch"),
                "torch_npu_boundary_fallback",
            ),
            (
                "internal_shape",
                "done",
                (0, "calls", 0, "mx_qr_input", "shape"),
                [256, 256],
            ),
            (
                "internal_dtype",
                "done",
                (0, "calls", 0, "mx_qr_input", "dtype"),
                "torch.float16",
            ),
            (
                "internal_contiguous",
                "done",
                (0, "calls", 0, "mx_qr_input", "contiguous"),
                False,
            ),
            ("input_sha", "done", (0, "input_file_sha256"), "0" * 64),
            (
                "physical_devices",
                "controller",
                ("physical_device_ids",),
                list(range(8)),
            ),
            (
                "raw_profile",
                "identity",
                (0, "raw_profile_retained"),
                False,
            ),
        )
        for label, target_name, path, value in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                prefix=f"qrv2_summary_{label}_"
            ) as raw:
                root = Path(raw)
                done, identities, controller_status = self.summary_fixture(root)
                targets = {
                    "done": done,
                    "identity": identities,
                    "controller": controller_status,
                }
                self.set_nested(targets[target_name], path, value)
                self.persist_summary_fixture(
                    root, done, identities, controller_status
                )
                with self.assertRaises(AssertionError):
                    self.execute_summary(root)

    def test_outer_failure_preserves_primary_and_appends_bounded_recovery_errors(self) -> None:
        with (
            mock.patch.object(
                controller,
                "host_script",
                side_effect=[
                    TimeoutError("case command timed out"),
                    (9, "", "cleanup command failed"),
                ],
            ) as remote_host,
            mock.patch.object(
                controller,
                "container_python",
                side_effect=PermissionError("inventory unavailable"),
            ),
        ):
            with self.assertRaises(RuntimeError) as caught:
                controller._run_case_with_recovery(
                    object(), "run case", "/installed/cloud", {}, "/diag/case"
                )

        message = str(caught.exception)
        self.assertTrue(message.startswith("TimeoutError: case command timed out"))
        self.assertIn("secondary installed inventory check: PermissionError", message)
        self.assertIn("secondary owned cleanup: RuntimeError: owned cleanup rc=9", message)
        cleanup_call = remote_host.call_args_list[1]
        cleanup_command = cleanup_call.args[1]
        self.assertIn("--cleanup-owned", cleanup_command)
        self.assertIn("--output-dir /diag/case/run", cleanup_command)
        self.assertNotIn("pkill", cleanup_command)
        self.assertNotIn("killall", cleanup_command)
        self.assertEqual(cleanup_call.kwargs, {"timeout": 120, "check": False})

    def test_outer_nonzero_case_runs_owned_cleanup_without_replacing_primary(self) -> None:
        installed = {"closed": True}
        with (
            mock.patch.object(
                controller,
                "host_script",
                side_effect=[(122, "", "host gate failed"), (0, "PASS", "")],
            ) as remote_host,
            mock.patch.object(
                controller,
                "container_python",
                return_value=(0, json.dumps(installed), ""),
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "^STEP358 host gate rc=122: host gate failed$"
            ):
                controller._run_case_with_recovery(
                    object(), "run case", "/installed/cloud", installed, "/diag/case"
                )
        self.assertEqual(remote_host.call_count, 2)

    def test_outer_success_does_not_repeat_owned_cleanup(self) -> None:
        installed = {"closed": True}
        with (
            mock.patch.object(
                controller, "host_script", return_value=(0, "PASS", "")
            ) as remote_host,
            mock.patch.object(
                controller,
                "container_python",
                return_value=(0, json.dumps(installed), ""),
            ),
        ):
            controller._run_case_with_recovery(
                object(), "run case", "/installed/cloud", installed, "/diag/case"
            )
        remote_host.assert_called_once_with(
            mock.ANY, "run case", timeout=1000, check=False
        )

    def test_v5_controller_unready_override_fails_before_machine_mapping(self) -> None:
        module = SimpleNamespace(parse_machine_info=mock.Mock())
        with mock.patch.object(controller, "V5_RELEASE_READY", False):
            with self.assertRaisesRegex(RuntimeError, "v5 release wheel has not been built"):
                controller.local_preflight(module)
        module.parse_machine_info.assert_not_called()

    def test_v5_controller_rejects_known_v4_wheel_before_machine_mapping(self) -> None:
        module = SimpleNamespace(parse_machine_info=mock.Mock())
        cases = (
            (
                "/diagnostics/step370_qrv2_lifetime_alpha_sync_v4_release/wheel.whl",
                "1" * 64,
                "v4 wheel path",
            ),
            (
                "/diagnostics/step373_qrv2_matmul_position_v5_release/wheel.whl",
                controller.FORBIDDEN_V4_WHEEL_SHA256,
                "v4 wheel SHA",
            ),
        )
        for wheel, digest, error in cases:
            with (
                self.subTest(error=error),
                mock.patch.object(controller, "V5_RELEASE_READY", True),
                mock.patch.object(controller, "REMOTE_WHEEL", wheel),
                mock.patch.object(controller, "REMOTE_WHEEL_SHA256", digest),
            ):
                with self.assertRaisesRegex(RuntimeError, error):
                    controller.local_preflight(module)
        module.parse_machine_info.assert_not_called()

    def test_release_identity_is_unique_v5_across_build_shadow_and_worker(self) -> None:
        expected = "QrV2_matmul_position_fix_v5"
        self.assertEqual(release.BIN_NAME, expected)
        self.assertEqual(prepare.EXPECTED_KERNEL, expected)
        self.assertEqual(prepare.EXPECTED_AIC, expected + "_0_mix_aic")
        self.assertEqual(prepare.EXPECTED_AIV, expected + "_0_mix_aiv")
        self.assertEqual(worker.CANDIDATE_AIC, expected + "_0_mix_aic")
        self.assertEqual(worker.CANDIDATE_AIV, expected + "_0_mix_aiv")
        self.assertEqual(controller.EXPECTED_RELEASE_KERNEL, expected)

    def finalize(
        self,
        value: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        *,
        a: torch.Tensor | None = None,
        a_before: torch.Tensor | None = None,
        case_id: str = "failure_contract",
    ) -> dict:
        eligible = oracle.is_mx_eligible(tuple(value.shape))
        records = []
        if eligible:
            padded = oracle.padded_length(tuple(value.shape))
            records = [
                {
                    "shape": [padded, padded],
                    "dtype": "torch.float32",
                    "contiguous": True,
                }
            ]
        return worker._finalize_call(
            torch,
            value,
            value.clone() if a is None else a,
            value.clone() if a_before is None else a_before,
            q,
            r,
            1.0,
            case_id=case_id,
            eligible=eligible,
            mx_call_records=records,
        )

    def failure_record(
        self, value: torch.Tensor, q: torch.Tensor, r: torch.Tensor, *, case_id: str
    ) -> dict:
        prefix = "QrV2 public math contract failed: "
        with self.assertRaisesRegex(
            RuntimeError, "^QrV2 public math contract failed: "
        ) as caught:
            self.finalize(value, q, r, case_id=case_id)
        message = str(caught.exception)
        self.assertTrue(message.startswith(prefix), message)
        record = json.loads(message[len(prefix) :])
        serialized = json.dumps(record, sort_keys=True, allow_nan=False)
        self.assertEqual(json.loads(serialized), record)
        return record

    def check_case(self, shape: tuple[int, int]) -> dict:
        generated = oracle.generate_case(
            oracle.CaseSpec("field_contract", shape, "randn", 358)
        )
        value = generated.tensor
        mode = "complete" if oracle.is_mx_eligible(shape) else "reduced"
        q, r = torch.linalg.qr(value, mode=mode)
        return self.finalize(
            value,
            q,
            r,
            case_id=f"field_contract_{shape[0]}x{shape[1]}",
        )

    def test_reduced_and_complete_result_fields_match_oracle_schema(self) -> None:
        for shape in ((81, 80), (81, 81)):
            with self.subTest(shape=shape):
                result = self.check_case(shape)
                self.assertTrue(result["contract_pass"])
                self.assertTrue(result["lower_triangle_exact_zero"])
                self.assertIn("fp64", result)
                self.assertIn("full_rank_projection", result)
                self.assertNotIn("lower_triangle", result)
                self.assertNotIn("fp64_summary", result)

    def test_shape_early_return_preserves_original_failure_and_missing_state(self) -> None:
        value = oracle.generate_case(
            oracle.CaseSpec("shape_early", (81, 81), "randn", 359)
        ).tensor
        q, r = torch.linalg.qr(value, mode="complete")
        record = self.failure_record(value, q[:, :-1], r, case_id="shape_early")

        self.assertFalse(record["shape_pass"])
        self.assertTrue(record["finite_pass"])
        self.assertEqual(record["failed_predicates"], ["shape"])
        for predicate in (
            "reconstruction",
            "orthogonality",
            "lower_triangle_exact_zero",
            "projection",
        ):
            self.assertEqual(record["predicate_status"][predicate], "not_evaluated")
        for field in (
            "reconstruction",
            "orthogonality",
            "lower_triangle_exact_zero",
            "fp64",
            "full_rank_projection",
        ):
            self.assertIsNone(record[field])

    def test_finite_early_return_preserves_original_failure_and_missing_state(self) -> None:
        value = oracle.generate_case(
            oracle.CaseSpec("finite_early", (81, 81), "randn", 360)
        ).tensor
        q, r = torch.linalg.qr(value, mode="complete")
        q[0, 0] = float("nan")
        record = self.failure_record(value, q, r, case_id="finite_early")

        self.assertTrue(record["shape_pass"])
        self.assertFalse(record["finite_pass"])
        self.assertTrue(record["input_finite"])
        self.assertFalse(record["q_finite"])
        self.assertTrue(record["r_finite"])
        self.assertEqual(record["nonfinite_count"], {"input": 0, "q": 1, "r": 0})
        self.assertEqual(record["failed_predicates"], ["finite"])
        self.assertEqual(record["predicate_status"]["shape"], "pass")
        self.assertEqual(record["predicate_status"]["finite"], "fail")
        self.assertIsNone(record["reconstruction"])
        self.assertIsNone(record["orthogonality"])
        self.assertIsNone(record["fp64"])
        self.assertIsNone(record["full_rank_projection"])

    def test_finite_outputs_with_overflowing_diagnostics_remain_json_safe(self) -> None:
        value = oracle.generate_case(
            oracle.CaseSpec("finite_overflow", (81, 81), "randn", 362)
        ).tensor
        maximum = torch.finfo(torch.float32).max
        q = torch.full((81, 81), maximum, dtype=torch.float32)
        r = torch.full((81, 81), maximum, dtype=torch.float32)
        record = self.failure_record(value, q, r, case_id="finite_overflow")

        self.assertTrue(record["input_finite"])
        self.assertTrue(record["q_finite"])
        self.assertTrue(record["r_finite"])
        self.assertTrue(record["finite_pass"])
        self.assertFalse(record["diagnostic_scalars_finite"])
        self.assertGreater(record["diagnostic_nonfinite_scalar_count"], 0)
        self.assertEqual(
            record["reconstruction"]["max_abs"],
            {"finite": False, "value": "positive_infinity"},
        )
        error = worker.QrContractFailure(record)
        with tempfile.TemporaryDirectory(prefix="qrv2_overflow_summary_") as raw:
            path = Path(raw) / "failure" / "rank0.json"
            path.parent.mkdir()
            worker._persist_failure_summary(path, error)
            persisted = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(persisted, record)

    def test_scalar_failure_summary_is_persisted_as_strict_json(self) -> None:
        summary = {
            "case_id": "persisted",
            "input_finite": True,
            "q_finite": False,
            "r_finite": True,
            "nonfinite_count": {"input": 0, "q": 1, "r": 0},
        }
        with tempfile.TemporaryDirectory(prefix="qrv2_failure_summary_") as raw:
            path = Path(raw) / "failure" / "rank0.json"
            path.parent.mkdir()
            worker._persist_failure_summary(path, worker.QrContractFailure(summary))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), summary)
            with self.assertRaises(FileExistsError):
                worker._persist_failure_summary(path, worker.QrContractFailure(summary))

    def test_failure_artifact_faults_never_escape_or_replace_original(self) -> None:
        error = worker.QrContractFailure({"case_id": "fault_injection"})
        original = "ORIGINAL_QR_TRACEBACK\n"

        def fail_directory(*_args, **_kwargs) -> None:
            raise PermissionError("mkdir denied")

        reported: list[str] = []
        directory_errors = worker._persist_failure_artifacts(
            Path("/unused/failure/rank0.txt"),
            error,
            original,
            make_directory=fail_directory,
            report_error=reported.append,
        )
        self.assertEqual(len(directory_errors), 1)
        self.assertIn("failure_directory: PermissionError", directory_errors[0])
        self.assertEqual(reported, directory_errors)

        with tempfile.TemporaryDirectory(prefix="qrv2_persist_faults_") as raw:
            failure = Path(raw) / "summary" / "rank0.txt"

            def fail_summary(_path: Path, _error: worker.QrContractFailure) -> None:
                raise OSError("summary full")

            reported = []
            summary_errors = worker._persist_failure_artifacts(
                failure,
                error,
                original,
                write_summary=fail_summary,
                report_error=reported.append,
            )
            self.assertEqual(len(summary_errors), 1)
            self.assertIn("failure_summary: OSError", summary_errors[0])
            traceback_text = failure.read_text(encoding="utf-8")
            self.assertTrue(traceback_text.startswith(original))
            self.assertIn(summary_errors[0], traceback_text)
            self.assertEqual(reported, summary_errors)

        with tempfile.TemporaryDirectory(prefix="qrv2_traceback_fault_") as raw:
            failure = Path(raw) / "traceback" / "rank0.txt"

            def fail_traceback(_path: Path, _payload: str) -> None:
                raise OSError("traceback full")

            reported = []
            traceback_errors = worker._persist_failure_artifacts(
                failure,
                error,
                original,
                write_traceback=fail_traceback,
                report_error=reported.append,
            )
            self.assertEqual(len(traceback_errors), 1)
            self.assertIn("failure_traceback: OSError", traceback_errors[0])
            self.assertEqual(reported, traceback_errors)
            self.assertTrue(failure.with_suffix(".json").is_file())
            self.assertFalse(failure.exists())

    def test_full_failure_report_contains_all_diagnostics_and_failed_predicates(self) -> None:
        value = oracle.generate_case(
            oracle.CaseSpec("full_failure", (81, 81), "randn", 361)
        ).tensor
        q, r = torch.linalg.qr(value, mode="complete")
        r[1, 0] = 1.0
        record = self.failure_record(value, q, r, case_id="full_failure")

        self.assertEqual(record["predicate_status"]["shape"], "pass")
        self.assertEqual(record["predicate_status"]["finite"], "pass")
        self.assertEqual(record["predicate_status"]["reconstruction"], "fail")
        self.assertEqual(
            record["predicate_status"]["lower_triangle_exact_zero"], "fail"
        )
        self.assertIn("reconstruction", record["failed_predicates"])
        self.assertIn("lower_triangle_exact_zero", record["failed_predicates"])
        self.assertEqual(record["not_evaluated_predicates"], [])
        for field in (
            "reconstruction",
            "orthogonality",
            "fp64",
            "full_rank_projection",
        ):
            self.assertIsInstance(record[field], dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
