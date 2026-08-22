#!/usr/bin/env python3
"""CPU-only unit tests for qrv2_release_oracle.py."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from qrv2_release_oracle import (
    CaseSpec,
    align_call_manifests,
    align_release_call_manifests,
    build_call_manifest,
    core_case_specs,
    evaluate_qr_outputs,
    evaluate_downstream_stages,
    generate_case,
    is_mx_eligible,
    pad_exactly_like_production,
    public_output_shapes,
    release_expected_calls,
    load_step260_case,
    load_known_step260_case,
    run_wrapper_contract_call,
    tensor_sha256,
    validate_production_wrapper_source,
    write_manifest,
)


ROOT = Path(__file__).resolve().parent
WRAPPER = ROOT / "linalg_official_26.0.7.py"


def cpu_complete_qr(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.linalg.qr(value.clone(), mode="complete")


class TestWrapperContract(unittest.TestCase):
    def test_boundary_eligibility_and_output_shapes(self) -> None:
        self.assertFalse(is_mx_eligible((80, 81)))
        self.assertFalse(is_mx_eligible((81, 80)))
        self.assertTrue(is_mx_eligible((81, 81)))
        self.assertEqual(public_output_shapes((80, 81)), ((80, 80), (80, 81)))
        self.assertEqual(public_output_shapes((81, 80)), ((81, 80), (80, 80)))
        self.assertEqual(public_output_shapes((81, 129)), ((81, 81), (81, 129)))
        self.assertEqual(public_output_shapes((129, 81)), ((129, 129), (129, 81)))

    def test_padding_is_square_aligned_contiguous_and_zero(self) -> None:
        value = torch.arange(81 * 129, dtype=torch.float32).reshape(81, 129)
        padded = pad_exactly_like_production(value)
        self.assertEqual(tuple(padded.shape), (192, 192))
        self.assertTrue(padded.is_contiguous())
        self.assertTrue(torch.equal(padded[:81, :129], value))
        self.assertEqual(torch.count_nonzero(padded[81:, :]).item(), 0)
        self.assertEqual(torch.count_nonzero(padded[:, 129:]).item(), 0)

    def test_audited_wrapper_ast_contract(self) -> None:
        result = validate_production_wrapper_source(WRAPPER)
        self.assertEqual(result["gate"], "PASS")
        with tempfile.TemporaryDirectory(prefix="qrv2_wrapper_drift_") as raw:
            drifted = Path(raw) / "linalg.py"
            drifted.write_text(
                WRAPPER.read_text(encoding="utf-8").replace(
                    "QR_AICPU_THRESHOLD_SHAPE = 80", "QR_AICPU_THRESHOLD_SHAPE = 81"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "threshold"):
                validate_production_wrapper_source(drifted)
            drifted.write_text(
                WRAPPER.read_text(encoding="utf-8").replace(
                    "pad = BLOCK_TILING - (pad) if (pad) else 0", "pad = 0"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "missing statements"):
                validate_production_wrapper_source(drifted)


class TestDeterministicCases(unittest.TestCase):
    def test_all_kinds_are_repeatable_and_do_not_change_global_rng(self) -> None:
        torch.manual_seed(1234)
        state = torch.random.get_rng_state().clone()
        for kind in (
            "identity",
            "randn",
            "low_magnitude",
            "ill_conditioned",
            "rank_deficient",
        ):
            spec = CaseSpec(f"{kind}-7x5", (7, 5), kind, seed=7)
            first = generate_case(spec).tensor
            second = generate_case(spec).tensor
            self.assertEqual(tensor_sha256(first), tensor_sha256(second))
            self.assertEqual(first.dtype, torch.float32)
            self.assertTrue(first.is_contiguous())
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_rank_deficient_construction_has_expected_numerical_rank(self) -> None:
        generated = generate_case(CaseSpec("rank-deficient", (10, 8), "rank_deficient"))
        singular = torch.linalg.svdvals(generated.tensor.to(torch.float64))
        threshold = 10 * torch.finfo(torch.float64).eps * singular[0]
        observed = int((singular > threshold).sum().item())
        self.assertEqual(observed, 4)
        self.assertEqual(generated.generator["constructed_rank"], 4)

    def test_step260_loader_consumes_only_a_with_weights_only_contract(self) -> None:
        capture_root = ROOT.parent / "step260_qr_bad_tensors"
        for rank in range(8):
            source = capture_root / f"rank{rank}_step10_ind0_192x192_BAD.pt"
            loaded = load_known_step260_case(source)
            self.assertEqual(tuple(loaded.tensor.shape), (192, 192))
            self.assertEqual(loaded.spec.kind, "step260_capture")
            self.assertTrue(torch.isfinite(loaded.tensor).all().item())
            self.assertIn("weights_only=True", loaded.generator["load_policy"])
        source = capture_root / "rank0_step10_ind0_192x192_BAD.pt"
        with self.assertRaisesRegex(RuntimeError, "SHA"):
            load_step260_case(source, expected_sha256="0" * 64)

    def test_core_matrix_contains_all_required_boundaries(self) -> None:
        shapes = {spec.shape for spec in core_case_specs()}
        for shape in ((80, 81), (81, 80), (81, 81), (192, 192), (192, 256), (256, 192)):
            self.assertIn(shape, shapes)


class TestOracleAndHarness(unittest.TestCase):
    def test_cpu_standin_passes_fallback_and_mx_contracts(self) -> None:
        for call_index, shape in enumerate(((12, 9), (81, 83), (83, 81))):
            generated = generate_case(CaseSpec(f"randn-{shape}", shape, "randn"))
            q, r, record = run_wrapper_contract_call(
                generated,
                kernel=cpu_complete_qr,
                mode="fixed",
                rank=0,
                call_index=call_index,
            )
            self.assertEqual((tuple(q.shape), tuple(r.shape)), public_output_shapes(shape))
            self.assertTrue(record["input_unmodified"])
            self.assertTrue(record["contract_pass"])
            if is_mx_eligible(shape):
                self.assertEqual(record["padded_before"]["shape"], record["padded_after"]["shape"])
            else:
                self.assertIsNone(record["padded_before"])

    def test_padded_work_buffer_mutation_is_observed_but_not_rejected(self) -> None:
        def mutating_kernel(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            q, r = torch.linalg.qr(value.clone(), mode="complete")
            value.fill_(7.0)
            return q, r

        generated = generate_case(CaseSpec("mutating", (81, 81), "randn"))
        _, _, record = run_wrapper_contract_call(
            generated, kernel=mutating_kernel, mode="fixed", rank=0, call_index=0
        )
        self.assertNotEqual(
            record["padded_before"]["sha256"], record["padded_after"]["sha256"]
        )
        self.assertTrue(record["input_unmodified"])
        self.assertTrue(record["contract_pass"])

    def test_corrupt_q_fails_componentwise_oracle(self) -> None:
        value = generate_case(CaseSpec("bad-q", (9, 7), "randn")).tensor
        q, r = torch.linalg.qr(value, mode="reduced")
        q = q.clone()
        q[0, 0] += 0.25
        report = evaluate_qr_outputs(
            value, q, r, mode="reduced", require_exact_lower_zero=True
        )
        self.assertFalse(report["contract_pass"])
        self.assertGreater(
            report["reconstruction"]["violation_count"]
            + report["orthogonality"]["violation_count"],
            0,
        )

    def test_finite_early_return_attributes_input_q_and_r_independently(self) -> None:
        value = torch.eye(3, dtype=torch.float32)
        variants = (
            ("input", value.clone(), value.clone(), value.clone()),
            ("q", value.clone(), value.clone(), value.clone()),
            ("r", value.clone(), value.clone(), value.clone()),
        )
        for label, candidate_input, q, r in variants:
            with self.subTest(label=label):
                target = {"input": candidate_input, "q": q, "r": r}[label]
                target[0, 0] = float("nan")
                report = evaluate_qr_outputs(
                    candidate_input,
                    q,
                    r,
                    mode="complete",
                    require_exact_lower_zero=True,
                )
                self.assertFalse(report["finite_pass"])
                self.assertFalse(report[f"{label}_finite"])
                for other in {"input", "q", "r"} - {label}:
                    self.assertTrue(report[f"{other}_finite"])
                self.assertEqual(report["nonfinite_count"][label], 1)
                self.assertEqual(
                    sum(report["nonfinite_count"].values()),
                    1,
                )
                self.assertFalse(report["contract_pass"])
                self.assertNotIn("reconstruction", report)

    def test_full_rank_projection_is_required_and_bidirectional(self) -> None:
        value = torch.eye(129, 81, dtype=torch.float32)
        q, r = torch.linalg.qr(value, mode="complete")
        report = evaluate_qr_outputs(
            value, q, r, mode="complete", require_exact_lower_zero=True
        )
        projection = report["full_rank_projection"]
        self.assertTrue(projection["required"])
        self.assertTrue(projection["pass"])
        self.assertIn("candidate_to_reference", projection)
        self.assertIn("reference_to_candidate", projection)

    def test_kernel_output_dtype_is_not_silently_converted(self) -> None:
        def fp64_kernel(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            q, r = torch.linalg.qr(value.to(torch.float64), mode="complete")
            return q, r

        generated = generate_case(CaseSpec("wrong-dtype", (81, 81), "randn"))
        with self.assertRaisesRegex(ValueError, "kernel Q must be float32"):
            run_wrapper_contract_call(
                generated, kernel=fp64_kernel, mode="fixed", rank=0, call_index=0
            )

    def test_generic_downstream_control_envelope(self) -> None:
        reference = {"project": torch.ones((4, 3), dtype=torch.float32)}
        passing = {"project": reference["project"] + 1.0e-7}
        report = evaluate_downstream_stages(
            passing,
            reference,
            control_max_by_stage={"project": 0.0},
            reduction_dim_by_stage={"project": 4},
        )
        self.assertTrue(report["contract_pass"])
        failing = {"project": reference["project"] + 0.1}
        report = evaluate_downstream_stages(
            failing,
            reference,
            control_max_by_stage={"project": 0.0},
            reduction_dim_by_stage={"project": 4},
        )
        self.assertFalse(report["contract_pass"])


class TestManifestAlignment(unittest.TestCase):
    def _record(self, mode: str, call_index: int, shape: tuple[int, int]) -> dict:
        generated = generate_case(CaseSpec(f"case-{call_index}", shape, "randn"))
        return run_wrapper_contract_call(
            generated,
            kernel=cpu_complete_qr,
            mode=mode,
            rank=3,
            call_index=call_index,
        )[2]

    def test_original_fixed_align_by_rank_call_and_input_sha(self) -> None:
        original = build_call_manifest(
            "original", [self._record("original", 0, (81, 81)), self._record("original", 1, (12, 9))]
        )
        fixed = build_call_manifest(
            "fixed", [self._record("fixed", 0, (81, 81)), self._record("fixed", 1, (12, 9))]
        )
        expected = {(3, 0): "case-0", (3, 1): "case-1"}
        result = align_call_manifests(original, fixed, expected_calls=expected)
        self.assertEqual(result["gate"], "ALIGNMENT_PASS")
        self.assertEqual(result["aligned_call_count"], 2)

    def test_alignment_fails_on_sha_drift_or_missing_call(self) -> None:
        original = build_call_manifest("original", [self._record("original", 0, (81, 81))])
        fixed = build_call_manifest("fixed", [self._record("fixed", 0, (81, 81))])
        drifted = copy.deepcopy(fixed)
        drifted["calls"][0]["input_before"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "input_before.sha256"):
            align_call_manifests(original, drifted, expected_calls={(3, 0): "case-0"})
        missing = copy.deepcopy(fixed)
        missing["calls"] = []
        missing["call_count"] = 0
        with self.assertRaisesRegex(ValueError, "key sets"):
            align_call_manifests(original, missing, expected_calls={(3, 0): "case-0"})

    def test_alignment_rejects_empty_coverage_and_stride_drift(self) -> None:
        empty_original = build_call_manifest("original", [])
        empty_fixed = build_call_manifest("fixed", [])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            align_call_manifests(empty_original, empty_fixed, expected_calls={})
        original = build_call_manifest("original", [self._record("original", 0, (81, 81))])
        fixed = build_call_manifest("fixed", [self._record("fixed", 0, (81, 81))])
        fixed["calls"][0]["input_before"]["stride"] = [999, 1]
        with self.assertRaisesRegex(ValueError, "input_before.stride"):
            align_call_manifests(
                original, fixed, expected_calls={(3, 0): "case-0"}
            )

    def test_release_coverage_is_built_in_and_non_shrinkable(self) -> None:
        expected = release_expected_calls()
        case_ids = set(expected.values())
        step260_ids = {f"rank{rank}_step10_ind0_192x192" for rank in range(8)}
        self.assertTrue(step260_ids.issubset(case_ids))
        self.assertEqual(sum(case_id.startswith("state_default_stream") for case_id in case_ids), 6)
        self.assertEqual(sum(case_id.startswith("state_dedicated_stream") for case_id in case_ids), 6)
        for spec in core_case_specs():
            self.assertIn(spec.case_id, case_ids)
        empty_original = build_call_manifest("original", [])
        empty_fixed = build_call_manifest("fixed", [])
        with self.assertRaisesRegex(ValueError, "expected_calls"):
            align_release_call_manifests(empty_original, empty_fixed)

    def test_alignment_rejects_fixed_semantic_regression(self) -> None:
        original = build_call_manifest("original", [self._record("original", 0, (81, 81))])
        fixed = build_call_manifest("fixed", [self._record("fixed", 0, (81, 81))])
        regressed = copy.deepcopy(fixed)
        regressed["calls"][0]["contract_pass"] = False
        with self.assertRaisesRegex(ValueError, "mathematical contract"):
            align_call_manifests(
                original, regressed, expected_calls={(3, 0): "case-0"}
            )

    def test_manifest_writer_refuses_symlink(self) -> None:
        manifest = build_call_manifest("original", [])
        with tempfile.TemporaryDirectory(prefix="qrv2_manifest_") as raw:
            root = Path(raw)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "manifest.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                write_manifest(link, manifest)

    def test_manifest_writer_refuses_existing_regular_file(self) -> None:
        manifest = build_call_manifest("original", [])
        with tempfile.TemporaryDirectory(prefix="qrv2_manifest_existing_") as raw:
            destination = Path(raw) / "manifest.json"
            write_manifest(destination, manifest)
            with self.assertRaisesRegex(RuntimeError, "existing manifest"):
                write_manifest(destination, manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
