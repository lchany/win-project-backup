#!/usr/bin/env python3
"""Focused offline world8/math aggregation tests for STEP392."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import step392_delta2_host_case as host


def call(rank: int) -> dict:
    component = {"violation_count": 0, "max_abs": 0.0, "max_bound": 0.0, "max_scaled": 0.0}
    return {
        "case_id": f"step260_rank{rank}_profiled", "shape": [192, 192], "dtype": "torch.float32",
        "input_sha256": f"{rank:x}" * 64, "eligible_mx_branch": True, "mx_qr_call_delta": 1,
        "mx_qr_input": {"shape": [192, 192], "dtype": "torch.float32", "contiguous": True},
        "expected_padded_shape": [192, 192], "wrapper_branch": "mx_fixed", "public_qr_mode": "complete",
        "cpu_fp32_projection_control_max": 0.0, "input_unmodified": True, "elapsed_ms": 1.0,
        "contract_pass": True, "shape_pass": True, "input_finite": True, "q_finite": True, "r_finite": True,
        "nonfinite_count": {"input": 0, "q": 0, "r": 0}, "finite_pass": True,
        "reconstruction": copy.deepcopy(component), "orthogonality": copy.deepcopy(component),
        "lower_triangle_exact_zero": True, "lower_triangle_required": True,
        "fp64": {"candidate_reconstruction_relative_fro": 0.0, "candidate_orthogonality_relative_fro": 0.0, "reference_reconstruction_relative_fro": 0.0, "reference_orthogonality_relative_fro": 0.0, "numerical_rank": 192, "rank_threshold": 0.0},
        "full_rank_projection": {"required": True, "candidate_to_reference": {"relative_fro": 0.0, "relative_max": 0.0}, "reference_to_candidate": {"relative_fro": 0.0, "relative_max": 0.0}, "control_max": 0.0, "tolerance": 1e-6, "pass": True},
        "predicate_status": {key: "pass" for key in ("input_unmodified", "shape", "finite", "reconstruction", "orthogonality", "lower_triangle_exact_zero", "projection")},
        "failed_predicates": [], "not_evaluated_predicates": [], "diagnostic_scalars_finite": True,
        "diagnostic_nonfinite_scalar_count": 0, "reconstruction_violation_count": 0,
        "orthogonality_violation_count": 0, "projection_pass": True,
    }


def fixture(root: Path) -> tuple[Path, dict[int, str]]:
    output = root / "output"; output.mkdir(); (output / "done").mkdir()
    hashes = {}
    for rank in range(8):
        hashes[rank] = f"{rank:x}" * 64
        done = {"rank": rank, "local_rank": rank, "world_size": 8, "input_file_sha256": hashes[rank], "call_count": 1, "eligible_call_count": 1, "mx_qr_call_count": 1, "eligible_fallback_count": 0, "all_contract_pass": True, "profiler_identity_pass": True, "state_diagnostic_only": False, "first_profiled_only": True, "calls": [call(rank)]}
        identity = {"pass": True, "candidate_aic_reference_count": 1, "expected_aic_reference_count": 1, "referenced_qrv2_entries": ["QrV2_qa_position_delta2_only_diagnostic_v1_0_mix_aic"], "kernel_details_sources": [], "hash_dictionary_sources": [], "task_track_sources": [], "raw_profile_retained": True, "diagnostic_identity": "QrV2_qa_position_delta2_only_diagnostic_v1", "diagnostic_aic_task_reference_count": 1, "diagnostic_aiv_task_reference_count": 0, "original_task_reference_count": 0, "v4_task_reference_count": 0, "v5_task_reference_count": 0, "legacy_v6_task_reference_count": 0, "unknown_qrv2_task_reference_count": 0}
        (output / "done" / f"rank{rank}.json").write_text(json.dumps(done))
        (output / f"profiler_identity_rank{rank}.json").write_text(json.dumps(identity))
        profile = output / f"profile_rank{rank}"; profile.mkdir(); (profile / "raw.bin").write_bytes(b"profile")
    return output, hashes


class Step392HostContractTests(unittest.TestCase):
    def test_world8_exact_identity_and_cpu_oracle_negative_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, hashes = fixture(Path(directory))
            result = host.validate_outputs(output, hashes)
            self.assertEqual(result["concrete_aic_hit_count"], 8)
            self.assertEqual(result["concrete_aiv_hit_count"], 0)
            self.assertEqual(result["forbidden_identity_hit_count"], 0)
            self.assertEqual(set(result["math_gate"].values()), {True})
        mutations = (
            lambda done, identity: done["calls"][0].update({"input_finite": False}),
            lambda done, identity: done["calls"][0]["reconstruction"].update({"violation_count": 1}),
            lambda done, identity: done["calls"][0].update({"lower_triangle_exact_zero": False}),
            lambda done, identity: done["calls"][0]["full_rank_projection"].update({"pass": False}),
            lambda done, identity: identity.update({"referenced_qrv2_entries": ["QrV2_vtv_direct_qa_legacy_probe_v6_0_mix_aic"]}),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                output, hashes = fixture(Path(directory)); dp=output/"done/rank0.json"; ip=output/"profiler_identity_rank0.json"
                done=json.loads(dp.read_text()); identity=json.loads(ip.read_text()); mutate(done, identity); dp.write_text(json.dumps(done)); ip.write_text(json.dumps(identity))
                with self.assertRaises(RuntimeError): host.validate_outputs(output, hashes)


if __name__ == "__main__":
    unittest.main()
