#!/usr/bin/env python3
"""Local, deterministic contracts for the MX QrV2 release validation.

This module intentionally imports only CPU PyTorch.  A future device harness can
inject the real ``mx_driving_cloud._C.qr`` as ``kernel`` without changing input
generation, padding, oracle, or original/fixed manifest alignment semantics.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as functional


BLOCK_TILING = 64
QR_AICPU_THRESHOLD_SHAPE = 80
MANIFEST_SCHEMA = "qrv2-release-call-manifest-v1"
SUPPORTED_CASE_KINDS = (
    "identity",
    "randn",
    "low_magnitude",
    "ill_conditioned",
    "rank_deficient",
)
CAPTURED_CASE_KIND = "step260_capture"
STEP260_INPUT_SHA256 = {
    "rank0_step10_ind0_192x192_BAD.pt": "23ad9198223159fc6aa67f79642c299fd86e0aaa2b7ae72bdea297fcb023ab55",
    "rank1_step10_ind0_192x192_BAD.pt": "2cb99d06aa9c96d61f0b615cf41fa579bd6779f7f97c97fa84693180c32adb5b",
    "rank2_step10_ind0_192x192_BAD.pt": "61dcbad02578e60ce7bb82b837f0b33fff2e0071fbde530a339dcad1ce2a692d",
    "rank3_step10_ind0_192x192_BAD.pt": "89266a246497f51d1c6db5e698ee1442abc91bd48c7dc539a09d2373c21b3ac1",
    "rank4_step10_ind0_192x192_BAD.pt": "e750ddcc8dd892ece49d04873910752c657f6d853f8e698daf03fa3fce3a73ca",
    "rank5_step10_ind0_192x192_BAD.pt": "bbceebf84c574e21e9262774c41e0c8bb5eb7f5add0d0cf123e4efbd6a95dc68",
    "rank6_step10_ind0_192x192_BAD.pt": "f2091ec0c618721ba95452fcca82288a2fc8148f40718f945a9e80646dd1d766",
    "rank7_step10_ind0_192x192_BAD.pt": "3dcc3f2bdb7945eaac7ce246128804dfecd89d381e27dc108e99d90d2df2121c",
}


@dataclass(frozen=True)
class CaseSpec:
    """A reproducible CPU input specification."""

    case_id: str
    shape: tuple[int, int]
    kind: str
    seed: int = 0

    def __post_init__(self) -> None:
        m, n = self.shape
        if m <= 0 or n <= 0:
            raise ValueError(f"case shape must be positive, got {self.shape!r}")
        if self.kind not in {*SUPPORTED_CASE_KINDS, CAPTURED_CASE_KIND}:
            raise ValueError(f"unsupported case kind: {self.kind!r}")
        if not self.case_id:
            raise ValueError("case_id must not be empty")


@dataclass(frozen=True)
class GeneratedCase:
    """A generated tensor plus provenance that can be put in a manifest."""

    spec: CaseSpec
    tensor: torch.Tensor
    generator: Mapping[str, Any]


def _require_cpu_float32_matrix(value: torch.Tensor, label: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{label} must be on CPU for the local oracle")
    if value.dtype != torch.float32:
        raise ValueError(f"{label} must be float32, got {value.dtype}")
    if value.ndim != 2:
        raise ValueError(f"{label} must be 2D, got shape={tuple(value.shape)}")


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash logical tensor values in contiguous row-major order."""

    if value.device.type != "cpu":
        value = value.detach().cpu()
    raw = value.detach().contiguous().numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def tensor_contract(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "stride": list(value.stride()),
        "numel": value.numel(),
        "sha256": tensor_sha256(value),
    }


def is_mx_eligible(shape: Sequence[int]) -> bool:
    if len(shape) != 2:
        raise ValueError(f"QR input must be 2D, got shape={tuple(shape)}")
    return min(int(shape[0]), int(shape[1])) > QR_AICPU_THRESHOLD_SHAPE


def padded_length(shape: Sequence[int]) -> int:
    if len(shape) != 2:
        raise ValueError(f"QR input must be 2D, got shape={tuple(shape)}")
    lda = max(int(shape[0]), int(shape[1]))
    return math.ceil(lda / BLOCK_TILING) * BLOCK_TILING


def public_output_shapes(shape: Sequence[int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(shape) != 2:
        raise ValueError(f"QR input must be 2D, got shape={tuple(shape)}")
    m, n = (int(shape[0]), int(shape[1]))
    if is_mx_eligible(shape):
        return (m, m), (m, n)
    k = min(m, n)
    return (m, k), (k, n)


def pad_exactly_like_production(value: torch.Tensor) -> torch.Tensor:
    """Reproduce the audited production wrapper's square, 64-aligned padding."""

    _require_cpu_float32_matrix(value, "value")
    m, n = value.shape
    length = padded_length(value.shape)
    padding = (0, length - n, 0, length - m)
    return functional.pad(value, padding).contiguous()


def _seeded_randn(shape: tuple[int, int], seed: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype)


def _orthogonal(rows: int, seed: int) -> torch.Tensor:
    matrix = _seeded_randn((rows, rows), seed, torch.float64)
    q, _ = torch.linalg.qr(matrix, mode="complete")
    return q


def generate_case(spec: CaseSpec) -> GeneratedCase:
    """Generate a case without reading or modifying global RNG state."""

    m, n = spec.shape
    if spec.kind == CAPTURED_CASE_KIND:
        raise ValueError("captured STEP260 cases must be loaded with load_step260_case")
    k = min(m, n)
    generator: dict[str, Any] = {
        "algorithm": spec.kind,
        "seed": spec.seed,
        "torch_version": torch.__version__,
        "generation_dtype": "float32",
    }
    if spec.kind == "identity":
        value = torch.eye(m, n, dtype=torch.float32)
    elif spec.kind == "randn":
        value = _seeded_randn(spec.shape, spec.seed, torch.float32)
    elif spec.kind == "low_magnitude":
        value = _seeded_randn(spec.shape, spec.seed, torch.float32) * 1.0e-8
        generator["scale"] = 1.0e-8
    else:
        if spec.kind == "ill_conditioned":
            u = _orthogonal(m, spec.seed)
            v = _orthogonal(n, spec.seed + 1)
            rank = k
            singular_values = torch.logspace(0.0, -6.0, rank, dtype=torch.float64)
            value64 = (u[:, :rank] * singular_values.unsqueeze(0)) @ v[:, :rank].mT
            value = value64.to(torch.float32).contiguous()
            generator.update(
                {
                    "factor_seed_u": spec.seed,
                    "factor_seed_v": spec.seed + 1,
                    "constructed_rank": rank,
                    "singular_log10_min": -6,
                }
            )
        else:
            rank = k // 2
            value = torch.zeros((m, n), dtype=torch.float32)
            if rank:
                diagonal = torch.logspace(0.0, -3.0, rank, dtype=torch.float32)
                indices = torch.arange(rank)
                value[indices, indices] = diagonal
            generator.update(
                {
                    "algorithm": "diagonal_exact_rank",
                    "constructed_rank": rank,
                    "singular_log10_min": -3,
                }
            )
    return GeneratedCase(spec=spec, tensor=value.contiguous(), generator=generator)


def make_case_specs(
    shapes: Iterable[tuple[int, int]], kinds: Iterable[str] = SUPPORTED_CASE_KINDS
) -> tuple[CaseSpec, ...]:
    return tuple(
        CaseSpec(
            case_id=f"{kind}_m{m}_n{n}_seed0",
            shape=(m, n),
            kind=kind,
            seed=0,
        )
        for m, n in shapes
        for kind in kinds
    )


def core_case_specs() -> tuple[CaseSpec, ...]:
    shapes = (
        (80, 81),
        (81, 80),
        (81, 81),
        (127, 127),
        (128, 128),
        (129, 129),
        (191, 191),
        (192, 192),
        (193, 193),
        (256, 256),
        (129, 81),
        (81, 129),
        (193, 129),
        (129, 193),
        (192, 256),
        (256, 192),
    )
    return make_case_specs(shapes)


def load_step260_case(path: Path, *, expected_sha256: str) -> GeneratedCase:
    """Load only the captured STEP260 input A through torch's weights-only path."""

    raw = path.read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if file_sha256 != expected_sha256:
        raise RuntimeError(f"STEP260 file SHA mismatch: {path.name}")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "A" not in payload:
        raise RuntimeError(f"STEP260 payload lacks tensor A: {path.name}")
    value = payload["A"]
    _require_cpu_float32_matrix(value, "STEP260 A")
    if tuple(value.shape) != (192, 192):
        raise RuntimeError(f"unexpected STEP260 A shape: {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"STEP260 input A is non-finite: {path.name}")
    case_id = path.stem.removesuffix("_BAD")
    return GeneratedCase(
        spec=CaseSpec(case_id=case_id, shape=(192, 192), kind=CAPTURED_CASE_KIND, seed=0),
        tensor=value.contiguous().clone(),
        generator={
            "algorithm": "step260_capture",
            "source_basename": path.name,
            "source_file_sha256": file_sha256,
            "load_policy": "torch.load(weights_only=True,map_location=cpu); consume key A only",
        },
    )


def load_known_step260_case(path: Path) -> GeneratedCase:
    expected = STEP260_INPUT_SHA256.get(path.name)
    if expected is None:
        raise RuntimeError(f"unrecognized STEP260 capture basename: {path.name}")
    return load_step260_case(path, expected_sha256=expected)


def _relative_fro(error: torch.Tensor, reference: torch.Tensor) -> float:
    tiny = torch.finfo(reference.dtype).tiny
    numerator = torch.linalg.vector_norm(error)
    denominator = torch.clamp(torch.linalg.vector_norm(reference), min=tiny)
    return float((numerator / denominator).item())


def _relative_max(error: torch.Tensor, reference: torch.Tensor) -> float:
    tiny = torch.finfo(reference.dtype).tiny
    numerator = error.abs().max()
    denominator = torch.clamp(reference.abs().max(), min=tiny)
    return float((numerator / denominator).item())


def _componentwise_summary(error: torch.Tensor, bound: torch.Tensor) -> dict[str, Any]:
    violations = torch.logical_or(~torch.isfinite(error), error > bound)
    safe_bound = torch.clamp(bound, min=torch.finfo(bound.dtype).tiny)
    return {
        "violation_count": int(violations.sum().item()),
        "max_abs": float(error.max().item()),
        "max_bound": float(bound.max().item()),
        "max_scaled": float((error / safe_bound).max().item()),
    }


def evaluate_qr_outputs(
    expected_a: torch.Tensor,
    q: torch.Tensor,
    r: torch.Tensor,
    *,
    mode: str,
    require_exact_lower_zero: bool,
    projection_control_max: float = 0.0,
) -> dict[str, Any]:
    """Evaluate FP32 outputs and record an independent FP64 oracle summary."""

    _require_cpu_float32_matrix(expected_a, "expected_a")
    _require_cpu_float32_matrix(q, "q")
    _require_cpu_float32_matrix(r, "r")
    if mode not in {"reduced", "complete"}:
        raise ValueError(f"unsupported QR mode: {mode!r}")

    m, n = expected_a.shape
    q_columns = min(m, n) if mode == "reduced" else m
    expected_q_shape = (m, q_columns)
    expected_r_shape = (q_columns, n)
    shape_pass = tuple(q.shape) == expected_q_shape and tuple(r.shape) == expected_r_shape
    input_finite_mask = torch.isfinite(expected_a)
    q_finite_mask = torch.isfinite(q)
    r_finite_mask = torch.isfinite(r)
    input_finite = bool(input_finite_mask.all().item())
    q_finite = bool(q_finite_mask.all().item())
    r_finite = bool(r_finite_mask.all().item())
    # Keep the release predicate unchanged; the component fields only make an
    # early return attributable to input A, Q, or R without serializing tensors.
    finite_pass = bool(input_finite and q_finite and r_finite)
    result: dict[str, Any] = {
        "mode": mode,
        "expected_q_shape": list(expected_q_shape),
        "expected_r_shape": list(expected_r_shape),
        "actual_q_shape": list(q.shape),
        "actual_r_shape": list(r.shape),
        "shape_pass": shape_pass,
        "input_finite": input_finite,
        "q_finite": q_finite,
        "r_finite": r_finite,
        "nonfinite_count": {
            "input": int((~input_finite_mask).sum().item()),
            "q": int((~q_finite_mask).sum().item()),
            "r": int((~r_finite_mask).sum().item()),
        },
        "finite_pass": finite_pass,
    }
    if not shape_pass or not finite_pass:
        result["contract_pass"] = False
        return result

    q_ref32, r_ref32 = torch.linalg.qr(expected_a, mode=mode)
    unit_roundoff = torch.finfo(torch.float32).eps / 2.0
    gamma_q = (q_columns * unit_roundoff) / (1.0 - q_columns * unit_roundoff)
    gamma_m = (m * unit_roundoff) / (1.0 - m * unit_roundoff)
    tiny32 = torch.finfo(torch.float32).tiny

    reconstruction_error = (q @ r - expected_a).abs()
    reconstruction_bound = 10.0 * (
        (q_ref32 @ r_ref32 - expected_a).abs() + gamma_q * (q.abs() @ r.abs())
    ) + tiny32
    identity = torch.eye(q_columns, dtype=torch.float32)
    orthogonality_error = (q.mT @ q - identity).abs()
    orthogonality_bound = 10.0 * (
        (q_ref32.mT @ q_ref32 - identity).abs() + gamma_m * (q.abs().mT @ q.abs())
    ) + tiny32
    reconstruction = _componentwise_summary(reconstruction_error, reconstruction_bound)
    orthogonality = _componentwise_summary(orthogonality_error, orthogonality_bound)
    lower = torch.tril(r, diagonal=-1)
    lower_exact_zero = bool(torch.count_nonzero(lower).item() == 0)

    expected64 = expected_a.to(torch.float64)
    q_ref64, r_ref64 = torch.linalg.qr(expected64, mode=mode)
    reference_reconstruction64 = q_ref64 @ r_ref64 - expected64
    identity64 = torch.eye(q_columns, dtype=torch.float64)
    reference_orthogonality64 = q_ref64.mT @ q_ref64 - identity64
    candidate_q64 = q.to(torch.float64)
    candidate_r64 = r.to(torch.float64)
    candidate_reconstruction64 = candidate_q64 @ candidate_r64 - expected64
    candidate_orthogonality64 = candidate_q64.mT @ candidate_q64 - identity64
    singular_values = torch.linalg.svdvals(expected64)
    if singular_values.numel() and float(singular_values[0].item()) != 0.0:
        rank_threshold = max(m, n) * torch.finfo(torch.float64).eps * singular_values[0]
        numerical_rank = int((singular_values > rank_threshold).sum().item())
    else:
        rank_threshold = torch.tensor(0.0, dtype=torch.float64)
        numerical_rank = 0

    full_rank = numerical_rank == min(m, n)
    if not math.isfinite(projection_control_max) or projection_control_max < 0.0:
        raise ValueError("projection_control_max must be finite and non-negative")
    projection: dict[str, Any] = {"required": full_rank}
    projection_pass = True
    if full_rank:
        k = min(m, n)
        candidate_projection = candidate_q64[:, :k] @ candidate_q64[:, :k].mT
        reference_projection = q_ref64[:, :k] @ q_ref64[:, :k].mT
        projection_error = candidate_projection - reference_projection
        gamma_k = (k * unit_roundoff) / (1.0 - k * unit_roundoff)
        projection_tolerance = max(
            1.0e-6, 10.0 * projection_control_max, 10.0 * gamma_k
        )
        candidate_to_reference = {
            "relative_fro": _relative_fro(projection_error, reference_projection),
            "relative_max": _relative_max(projection_error, reference_projection),
        }
        reference_to_candidate = {
            "relative_fro": _relative_fro(projection_error, candidate_projection),
            "relative_max": _relative_max(projection_error, candidate_projection),
        }
        projection_pass = all(
            metric <= projection_tolerance
            for direction in (candidate_to_reference, reference_to_candidate)
            for metric in direction.values()
        )
        projection.update(
            {
                "candidate_to_reference": candidate_to_reference,
                "reference_to_candidate": reference_to_candidate,
                "control_max": projection_control_max,
                "tolerance": projection_tolerance,
                "pass": projection_pass,
            }
        )
    lower_pass = lower_exact_zero if require_exact_lower_zero else True
    result.update(
        {
            "reconstruction": reconstruction,
            "orthogonality": orthogonality,
            "lower_triangle_exact_zero": lower_exact_zero,
            "lower_triangle_required": require_exact_lower_zero,
            "fp64": {
                "candidate_reconstruction_relative_fro": _relative_fro(
                    candidate_reconstruction64, expected64
                ),
                "candidate_orthogonality_relative_fro": _relative_fro(
                    candidate_orthogonality64, identity64
                ),
                "reference_reconstruction_relative_fro": _relative_fro(
                    reference_reconstruction64, expected64
                ),
                "reference_orthogonality_relative_fro": _relative_fro(
                    reference_orthogonality64, identity64
                ),
                "numerical_rank": numerical_rank,
                "rank_threshold": float(rank_threshold.item()),
            },
            "full_rank_projection": projection,
            "contract_pass": (
                reconstruction["violation_count"] == 0
                and orthogonality["violation_count"] == 0
                and lower_pass
                and projection_pass
            ),
        }
    )
    return result


def evaluate_downstream_stages(
    candidate_stages: Mapping[str, torch.Tensor],
    reference_stages: Mapping[str, torch.Tensor],
    *,
    control_max_by_stage: Mapping[str, float],
    reduction_dim_by_stage: Mapping[str, int],
) -> dict[str, Any]:
    """Apply the plan's generic SOAP-stage control envelope.

    The caller owns the real ``project -> precondition -> project_back ->
    parameter_delta`` integration and must provide the maximum pairwise result
    from three identical torch-vs-torch controls for every stage.
    """

    expected_names = set(reference_stages)
    if not expected_names or set(candidate_stages) != expected_names:
        raise ValueError("candidate/reference downstream stage sets must match and be non-empty")
    if set(control_max_by_stage) != expected_names or set(reduction_dim_by_stage) != expected_names:
        raise ValueError("control and reduction-dimension stage sets must match tensors")
    unit_roundoff = torch.finfo(torch.float32).eps / 2.0
    reports: dict[str, Any] = {}
    all_pass = True
    for name in sorted(expected_names):
        candidate = candidate_stages[name]
        reference = reference_stages[name]
        _require_cpu_float32_matrix(candidate, f"candidate stage {name}")
        _require_cpu_float32_matrix(reference, f"reference stage {name}")
        if tuple(candidate.shape) != tuple(reference.shape):
            raise ValueError(f"downstream stage {name} shape mismatch")
        if not bool(torch.isfinite(candidate).all().item() and torch.isfinite(reference).all().item()):
            raise ValueError(f"downstream stage {name} contains non-finite values")
        control_max = float(control_max_by_stage[name])
        reduction_dim = int(reduction_dim_by_stage[name])
        if not math.isfinite(control_max) or control_max < 0.0:
            raise ValueError(f"downstream stage {name} has invalid control_max")
        if reduction_dim <= 0 or reduction_dim * unit_roundoff >= 1.0:
            raise ValueError(f"downstream stage {name} has invalid reduction dimension")
        gamma_d = (reduction_dim * unit_roundoff) / (
            1.0 - reduction_dim * unit_roundoff
        )
        tolerance = max(1.0e-6, 10.0 * control_max, 10.0 * gamma_d)
        error = candidate.to(torch.float64) - reference.to(torch.float64)
        reference64 = reference.to(torch.float64)
        rel_f = _relative_fro(error, reference64)
        rel_max = _relative_max(error, reference64)
        stage_pass = rel_f <= tolerance and rel_max <= tolerance
        reports[name] = {
            "relative_fro": rel_f,
            "relative_max": rel_max,
            "control_max": control_max,
            "reduction_dim": reduction_dim,
            "tolerance": tolerance,
            "pass": stage_pass,
        }
        all_pass = all_pass and stage_pass
    return {"contract_pass": all_pass, "stages": reports}


def validate_production_wrapper_source(path: Path) -> dict[str, Any]:
    """Fail closed if the audited wrapper's padding/crop contract drifts."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    constants: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, int):
                    constants[target.id] = node.value.value
    if constants.get("BLOCK_TILING") != BLOCK_TILING:
        raise RuntimeError("production BLOCK_TILING differs from audited value 64")
    if constants.get("QR_AICPU_THRESHOLD_SHAPE") != QR_AICPU_THRESHOLD_SHAPE:
        raise RuntimeError("production QR threshold differs from audited value 80")

    forward: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "QR":
            forward = next(
                (
                    child
                    for child in node.body
                    if isinstance(child, ast.FunctionDef) and child.name == "forward"
                ),
                None,
            )
            break
    if forward is None:
        raise RuntimeError("production QR.forward function is missing")

    statements = {ast.unparse(node) for node in ast.walk(forward) if isinstance(node, ast.stmt)}
    required = {
        "if dim[0] <= QR_AICPU_THRESHOLD_SHAPE or dim[1] <= QR_AICPU_THRESHOLD_SHAPE:\n    return torch.linalg.qr(A)",
        "lda = max(dim[0], dim[1])",
        "pad = lda % BLOCK_TILING",
        "pad = BLOCK_TILING - pad if pad else 0",
        "lda_pad = lda + pad",
        "padding = (0, pad_n, 0, pad_m)",
        "A = F.pad(A, padding).contiguous()",
        "Q = Q[:dim[0], :dim[0]]",
        "R = R[:dim[0], :dim[1]]",
        "R = torch.triu(R)",
    }
    missing = sorted(required - statements)
    if missing:
        raise RuntimeError(f"production wrapper contract drift: missing statements {missing!r}")

    def dotted_name(node: ast.AST) -> str | None:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        parts.append(node.id)
        return ".".join(reversed(parts))

    kernel_call_found = False
    for node in ast.walk(forward):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Tuple):
            continue
        target_names = [item.id for item in target.elts if isinstance(item, ast.Name)]
        call = node.value
        if (
            target_names == ["Q", "R"]
            and isinstance(call, ast.Call)
            and dotted_name(call.func) == "mx_driving_cloud._C.qr"
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "A"
            and not call.keywords
        ):
            kernel_call_found = True
            break
    if not kernel_call_found:
        raise RuntimeError("production wrapper _C.qr assignment contract drift")
    return {
        "gate": "PASS",
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "block_tiling": BLOCK_TILING,
        "threshold": QR_AICPU_THRESHOLD_SHAPE,
    }


Kernel = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def run_wrapper_contract_call(
    generated: GeneratedCase,
    *,
    kernel: Kernel,
    mode: str,
    rank: int,
    call_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Run one call and return JSON-safe evidence for later original/fixed A/B."""

    if mode not in {"original", "fixed"}:
        raise ValueError("mode must be 'original' or 'fixed'")
    if rank < 0 or call_index < 0:
        raise ValueError("rank and call_index must be non-negative")
    value = generated.tensor
    _require_cpu_float32_matrix(value, "generated.tensor")
    original_before = value.clone()
    input_before = tensor_contract(original_before)
    branch = "mx" if is_mx_eligible(value.shape) else "torch"

    padded_before_contract: dict[str, Any] | None = None
    padded_after_contract: dict[str, Any] | None = None
    internal_report: dict[str, Any] | None = None
    if branch == "torch":
        q_public, r_public = torch.linalg.qr(value, mode="reduced")
    else:
        padded = pad_exactly_like_production(value)
        padded_reference = torch.zeros(
            (padded_length(value.shape), padded_length(value.shape)), dtype=torch.float32
        )
        padded_reference[: value.shape[0], : value.shape[1]].copy_(value)
        if not torch.equal(padded, padded_reference):
            raise RuntimeError("test padding differs from independent zero-pad reference")
        padded_before = padded.clone()
        padded_before_contract = tensor_contract(padded_before)
        q_padded, r_padded = kernel(padded)
        _require_cpu_float32_matrix(q_padded, "kernel Q")
        _require_cpu_float32_matrix(r_padded, "kernel R")
        padded_after_contract = tensor_contract(padded)
        internal_report = evaluate_qr_outputs(
            padded_before,
            q_padded,
            r_padded,
            mode="complete",
            require_exact_lower_zero=False,
        )
        m, n = value.shape
        q_public = q_padded[:m, :m]
        r_public = torch.triu(r_padded[:m, :n])

    _require_cpu_float32_matrix(q_public, "public Q")
    _require_cpu_float32_matrix(r_public, "public R")
    q_public = q_public.detach().contiguous()
    r_public = r_public.detach().contiguous()
    original_after = tensor_contract(value)
    input_unmodified = input_before["sha256"] == original_after["sha256"]
    public_mode = "complete" if branch == "mx" else "reduced"
    public_report = evaluate_qr_outputs(
        original_before,
        q_public,
        r_public,
        mode=public_mode,
        require_exact_lower_zero=True,
    )
    record = {
        "mode": mode,
        "rank": rank,
        "call_index": call_index,
        "case_id": generated.spec.case_id,
        "case_spec": asdict(generated.spec),
        "generator": dict(generated.generator),
        "branch": branch,
        "input_before": input_before,
        "input_after": original_after,
        "input_unmodified": input_unmodified,
        "padded_before": padded_before_contract,
        "padded_after": padded_after_contract,
        "padded_input_may_be_work_buffer": branch == "mx",
        "q_public": tensor_contract(q_public),
        "r_public": tensor_contract(r_public),
        "internal_math": internal_report,
        "public_math": public_report,
        "contract_pass": bool(
            input_unmodified
            and public_report["contract_pass"]
            and (internal_report is None or internal_report["contract_pass"])
        ),
    }
    return q_public, r_public, record


def build_call_manifest(mode: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if mode not in {"original", "fixed"}:
        raise ValueError("manifest mode must be 'original' or 'fixed'")
    normalized = [dict(record) for record in records]
    keys: set[tuple[int, int]] = set()
    for record in normalized:
        if record.get("mode") != mode:
            raise ValueError("record mode differs from manifest mode")
        key = (int(record["rank"]), int(record["call_index"]))
        if key in keys:
            raise ValueError(f"duplicate rank/call_index key: {key!r}")
        keys.add(key)
    normalized.sort(key=lambda row: (int(row["rank"]), int(row["call_index"])))
    return {"schema": MANIFEST_SCHEMA, "mode": mode, "call_count": len(normalized), "calls": normalized}


def release_expected_calls() -> dict[tuple[int, int], str]:
    """Authoritative, non-shrinkable Section 6.3 release coverage schedule."""

    expected: dict[tuple[int, int], str] = {}
    rank0_index = 0
    for spec in core_case_specs():
        expected[(0, rank0_index)] = spec.case_id
        rank0_index += 1
    for rank in range(8):
        index = rank0_index if rank == 0 else 0
        expected[(rank, index)] = f"rank{rank}_step10_ind0_192x192"
        if rank == 0:
            rank0_index += 1
    state_shapes = (96, 192, 256, 192, 512, 192)
    for context in ("default_stream", "dedicated_stream"):
        for ordinal, size in enumerate(state_shapes):
            expected[(0, rank0_index)] = (
                f"state_{context}_{ordinal}_randn_m{size}_n{size}_seed0"
            )
            rank0_index += 1
    return expected


def align_release_call_manifests(
    original: Mapping[str, Any], fixed: Mapping[str, Any]
) -> dict[str, Any]:
    result = align_call_manifests(
        original, fixed, expected_calls=release_expected_calls()
    )
    result["gate"] = "PASS"
    result["coverage_profile"] = "section-6.3-release-v1"
    return result


def align_call_manifests(
    original: Mapping[str, Any],
    fixed: Mapping[str, Any],
    *,
    expected_calls: Mapping[tuple[int, int], str],
) -> dict[str, Any]:
    """Require exact per-rank/call input closure without comparing raw Q/R."""

    for manifest, expected_mode in ((original, "original"), (fixed, "fixed")):
        if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("mode") != expected_mode:
            raise ValueError(f"invalid {expected_mode} manifest header")
        if int(manifest.get("call_count", -1)) != len(manifest.get("calls", [])):
            raise ValueError(f"invalid {expected_mode} manifest call_count")
    original_rows = {(row["rank"], row["call_index"]): row for row in original["calls"]}
    fixed_rows = {(row["rank"], row["call_index"]): row for row in fixed["calls"]}
    if len(original_rows) != len(original["calls"]) or len(fixed_rows) != len(fixed["calls"]):
        raise ValueError("manifest contains duplicate rank/call_index keys")
    if original_rows.keys() != fixed_rows.keys():
        raise ValueError("original/fixed call key sets differ")
    if not expected_calls:
        raise ValueError("expected_calls must be a non-empty authoritative coverage contract")
    if original_rows.keys() != expected_calls.keys():
        raise ValueError("manifests do not exactly cover expected_calls")
    for key, expected_case_id in expected_calls.items():
        if original_rows[key].get("case_id") != expected_case_id:
            raise ValueError(f"call {key!r} does not match expected case_id")

    fields = ("case_id", "branch")
    nested_fields = (
        ("input_before", "shape"),
        ("input_before", "dtype"),
        ("input_before", "stride"),
        ("input_before", "numel"),
        ("input_before", "sha256"),
        ("padded_before", "shape"),
        ("padded_before", "dtype"),
        ("padded_before", "stride"),
        ("padded_before", "numel"),
        ("padded_before", "sha256"),
    )
    for key in sorted(original_rows):
        left, right = original_rows[key], fixed_rows[key]
        for field in fields:
            if left.get(field) != right.get(field):
                raise ValueError(f"call {key!r} differs at {field}")
        for parent, field in nested_fields:
            left_parent, right_parent = left.get(parent), right.get(parent)
            if (left_parent is None) != (right_parent is None):
                raise ValueError(f"call {key!r} differs at {parent}")
            if left_parent is not None and left_parent.get(field) != right_parent.get(field):
                raise ValueError(f"call {key!r} differs at {parent}.{field}")
    repaired_calls = 0
    stable_valid_calls = 0
    metric_observations: list[dict[str, Any]] = []
    for key in sorted(original_rows):
        left, right = original_rows[key], fixed_rows[key]
        if not bool(right.get("contract_pass")):
            raise ValueError(f"fixed call {key!r} fails the mathematical contract")
        if bool(left.get("contract_pass")):
            stable_valid_calls += 1
        else:
            repaired_calls += 1
        observation: dict[str, Any] = {"rank": key[0], "call_index": key[1]}
        for report_name in ("internal_math", "public_math"):
            left_report, right_report = left.get(report_name), right.get(report_name)
            if (left_report is None) != (right_report is None):
                raise ValueError(f"call {key!r} differs at {report_name} presence")
            if right_report is None:
                continue
            if not bool(right_report.get("contract_pass")):
                raise ValueError(f"fixed call {key!r} fails {report_name}")
            report_metrics: dict[str, Any] = {}
            for metric in ("reconstruction", "orthogonality"):
                left_metric, right_metric = left_report.get(metric), right_report.get(metric)
                if right_metric is None or int(right_metric.get("violation_count", -1)) != 0:
                    raise ValueError(f"fixed call {key!r} fails {report_name}.{metric}")
                report_metrics[metric] = {
                    "original_max_scaled": (
                        None if left_metric is None else left_metric.get("max_scaled")
                    ),
                    "fixed_max_scaled": right_metric.get("max_scaled"),
                    "fixed_within_oracle_bound": float(right_metric["max_scaled"]) <= 1.0,
                }
                if not report_metrics[metric]["fixed_within_oracle_bound"]:
                    raise ValueError(
                        f"fixed call {key!r} exceeds {report_name}.{metric} oracle bound"
                    )
            observation[report_name] = report_metrics
        metric_observations.append(observation)
    return {
        "gate": "ALIGNMENT_PASS",
        "schema": "qrv2-release-ab-alignment-v1",
        "aligned_call_count": len(original_rows),
        "stable_valid_call_count": stable_valid_calls,
        "repaired_call_count": repaired_calls,
        "keys": [[rank, call_index] for rank, call_index in sorted(original_rows)],
        "semantic_metrics": metric_observations,
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Atomically create a JSON manifest; existing destinations are immutable."""

    if path.is_symlink():
        raise RuntimeError(f"refusing to overwrite symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite existing manifest: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _cpu_kernel(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.linalg.qr(value, mode="complete")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper", type=Path, help="audited production linalg.py source")
    parser.add_argument("--shape", nargs=2, type=int, metavar=("M", "N"), default=(81, 81))
    parser.add_argument("--kind", choices=SUPPORTED_CASE_KINDS, default="randn")
    args = parser.parse_args()
    if args.wrapper:
        validate_production_wrapper_source(args.wrapper)
    shape = tuple(args.shape)
    spec = CaseSpec(f"{args.kind}_m{shape[0]}_n{shape[1]}_seed0", shape, args.kind)
    generated = generate_case(spec)
    _, _, record = run_wrapper_contract_call(
        generated, kernel=_cpu_kernel, mode="fixed", rank=0, call_index=0
    )
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if record["contract_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
