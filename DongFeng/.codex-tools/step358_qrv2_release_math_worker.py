#!/usr/bin/env python3
"""Eight-rank NPU math/identity worker for the QrV2 v5 release gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


VISIBLE = "8,9,10,11,12,13,14,15"
CANDIDATE_AIC = "QrV2_matmul_position_fix_v5_0_mix_aic"
CANDIDATE_AIV = "QrV2_matmul_position_fix_v5_0_mix_aiv"
ORIGINAL_AIC = "QrV2_566c2e1c0e6c8c92152ad84416d77006_0_mix_aic"
ORIGINAL_AIV = "QrV2_566c2e1c0e6c8c92152ad84416d77006_0_mix_aiv"


class QrContractFailure(RuntimeError):
    """Carry the JSON-safe oracle summary to the rank failure artifact."""

    def __init__(self, summary: dict[str, Any]) -> None:
        self.summary = summary
        super().__init__(
            "QrV2 public math contract failed: "
            + json.dumps(summary, sort_keys=True, allow_nan=False)
        )


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        return True
    except ValueError:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_profile(profile_root: Path, *, expected_aic_references: int) -> dict[str, Any]:
    names, kernel_sources = legacy.collect_profile_names(profile_root)
    if not any(Path(name.strip()).name == "QrV2" for name in names):
        raise RuntimeError("profiler contains no generic QrV2 execution")
    mappings, references, dictionary_sources, task_sources = legacy.collect_runtime_identity(
        profile_root
    )
    referenced = {
        name: references[hash_value]
        for hash_value, name in mappings.items()
        if references[hash_value] > 0 and name.startswith("QrV2")
    }
    if referenced.get(CANDIDATE_AIC, 0) != expected_aic_references:
        raise RuntimeError(
            "candidate concrete AIC task count does not close to profiled _C.qr calls: "
            f"expected={expected_aic_references}, "
            f"actual={referenced.get(CANDIDATE_AIC, 0)}"
        )
    allowed = {CANDIDATE_AIC, CANDIDATE_AIV}
    forbidden = {name: count for name, count in referenced.items() if name not in allowed}
    if forbidden or ORIGINAL_AIC in referenced or ORIGINAL_AIV in referenced:
        raise RuntimeError(f"unexpected task-referenced QrV2 identity: {sorted(forbidden)}")
    return {
        "pass": True,
        "candidate_aic_reference_count": referenced[CANDIDATE_AIC],
        "expected_aic_reference_count": expected_aic_references,
        "referenced_qrv2_entries": sorted(referenced),
        "kernel_details_sources": kernel_sources,
        "hash_dictionary_sources": dictionary_sources,
        "task_track_sources": task_sources,
        "raw_profile_retained": True,
    }


def _predicate_status(value: Any) -> str:
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return "not_evaluated"


def _normalize_json_diagnostic(value: Any) -> tuple[Any, int]:
    """Recursively replace non-finite floats with explicit JSON-safe records."""

    if isinstance(value, float):
        if math.isfinite(value):
            return value, 0
        if math.isnan(value):
            label = "nan"
        elif value > 0:
            label = "positive_infinity"
        else:
            label = "negative_infinity"
        return {"finite": False, "value": label}, 1
    if isinstance(value, dict):
        normalized: dict[Any, Any] = {}
        nonfinite_count = 0
        for key, item in value.items():
            normalized_item, item_count = _normalize_json_diagnostic(item)
            normalized[key] = normalized_item
            nonfinite_count += item_count
        return normalized, nonfinite_count
    if isinstance(value, (list, tuple)):
        normalized_items = []
        nonfinite_count = 0
        for item in value:
            normalized_item, item_count = _normalize_json_diagnostic(item)
            normalized_items.append(normalized_item)
            nonfinite_count += item_count
        return normalized_items, nonfinite_count
    return value, 0


def _persist_failure_summary(path: Path, error: QrContractFailure) -> None:
    """Create, but never replace, the scalar-only rank failure JSON."""

    payload = (
        json.dumps(error.summary, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_failure_traceback(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _best_effort_stderr(message: str) -> None:
    try:
        os.write(2, (message + "\n").encode("utf-8", errors="replace"))
    except BaseException:
        pass


def _persist_failure_artifacts(
    failure: Path,
    error: BaseException,
    original_traceback: str,
    *,
    make_directory: Callable[..., Any] = Path.mkdir,
    write_summary: Callable[[Path, QrContractFailure], None] = _persist_failure_summary,
    write_traceback: Callable[[Path, str], None] = _write_failure_traceback,
    report_error: Callable[[str], None] = _best_effort_stderr,
) -> list[str]:
    """Best-effort persistence that never replaces the original exception."""

    persistence_errors: list[str] = []
    try:
        make_directory(failure.parent, parents=True, exist_ok=True)
    except BaseException as persistence_error:
        persistence_errors.append(
            "failure_directory: "
            f"{type(persistence_error).__name__}: {persistence_error}"
        )
        report_error(persistence_errors[-1])
        return persistence_errors

    if isinstance(error, QrContractFailure):
        try:
            write_summary(failure.with_suffix(".json"), error)
        except BaseException as persistence_error:
            persistence_errors.append(
                "failure_summary: "
                f"{type(persistence_error).__name__}: {persistence_error}"
            )

    traceback_payload = original_traceback
    if persistence_errors:
        traceback_payload += (
            "\nFailure-artifact persistence errors did not replace the original "
            "exception:\n" + "\n".join(persistence_errors) + "\n"
        )
    try:
        write_traceback(failure, traceback_payload)
    except BaseException as persistence_error:
        persistence_errors.append(
            "failure_traceback: "
            f"{type(persistence_error).__name__}: {persistence_error}"
        )
    for message in persistence_errors:
        report_error(message)
    return persistence_errors


def _qr_failure_summary(
    report: dict[str, Any],
    *,
    case_id: str,
    shape: list[int],
    mode: str,
    input_unmodified: bool,
    projection_control_max: float,
) -> dict[str, Any]:
    """Build a JSON-safe summary even when the oracle returned early."""

    reconstruction = report.get("reconstruction")
    orthogonality = report.get("orthogonality")
    projection = report.get("full_rank_projection")
    reconstruction_pass = (
        reconstruction.get("violation_count") == 0
        if isinstance(reconstruction, dict)
        else None
    )
    orthogonality_pass = (
        orthogonality.get("violation_count") == 0
        if isinstance(orthogonality, dict)
        else None
    )
    if isinstance(projection, dict):
        projection_pass = (
            projection.get("pass")
            if projection.get("required") is True
            else True
        )
    else:
        projection_pass = None
    predicate_status = {
        "input_unmodified": _predicate_status(input_unmodified),
        "shape": _predicate_status(report.get("shape_pass")),
        "finite": _predicate_status(report.get("finite_pass")),
        "reconstruction": _predicate_status(reconstruction_pass),
        "orthogonality": _predicate_status(orthogonality_pass),
        "lower_triangle_exact_zero": _predicate_status(
            report.get("lower_triangle_exact_zero")
        ),
        "projection": _predicate_status(projection_pass),
    }
    raw_summary = {
        "case_id": case_id,
        "shape": shape,
        "mode": mode,
        "input_unmodified": input_unmodified,
        "contract_pass": report.get("contract_pass"),
        "shape_pass": report.get("shape_pass"),
        "input_finite": report.get("input_finite"),
        "q_finite": report.get("q_finite"),
        "r_finite": report.get("r_finite"),
        "nonfinite_count": report.get("nonfinite_count"),
        "finite_pass": report.get("finite_pass"),
        "reconstruction": reconstruction,
        "reconstruction_violation_count": (
            reconstruction.get("violation_count")
            if isinstance(reconstruction, dict)
            else None
        ),
        "reconstruction_max_scaled": (
            reconstruction.get("max_scaled")
            if isinstance(reconstruction, dict)
            else None
        ),
        "orthogonality": orthogonality,
        "orthogonality_violation_count": (
            orthogonality.get("violation_count")
            if isinstance(orthogonality, dict)
            else None
        ),
        "orthogonality_max_scaled": (
            orthogonality.get("max_scaled")
            if isinstance(orthogonality, dict)
            else None
        ),
        "lower_triangle_exact_zero": report.get("lower_triangle_exact_zero"),
        "lower_triangle_required": report.get("lower_triangle_required"),
        "fp64": report.get("fp64"),
        "full_rank_projection": projection,
        "projection_pass": projection_pass,
        "cpu_fp32_projection_control_max": projection_control_max,
        "predicate_status": predicate_status,
        "failed_predicates": sorted(
            name for name, status in predicate_status.items() if status == "fail"
        ),
        "not_evaluated_predicates": sorted(
            name
            for name, status in predicate_status.items()
            if status == "not_evaluated"
        ),
    }
    normalized_summary, diagnostic_nonfinite_count = _normalize_json_diagnostic(
        raw_summary
    )
    if not isinstance(normalized_summary, dict):
        raise RuntimeError("normalized QR failure summary must remain a dictionary")
    normalized_summary["diagnostic_scalars_finite"] = (
        diagnostic_nonfinite_count == 0
    )
    normalized_summary["diagnostic_nonfinite_scalar_count"] = (
        diagnostic_nonfinite_count
    )
    json.dumps(normalized_summary, sort_keys=True, allow_nan=False)
    return normalized_summary


def _finalize_call(
    torch: Any,
    a_cpu: Any,
    a: Any,
    a_before: Any,
    q: Any,
    r: Any,
    elapsed_ms: float,
    *,
    case_id: str,
    eligible: bool,
    mx_call_records: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_delta = 1 if eligible else 0
    if len(mx_call_records) != expected_delta:
        raise RuntimeError(
            "public wrapper branch ledger mismatch: "
            f"eligible={eligible}, records={len(mx_call_records)}"
        )
    expected_padded_shape: list[int] | None = None
    if eligible:
        lda = max(int(a_cpu.shape[0]), int(a_cpu.shape[1]))
        lda_pad = ((lda + 63) // 64) * 64
        expected_padded_shape = [lda_pad, lda_pad]
        record = mx_call_records[0]
        if (
            record.get("shape") != expected_padded_shape
            or record.get("dtype") != "torch.float32"
            or record.get("contiguous") is not True
        ):
            raise RuntimeError(
                "internal _C.qr input contract mismatch: "
                f"expected={expected_padded_shape}, actual={record}"
            )
    input_unmodified = bool(torch.equal(a, a_before))
    q_cpu = q.detach().cpu().contiguous()
    r_cpu = r.detach().cpu().contiguous()
    public_mode = "complete" if eligible else "reduced"
    q_control, r_control = torch.linalg.qr(a_cpu, mode=public_mode)
    control_probe = oracle.evaluate_qr_outputs(
        a_cpu,
        q_control,
        r_control,
        mode=public_mode,
        require_exact_lower_zero=True,
        projection_control_max=1.0,
    )
    if (
        control_probe["shape_pass"] is not True
        or control_probe["finite_pass"] is not True
        or control_probe["reconstruction"]["violation_count"] != 0
        or control_probe["orthogonality"]["violation_count"] != 0
        or control_probe["lower_triangle_exact_zero"] is not True
    ):
        raise RuntimeError("official CPU FP32 QR control failed its base contract")
    projection = control_probe["full_rank_projection"]
    projection_control_max = 0.0
    if projection.get("required"):
        projection_control_max = max(
            float(projection[direction][metric])
            for direction in ("candidate_to_reference", "reference_to_candidate")
            for metric in ("relative_fro", "relative_max")
        )
    report = oracle.evaluate_qr_outputs(
        a_cpu,
        q_cpu,
        r_cpu,
        mode=public_mode,
        require_exact_lower_zero=True,
        projection_control_max=projection_control_max,
    )
    if not input_unmodified or not report["contract_pass"]:
        failure_summary = _qr_failure_summary(
            report,
            case_id=case_id,
            shape=list(a_cpu.shape),
            mode=public_mode,
            input_unmodified=input_unmodified,
            projection_control_max=projection_control_max,
        )
        raise QrContractFailure(failure_summary)
    if not math.isfinite(elapsed_ms):
        raise RuntimeError("QrV2 elapsed time is non-finite")
    return {
        "case_id": case_id,
        "shape": list(a_cpu.shape),
        "dtype": str(a_cpu.dtype),
        "input_sha256": oracle.tensor_sha256(a_cpu),
        "eligible_mx_branch": eligible,
        "mx_qr_call_delta": len(mx_call_records),
        "mx_qr_input": mx_call_records[0] if mx_call_records else None,
        "expected_padded_shape": expected_padded_shape,
        "wrapper_branch": "mx_fixed" if eligible else "torch_npu_boundary_fallback",
        "public_qr_mode": public_mode,
        "cpu_fp32_projection_control_max": projection_control_max,
        "input_unmodified": input_unmodified,
        "elapsed_ms": elapsed_ms,
        "contract_pass": True,
        "shape_pass": report["shape_pass"],
        "input_finite": report["input_finite"],
        "q_finite": report["q_finite"],
        "r_finite": report["r_finite"],
        "nonfinite_count": report["nonfinite_count"],
        "finite_pass": report["finite_pass"],
        "reconstruction": report["reconstruction"],
        "orthogonality": report["orthogonality"],
        "lower_triangle_exact_zero": report["lower_triangle_exact_zero"],
        "lower_triangle_required": report["lower_triangle_required"],
        "fp64": report["fp64"],
        "full_rank_projection": report["full_rank_projection"],
    }


def evaluate_call(torch: Any, mx: Any, case_id: str, a_cpu: Any, device: Any, ledger: list[dict[str, Any]], *, profile_root: Path) -> dict[str, Any]:
    a = a_cpu.to(device)
    a_before = a.detach().clone()
    before = len(ledger)
    with legacy.profile_context(profile_root) as prof:
        start = time.perf_counter()
        q, r = mx.linalg.qr(a)
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        prof.step()
    return _finalize_call(
        torch,
        a_cpu,
        a,
        a_before,
        q,
        r,
        elapsed_ms,
        case_id=case_id,
        eligible=min(a_cpu.shape) > 80,
        mx_call_records=ledger[before:],
    )


def evaluate_sequence(torch: Any, mx: Any, cases: list[tuple[str, Any]], device: Any, ledger: list[dict[str, Any]], *, dedicated_stream: bool) -> list[dict[str, Any]]:
    pending: list[tuple[str, Any, Any, Any, Any, Any, float, bool, list[dict[str, Any]]]] = []

    def launch() -> None:
        for case_id, a_cpu in cases:
            a = a_cpu.to(device)
            a_before = a.detach().clone()
            before = len(ledger)
            start = time.perf_counter()
            q, r = mx.linalg.qr(a)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            pending.append(
                (
                    case_id,
                    a_cpu,
                    a,
                    a_before,
                    q,
                    r,
                    elapsed_ms,
                    min(a_cpu.shape) > 80,
                    ledger[before:],
                )
            )

    if dedicated_stream:
        stream = torch.npu.Stream()
        producer_ready = torch.npu.Event()
        producer_ready.record(torch.npu.current_stream())
        stream.wait_event(producer_ready)
        with torch.npu.stream(stream):
            launch()
            completed = torch.npu.Event()
            completed.record(stream)
        torch.npu.current_stream().wait_event(completed)
    else:
        launch()
    # Exactly one global synchronization per stateful sequence.
    torch.npu.synchronize()
    return [
        _finalize_call(
            torch,
            a_cpu,
            a,
            a_before,
            q,
            r,
            elapsed_ms,
            case_id=case_id,
            eligible=eligible,
            mx_call_records=records,
        )
        for case_id, a_cpu, a, a_before, q, r, elapsed_ms, eligible, records in pending
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shadow-root", required=True, type=Path)
    parser.add_argument("--installed-custom-opp", required=True, type=Path)
    parser.add_argument("--first-profiled-only", action="store_true")
    args = parser.parse_args()
    rank_hint = os.environ.get("RANK", "unknown")
    output_hint = args.output_dir.absolute()
    failure = output_hint / "failure" / f"rank{rank_hint}.txt"
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if world_size != 8 or rank != local_rank or rank not in range(8) or visible != VISIBLE:
            raise RuntimeError("rank/world/rear8 contract failed before imports")
        output = args.output_dir.resolve(strict=True)
        shadow = args.shadow_root.resolve(strict=True)
        installed_custom = args.installed_custom_opp.resolve(strict=True)
        if shadow.is_symlink() or installed_custom.is_symlink():
            raise RuntimeError("shadow/installed custom OPP must not be symlinks")
        expected_package = shadow / "mx_driving_cloud"
        expected_custom = expected_package / "packages/vendors/customize"
        if not expected_custom.is_dir() or not installed_custom.is_dir():
            raise RuntimeError("shadow or installed custom OPP is missing")
        pre_spec = __import__("importlib.util").util.find_spec("mx_driving_cloud")
        if pre_spec is None or pre_spec.origin is None or not inside(Path(pre_spec.origin), shadow):
            raise RuntimeError("mx_driving_cloud import would not originate from physical shadow")
        startup_parts = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")
        if len(startup_parts) != 1:
            raise RuntimeError("startup custom OPP must contain only the complete shadow")
        if Path(startup_parts[0]).resolve(strict=True) != expected_custom:
            raise RuntimeError("shadow custom OPP is not first before imports")

        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        os.environ.pop("MX_QR_VALIDATION_BYPASS", None)
        # These imports are intentionally delayed until after the static package/OPP gate.
        global oracle, legacy
        import qrv2_release_oracle as oracle
        import step343_qrv2_cold_case as legacy

        import torch
        import torch.distributed as dist
        import torch_npu
        import mx_driving_cloud
        import mx_driving

        module_paths = {
            "cloud_init": Path(mx_driving_cloud.__file__),
            "cloud_extension": Path(mx_driving_cloud._C.__file__),
            "cloud_linalg": Path(mx_driving_cloud.ops.linalg.__file__),
        }
        if not all(inside(path, shadow) for path in module_paths.values()):
            raise RuntimeError("one or more mx_driving_cloud modules escaped the physical shadow")
        wrapper_contract = oracle.validate_production_wrapper_source(
            module_paths["cloud_linalg"]
        )
        base_custom = (
            Path(mx_driving.__file__).resolve(strict=True).parent
            / "packages/vendors/customize"
        ).resolve(strict=True)
        if not base_custom.is_dir() or base_custom in {expected_custom, installed_custom}:
            raise RuntimeError("base mx custom OPP is missing or aliases another vendor root")
        opp_parts = [Path(part).resolve(strict=True) for part in os.environ.get(
            "ASCEND_CUSTOM_OPP_PATH", ""
        ).split(":")]
        expected_after_import = [
            expected_custom,
            base_custom,
            expected_custom,
        ]
        if opp_parts != expected_after_import:
            raise RuntimeError(
                "mx import custom OPP transition differs from audited order: "
                f"role_count={len(opp_parts)}"
            )
        restored_opp = [expected_custom, base_custom]
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(map(str, restored_opp))
        if [
            Path(part).resolve(strict=True)
            for part in os.environ["ASCEND_CUSTOM_OPP_PATH"].split(":")
        ] != restored_opp:
            raise RuntimeError("custom OPP stable-dedup restoration failed")
        real_mx_qr = mx_driving_cloud._C.qr
        mx_call_ledger: list[dict[str, Any]] = []

        def counted_mx_qr(value: Any):
            mx_call_ledger.append(
                {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "contiguous": bool(value.is_contiguous()),
                }
            )
            return real_mx_qr(value)

        mx_driving_cloud._C.qr = counted_mx_qr

        available = bool(torch.npu.is_available())
        device_count = int(torch.npu.device_count())
        if not available or device_count != 8 or not getattr(torch_npu, "__version__", None):
            raise RuntimeError("torch_npu/device_count gate failed")
        torch.npu.set_device(local_rank)
        dist.init_process_group(backend="hccl")
        device = torch.device(f"npu:{local_rank}")
        for name in ("ready", "done", "failure"):
            (output / name).mkdir(exist_ok=True)
        legacy.atomic_json(
            output / "ready" / f"rank{rank}.json",
            {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "visible": visible,
                "npu_available": available,
                "device_count": device_count,
                "container_pid": os.getpid(),
                "gate_pass": True,
                "shadow_gate": True,
                "opp_first_shadow": True,
                "module_file_sha256": {
                    key: sha256_file(path) for key, path in module_paths.items()
                },
                "custom_opp_role_sequence": ["shadow", "base"],
                "wrapper_contract": wrapper_contract,
            },
        )
        legacy.wait_release(output / "release_after_npu_smi")
        dist.barrier()

        input_path = args.input_dir / f"rank{rank}_step10_ind0_192x192_BAD.pt"
        raw_sha = sha256_file(input_path)
        captured = oracle.load_known_step260_case(input_path)
        a_cpu = captured.tensor
        torch.npu.synchronize()

        calls: list[dict[str, Any]] = []
        profile_root = output / f"profile_rank{rank}"
        profile_root.mkdir(exist_ok=False)
        calls.append(
            evaluate_call(
                torch,
                mx_driving_cloud,
                f"step260_rank{rank}_profiled",
                a_cpu,
                device,
                mx_call_ledger,
                profile_root=profile_root,
            )
        )
        identity = verify_profile(profile_root, expected_aic_references=1)
        legacy.atomic_json(output / f"profiler_identity_rank{rank}.json", identity)
        diagnostic_only = os.environ.get("STEP358_STATE_DIAGNOSTIC_ONLY") == "1"
        if not args.first_profiled_only:
            calls.extend(
                evaluate_sequence(
                    torch,
                    mx_driving_cloud,
                    [
                        (f"step260_rank{rank}_repeat{ordinal}", a_cpu)
                        for ordinal in range(3)
                    ] if not diagnostic_only else [],
                    device,
                    mx_call_ledger,
                    dedicated_stream=False,
                )
            )
        if rank == 0 and not args.first_profiled_only:
            if not diagnostic_only:
                core_inputs = [
                    (spec.case_id, oracle.generate_case(spec).tensor)
                    for spec in oracle.core_case_specs()
                ]
                calls.extend(
                    evaluate_sequence(
                        torch,
                        mx_driving_cloud,
                        core_inputs,
                        device,
                        mx_call_ledger,
                        dedicated_stream=False,
                    )
                )
            state_shapes = (96, 192, 256, 192, 512, 192)
            state_inputs = []
            for index, size in enumerate(state_shapes):
                case_id = f"state_default_{index}_{size}x{size}"
                state_inputs.append((
                    case_id,
                    oracle.generate_case(
                        oracle.CaseSpec(
                        case_id=case_id,
                        shape=(size, size),
                        kind="randn",
                        seed=0,
                        )
                    ).tensor,
                ))
            calls.extend(
                evaluate_sequence(
                    torch,
                    mx_driving_cloud,
                    state_inputs,
                    device,
                    mx_call_ledger,
                    dedicated_stream=False,
                )
            )
            if not diagnostic_only:
                dedicated_inputs = [
                    (case_id.replace("state_default", "state_dedicated"), tensor)
                    for case_id, tensor in state_inputs
                ]
                calls.extend(
                    evaluate_sequence(
                        torch,
                        mx_driving_cloud,
                        dedicated_inputs,
                        device,
                        mx_call_ledger,
                        dedicated_stream=True,
                    )
                )
        dist.barrier()
        eligible_calls = sum(bool(row["eligible_mx_branch"]) for row in calls)
        eligible_fallbacks = sum(
            bool(row["eligible_mx_branch"]) and row["mx_qr_call_delta"] != 1
            for row in calls
        )
        if len(mx_call_ledger) != eligible_calls or eligible_fallbacks != 0:
            raise RuntimeError("eligible public-call ledger does not close to _C.qr calls")
        result: dict[str, Any] = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "input_file_sha256": raw_sha,
            "call_count": len(calls),
            "eligible_call_count": eligible_calls,
            "mx_qr_call_count": len(mx_call_ledger),
            "eligible_fallback_count": eligible_fallbacks,
            "all_contract_pass": all(row["contract_pass"] for row in calls),
            "profiler_identity_pass": True,
            "state_diagnostic_only": diagnostic_only,
            "first_profiled_only": args.first_profiled_only,
            "calls": calls,
        }
        legacy.atomic_json(output / "done" / f"rank{rank}.json", result)
        dist.barrier()
        dist.destroy_process_group()
        return 0
    except BaseException as error:
        original_traceback = traceback.format_exc()
        try:
            if "dist" in locals() and dist.is_initialized():
                dist.destroy_process_group()
        except BaseException:
            pass
        _persist_failure_artifacts(failure, error, original_traceback)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
