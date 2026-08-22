#!/usr/bin/env python3
"""One cold mx QrV2 call per rank, with rank-0 kernel-hit profiling."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any


VISIBLE = "8,9,10,11,12,13,14,15"
ORIGINAL_KERNEL = "QrV2_566c2e1c0e6c8c92152ad84416d77006"
CANDIDATE_KERNEL = "QrV2_step338_lifetime_fix"
ORIGINAL_AIC = "QrV2_566c2e1c0e6c8c92152ad84416d77006_0_mix_aic"
CANDIDATE_AIC = "QrV2_step338_lifetime_fix_mix_aic"
ORIGINAL_AIV = "QrV2_566c2e1c0e6c8c92152ad84416d77006_0_mix_aiv"
CANDIDATE_AIV = "QrV2_step338_lifetime_fix_mix_aiv"


def opp_path_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def requested_opp_contract(
    mode: str,
    requested: str | None,
    expected_installed: str,
    expected_overlay: str | None = None,
) -> dict[str, Any]:
    if requested is None or not requested:
        raise RuntimeError("ASCEND_CUSTOM_OPP_PATH is missing before imports")
    paths = tuple(requested.split(":"))
    if any(not path for path in paths):
        raise RuntimeError("requested custom OPP path contains an empty item")
    if mode == "original":
        if len(paths) != 1 or paths[0] != expected_installed or expected_overlay is not None:
            raise RuntimeError("original requested custom OPP must contain exactly installed")
        roles = ("installed",)
        installed = paths[0]
    elif mode == "candidate":
        if (
            len(paths) != 2
            or len(set(paths)) != 2
            or expected_overlay is None
            or paths != (expected_overlay, expected_installed)
        ):
            raise RuntimeError("candidate requested custom OPP must be unique overlay,installed")
        roles = ("overlay", "installed")
        installed = paths[1]
    else:
        raise ValueError(f"unsupported custom OPP mode: {mode}")
    return {
        "requested_raw": requested,
        "requested_paths": paths,
        "cloud_path": installed,
        "startup_roles": roles,
        "startup_sha256": opp_path_sha256(requested),
    }


def stable_dedup(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def validate_opp_import_transition(
    contract: dict[str, Any], after_import: str | None, actual_cloud: str, actual_base: str
) -> dict[str, Any]:
    if after_import is None:
        raise RuntimeError("ASCEND_CUSTOM_OPP_PATH disappeared during imports")
    if actual_cloud != contract["cloud_path"]:
        raise RuntimeError("derived cloud vendor root differs from requested cloud root")
    if not actual_base or actual_base in {actual_cloud, *contract["requested_paths"]}:
        raise RuntimeError("derived base vendor root is missing, duplicate, or unknown")
    after_paths = tuple(after_import.split(":"))
    expected_paths = (actual_cloud, actual_base) + tuple(contract["requested_paths"])
    if after_paths != expected_paths:
        raise RuntimeError("mx imports did not produce cloud,base plus requested in exact order")
    desired_paths = stable_dedup(tuple(contract["requested_paths"]) + after_paths)
    if len(desired_paths) != len(set(desired_paths)):
        raise RuntimeError("restored custom OPP path calculation contains duplicates")
    desired_roles = (
        ("cloud", "base")
        if contract["startup_roles"] == ("installed",)
        else ("overlay", "cloud", "base")
    )
    startup_roles = (
        ["cloud"]
        if contract["startup_roles"] == ("installed",)
        else ["overlay", "cloud"]
    )
    after_roles = ["cloud", "base", *startup_roles]
    return {
        "startup_role_sequence": startup_roles,
        "after_import_role_sequence": after_roles,
        "startup_path_sha256": contract["startup_sha256"],
        "after_import_path_sha256": opp_path_sha256(after_import),
        "_desired_raw": ":".join(desired_paths),
        "_desired_roles": desired_roles,
    }


def validate_opp_restoration(
    contract: dict[str, Any], transition: dict[str, Any], restored: str | None
) -> dict[str, Any]:
    desired_raw = transition.get("_desired_raw")
    if restored != desired_raw:
        raise RuntimeError("failed to restore exact stable-dedup custom OPP path")
    public_transition = {key: value for key, value in transition.items() if not key.startswith("_")}
    return {
        **public_transition,
        "restored_role_sequence": list(transition["_desired_roles"]),
        "restored_path_sha256": opp_path_sha256(restored),
        "restored_exact": True,
        "restored_paths_unique": True,
    }


def vendor_root_from_module_file(module_file: str | None) -> str:
    if not module_file:
        raise RuntimeError("imported MX module has no __file__")
    module_path = Path(module_file)
    if not module_path.is_absolute() or not module_path.is_file() or module_path.is_symlink():
        raise RuntimeError("imported MX module file contract failed")
    vendor_root = module_path.parent / "packages/vendors/customize"
    if not vendor_root.is_absolute() or not vendor_root.is_dir() or vendor_root.is_symlink():
        raise RuntimeError("derived MX vendor root contract failed")
    return str(vendor_root)


def validate_kernel_argument_contract(mode: str, expected: str, original: str) -> None:
    if original != ORIGINAL_KERNEL:
        raise RuntimeError("original kernel argument differs from audited OPP kernelName/binFileName")
    mode_expected = original if mode == "original" else CANDIDATE_KERNEL
    if expected != mode_expected:
        raise RuntimeError("expected kernel argument is inconsistent with mode")


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def scalar_stats(tensor: Any) -> dict[str, int]:
    import torch

    value = tensor.detach().float().cpu()
    return {
        "nan": int(torch.isnan(value).sum().item()),
        "posinf": int(torch.isposinf(value).sum().item()),
        "neginf": int(torch.isneginf(value).sum().item()),
        "finite": int(torch.isfinite(value).sum().item()),
        "numel": value.numel(),
    }


def tensor_content_sha256(tensor: Any) -> str:
    value = tensor.detach().float().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def first_coordinate(mask: Any) -> list[int] | None:
    import torch

    coordinates = torch.nonzero(mask, as_tuple=False)
    return coordinates[0].tolist() if coordinates.numel() else None


def evaluate_math_contract(
    a: Any,
    q: Any,
    r: Any,
    actual: Any,
    output: Path,
    metadata: dict[str, Any],
    expected_a: Any | None = None,
) -> dict[str, Any]:
    import torch

    tensors = {
        "INPUT_A": a.detach().float().cpu(),
        "Q": q.detach().float().cpu(),
        "R": r.detach().float().cpu(),
        "ACTUAL_QR": actual.detach().float().cpu(),
    }
    tensors["EXPECTED_A"] = (
        tensors["INPUT_A"].clone()
        if expected_a is None
        else expected_a.detach().float().cpu().contiguous().clone()
    )
    tensors["DIFF"] = tensors["ACTUAL_QR"] - tensors["EXPECTED_A"]
    a_sha_before = tensor_content_sha256(tensors["EXPECTED_A"])
    a_sha_after = tensor_content_sha256(tensors["INPUT_A"])
    input_a_unmodified = a_sha_before == a_sha_after
    stats = {name: scalar_stats(value) for name, value in tensors.items()}
    all_finite = all(row["finite"] == row["numel"] for row in stats.values())
    n = tensors["EXPECTED_A"].shape[-1]
    unit_roundoff = torch.finfo(torch.float32).eps / 2.0
    gamma_n = (n * unit_roundoff) / (1.0 - n * unit_roundoff)
    oracle_q, oracle_r = torch.linalg.qr(tensors["EXPECTED_A"])
    oracle_recon_error = (oracle_q @ oracle_r - tensors["EXPECTED_A"]).abs()
    identity = torch.eye(n, dtype=torch.float32)
    oracle_orth_error = (oracle_q.transpose(-2, -1) @ oracle_q - identity).abs()
    oracle_lower_error = torch.tril(oracle_r, diagonal=-1).abs()

    recon_allowed = oracle_recon_error + gamma_n * (
        tensors["Q"].abs() @ tensors["R"].abs()
    )
    orth_diff = tensors["Q"].transpose(-2, -1) @ tensors["Q"] - identity
    orth_allowed = oracle_orth_error + gamma_n * (
        tensors["Q"].abs().transpose(-2, -1) @ tensors["Q"].abs()
    )
    lower_abs = torch.tril(tensors["R"], diagonal=-1).abs()
    recon_violation = torch.isfinite(tensors["DIFF"]) & (tensors["DIFF"].abs() > recon_allowed)
    orth_violation = torch.isfinite(orth_diff) & (orth_diff.abs() > orth_allowed)
    lower_violation = torch.isfinite(lower_abs) & (lower_abs > oracle_lower_error)
    mismatch_counts = {
        "reconstruction": int(recon_violation.sum().item()),
        "orthogonality": int(orth_violation.sum().item()),
        "lower_triangle": int(lower_violation.sum().item()),
    }
    contract_pass = all_finite and input_a_unmodified and not any(mismatch_counts.values())

    first_anomaly: dict[str, Any] | None = None
    for name, value in tensors.items():
        coordinate = first_coordinate(~torch.isfinite(value))
        if coordinate is not None:
            scalar = value[tuple(coordinate)].item()
            first_anomaly = {"tensor": name, "coordinate": coordinate, "value": str(scalar)}
            break
    if first_anomaly is None:
        mutation = tensors["INPUT_A"] != tensors["EXPECTED_A"]
        coordinate = first_coordinate(mutation)
        if coordinate is not None:
            first_anomaly = {"tensor": "INPUT_A_MUTATION", "coordinate": coordinate}
    if first_anomaly is None:
        for label, violation, observed, allowed in (
            ("ACTUAL_QR_vs_EXPECTED_A", recon_violation, tensors["DIFF"].abs(), recon_allowed),
            ("Q_ORTHOGONALITY", orth_violation, orth_diff.abs(), orth_allowed),
            ("R_LOWER_TRIANGLE", lower_violation, lower_abs, oracle_lower_error),
        ):
            coordinate = first_coordinate(violation)
            if coordinate is not None:
                index = tuple(coordinate)
                first_anomaly = {
                    "tensor": label,
                    "coordinate": coordinate,
                    "observed_abs": float(observed[index].item()),
                    "allowed_abs": float(allowed[index].item()),
                }
                break

    if all_finite:
        diff_abs = tensors["DIFF"].abs()
        expected_norm = float(torch.linalg.vector_norm(tensors["EXPECTED_A"]).item())
        diff_norm = float(torch.linalg.vector_norm(tensors["DIFF"]).item())
        max_abs = float(diff_abs.max().item())
        positive_allowed = torch.clamp(recon_allowed, min=torch.finfo(torch.float32).tiny)
        max_scaled_relative = float((diff_abs / positive_allowed).max().item())
        l2_relative = diff_norm / max(expected_norm, torch.finfo(torch.float32).tiny)
        orthogonality = float(
            (tensors["Q"].transpose(-2, -1) @ tensors["Q"] - identity).abs().max().item()
        )
        lower = float(torch.tril(tensors["R"], diagonal=-1).abs().max().item())
    else:
        max_abs = None
        max_scaled_relative = None
        l2_relative = None
        orthogonality = None
        lower = None

    result: dict[str, Any] = {
        "math_contract_pass": contract_pass,
        "input_a_content_sha256_before": a_sha_before,
        "input_a_content_sha256_after": a_sha_after,
        "input_a_unmodified": input_a_unmodified,
        "tolerance_basis": "FP32 gamma_n forward-error bound plus same-A torch.linalg.qr oracle error",
        "fp32_unit_roundoff": unit_roundoff,
        "matrix_n": n,
        "gamma_n": gamma_n,
        "tolerance_mismatch_counts": mismatch_counts,
        "first_anomaly": first_anomaly,
        "tensor_stats": stats,
        "recon_max_abs": max_abs,
        "recon_max_scaled_relative": max_scaled_relative,
        "recon_l2_relative": l2_relative,
        "orthogonality_max_abs": orthogonality,
        "r_lower_max_abs": lower,
        "failure_bundle": None,
        "full_tensor_log": None,
    }
    if not contract_pass:
        failure_dir = output / "failure_bundles" / f"rank{metadata['rank']}"
        failure_dir.mkdir(parents=True, exist_ok=False)
        bundle_path = failure_dir / "tensor_bundle.pt"
        log_path = failure_dir / "full_tensor.log"
        result["failure_bundle"] = str(bundle_path)
        result["full_tensor_log"] = str(log_path)
        bundle_metadata = {**metadata, **result}
        temporary = bundle_path.with_suffix(".pt.tmp")
        torch.save({**tensors, "metadata": bundle_metadata}, temporary)
        temporary.replace(bundle_path)
        with log_path.open("w", encoding="utf-8") as stream, contextlib.redirect_stdout(stream):
            print("METADATA_JSON")
            print(json.dumps(bundle_metadata, ensure_ascii=False, indent=2, sort_keys=True))
            torch.set_printoptions(profile="full", precision=10, linewidth=240, sci_mode=True)
            for name in ("INPUT_A", "Q", "R", "ACTUAL_QR", "EXPECTED_A", "DIFF"):
                print(name)
                print(tensors[name])
    return result


def profile_context(output: Path):
    from torch_npu import profiler

    handler = profiler.tensorboard_trace_handler(
        str(output), worker_name="rank0", analyse_flag=True, async_mode=False
    )
    return profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=handler,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        experimental_config=profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level0,
            aic_metrics=profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            data_simplification=True,
            export_type="text",
        ),
    )


def collect_profile_names(root: Path) -> tuple[list[str], list[str]]:
    names: set[str] = set()
    sources: list[str] = []
    for path in root.rglob("kernel_details.csv"):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("kernel_details.csv source is not a regular file")
        sources.append(str(path.relative_to(root)))
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                kernel_columns = [
                    key
                    for key in (reader.fieldnames or [])
                    if key.strip().lower() in {"name", "kernel name"}
                ]
                if not kernel_columns:
                    raise RuntimeError("kernel_details.csv lacks an explicit kernel column")
                for row in reader:
                    for key in kernel_columns:
                        if row.get(key):
                            names.add(str(row[key]))
        except (UnicodeDecodeError, csv.Error) as exc:
            raise RuntimeError("malformed kernel_details.csv") from exc
    if not sources:
        raise RuntimeError("profiler produced no kernel_details.csv")
    return sorted(names), sorted(sources)


def parse_hash_dictionary(data: bytes) -> list[tuple[int, str]]:
    """Parse audited CANN `<uint64 decimal>:<name>\n` dictionary slices."""
    if not data or not data.endswith(b"\n") or b"\r" in data or b"\x00" in data:
        raise RuntimeError("profiler hash dictionary violates LF-only text framing")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("profiler hash dictionary is not UTF-8") from exc
    rows: list[tuple[int, str]] = []
    for line in text[:-1].split("\n"):
        hash_text, separator, name = line.partition(":")
        if (
            separator != ":"
            or re.fullmatch(r"[0-9]+", hash_text) is None
            or not name
            or name != name.strip()
            or any(ord(char) < 0x20 for char in name)
        ):
            raise RuntimeError("malformed profiler hash dictionary row")
        hash_value = int(hash_text)
        if not 0 <= hash_value <= (1 << 64) - 1:
            raise RuntimeError("profiler hash dictionary value exceeds uint64")
        rows.append((hash_value, name))
    if not rows:
        raise RuntimeError("empty profiler hash dictionary")
    return rows


def collect_runtime_identity(profile_root: Path) -> tuple[dict[int, str], dict[int, int], list[str], list[str]]:
    dictionary_paths = sorted(
        path
        for path in profile_root.glob("**/host/data/unaging.additional.hash_dic.slice_*")
        if not path.name.endswith(".done")
    )
    task_paths = sorted(
        path
        for path in profile_root.glob("**/host/data/aging.compact.task_track.slice_*")
        if not path.name.endswith(".done")
    )
    if not dictionary_paths or not task_paths:
        raise RuntimeError("profiler raw hash dictionary or task_track slices are missing")
    mappings: dict[int, str] = {}
    dictionary_sources: list[str] = []
    for path in dictionary_paths:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("profiler hash dictionary source is not a regular file")
        dictionary_sources.append(str(path.relative_to(profile_root)))
        for hash_value, name in parse_hash_dictionary(path.read_bytes()):
            previous = mappings.get(hash_value)
            if previous is not None and previous != name:
                raise RuntimeError("conflicting duplicate profiler hash mapping")
            mappings[hash_value] = name
    references = {hash_value: 0 for hash_value in mappings}
    task_sources: list[str] = []
    for path in task_paths:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("profiler task_track source is not a regular file")
        task_sources.append(str(path.relative_to(profile_root)))
        data = path.read_bytes()
        if not data or len(data) % 64 != 0:
            raise RuntimeError("profiler task_track slice size is not a positive multiple of 64")
        for record_offset in range(0, len(data), 64):
            hash_value = int.from_bytes(
                data[record_offset + 40 : record_offset + 48], "little", signed=False
            )
            if hash_value in references:
                references[hash_value] += 1
    return mappings, references, dictionary_sources, task_sources


def verify_profile_hit(profile_root: Path, mode: str) -> dict[str, Any]:
    if mode not in {"original", "candidate"}:
        raise ValueError(f"unsupported profiler identity mode: {mode}")
    names, sources = collect_profile_names(profile_root)
    generic_qrv2_matches = [name for name in names if Path(name.strip()).name == "QrV2"]
    if not generic_qrv2_matches:
        raise RuntimeError("kernel_details.csv contains no generic QrV2 execution evidence")
    mappings, references, dictionary_sources, task_sources = collect_runtime_identity(profile_root)
    expected_name = ORIGINAL_AIC if mode == "original" else CANDIDATE_AIC
    expected_entries = [
        {"hash": hash_value, "name": name, "reference_count": references[hash_value]}
        for hash_value, name in mappings.items()
        if name == expected_name
    ]
    if len(expected_entries) != 1 or expected_entries[0]["reference_count"] < 1:
        raise RuntimeError("expected concrete profiler kernel identity is not task_track-referenced")
    forbidden_names = (
        {CANDIDATE_AIC, CANDIDATE_AIV}
        if mode == "original"
        else {ORIGINAL_AIC, ORIGINAL_AIV}
    )
    forbidden_entries = [
        {"hash": hash_value, "name": name, "reference_count": references[hash_value]}
        for hash_value, name in mappings.items()
        if name in forbidden_names and references[hash_value] > 0
    ]
    if forbidden_entries:
        raise RuntimeError(f"{mode} task_track references a forbidden opposite-mode AIC/AIV identity")
    result = {
        "identity_gate": "hash_dictionary_plus_task_track_reference",
        "mode": mode,
        "generic_qrv2_csv_matches": generic_qrv2_matches,
        "kernel_details_sources": sources,
        "hash_dictionary_sources": dictionary_sources,
        "task_track_sources": task_sources,
        "matched_entry": expected_entries[0],
        "reference_count": expected_entries[0]["reference_count"],
        "forbidden_referenced_entries": forbidden_entries,
        "profile_raw_retained": True,
        "pass": True,
    }
    return result


def wait_release(path: Path, timeout_seconds: int = 120) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for live npu-smi release")
        time.sleep(0.1)


def run(args: argparse.Namespace) -> int:
    rank_label = os.environ.get("RANK", "unknown")
    if not rank_label.isdigit():
        rank_label = "unknown"
    output_hint = Path(args.output_dir).absolute()
    failure_path = output_hint / "failure" / f"rank{rank_label}.txt"
    try:
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        os.environ.pop("MX_QR_VALIDATION_BYPASS", None)
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        output = Path(args.output_dir).resolve(strict=True)
        for name in ("ready", "done", "failure"):
            (output / name).mkdir(exist_ok=True)
        failure_path = output / "failure" / f"rank{rank}.txt"
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if not (
            world_size == 8
            and rank in range(8)
            and local_rank in range(8)
            and rank == local_rank
            and visible == VISIBLE
        ):
            raise RuntimeError("rank/world/visible startup contract failed before imports")
        installed_path = Path(args.installed_custom_opp)
        if not installed_path.is_absolute() or not installed_path.is_dir() or installed_path.is_symlink():
            raise RuntimeError("installed custom OPP startup directory contract failed")
        if args.mode == "candidate":
            overlay_path = Path(args.overlay_custom_opp)
            if not overlay_path.is_absolute() or not overlay_path.is_dir() or overlay_path.is_symlink():
                raise RuntimeError("overlay custom OPP startup directory contract failed")
        elif args.overlay_custom_opp is not None:
            raise RuntimeError("original mode must not receive an overlay custom OPP")
        validate_kernel_argument_contract(args.mode, args.expected_kernel, args.original_kernel)
        requested_raw = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
        startup_contract = requested_opp_contract(
            args.mode, requested_raw, args.installed_custom_opp, args.overlay_custom_opp
        )
        import torch
        import torch.distributed as dist
        import torch_npu
        import mx_driving_cloud
        import mx_driving

        after_import_raw = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
        actual_cloud = vendor_root_from_module_file(mx_driving_cloud.__file__)
        actual_base = vendor_root_from_module_file(mx_driving.__file__)
        import_transition = validate_opp_import_transition(
            startup_contract, after_import_raw, actual_cloud, actual_base
        )
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = import_transition["_desired_raw"]
        opp_transition = validate_opp_restoration(
            startup_contract, import_transition, os.environ.get("ASCEND_CUSTOM_OPP_PATH")
        )

        # No torch.npu API, device query, set_device, or distributed init may precede this point.
        available = bool(torch.npu.is_available())
        device_count = int(torch.npu.device_count())
        gate_pass = (
            available
            and device_count == 8
            and getattr(torch_npu, "__version__", None) is not None
        )
        if not gate_pass:
            raise RuntimeError(
                f"torch_npu/rank gate failed rank={rank} local={local_rank} world={world_size} "
                f"visible={visible!r} available={available} count={device_count}"
            )
        torch.npu.set_device(local_rank)
        dist.init_process_group(backend="hccl")
        input_path = Path(args.input_dir) / f"rank{rank}_step10_ind0_192x192_BAD.pt"
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        loaded = torch.load(input_path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict) or "A" not in loaded:
            raise RuntimeError(f"invalid STEP260 payload: {input_path.name}")
        a_cpu = loaded["A"].detach().float().contiguous()
        if tuple(a_cpu.shape) != (192, 192) or not bool(torch.isfinite(a_cpu).all()):
            raise RuntimeError(f"invalid A contract for {input_path.name}")
        a = a_cpu.to(torch.device(f"npu:{local_rank}"))
        torch.npu.synchronize()
        atomic_json(output / "ready" / f"rank{rank}.json", {
            "mode": args.mode,
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "visible": visible,
            "npu_available": available,
            "device_count": device_count,
            "torch_version": torch.__version__,
            "torch_npu_version": torch_npu.__version__,
            "container_pid": os.getpid(),
            "input_file": input_path.name,
            "custom_opp_transition": opp_transition,
            "gate_pass": gate_pass,
        })
        wait_release(output / "release_after_npu_smi")
        dist.barrier()

        profile_root = output / "profile_rank0"
        if rank == 0:
            if profile_root.exists() and any(profile_root.iterdir()):
                raise RuntimeError("refusing to overwrite non-empty profiler directory")
            profile_root.mkdir(exist_ok=True)
            with profile_context(profile_root) as prof:
                start = time.perf_counter()
                q, r = mx_driving_cloud.linalg.qr(a)
                torch.npu.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                prof.step()
        else:
            start = time.perf_counter()
            q, r = mx_driving_cloud.linalg.qr(a)
            torch.npu.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        dist.barrier()

        # This work is deliberately outside the QR timing/profiler window.
        actual = q @ r
        torch.npu.synchronize()

        result: dict[str, Any] = {
            "mode": args.mode,
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "input_file": input_path.name,
            "operator": f"{mx_driving_cloud.linalg.qr.__module__}.{mx_driving_cloud.linalg.qr.__name__}",
            "qr_elapsed_ms": elapsed_ms,
        }
        contract = evaluate_math_contract(
            a,
            q,
            r,
            actual,
            output,
            {
                "schema": "step347-failure-bundle-v1",
                "mode": args.mode,
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "input_file": input_path.name,
                "qr_elapsed_ms": elapsed_ms,
                "timing_excludes_contract_dump": True,
            },
            expected_a=a_cpu,
        )
        result.update(contract)
        result["A"] = contract["tensor_stats"]["INPUT_A"]
        result["Q"] = contract["tensor_stats"]["Q"]
        result["R"] = contract["tensor_stats"]["R"]
        for value in result.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError("result contains a non-finite scalar")
        if rank == 0:
            hit = verify_profile_hit(profile_root, args.mode)
            atomic_json(output / "profiler_hit.json", hit)
            result["profiler_hit_pass"] = True
        atomic_json(output / "done" / f"rank{rank}.json", result)
        dist.barrier()
        dist.destroy_process_group()
        return 0
    except BaseException:
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def summarize(output: Path) -> int:
    rows = [json.loads((output / "done" / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(8)]
    hit = json.loads((output / "profiler_hit.json").read_text(encoding="utf-8"))
    if [row["rank"] for row in rows] != list(range(8)) or not hit.get("pass"):
        raise RuntimeError("result inventory or profiler hit gate failed")
    summary = {
        "schema": "step347-cold-qrv2-v1",
        "mode": rows[0]["mode"],
        "rank_count": len(rows),
        "all_a_finite": all(row["A"]["finite"] == row["A"]["numel"] for row in rows),
        "all_q_finite": all(row["Q"]["finite"] == row["Q"]["numel"] for row in rows),
        "all_r_finite": all(row["R"]["finite"] == row["R"]["numel"] for row in rows),
        "all_math_contract_pass": all(row["math_contract_pass"] for row in rows),
        "all_input_a_unmodified": all(row["input_a_unmodified"] for row in rows),
        "qr_elapsed_ms": [row["qr_elapsed_ms"] for row in rows],
        "profiler_hit": hit,
        "rows": rows,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({
        "mode": summary["mode"],
        "rank_count": 8,
        "all_a_finite": summary["all_a_finite"],
        "all_q_finite": summary["all_q_finite"],
        "all_r_finite": summary["all_r_finite"],
        "all_math_contract_pass": summary["all_math_contract_pass"],
        "profiler_hit": hit["pass"],
    }, sort_keys=True))
    return 0


def summarize_ab(root: Path) -> int:
    original = json.loads((root / "original" / "summary.json").read_text(encoding="utf-8"))
    candidate = json.loads((root / "candidate" / "summary.json").read_text(encoding="utf-8"))
    if original["mode"] != "original" or candidate["mode"] != "candidate":
        raise RuntimeError("A/B mode labels are invalid")
    if original["rank_count"] != 8 or candidate["rank_count"] != 8:
        raise RuntimeError("A/B rank inventory is invalid")
    original_hit = original.get("profiler_hit", {}).get("pass") is True
    candidate_hit = candidate.get("profiler_hit", {}).get("pass") is True
    original_rows = {int(row["rank"]): row for row in original["rows"]}
    candidate_rows = {int(row["rank"]): row for row in candidate["rows"]}
    if set(original_rows) != set(range(8)) or set(candidate_rows) != set(range(8)):
        raise RuntimeError("A/B rank keys are invalid")
    comparisons = []
    for rank in range(8):
        left = original_rows[rank]
        right = candidate_rows[rank]
        if left["input_file"] != right["input_file"]:
            raise RuntimeError(f"rank {rank} input identity differs between modes")
        if left["input_a_content_sha256_before"] != right["input_a_content_sha256_before"]:
            raise RuntimeError(f"rank {rank} loaded A content SHA differs between modes")
        comparisons.append({
            "rank": rank,
            "input_file": left["input_file"],
            "original_qr_elapsed_ms": left["qr_elapsed_ms"],
            "candidate_qr_elapsed_ms": right["qr_elapsed_ms"],
            "candidate_minus_original_ms": right["qr_elapsed_ms"] - left["qr_elapsed_ms"],
            "original": {key: left[key] for key in (
                "A", "Q", "R", "tensor_stats", "math_contract_pass", "tolerance_basis",
                "fp32_unit_roundoff", "matrix_n", "gamma_n", "tolerance_mismatch_counts", "first_anomaly", "recon_max_abs",
                "recon_max_scaled_relative", "recon_l2_relative", "orthogonality_max_abs",
                "r_lower_max_abs", "failure_bundle", "full_tensor_log"
            )},
            "candidate": {key: right[key] for key in (
                "A", "Q", "R", "tensor_stats", "math_contract_pass", "tolerance_basis",
                "fp32_unit_roundoff", "matrix_n", "gamma_n", "tolerance_mismatch_counts", "first_anomaly", "recon_max_abs",
                "recon_max_scaled_relative", "recon_l2_relative", "orthogonality_max_abs",
                "r_lower_max_abs", "failure_bundle", "full_tensor_log"
            )},
        })
    payload = {
        "schema": "step347-cold-qrv2-ab-v1",
        "training_started": False,
        "raw_profiles_retained_remote": True,
        "original_kernel_hit": original["profiler_hit"],
        "candidate_kernel_hit": candidate["profiler_hit"],
        "rows": comparisons,
    }
    failure_reasons = []
    if not original_hit:
        failure_reasons.append("original_profiler_hit_failed")
    if not candidate_hit:
        failure_reasons.append("candidate_profiler_hit_failed")
    if original.get("all_a_finite") is not True:
        failure_reasons.append("original_A_nonfinite")
    if candidate.get("all_a_finite") is not True:
        failure_reasons.append("candidate_A_nonfinite")
    if candidate.get("all_q_finite") is not True:
        failure_reasons.append("candidate_Q_nonfinite")
    if candidate.get("all_r_finite") is not True:
        failure_reasons.append("candidate_R_nonfinite")
    if candidate.get("all_math_contract_pass") is not True:
        failure_reasons.append("candidate_math_contract_failed")
    if original.get("all_input_a_unmodified") is not True:
        failure_reasons.append("original_modified_A_in_place")
    if candidate.get("all_input_a_unmodified") is not True:
        failure_reasons.append("candidate_modified_A_in_place")
    candidate_pass = not any(reason.startswith("candidate_") for reason in failure_reasons)
    original_numeric_pass = (
        original.get("all_q_finite") is True
        and original.get("all_r_finite") is True
        and original.get("all_math_contract_pass") is True
    )
    payload["decision"] = "COLD_VALIDATION_FAIL" if failure_reasons else "COLD_VALIDATION_PASS"
    payload["failure_reasons"] = failure_reasons
    payload["observed_improvement"] = bool(candidate_pass and not original_numeric_pass)
    payload["causal_fix_proven"] = False
    atomic_json(root / "comparison.json", payload)
    print(json.dumps({
        "ab_summary": payload["decision"],
        "failure_reasons": failure_reasons,
        "rank_count": 8,
        "original_kernel_hit": original_hit,
        "candidate_kernel_hit": candidate_hit,
        "observed_improvement": payload["observed_improvement"],
        "causal_fix_proven": False,
        "training_started": False,
    }, sort_keys=True))
    return 3 if failure_reasons else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("original", "candidate"))
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-kernel")
    parser.add_argument("--original-kernel")
    parser.add_argument("--installed-custom-opp")
    parser.add_argument("--overlay-custom-opp")
    parser.add_argument("--summarize")
    parser.add_argument("--summarize-ab")
    args = parser.parse_args()
    if args.summarize and args.summarize_ab:
        parser.error("--summarize and --summarize-ab are mutually exclusive")
    if args.summarize:
        return summarize(Path(args.summarize).resolve(strict=True))
    if args.summarize_ab:
        return summarize_ab(Path(args.summarize_ab).resolve(strict=True))
    required = (
        args.mode,
        args.input_dir,
        args.output_dir,
        args.expected_kernel,
        args.original_kernel,
        args.installed_custom_opp,
    )
    if any(value is None for value in required):
        parser.error("case mode lacks required worker arguments")
    if args.mode == "candidate" and args.overlay_custom_opp is None:
        parser.error("candidate case requires --overlay-custom-opp")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
