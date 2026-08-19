#!/usr/bin/env python3
"""STEP-215-E fail-closed SOAP two-QR-cycle and resume gate.

This harness never edits the business repository.  A project-specific adapter
must construct the real model/optimizer state from the requested config and
checkpoint.  See step215_e_real_soap_adapter_TEMPLATE.py for the contract.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EXPECTED_VISIBLE = "8,9,10,11,12,13,14,15"
TARGET_RELATIVE = Path("projects/mmdet3d_plugin/optimizers/soap.py")
TARGET_FUNCTION = "get_orthogonal_matrix_QR"
TARGET_NEEDLE = "Q, _ = torch.linalg.qr(power_iter)"
REQUIRED_READINESS = (
    "real_soap_optimizer",
    "loads_requested_checkpoint",
    "uses_requested_config",
    "deterministic_replay_gradient",
    "state_view_includes_parameters",
    "state_view_includes_optimizer_state",
    "state_view_includes_q",
    "state_view_includes_gg_exp_avg_exp_avg_sq_step",
    "checkpoint_roundtrip",
    "sort_observable_through_torch_argsort",
)
REQUIRED_METHODS = (
    "build_trial",
    "make_gradient",
    "apply_gradient",
    "state_view",
    "layout_view",
    "save_trial",
    "load_trial",
    "destroy_trial",
)
Q_LIMIT = 5.0e-5
OTHER_LIMIT = 1.0e-4
Q_ORTHOGONALITY_DEFAULT_LIMIT = 1.0e-5
Q_ORTHOGONALITY_CALIBRATED_MAX = 2.0e-5
Q_ORTHOGONALITY_LIMIT = Q_ORTHOGONALITY_DEFAULT_LIMIT
BASIS_RELAXED_COMPARISONS = frozenset(
    {
        "cycle1-baselineA-candidate-adaptive",
        "cycle2-baselineA-candidate-adaptive",
    }
)
APPROVED_ACTIVE_QR_INVENTORY = {
    "1x1": 106,
    "3x3": 30,
    "4x4": 6,
    "7x7": 37,
    "8x8": 1,
    "11x11": 1,
    "22x22": 1,
    "32x32": 4,
    "40x40": 9,
    "64x64": 28,
    "96x96": 3,
    "120x120": 1,
    "128x128": 18,
    "160x160": 1,
    "192x192": 32,
    "220x220": 4,
    "256x256": 181,
    "352x352": 1,
    "440x440": 4,
    "512x512": 43,
    "768x768": 22,
    "1024x1024": 6,
    "2560x2560": 4,
}
APPROVED_ACTIVE_QR_COUNT = 543
APPROVED_ACTIVE_SQUARE_SIZES = frozenset(
    int(key.split("x", 1)[0]) for key in APPROVED_ACTIVE_QR_INVENTORY
)


def guarded_candidate_shape(shape: Iterable[int]) -> bool:
    dims = tuple(int(x) for x in shape)
    return len(dims) == 2 and dims[0] == dims[1] and dims[0] in APPROVED_ACTIVE_SQUARE_SIZES


def assert_guard_contract() -> None:
    if sum(APPROVED_ACTIVE_QR_INVENTORY.values()) != APPROVED_ACTIVE_QR_COUNT:
        raise RuntimeError("approved QR inventory does not sum to 543")
    if guarded_candidate_shape((5120, 5120)) or guarded_candidate_shape((2, 2)):
        raise RuntimeError("5120 or an unknown shape entered the geqrf+orgqr allow-list")
    if not all(
        guarded_candidate_shape(tuple(map(int, key.split("x"))))
        for key in APPROVED_ACTIVE_QR_INVENTORY
    ):
        raise RuntimeError("an approved active shape is absent from the allow-list")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--expected-soap-sha256", required=True)
    p.add_argument("--max-optimizer-steps", type=int, default=200)
    p.add_argument(
        "--q-orthogonality-limit",
        type=float,
        default=Q_ORTHOGONALITY_DEFAULT_LIMIT,
        help=(
            "explicit max_abs(Q^H Q-I) limit; default 1e-5 and calibrated values "
            "are fail-closed at an absolute maximum of 2e-5"
        ),
    )
    p.add_argument(
        "--basis-relaxed-diagnostic",
        action="store_true",
        help=(
            "diagnostic only: ignore raw Q distance exclusively for the two "
            "baseline-vs-candidate cycle comparisons; all Q finite/orthogonality, "
            "candidate resume, non-Q, schema, step, sort and inventory gates remain active"
        ),
    )
    return p.parse_args(argv)


def configure_q_orthogonality_limit(requested: float) -> float:
    global Q_ORTHOGONALITY_LIMIT
    value = float(requested)
    if not math.isfinite(value) or value < Q_ORTHOGONALITY_DEFAULT_LIMIT:
        raise RuntimeError("Q orthogonality limit must be finite and at least 1e-5")
    if value > Q_ORTHOGONALITY_CALIBRATED_MAX:
        raise RuntimeError("Q orthogonality limit exceeds calibrated hard maximum 2e-5")
    Q_ORTHOGONALITY_LIMIT = value
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def git_status(repo: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


class ParentMap(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[ast.AST] = []
        self.qr_functions: list[str | None] = []

    def generic_visit(self, node: ast.AST) -> None:
        self.stack.append(node)
        super().generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        is_qr = (
            isinstance(f, ast.Attribute)
            and f.attr == "qr"
            and isinstance(f.value, ast.Attribute)
            and f.value.attr == "linalg"
            and isinstance(f.value.value, ast.Name)
            and f.value.value.id == "torch"
        )
        if is_qr:
            owner = next(
                (x.name for x in reversed(self.stack) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))),
                None,
            )
            self.qr_functions.append(owner)
        self.generic_visit(node)


def assert_source(repo: Path, expected_sha: str) -> tuple[Path, str]:
    soap = (repo / TARGET_RELATIVE).resolve(strict=True)
    if repo.resolve() not in soap.parents:
        raise RuntimeError("soap path escaped repo")
    digest = sha256_file(soap)
    if digest.lower() != expected_sha.lower():
        raise RuntimeError(f"soap SHA mismatch: actual={digest}")
    text = soap.read_text(encoding="utf-8")
    if text.count(TARGET_NEEDLE) != 1:
        raise RuntimeError("target QR textual context is not unique")
    visitor = ParentMap()
    visitor.visit(ast.parse(text, filename=str(soap)))
    if visitor.qr_functions != [TARGET_FUNCTION]:
        raise RuntimeError(f"unexpected torch.linalg.qr AST contexts: {visitor.qr_functions}")
    return soap, digest


def load_adapter(path: Path, context: dict[str, Any]) -> Any:
    spec = importlib.util.spec_from_file_location("step215_e_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load adapter module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    readiness = getattr(module, "READINESS", None)
    missing_flags = [x for x in REQUIRED_READINESS if not isinstance(readiness, dict) or readiness.get(x) is not True]
    if missing_flags:
        raise RuntimeError("adapter is fail-closed; missing readiness: " + ",".join(missing_flags))
    factory = getattr(module, "create_adapter", None)
    if not callable(factory):
        raise RuntimeError("adapter lacks create_adapter(context)")
    adapter = factory(context)
    missing_methods = [x for x in REQUIRED_METHODS if not callable(getattr(adapter, x, None))]
    if missing_methods:
        raise RuntimeError("adapter missing methods: " + ",".join(missing_methods))
    return adapter


def tensor_bytes(t: Any) -> bytes:
    x = t.detach().contiguous().cpu()
    try:
        return x.numpy().tobytes()
    except Exception:
        import torch

        return bytes(x.view(torch.uint8).reshape(-1).tolist())


def digest_tree(obj: Any) -> str:
    h = hashlib.sha256()

    def visit(x: Any) -> None:
        if hasattr(x, "detach") and hasattr(x, "dtype") and hasattr(x, "shape"):
            h.update(b"tensor\0")
            h.update(str(x.dtype).encode())
            h.update(str(tuple(x.shape)).encode())
            h.update(tensor_bytes(x))
        elif isinstance(x, dict):
            h.update(b"dict\0")
            for key in sorted(x, key=lambda z: repr(z)):
                visit(key)
                visit(x[key])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode() + b"\0")
            for item in x:
                visit(item)
        elif isinstance(x, (str, int, float, bool, type(None))):
            h.update((type(x).__name__ + ":" + repr(x)).encode())
        else:
            raise TypeError(f"unsupported replay/state leaf {type(x)!r}")

    visit(obj)
    return h.hexdigest()


def clone_tree(obj: Any) -> Any:
    if hasattr(obj, "detach") and hasattr(obj, "clone"):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: clone_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_tree(v) for v in obj)
    return copy.deepcopy(obj)


def flatten(obj: Any, path: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    out: dict[tuple[str, ...], Any] = {}
    if isinstance(obj, dict):
        for key in sorted(obj, key=lambda z: repr(z)):
            out.update(flatten(obj[key], path + (str(key),)))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            out.update(flatten(value, path + (str(i),)))
    else:
        out[path] = obj
    return out


def schema(obj: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in flatten(obj).items():
        name = "/".join(key)
        if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "device"):
            result[name] = f"tensor:{tuple(value.shape)}:{value.dtype}:{value.device}"
        else:
            result[name] = type(value).__name__
    return result


def is_q_path(path: tuple[str, ...]) -> bool:
    return any(part == "Q" for part in path)


def is_step_path(path: tuple[str, ...]) -> bool:
    return bool(path) and path[-1] == "step"


def finite_tensor(x: Any) -> bool:
    import torch

    if not (x.is_floating_point() or x.is_complex()):
        return True
    return bool(torch.isfinite(x).all().item())


def validate_q_contract(label: str, obj: Any) -> dict[str, Any]:
    """Validate every materialized Q independently of cross-track basis choice."""
    import torch

    q_count = 0
    worst_orthogonality_max_abs = 0.0
    worst_path = ""
    for path, value in flatten(obj).items():
        if not is_q_path(path) or not hasattr(value, "dtype") or not hasattr(value, "shape"):
            continue
        q_count += 1
        path_text = "/".join(path)
        if not finite_tensor(value):
            raise RuntimeError(f"{label}: nonfinite Q at {path_text}")
        if not (value.is_floating_point() or value.is_complex()):
            raise RuntimeError(f"{label}: Q is not floating/complex at {path_text}")
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise RuntimeError(f"{label}: Q is not square at {path_text}")
        q = value.detach().to(torch.complex128 if value.is_complex() else torch.float64)
        identity = torch.eye(q.shape[1], dtype=q.dtype, device=q.device)
        gram = q.mH @ q if q.is_complex() else q.transpose(-2, -1) @ q
        orthogonality_max_abs = float(torch.max(torch.abs(gram - identity)).item())
        if not math.isfinite(orthogonality_max_abs):
            raise RuntimeError(f"{label}: nonfinite Q orthogonality metric at {path_text}")
        if orthogonality_max_abs > Q_ORTHOGONALITY_LIMIT:
            raise RuntimeError(
                f"{label}: Q orthogonality max-abs {orthogonality_max_abs} "
                f"> {Q_ORTHOGONALITY_LIMIT} at {path_text}"
            )
        if orthogonality_max_abs > worst_orthogonality_max_abs:
            worst_orthogonality_max_abs = orthogonality_max_abs
            worst_path = path_text
    if q_count == 0:
        raise RuntimeError(f"{label}: no materialized Q tensor found")
    return {
        "q_tensor_count": q_count,
        "q_orthogonality_limit": Q_ORTHOGONALITY_LIMIT,
        "q_worst_orthogonality_max_abs": worst_orthogonality_max_abs,
        "q_worst_orthogonality_path": worst_path,
    }


def basis_relaxed_for_comparison(label: str, enabled: bool) -> bool:
    """Keep relaxation scoped to the two explicit cross-implementation comparisons."""
    return bool(enabled and label in BASIS_RELAXED_COMPARISONS)


def enforce_distance_limits(
    label: str,
    q_worst_nrmse: float,
    other_worst_relative_l2: float,
    other_worst_path: str,
    other_global_relative_l2: float,
    q_limit: float,
    other_limit: float,
    ignore_q_distance: bool,
) -> None:
    """Enforce relaxation as a Q-only exception; non-Q checks are unconditional."""
    if not (0.0 < q_limit <= Q_LIMIT and 0.0 < other_limit <= OTHER_LIMIT):
        raise RuntimeError(f"{label}: invalid effective limits")
    if q_worst_nrmse > q_limit and not ignore_q_distance:
        raise RuntimeError(f"{label}: Q NRMSE {q_worst_nrmse} > {q_limit}")
    if other_worst_relative_l2 > other_limit:
        raise RuntimeError(
            f"{label}: non-Q tensor relative-L2 {other_worst_relative_l2} "
            f"> {other_limit} at {other_worst_path}"
        )
    if other_global_relative_l2 > other_limit:
        raise RuntimeError(
            f"{label}: other relative-L2 {other_global_relative_l2} > {other_limit}"
        )


def compare_views(
    label: str,
    left: Any,
    right: Any,
    q_limit: float = Q_LIMIT,
    other_limit: float = OTHER_LIMIT,
    ignore_q_distance: bool = False,
) -> dict[str, Any]:
    import torch

    ls, rs = schema(left), schema(right)
    if ls != rs:
        raise RuntimeError(f"{label}: schema mismatch")
    left_q_contract = validate_q_contract(f"{label}:left", left)
    right_q_contract = validate_q_contract(f"{label}:right", right)
    lf, rf = flatten(left), flatten(right)
    q_worst_nrmse = 0.0
    q_worst_path = ""
    other_num = 0.0
    other_den = 0.0
    other_worst_relative_l2 = 0.0
    other_worst_path = ""
    for path in lf:
        a, b = lf[path], rf[path]
        if hasattr(a, "dtype") and hasattr(a, "shape"):
            if not finite_tensor(a) or not finite_tensor(b):
                raise RuntimeError(f"{label}: nonfinite at {'/'.join(path)}")
            if is_step_path(path):
                if not torch.equal(a, b):
                    raise RuntimeError(f"{label}: state.step tensor mismatch")
                continue
            if not (a.is_floating_point() or a.is_complex()):
                if not torch.equal(a, b):
                    raise RuntimeError(f"{label}: discrete tensor mismatch at {'/'.join(path)}")
                continue
            da = a.detach().to(torch.float64)
            db = b.detach().to(torch.float64)
            diff2 = float(torch.sum((da - db) ** 2).item())
            den2 = max(float(torch.sum(da**2).item()), 1.0e-30)
            if is_q_path(path):
                nrmse = math.sqrt(diff2 / max(da.numel(), 1)) / max(
                    math.sqrt(den2 / max(da.numel(), 1)), 1.0e-30
                )
                if nrmse > q_worst_nrmse:
                    q_worst_nrmse, q_worst_path = nrmse, "/".join(path)
            else:
                other_num += diff2
                other_den += den2
                relative_l2 = math.sqrt(diff2 / max(den2, 1.0e-30))
                if relative_l2 > other_worst_relative_l2:
                    other_worst_relative_l2 = relative_l2
                    other_worst_path = "/".join(path)
        elif is_step_path(path):
            if a != b:
                raise RuntimeError(f"{label}: state.step scalar mismatch")
        elif a != b:
            raise RuntimeError(f"{label}: non-tensor state mismatch at {'/'.join(path)}")
    other_rel_l2 = math.sqrt(other_num / max(other_den, 1.0e-30))
    enforce_distance_limits(
        label,
        q_worst_nrmse,
        other_worst_relative_l2,
        other_worst_path,
        other_rel_l2,
        q_limit,
        other_limit,
        ignore_q_distance,
    )
    return {
        "label": label,
        "q_worst_nrmse": q_worst_nrmse,
        "q_worst_path": q_worst_path,
        "other_global_relative_l2": other_rel_l2,
        "q_limit": q_limit,
        "q_distance_checked": not ignore_q_distance,
        "basis_relaxed_diagnostic": ignore_q_distance,
        "left_q_contract": left_q_contract,
        "right_q_contract": right_q_contract,
        "other_worst_tensor_relative_l2": other_worst_relative_l2,
        "other_worst_tensor_path": other_worst_path,
        "other_limit": other_limit,
    }


def adaptive_limits(*baseline_metrics: dict[str, Any]) -> tuple[float, float]:
    if not baseline_metrics:
        raise RuntimeError("adaptive limits require baseline self-noise")
    q_noise = max(float(x["q_worst_nrmse"]) for x in baseline_metrics)
    other_noise = max(float(x["other_global_relative_l2"]) for x in baseline_metrics)
    return (
        min(Q_LIMIT, max(1.0e-5, 2.0 * q_noise)),
        min(OTHER_LIMIT, max(1.0e-5, 2.0 * other_noise)),
    )


class OpObserver(contextlib.AbstractContextManager):
    def __init__(self, torch: Any, soap_path: Path, candidate: bool) -> None:
        self.torch = torch
        self.soap_path = soap_path
        self.candidate = candidate
        self.qr_shapes: Counter[tuple[int, ...]] = Counter()
        self.sort_digests: list[str] = []
        self.sort_events: list[dict[str, Any]] = []
        self.candidate_guarded_count = 0
        self.candidate_fallback_count = 0
        self._qr = None
        self._argsort = None

    def _is_target_caller(self) -> bool:
        frame = inspect.currentframe()
        assert frame is not None
        frame = frame.f_back
        while frame is not None and frame.f_code.co_filename == __file__:
            frame = frame.f_back
        return bool(
            frame
            and Path(frame.f_code.co_filename).resolve() == self.soap_path
            and frame.f_code.co_name == TARGET_FUNCTION
        )

    def __enter__(self) -> "OpObserver":
        self._qr = self.torch.linalg.qr
        self._argsort = self.torch.argsort

        def qr_wrapper(a: Any, mode: str = "reduced", *, out: Any = None) -> Any:
            if not self._is_target_caller():
                return self._qr(a, mode=mode, out=out) if out is not None else self._qr(a, mode=mode)
            if mode != "reduced" or out is not None or a.ndim != 2 or a.shape[0] != a.shape[1]:
                raise RuntimeError("target QR call violated reduced/square/no-out contract")
            if self.torch.is_grad_enabled() or a.requires_grad:
                raise RuntimeError("target QR left SOAP no-grad boundary")
            self.qr_shapes[tuple(int(x) for x in a.shape)] += 1
            if not self.candidate:
                return self._qr(a, mode=mode)
            if not guarded_candidate_shape(a.shape):
                self.candidate_fallback_count += 1
                return self._qr(a, mode=mode)
            self.candidate_guarded_count += 1
            packed, tau = self.torch.geqrf(a)
            q = self.torch.orgqr(packed, tau).contiguous()
            if q.grad_fn is not None:
                raise RuntimeError("candidate Q unexpectedly has grad_fn")
            if not q.is_contiguous():
                raise RuntimeError("candidate Q layout normalization failed")
            return q, None

        def argsort_wrapper(input: Any, *args: Any, **kwargs: Any) -> Any:
            result = self._argsort(input, *args, **kwargs)
            if self._is_target_caller():
                descending = kwargs.get("descending", args[1] if len(args) > 1 else False)
                stable = kwargs.get("stable", args[2] if len(args) > 2 else False)
                if descending is not True or stable is not True:
                    raise RuntimeError("SOAP sort lost descending=True/stable=True")
                input_digest = hashlib.sha256(tensor_bytes(input)).hexdigest()
                result_digest = hashlib.sha256(tensor_bytes(result)).hexdigest()
                self.sort_digests.append(result_digest)
                self.sort_events.append(
                    {
                        "index": len(self.sort_events),
                        "input_digest": input_digest,
                        "result_digest": result_digest,
                        "input_shape": [int(x) for x in input.shape],
                        "input_dtype": str(input.dtype),
                    }
                )
            return result

        self.torch.linalg.qr = qr_wrapper
        self.torch.argsort = argsort_wrapper
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.torch.linalg.qr = self._qr
        self.torch.argsort = self._argsort


def cycle_payload(observer: OpObserver, step: int) -> dict[str, Any]:
    return {
        "logical_step": step,
        "qr_count": sum(observer.qr_shapes.values()),
        "qr_shapes": {"x".join(map(str, k)): v for k, v in sorted(observer.qr_shapes.items())},
        "sort_count": len(observer.sort_digests),
        "sort_digest": hashlib.sha256("".join(observer.sort_digests).encode()).hexdigest(),
        "sort_events": observer.sort_events,
        "candidate_guarded_count": observer.candidate_guarded_count,
        "candidate_fallback_count": observer.candidate_fallback_count,
    }


def event_diff(continuous: dict[str, Any], resumed: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(continuous) | set(resumed))
    differing = {
        key: {"continuous": continuous.get(key), "resume": resumed.get(key)}
        for key in keys
        if continuous.get(key) != resumed.get(key)
    }
    left_sorts = continuous.get("sort_events", [])
    right_sorts = resumed.get("sort_events", [])
    first_sort_difference = None
    for index in range(max(len(left_sorts), len(right_sorts))):
        left = left_sorts[index] if index < len(left_sorts) else None
        right = right_sorts[index] if index < len(right_sorts) else None
        if left != right:
            first_sort_difference = {
                "index": index,
                "continuous": left,
                "resume": right,
            }
            break
    return {
        "differing_fields": differing,
        "first_sort_difference": first_sort_difference,
    }


def assert_cycle_inventory(name: str, candidate: bool, event: dict[str, Any]) -> None:
    if event["qr_count"] != APPROVED_ACTIVE_QR_COUNT:
        raise RuntimeError(f"{name}: QR count {event['qr_count']} != 543")
    if event["qr_shapes"] != APPROVED_ACTIVE_QR_INVENTORY:
        raise RuntimeError(f"{name}: QR inventory differs from approved active 23-shape contract")
    if "5120x5120" in event["qr_shapes"]:
        raise RuntimeError(f"{name}: forbidden 5120 shape appeared in an active cycle")
    if candidate:
        if event["candidate_guarded_count"] != APPROVED_ACTIVE_QR_COUNT:
            raise RuntimeError(f"{name}: candidate did not guard all 543 approved calls")
        if event["candidate_fallback_count"] != 0:
            raise RuntimeError(f"{name}: unexpected fallback inside approved active inventory")
    elif event["candidate_guarded_count"] or event["candidate_fallback_count"]:
        raise RuntimeError(f"{name}: baseline unexpectedly entered candidate routing")


def execute_step(adapter: Any, trial: Any, gradient: Any, logical_step: int, torch: Any, soap: Path, candidate: bool) -> dict[str, Any]:
    before = digest_tree(gradient)
    with OpObserver(torch, soap, candidate) as observer:
        adapter.apply_gradient(trial, clone_tree(gradient), logical_step)
    if digest_tree(gradient) != before:
        raise RuntimeError("canonical replay gradient mutated")
    return cycle_payload(observer, logical_step)


def run_track(
    name: str,
    adapter: Any,
    context: dict[str, Any],
    torch: Any,
    soap: Path,
    gradient_digests: dict[int, str],
    discover_gradients: bool,
    expected_cycle_steps: list[int] | None,
    max_steps: int,
    track_dir: Path,
) -> dict[str, Any]:
    candidate = name == "candidate"
    track_dir.mkdir(parents=True, exist_ok=False)
    continuous = adapter.build_trial(context)
    initial_view = clone_tree(adapter.state_view(continuous))
    if not isinstance(initial_view, dict) or set(initial_view) != {"parameters", "optimizer_state"}:
        raise RuntimeError("state_view must have exactly parameters and optimizer_state roots")
    if not any(is_q_path(path) for path in flatten(initial_view)):
        raise RuntimeError("state_view contains no persistent Q tensor")
    cycles: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    cycle1_view = None
    resumed = None
    checkpoint_path = track_dir / "cycle1_checkpoint.pt"
    for step in range(max_steps):
        gradient = adapter.make_gradient(context, step)
        gradient_digest = digest_tree(gradient)
        if discover_gradients:
            gradient_digests[step] = gradient_digest
        elif gradient_digests.get(step) != gradient_digest:
            raise RuntimeError(f"{name}: replay gradient digest mismatch at step {step}")
        if candidate and resumed is not None and expected_cycle_steps and step == expected_cycle_steps[1]:
            atomic_json(
                track_dir / f"step{step}_pre_layout.json",
                {
                    "logical_step": step,
                    "continuous": adapter.layout_view(continuous),
                    "resume": adapter.layout_view(resumed),
                },
            )
        event = execute_step(adapter, continuous, gradient, step, torch, soap, candidate)
        event["gradient_digest"] = gradient_digest
        resumed_event = None
        if resumed is not None:
            resumed_event = execute_step(adapter, resumed, gradient, step, torch, soap, candidate)
            if resumed_event != {k: v for k, v in event.items() if k != "gradient_digest"}:
                continuous_event = {k: v for k, v in event.items() if k != "gradient_digest"}
                atomic_json(
                    track_dir / f"op_event_mismatch_step{step}.json",
                    {
                        "track": name,
                        "logical_step": step,
                        "continuous_event": continuous_event,
                        "resume_event": resumed_event,
                        **event_diff(continuous_event, resumed_event),
                    },
                )
                raise RuntimeError(f"{name}: continuous/resume op event mismatch at step {step}")
        if event["qr_count"]:
            assert_cycle_inventory(name, candidate, event)
            if event["sort_count"] == 0:
                raise RuntimeError(f"{name}: QR cycle had no observed stable sort")
            cycles.append(event)
            view = clone_tree(adapter.state_view(continuous))
            if len(cycles) == 1:
                cycle1_view = view
                if candidate:
                    atomic_json(
                        track_dir / "cycle1_pre_save_layout.json",
                        adapter.layout_view(continuous),
                    )
                adapter.save_trial(continuous, checkpoint_path)
                if not checkpoint_path.is_file():
                    raise RuntimeError("adapter did not create cycle1 checkpoint")
                checkpoint_sha = sha256_file(checkpoint_path)
                resumed = adapter.load_trial(context, checkpoint_path)
                if candidate:
                    atomic_json(
                        track_dir / "cycle1_post_load_layout.json",
                        adapter.layout_view(resumed),
                    )
                resumed_view = clone_tree(adapter.state_view(resumed))
                comparisons.append(compare_views(f"{name}:cycle1-save-load", view, resumed_view))
            elif len(cycles) == 2:
                if resumed is None or resumed_event is None:
                    raise RuntimeError("resume branch absent at second cycle")
                comparisons.append(
                    compare_views(f"{name}:cycle2-continuous-resume", view, clone_tree(adapter.state_view(resumed)))
                )
                break
    if len(cycles) != 2:
        raise RuntimeError(f"{name}: found {len(cycles)} QR cycles before max steps")
    steps = [x["logical_step"] for x in cycles]
    if expected_cycle_steps is not None and steps != expected_cycle_steps:
        raise RuntimeError(f"{name}: QR cycle steps {steps} != {expected_cycle_steps}")
    final_cont = clone_tree(adapter.state_view(continuous))
    final_resume = clone_tree(adapter.state_view(resumed))
    adapter.destroy_trial(continuous)
    adapter.destroy_trial(resumed)
    return {
        "name": name,
        "candidate": candidate,
        "initial_view": initial_view,
        "cycle1_view": cycle1_view,
        "final_continuous_view": final_cont,
        "final_resume_view": final_resume,
        "cycles": cycles,
        "checkpoint_sha256": checkpoint_sha,
        "resume_comparisons": comparisons,
    }


def public_track(track: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in track.items() if not k.endswith("_view")}


def main() -> int:
    args = parse_args()
    configure_q_orthogonality_limit(args.q_orthogonality_limit)
    assert_guard_contract()
    repo = Path(args.repo).resolve(strict=True)
    config = Path(args.config).resolve(strict=True)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    adapter_path = Path(args.adapter).resolve(strict=True)
    output = Path(args.output_dir).resolve(strict=True)
    if repo in adapter_path.parents:
        raise RuntimeError("adapter must remain outside the business repository")
    if output == repo or repo in output.parents:
        raise RuntimeError("output must remain outside the business repository")
    if args.max_optimizer_steps < 2:
        raise RuntimeError("max optimizer steps must be >=2")
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    failure = output / "failure" / f"rank{rank}.txt"
    failure.parent.mkdir(exist_ok=True)
    try:
        if world_size != 8 or rank not in range(8) or local_rank not in range(8) or visible != EXPECTED_VISIBLE:
            raise RuntimeError("requires world8 mapped to ASCEND_RT_VISIBLE_DEVICES=8..15")
        if os.environ.get("TASK_QUEUE_ENABLE") is not None:
            raise RuntimeError("TASK_QUEUE_ENABLE must be absent for this single-variable gate")
        if os.environ.get("PYTORCH_NPU_ALLOC_CONF") != "expandable_segments:True":
            raise RuntimeError("allocator contract mismatch")
        soap, soap_sha = assert_source(repo, args.expected_soap_sha256)
        status_before = git_status(repo)
        import torch
        import torch.distributed as dist
        import torch_npu  # noqa: F401

        dist.init_process_group("hccl")
        torch.npu.set_device(local_rank)
        device = f"npu:{local_rank}"
        context = {
            "repo": str(repo),
            "config": str(config),
            "checkpoint": str(checkpoint),
            "output_dir": str(output),
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "device": device,
        }
        adapter = load_adapter(adapter_path, context)
        ready_dir = output / "ready"
        ready_dir.mkdir(exist_ok=True)
        atomic_json(
            ready_dir / f"rank{rank}.json",
            {"rank": rank, "local_rank": local_rank, "world_size": world_size, "visible": visible, "gate_pass": True},
        )
        release = output / "release_after_npu_smi"
        deadline = time.monotonic() + 180.0
        while not release.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("controller did not release ranks")
            time.sleep(0.1)
        rank_dir = output / f"rank{rank}"
        rank_dir.mkdir(exist_ok=True)
        gradient_digests: dict[int, str] = {}
        a = run_track("baseline-A", adapter, context, torch, soap, gradient_digests, True, None, args.max_optimizer_steps, rank_dir / "baseline-A")
        cycle_steps = [x["logical_step"] for x in a["cycles"]]
        b = run_track("baseline-B", adapter, context, torch, soap, gradient_digests, False, cycle_steps, args.max_optimizer_steps, rank_dir / "baseline-B")
        c = run_track("candidate", adapter, context, torch, soap, gradient_digests, False, cycle_steps, args.max_optimizer_steps, rank_dir / "candidate")
        if digest_tree(a["initial_view"]) != digest_tree(b["initial_view"]) or digest_tree(a["initial_view"]) != digest_tree(c["initial_view"]):
            raise RuntimeError("three tracks did not start from the exact same snapshot")
        comparisons = []
        comparisons.append(compare_views("initial-baselineA-baselineB", a["initial_view"], b["initial_view"]))
        comparisons.append(compare_views("initial-baselineA-candidate", a["initial_view"], c["initial_view"]))
        baseline_cycle1 = compare_views("cycle1-baselineA-baselineB", a["cycle1_view"], b["cycle1_view"])
        comparisons.append(baseline_cycle1)
        q1_limit, other1_limit = adaptive_limits(baseline_cycle1)
        comparisons.append(
            compare_views(
                "cycle1-baselineA-candidate-adaptive",
                a["cycle1_view"],
                c["cycle1_view"],
                q_limit=q1_limit,
                other_limit=other1_limit,
                ignore_q_distance=basis_relaxed_for_comparison(
                    "cycle1-baselineA-candidate-adaptive", args.basis_relaxed_diagnostic
                ),
            )
        )
        baseline_cycle2 = compare_views(
            "cycle2-baselineA-baselineB", a["final_continuous_view"], b["final_continuous_view"]
        )
        comparisons.append(baseline_cycle2)
        q2_limit, other2_limit = adaptive_limits(baseline_cycle2)
        comparisons.append(
            compare_views(
                "cycle2-baselineA-candidate-adaptive",
                a["final_continuous_view"],
                c["final_continuous_view"],
                q_limit=q2_limit,
                other_limit=other2_limit,
                ignore_q_distance=basis_relaxed_for_comparison(
                    "cycle2-baselineA-candidate-adaptive", args.basis_relaxed_diagnostic
                ),
            )
        )
        baseline_resume_a = compare_views(
            "resume-self-baselineA", a["final_continuous_view"], a["final_resume_view"]
        )
        baseline_resume_b = compare_views(
            "resume-self-baselineB", b["final_continuous_view"], b["final_resume_view"]
        )
        comparisons.extend((baseline_resume_a, baseline_resume_b))
        qr_limit, otherr_limit = adaptive_limits(baseline_resume_a, baseline_resume_b)
        comparisons.append(
            compare_views(
                "resume-candidate-adaptive",
                c["final_continuous_view"],
                c["final_resume_view"],
                q_limit=qr_limit,
                other_limit=otherr_limit,
            )
        )
        if [x["qr_count"] for x in a["cycles"]] != [x["qr_count"] for x in b["cycles"]] or [x["qr_count"] for x in a["cycles"]] != [x["qr_count"] for x in c["cycles"]]:
            raise RuntimeError("QR call counts differ across tracks")
        if [x["qr_shapes"] for x in a["cycles"]] != [x["qr_shapes"] for x in b["cycles"]] or [x["qr_shapes"] for x in a["cycles"]] != [x["qr_shapes"] for x in c["cycles"]]:
            raise RuntimeError("QR shapes differ across tracks")
        if [x["sort_digest"] for x in a["cycles"]] != [x["sort_digest"] for x in b["cycles"]] or [x["sort_digest"] for x in a["cycles"]] != [x["sort_digest"] for x in c["cycles"]]:
            raise RuntimeError("sort_idx digest differs across tracks")
        if sha256_file(soap) != soap_sha or git_status(repo) != status_before:
            raise RuntimeError("business repository changed during harness")
        result = {
            "status": "PASS",
            "rank": rank,
            "soap_sha256": soap_sha,
            "config_sha256": sha256_file(config),
            "checkpoint_sha256": sha256_file(checkpoint),
            "adapter_sha256": sha256_file(adapter_path),
            "gate_mode": (
                "basis_relaxed_diagnostic" if args.basis_relaxed_diagnostic else "strict_raw_q"
            ),
            "basis_relaxed_diagnostic": bool(args.basis_relaxed_diagnostic),
            "cycle_steps": cycle_steps,
            "gradient_digests": {str(k): v for k, v in gradient_digests.items()},
            "tracks": [public_track(x) for x in (a, b, c)],
            "cross_track_comparisons": comparisons,
            "adaptive_limits": {
                "cycle1": {"q": q1_limit, "other": other1_limit},
                "cycle2": {"q": q2_limit, "other": other2_limit},
                "resume": {"q": qr_limit, "other": otherr_limit},
            },
        }
        done = output / "done"
        done.mkdir(exist_ok=True)
        atomic_json(done / f"rank{rank}.json", result)
        dist.barrier()
        if rank == 0:
            rows = [json.loads((done / f"rank{i}.json").read_text(encoding="utf-8")) for i in range(8)]
            atomic_json(
                output / "world_summary.json",
                {
                    "status": "PASS",
                    "rank_count": 8,
                    "all_rank_pass": all(x["status"] == "PASS" for x in rows),
                    "q_limit": Q_LIMIT,
                    "other_state_parameter_limit": OTHER_LIMIT,
                    "q_orthogonality_limit": Q_ORTHOGONALITY_LIMIT,
                    "gate_mode": (
                        "basis_relaxed_diagnostic"
                        if args.basis_relaxed_diagnostic
                        else "strict_raw_q"
                    ),
                    "basis_relaxed_diagnostic": bool(args.basis_relaxed_diagnostic),
                    "cycle_steps_by_rank": [x["cycle_steps"] for x in rows],
                },
            )
        dist.barrier()
        dist.destroy_process_group()
        return 0
    except Exception as exc:
        failure.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    sys.exit(main())
