#!/usr/bin/env python3
"""Pure-stdlib fail-closed policy for the STEP-216 Brockett core probe."""
from __future__ import annotations

import json
import hashlib
import inspect
import os
import re
from pathlib import Path
from typing import Any


SCHEMA = "step216_turbosoap_brockett_core_v1"
REVISION = "1339218c180312b6ed1b04013fd910df9aff6ee7"
SOURCE_BLOB = "d1563b35096440d4374c4a2e784dd652d804954e"
SOAP_BOUND_SIGNATURE = "(grad, state, merge_dims=False, max_precond_dim=10000)"
APPROVED_COUNTS = {
    1: 106, 3: 30, 4: 6, 7: 37, 8: 1, 11: 1, 22: 1, 32: 4,
    40: 9, 64: 28, 96: 3, 120: 1, 128: 18, 160: 1, 192: 32,
    220: 4, 256: 181, 352: 1, 440: 4, 512: 43, 768: 22,
    1024: 6, 2560: 4,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_verified_policy(path: str | Path) -> dict[str, Any]:
    """Load only the pinned, explicitly verified TurboSOAP core-probe contract."""
    source = Path(path).resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(set(payload) == {"schema", "authority", "probe_scope", "algorithm", "retraction"}, "policy top-level schema changed")
    _require(payload["schema"] == SCHEMA, "policy schema is not approved")

    authority = payload["authority"]
    _require(authority.get("status") == "verified_primary_source", "community source is not verified")
    _require(authority.get("revision") == REVISION, "community revision changed")
    _require(authority.get("source_blob") == SOURCE_BLOB, "community source blob changed")
    _require(authority.get("license") == "Apache-2.0", "community license changed")
    _require(authority.get("source_url") == f"https://github.com/ethansmith2000/TurboSOAP/blob/{REVISION}/soap.py", "community source URL changed")

    scope = payload["probe_scope"]
    _require(scope == {
        "candidate_accept_controller": "disabled",
        "dtype": "float32",
        "persistent_direction_ema": "disabled",
        "purpose": "isolated_core_operator_screen_only",
        "stable_sort": True,
    }, "probe scope widened or changed")

    algorithm = payload["algorithm"]
    _require(algorithm == {
        "basis_weight_exponent": 1.0,
        "basis_weight_normalization": "mean_one",
        "basis_weight_order": "n_to_1",
        "direction": "plus",
        "direction_normalizer": "disabled_for_core_probe",
        "eigengap_preconditioner": "disabled_for_core_probe",
        "eta": 0.01,
        "matrix_normalization": "abs_trace_over_n",
        "matrix_normalization_floor": 1e-12,
        "substeps": 1,
    }, "Brockett core parameters differ from the pinned probe")

    retraction = payload["retraction"]
    _require(retraction == {
        "coefficient_g": -10.0,
        "coefficient_g2": 3.0,
        "coefficient_z": 15.0,
        "count": 1,
        "kind": "scaled_single_cubic_polar",
        "outer_scale": 0.125,
        "row_sum_target": 1.25,
        "scale_epsilon": 1e-7,
    }, "cubic polar retraction parameters changed")
    return payload


def dispatch_candidate(dimension: int, *, square: bool, dtype: str, contiguous: bool, requires_grad: bool) -> bool:
    """The production-like dispatch guard; every unknown contract falls back."""
    return (
        dimension in APPROVED_COUNTS
        and dimension != 5120
        and square
        and dtype == "torch.float32"
        and contiguous
        and not requires_grad
    )


def assert_inventory(counts: dict[int, int]) -> None:
    normalized = {int(k): int(v) for k, v in counts.items() if int(v) != 0}
    _require(normalized == APPROVED_COUNTS, "active Q inventory is not the approved 23-shape contract")
    _require(sum(normalized.values()) == 543 and normalized.get(5120, 0) == 0, "active Q inventory is not 543/no-5120")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_contract(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "step216_source_contract_v3", "source contract schema changed")
    _require(payload.get("runnable") is True, "source contract is not runnable")
    _require(set(payload) == {"schema", "runnable", "runtime_artifacts", "source_files", "runtime_protocol_schema", "soap_runtime_schema"}, "source contract fields changed")
    _require(payload["runtime_protocol_schema"] == {
        "controller_execution": "host_python",
        "container_runner_controller": "disabled",
        "ready_pid_field": "container_pid",
        "npu_smi_pid_field": "host_pid",
        "host_to_container_mapping": "proc_status_nspid_last",
    }, "runtime PID protocol changed")
    runtime = payload["runtime_artifacts"]
    _require(set(runtime) == {"adapter", "config", "checkpoint", "soap", "community_config"}, "runtime artifact set changed")
    files = payload["source_files"]
    _require(isinstance(files, list) and len(files) >= 8, "source file inventory is incomplete")
    names = [row.get("name") for row in files]
    _require(len(names) == len(set(names)) and "step216_a_source_contract.json" not in names, "source manifest is recursive or duplicated")
    for row in list(runtime.values()) + files:
        _require(set(row) >= {"name", "sha256", "bytes"}, "source identity row is incomplete")
        _require(isinstance(row["bytes"], int) and row["bytes"] > 0, "source bytes are not pinned")
        _require(isinstance(row["sha256"], str) and len(row["sha256"]) == 64, "source SHA256 is not pinned")
    schema = payload["soap_runtime_schema"]
    _require(schema == {
        "checkpoint_top_level_keys": ["meta", "optimizer", "state_dict"],
        "optimizer_group_count": 767,
        "stateful_group_count": 559,
        "state_keys": ["GG", "Q", "exp_avg", "exp_avg_sq", "precondition_frequency", "shampoo_beta", "step"],
        "state_step": 26,
        "q_factor_count": 543,
        "q_shape_count": 23,
        "project_bound_signature": SOAP_BOUND_SIGNATURE,
        "project_back_bound_signature": SOAP_BOUND_SIGNATURE,
    }, "SOAP runtime schema contract changed")
    return payload


def verify_identity(path: str | Path, identity: dict[str, Any]) -> Path:
    source = Path(path).resolve(strict=True)
    _require(not source.is_symlink(), f"identity path is a symlink: {source.name}")
    _require(source.name == identity["name"], f"identity basename changed: {source.name}")
    _require(source.stat().st_size == identity["bytes"], f"identity bytes changed: {source.name}")
    _require(sha256_file(source) == identity["sha256"], f"identity SHA changed: {source.name}")
    return source


def verify_source_package(contract: dict[str, Any], source_root: str | Path) -> None:
    root = Path(source_root).resolve(strict=True)
    for identity in contract["source_files"]:
        verify_identity(root / identity["name"], identity)


def assert_tool_layout(
    tool_root: str | Path, repo: str | Path, adapter: str | Path, output: str | Path
) -> dict[str, Path]:
    """Require tools/state outside the business repo after realpath resolution."""
    tool = Path(tool_root).resolve(strict=True)
    business = Path(repo).resolve(strict=True)
    adapter_path = Path(adapter).resolve(strict=True)
    output_path = Path(output).resolve(strict=False)

    def within(child: Path, parent: Path) -> bool:
        return os.path.commonpath((str(child), str(parent))) == str(parent)

    _require(tool != business and not within(tool, business), "tool root must be outside business repo")
    _require(within(adapter_path, tool / "harness"), "adapter must be inside tool-root harness")
    _require(within(output_path, tool / "runs"), "output must be inside tool-root runs")
    _require(not within(adapter_path, business), "adapter must be outside business repo")
    _require(not within(output_path, business), "output must be outside business repo")
    return {"tool_root": tool, "repo": business, "adapter": adapter_path, "output": output_path}


def parse_positive_pgid(text: str) -> int:
    """Parse the exact on-disk PGID grammar used by fail-closed cleanup."""
    _require(re.fullmatch(r"[1-9][0-9]*\n?", text) is not None, "PGID is not a strict positive integer")
    value = int(text.strip())
    _require(value > 1, "PGID must not target init")
    return value


def assert_bound_project_signature(method: Any, expected: str) -> None:
    signature = inspect.signature(method)
    parameters = list(signature.parameters.values())
    _require(expected == SOAP_BOUND_SIGNATURE, "source contract method signature is not approved")
    _require(
        [parameter.name for parameter in parameters]
        == ["grad", "state", "merge_dims", "max_precond_dim"],
        f"SOAP method parameters changed: {signature}",
    )
    _require(parameters[2].default is False, f"SOAP merge_dims default changed: {signature}")
    _require(parameters[3].default == 10000, f"SOAP max_precond_dim default changed: {signature}")


def state_q_view(state: dict[str, Any], q_values: list[Any]) -> dict[str, Any]:
    """Make a non-mutating shallow state view with an isolated Q list."""
    _require("Q" in state, "SOAP state has no Q")
    view = dict(state)
    view["Q"] = list(q_values)
    return view


def project_roundtrip_views(
    optimizer: Any,
    exp_avg: Any,
    state: dict[str, Any],
    baseline_q: list[Any],
    candidate_q: list[Any],
) -> tuple[Any, Any, Any]:
    old_view = state_q_view(state, state["Q"])
    baseline_view = state_q_view(state, baseline_q)
    candidate_view = state_q_view(state, candidate_q)
    original = optimizer.project_back(
        exp_avg, old_view, merge_dims=False, max_precond_dim=10000
    )
    baseline = optimizer.project(
        original, baseline_view, merge_dims=False, max_precond_dim=10000
    )
    candidate = optimizer.project(
        original, candidate_view, merge_dims=False, max_precond_dim=10000
    )
    return original, baseline, candidate


def _main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    contract = load_source_contract(args.source_contract)
    verify_source_package(contract, args.source_root)
    print("source_contract=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
