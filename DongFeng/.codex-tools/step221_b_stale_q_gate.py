#!/usr/bin/env python3
"""STEP-221 Stage B: single-card correctness gate for the stale-Q candidate.

Runs on the real iter30 checkpoint (559 SOAP states / 543 Q factors) using the
patched SOAP loaded from the tool root. The business repository is only read.

Gates (all must pass; any failure closes the candidate):
  T1 trio_equivalence   _qr_plan+_qr_finish+_qr_install == get_orthogonal_matrix_QR,
                        bit-for-bit on both Q and exp_avg_sq, for all 559 states.
  T2 async_exact        With k>0 the installed Q of the first cycle is bit-identical
                        to the synchronous run's Q for that same cycle.
  T3 pending_discipline At most one pending basis per state; install lands exactly
                        on step_at_submit + k; no pending survives the run.
  T4 schema_stable      state_dict() during a pending window force-installs and
                        yields exactly the 7-key SOAP state contract.
  T6 memory             allocated/reserved deltas across the pending window are recorded.

Real save/resume equivalence is deliberately left to Stage C, where the actual
training checkpoint path exercises it end to end.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch_npu  # noqa: F401

EXPECTED_CHECKPOINT_SHA256 = "f001a7d55c19b74d84dd1384f262acef786237822e9581203176853d735f997d"
EXPECTED_CHECKPOINT_BYTES = 1_607_991_401
EXPECTED_STATE_KEYS = {
    "GG", "Q", "exp_avg", "exp_avg_sq", "precondition_frequency", "shampoo_beta", "step",
}
EXPECTED_Q_INVENTORY = {
    1: 106, 3: 30, 4: 6, 7: 37, 8: 1, 11: 1, 22: 1, 32: 4,
    40: 9, 64: 28, 96: 3, 120: 1, 128: 18, 160: 1, 192: 32,
    220: 4, 256: 181, 352: 1, 440: 4, 512: 43, 768: 22,
    1024: 6, 2560: 4,
}


def fail(message: str) -> None:
    raise SystemExit(f"GATE_FAIL: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def q_digest(qlist: Any) -> str:
    parts = []
    for value in qlist:
        parts.append(tensor_digest(value) if torch.is_tensor(value) else "empty")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def all_q_digest(optimizer, params) -> str:
    """Aggregate digest over every Q factor of every state."""
    digest = hashlib.sha256()
    for parameter in params:
        digest.update(q_digest(optimizer.state[parameter]["Q"]).encode())
    return digest.hexdigest()


def load_soap_class(soap_path: Path):
    spec = importlib.util.spec_from_file_location("step221_patched_soap", str(soap_path))
    if spec is None or spec.loader is None:
        fail("cannot import patched SOAP")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "SOAP", None)
    if cls is None:
        fail("patched module does not define SOAP")
    for name in ("_qr_plan", "_qr_finish", "_qr_install", "_stale_q_submit",
                 "_stale_q_install_if_due", "_stale_q_eligible", "state_dict"):
        if not hasattr(cls, name):
            fail(f"patched SOAP lacks {name}")
    return cls


def load_checkpoint_subset(checkpoint: Path) -> tuple[dict[str, Any], tuple[int, ...]]:
    if checkpoint.stat().st_size != EXPECTED_CHECKPOINT_BYTES:
        fail("checkpoint byte size changed")
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        fail("checkpoint SHA is not the qualified iter30 checkpoint")
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if set(payload) != {"meta", "optimizer", "state_dict"}:
        fail("checkpoint top-level schema changed")
    source = payload["optimizer"]
    del payload
    groups = source["param_groups"]
    states = source["state"]
    if len(groups) != 767:
        fail("checkpoint no longer has 767 parameter groups")
    flat_ids = [int(group["params"][0]) for group in groups]
    source_ids = tuple(pid for pid in flat_ids if pid in states)
    if len(source_ids) != 559:
        fail("checkpoint no longer has 559 stateful parameters")
    if any(set(states[pid]) != EXPECTED_STATE_KEYS for pid in source_ids):
        fail("SOAP state key schema changed")
    counts: Counter[int] = Counter()
    for pid in source_ids:
        for tensor in states[pid]["Q"]:
            if torch.is_tensor(tensor):
                counts[int(tensor.shape[0])] += 1
    if dict(sorted(counts.items())) != EXPECTED_Q_INVENTORY:
        fail("Q inventory is not the approved 23-shape/543 contract")
    group_by_id = {int(g["params"][0]): g for g in groups}
    remap = {old: new for new, old in enumerate(source_ids)}
    subset = {
        "state": {remap[old]: states[old] for old in source_ids},
        "param_groups": [
            {**{k: v for k, v in group_by_id[old].items() if k != "params"},
             "params": [remap[old]]}
            for old in source_ids
        ],
    }
    return subset, source_ids


def build_optimizer(soap_cls, subset: dict[str, Any], device: torch.device):
    params = []
    runtime_groups = []
    for index, group in enumerate(subset["param_groups"]):
        exp_avg = subset["state"][index]["exp_avg"]
        parameter = torch.nn.Parameter(
            torch.zeros(tuple(exp_avg.shape), dtype=exp_avg.dtype, device=device),
            requires_grad=True,
        )
        params.append(parameter)
        runtime_groups.append(
            {**{k: copy.deepcopy(v) for k, v in group.items() if k != "params"},
             "params": [parameter]}
        )
    optimizer = soap_cls(runtime_groups)
    optimizer.load_state_dict(subset)
    return optimizer, params


def apply_gradient(optimizer, params, source_ids, logical_step: int) -> None:
    for parameter, source_id in zip(params, source_ids):
        value = (((int(source_id) % 29) - 14) * 1.0e-4) + ((logical_step + 1) * 1.0e-6)
        parameter.grad = torch.empty_like(parameter).fill_(value)
    optimizer.step()


def test_trio_equivalence(optimizer, params, limit: int) -> dict[str, Any]:
    """T1: the refactored trio must reproduce the untouched function exactly."""
    checked = 0
    for parameter in params:
        state = optimizer.state[parameter]
        if state.get("Q") is None:
            continue
        reference = {
            "GG": [t.clone() if torch.is_tensor(t) else t for t in state["GG"]],
            "Q": [t.clone() if torch.is_tensor(t) else t for t in state["Q"]],
            "exp_avg_sq": state["exp_avg_sq"].clone(),
            "precondition_frequency": state["precondition_frequency"],
        }
        candidate_state = {
            "GG": [t.clone() if torch.is_tensor(t) else t for t in state["GG"]],
            "Q": [t.clone() if torch.is_tensor(t) else t for t in state["Q"]],
            "exp_avg_sq": state["exp_avg_sq"].clone(),
            "precondition_frequency": state["precondition_frequency"],
        }
        baseline_q = optimizer.get_orthogonal_matrix_QR(reference, 10000, False)
        plan = optimizer._qr_plan(candidate_state, 10000, False)
        qlist = optimizer._qr_finish(plan)
        trio_q = optimizer._qr_install(candidate_state, plan, qlist, 10000, False)
        if q_digest(baseline_q) != q_digest(trio_q):
            fail(f"T1 trio Q mismatch on parameter index {checked}")
        if tensor_digest(reference["exp_avg_sq"]) != tensor_digest(candidate_state["exp_avg_sq"]):
            fail(f"T1 trio exp_avg_sq mismatch on parameter index {checked}")
        checked += 1
        del reference, candidate_state, plan, qlist, baseline_q, trio_q
        if limit and checked >= limit:
            break
    torch.npu.empty_cache()
    return {"states_checked": checked}


def run_steps(optimizer, params, source_ids, steps: int, k: int) -> dict[str, Any]:
    timeline = []
    cycle_digests = []
    max_pending = 0
    torch.npu.reset_peak_memory_stats()
    base_alloc = torch.npu.memory_allocated()
    base_reserved = torch.npu.memory_reserved()
    for logical in range(steps):
        started = time.perf_counter()
        apply_gradient(optimizer, params, source_ids, logical)
        # Default stream only. A global npu.synchronize() would also wait on the
        # side stream and hide the very overlap under test; a real training step
        # syncs the default stream via .item()/DDP, not every stream.
        torch.npu.current_stream().synchronize()
        elapsed = time.perf_counter() - started
        pending = sum(
            1 for parameter in params
            if optimizer.state[parameter].get("_stale_q_pending") is not None
        )
        max_pending = max(max_pending, pending)
        step_value = int(optimizer.state[params[0]]["step"])
        timeline.append({
            "logical": logical, "state_step": step_value,
            "wall_s": elapsed, "pending_states": pending,
        })
        if k == 0 and step_value % 10 == 0:
            cycle_digests.append({
                "state_step": step_value,
                "q": all_q_digest(optimizer, params),
            })
        if k > 0 and pending == 0 and len(timeline) > 1 and timeline[-2]["pending_states"] > 0:
            cycle_digests.append({
                "state_step": step_value,
                "q": all_q_digest(optimizer, params),
            })
    torch.npu.synchronize()
    return {
        "timeline": timeline,
        "cycle_digests": cycle_digests,
        "max_pending_states": max_pending,
        "alloc_delta": torch.npu.max_memory_allocated() - base_alloc,
        "reserved_delta": torch.npu.max_memory_reserved() - base_reserved,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soap", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--trio-limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    soap_cls = load_soap_class(Path(args.soap).resolve(strict=True))
    subset, source_ids = load_checkpoint_subset(Path(args.checkpoint).resolve(strict=True))
    report: dict[str, Any] = {"k": args.k, "steps": args.steps}

    os.environ["SOAP_STALE_Q_K"] = "0"
    optimizer, params = build_optimizer(soap_cls, subset, device)
    report["T1_trio_equivalence"] = test_trio_equivalence(optimizer, params, args.trio_limit)
    del optimizer, params
    torch.npu.empty_cache()

    optimizer, params = build_optimizer(soap_cls, subset, device)
    if optimizer._stale_q_k() != 0:
        fail("k=0 baseline did not read SOAP_STALE_Q_K=0")
    report["baseline"] = run_steps(optimizer, params, source_ids, args.steps, 0)
    if report["baseline"]["max_pending_states"] != 0:
        fail("T3 baseline created a pending basis with k=0")
    del optimizer, params
    torch.npu.empty_cache()

    os.environ["SOAP_STALE_Q_K"] = str(args.k)
    optimizer, params = build_optimizer(soap_cls, subset, device)
    if optimizer._stale_q_k() != args.k:
        fail("candidate did not read the requested SOAP_STALE_Q_K")
    report["candidate"] = run_steps(optimizer, params, source_ids, args.steps, args.k)

    if report["candidate"]["max_pending_states"] not in (0, 559):
        fail(f"T3 unexpected pending count {report['candidate']['max_pending_states']}")
    if any(
        entry["pending_states"] not in (0, 559)
        for entry in report["candidate"]["timeline"]
    ):
        fail("T3 pending set was not uniform across states")

    base_first = report["baseline"]["cycle_digests"]
    cand_first = report["candidate"]["cycle_digests"]
    if not base_first or not cand_first:
        fail("no cycle was observed; increase --steps")
    # The install check runs before this step's `state["step"] += 1`, so the
    # post-step counters differ by k+1 while the number of steps that actually
    # consumed the stale basis is exactly k.
    counter_delta = cand_first[0]["state_step"] - base_first[0]["state_step"]
    report["T2_async_exact"] = {
        "baseline_first_cycle": base_first[0],
        "candidate_first_install": cand_first[0],
        "bitwise_equal": base_first[0]["q"] == cand_first[0]["q"],
        "post_step_counter_delta": counter_delta,
        "stale_steps": counter_delta - 1,
    }
    if not report["T2_async_exact"]["bitwise_equal"]:
        fail("T2 asynchronous Q is not bit-identical to the synchronous Q")
    if report["T2_async_exact"]["stale_steps"] != args.k:
        fail(f"T3 stale steps {report['T2_async_exact']['stale_steps']} != k={args.k}")

    apply_gradient(optimizer, params, source_ids, args.steps)
    while all(
        optimizer.state[parameter].get("_stale_q_pending") is None for parameter in params
    ):
        apply_gradient(optimizer, params, source_ids, args.steps)
        if int(optimizer.state[params[0]]["step"]) > 26 + args.steps + 12:
            break
    pending_before = sum(
        1 for parameter in params
        if optimizer.state[parameter].get("_stale_q_pending") is not None
    )
    persisted = optimizer.state_dict()
    pending_after = sum(
        1 for parameter in params
        if optimizer.state[parameter].get("_stale_q_pending") is not None
    )
    key_sets = {frozenset(value) for value in persisted["state"].values()}
    report["T4_schema_stable"] = {
        "pending_before_state_dict": pending_before,
        "pending_after_state_dict": pending_after,
        "distinct_key_sets": [sorted(s) for s in key_sets],
    }
    if pending_after != 0:
        fail("T4 state_dict did not flush the pending basis")
    if key_sets != {frozenset(EXPECTED_STATE_KEYS)}:
        fail("T4 persisted state schema is not the 7-key SOAP contract")

    report["T6_memory"] = {
        "baseline_alloc_delta": report["baseline"]["alloc_delta"],
        "baseline_reserved_delta": report["baseline"]["reserved_delta"],
        "candidate_alloc_delta": report["candidate"]["alloc_delta"],
        "candidate_reserved_delta": report["candidate"]["reserved_delta"],
        "extra_alloc": report["candidate"]["alloc_delta"] - report["baseline"]["alloc_delta"],
        "extra_reserved": report["candidate"]["reserved_delta"] - report["baseline"]["reserved_delta"],
    }
    report["decision"] = "STAGE_B_LOCAL_PASS"
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("GATE_DONE", args.out)
    print("decision=", report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
