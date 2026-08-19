#!/usr/bin/env python3
"""Real SOAP optimizer-state adapter for the STEP-215-E local gate.

This adapter deliberately validates only the stateful optimizer subset stored
in the requested authoritative checkpoint.  The checkpoint contains 767
single-parameter groups, but only 559 parameter ids have SOAP state.  Those
559 states are reconstructed with independent Parameters matching exp_avg
shape/dtype; the 208 stateless parameters are omitted because the checkpoint
does not encode their shapes or a parameter-name mapping.

Consequently this is a real SOAP persistent-state/resume gate, not a model,
loss, data, DDP, or full-parameter checkpoint gate.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import inspect
from collections import Counter
from pathlib import Path
from typing import Any


READINESS = {
    "real_soap_optimizer": True,
    "loads_requested_checkpoint": True,
    "uses_requested_config": True,
    "deterministic_replay_gradient": True,
    "state_view_includes_parameters": True,
    "state_view_includes_optimizer_state": True,
    "state_view_includes_q": True,
    "state_view_includes_gg_exp_avg_exp_avg_sq_step": True,
    "checkpoint_roundtrip": True,
    "sort_observable_through_torch_argsort": True,
}

VALIDATION_SCOPE = {
    "kind": "real_soap_checkpoint_stateful_subset",
    "source_param_group_count": 767,
    "stateful_param_count": 559,
    "omitted_stateless_param_count": 208,
    "surrogate_parameter_values": True,
    "validates_full_model_parameters": False,
    "validates_model_forward_or_loss": False,
    "validates_training_data_or_ddp": False,
}

EXPECTED_CONFIG_SHA256 = "02aca0c7a33e05e972e268aecc1932bbf10f611fa523ec4696b94d58cd7f56a5"
EXPECTED_CHECKPOINT_SHA256 = "f001a7d55c19b74d84dd1384f262acef786237822e9581203176853d735f997d"
EXPECTED_CHECKPOINT_BYTES = 1_607_991_401
EXPECTED_STATE_KEYS = {
    "GG",
    "Q",
    "exp_avg",
    "exp_avg_sq",
    "precondition_frequency",
    "shampoo_beta",
    "step",
}
EXPECTED_Q_INVENTORY = {
    1: 106,
    3: 30,
    4: 6,
    7: 37,
    8: 1,
    11: 1,
    22: 1,
    32: 4,
    40: 9,
    64: 28,
    96: 3,
    120: 1,
    128: 18,
    160: 1,
    192: 32,
    220: 4,
    256: 181,
    352: 1,
    440: 4,
    512: 43,
    768: 22,
    1024: 6,
    2560: 4,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cpu_tree(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def tensor_layout_summary(tensor: Any) -> dict[str, Any]:
    value = tensor.detach()
    cpu = value.contiguous().cpu()
    try:
        raw = cpu.numpy().tobytes()
    except Exception:
        raw = bytes(cpu.view(__import__("torch").uint8).reshape(-1).tolist())
    return {
        "shape": [int(x) for x in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "stride": [int(x) for x in value.stride()],
        "storage_offset": int(value.storage_offset()),
        "contiguous": bool(value.is_contiguous()),
        "value_sha256": hashlib.sha256(raw).hexdigest(),
    }


class RealSOAPStatefulSubsetAdapter:
    def __init__(self, context: dict[str, Any]) -> None:
        import torch

        self.torch = torch
        self.repo = Path(context["repo"]).resolve(strict=True)
        self.config_path = Path(context["config"]).resolve(strict=True)
        self.checkpoint_path = Path(context["checkpoint"]).resolve(strict=True)
        self.device = context["device"]
        self.soap_path = (
            self.repo / "projects/mmdet3d_plugin/optimizers/soap.py"
        ).resolve(strict=True)
        if self.repo not in self.soap_path.parents:
            raise RuntimeError("SOAP source escaped the requested repository")
        self._validate_config()
        self.SOAP = self._load_real_soap_class()
        self.source_optimizer, self.source_ids = self._load_source_optimizer()

    def _validate_config(self) -> None:
        if sha256_file(self.config_path) != EXPECTED_CONFIG_SHA256:
            raise RuntimeError("requested config SHA is not the qualified STEP-215 config")
        from mmcv import Config

        cfg = Config.fromfile(str(self.config_path))
        opt = cfg.optimizer
        if opt.get("type") != "SOAP":
            raise RuntimeError("requested config does not select SOAP")
        if int(opt.get("precondition_frequency", -1)) != 10:
            raise RuntimeError("requested config precondition_frequency is not 10")
        if int(opt.get("one_sided_dim_threshold", -1)) != 1024:
            raise RuntimeError("requested config one_sided_dim_threshold is not 1024")
        custom = opt.get("paramwise_cfg", {}).get("custom_keys", {})
        if float(custom.get("img_backbone", {}).get("lr_mult", -1.0)) != 0.25:
            raise RuntimeError("requested config img_backbone lr multiplier changed")

    def _load_real_soap_class(self) -> Any:
        from mmcv.runner.optimizer import OPTIMIZERS

        existing = OPTIMIZERS.get("SOAP")
        if existing is not None:
            source = Path(inspect.getsourcefile(existing) or "").resolve()
            if source != self.soap_path:
                raise RuntimeError("an unexpected SOAP class is already registered")
            return existing
        spec = importlib.util.spec_from_file_location(
            "step215_authoritative_soap", self.soap_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot import authoritative SOAP source")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, "SOAP", None)
        if cls is None or Path(inspect.getsourcefile(cls) or "").resolve() != self.soap_path:
            raise RuntimeError("authoritative SOAP class was not loaded from soap.py")
        return cls

    def _load_source_optimizer(self) -> tuple[dict[str, Any], tuple[int, ...]]:
        if self.checkpoint_path.stat().st_size != EXPECTED_CHECKPOINT_BYTES:
            raise RuntimeError("requested checkpoint byte size changed")
        if sha256_file(self.checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError("requested checkpoint SHA is not the qualified iter30 checkpoint")
        payload = self.torch.load(
            str(self.checkpoint_path), map_location="cpu", weights_only=False
        )
        if set(payload) != {"meta", "optimizer", "state_dict"}:
            raise RuntimeError("requested checkpoint top-level schema changed")
        source = payload["optimizer"]
        del payload
        groups = source.get("param_groups")
        states = source.get("state")
        if not isinstance(groups, list) or not isinstance(states, dict):
            raise RuntimeError("checkpoint optimizer schema is missing groups/state")
        if len(groups) != VALIDATION_SCOPE["source_param_group_count"]:
            raise RuntimeError("checkpoint no longer has 767 parameter groups")
        if any(len(group.get("params", [])) != 1 for group in groups):
            raise RuntimeError("checkpoint parameter groups are no longer one-parameter groups")
        flat_ids = [int(group["params"][0]) for group in groups]
        if len(set(flat_ids)) != len(flat_ids):
            raise RuntimeError("checkpoint optimizer parameter ids are not unique")
        source_ids = tuple(pid for pid in flat_ids if pid in states)
        if len(source_ids) != VALIDATION_SCOPE["stateful_param_count"]:
            raise RuntimeError("checkpoint no longer has 559 stateful parameters")
        if set(states) != set(source_ids):
            raise RuntimeError("checkpoint state ids do not match the stateful group subset")
        if any(set(states[pid]) != EXPECTED_STATE_KEYS for pid in source_ids):
            raise RuntimeError("checkpoint SOAP state key schema changed")
        if any(int(states[pid]["step"]) != 26 for pid in source_ids):
            raise RuntimeError("checkpoint SOAP state.step is no longer uniformly 26")
        self._assert_q_inventory(states.values())
        group_by_id = {int(group["params"][0]): group for group in groups}
        remap = {old: new for new, old in enumerate(source_ids)}
        subset = {
            "state": {remap[old]: states[old] for old in source_ids},
            "param_groups": [
                {**{k: v for k, v in group_by_id[old].items() if k != "params"}, "params": [remap[old]]}
                for old in source_ids
            ],
        }
        return subset, source_ids

    def _assert_q_inventory(self, states: Any) -> None:
        counts: Counter[int] = Counter()
        for state in states:
            q = state.get("Q")
            if not isinstance(q, list):
                raise RuntimeError("SOAP Q is not a list")
            for tensor in q:
                if self.torch.is_tensor(tensor):
                    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
                        raise RuntimeError("SOAP Q contains a non-square tensor")
                    counts[int(tensor.shape[0])] += 1
        if dict(sorted(counts.items())) != EXPECTED_Q_INVENTORY:
            raise RuntimeError("checkpoint Q inventory is not the approved 23-shape/543 contract")
        if counts.get(5120, 0) != 0 or sum(counts.values()) != 543:
            raise RuntimeError("checkpoint Q inventory contains 5120 or does not total 543")

    def _build_from_state(
        self,
        state_dict: dict[str, Any],
        parameter_values: list[Any] | None = None,
    ) -> dict[str, Any]:
        groups = state_dict["param_groups"]
        states = state_dict["state"]
        if len(groups) != len(self.source_ids) or set(states) != set(range(len(self.source_ids))):
            raise RuntimeError("stateful-subset optimizer schema changed")
        params = []
        runtime_groups = []
        for index, group in enumerate(groups):
            state = states[index]
            exp_avg = state.get("exp_avg")
            if not self.torch.is_tensor(exp_avg):
                raise RuntimeError("stateful parameter lacks exp_avg")
            parameter = self.torch.nn.Parameter(
                self.torch.zeros(
                    tuple(exp_avg.shape), dtype=exp_avg.dtype, device=self.device
                ),
                requires_grad=True,
            )
            if parameter_values is not None:
                value = parameter_values[index].to(
                    device=self.device, dtype=parameter.dtype
                )
                if tuple(value.shape) != tuple(parameter.shape):
                    raise RuntimeError("resume surrogate parameter shape mismatch")
                parameter.data.copy_(value)
            params.append(parameter)
            runtime_groups.append(
                {**{k: copy.deepcopy(v) for k, v in group.items() if k != "params"}, "params": [parameter]}
            )
        optimizer = self.SOAP(runtime_groups)
        optimizer.load_state_dict(state_dict)
        self._assert_q_inventory(optimizer.state.values())
        return {"parameters": params, "optimizer": optimizer, "source_ids": self.source_ids}

    def build_trial(self, context: dict[str, Any]) -> dict[str, Any]:
        if context["device"] != self.device:
            raise RuntimeError("adapter device context changed")
        return self._build_from_state(self.source_optimizer)

    def make_gradient(self, context: dict[str, Any], logical_step: int) -> dict[str, Any]:
        return {
            "logical_step": int(logical_step),
            "formula": "constant_per_source_param_v1",
        }

    def apply_gradient(
        self, trial: dict[str, Any], gradient: dict[str, Any], logical_step: int
    ) -> None:
        if gradient != {
            "logical_step": int(logical_step),
            "formula": "constant_per_source_param_v1",
        }:
            raise RuntimeError("deterministic replay gradient descriptor changed")
        for parameter, source_id in zip(trial["parameters"], trial["source_ids"]):
            value = (((int(source_id) % 29) - 14) * 1.0e-4) + (
                (int(logical_step) + 1) * 1.0e-6
            )
            parameter.grad = self.torch.empty_like(parameter).fill_(value)
        trial["optimizer"].step()

    def state_view(self, trial: dict[str, Any]) -> dict[str, Any]:
        return {
            "parameters": cpu_tree(trial["parameters"]),
            "optimizer_state": {
                "scope": copy.deepcopy(VALIDATION_SCOPE),
                "source_param_ids": tuple(int(x) for x in trial["source_ids"]),
                "state_dict": cpu_tree(trial["optimizer"].state_dict()),
            },
        }

    def layout_view(self, trial: dict[str, Any]) -> dict[str, Any]:
        states = trial["optimizer"].state
        rows: list[dict[str, Any]] = []
        for index, parameter in enumerate(trial["parameters"]):
            state = states[parameter]
            selected: dict[str, Any] = {
                "parameter": tensor_layout_summary(parameter),
            }
            for key in ("GG", "Q"):
                values = state.get(key)
                if isinstance(values, list):
                    selected[key] = [
                        tensor_layout_summary(value)
                        if self.torch.is_tensor(value)
                        else {"non_tensor_type": type(value).__name__}
                        for value in values
                    ]
            value = state.get("exp_avg_sq")
            if self.torch.is_tensor(value):
                selected["exp_avg_sq"] = tensor_layout_summary(value)
            rows.append(
                {
                    "stateful_index": index,
                    "source_param_id": int(trial["source_ids"][index]),
                    "tensors": selected,
                }
            )
        return {
            "scope": "parameters_and_Q_GG_exp_avg_sq_layout_digest_v1",
            "stateful_parameter_count": len(rows),
            "rows": rows,
        }

    def save_trial(self, trial: dict[str, Any], checkpoint_path: Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        payload = {
            "format": "step215_real_soap_stateful_subset_v1",
            "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "source_param_ids": tuple(int(x) for x in trial["source_ids"]),
            "parameters": cpu_tree(trial["parameters"]),
            "optimizer": cpu_tree(trial["optimizer"].state_dict()),
        }
        self.torch.save(payload, str(checkpoint_path))

    def load_trial(self, context: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
        payload = self.torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        if payload.get("format") != "step215_real_soap_stateful_subset_v1":
            raise RuntimeError("adapter resume checkpoint format changed")
        if payload.get("source_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError("adapter resume source checkpoint identity changed")
        if tuple(payload.get("source_param_ids", ())) != self.source_ids:
            raise RuntimeError("adapter resume source parameter ids changed")
        return self._build_from_state(payload["optimizer"], payload["parameters"])

    def destroy_trial(self, trial: dict[str, Any]) -> None:
        for parameter in trial.get("parameters", []):
            parameter.grad = None
        trial.clear()
        gc.collect()
        if hasattr(self.torch, "npu"):
            self.torch.npu.empty_cache()


def create_adapter(context: dict[str, Any]) -> RealSOAPStatefulSubsetAdapter:
    return RealSOAPStatefulSubsetAdapter(context)
