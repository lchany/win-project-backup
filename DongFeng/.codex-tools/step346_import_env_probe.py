#!/usr/bin/env python3
"""Observe import-time custom OPP environment changes without touching an NPU."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any


ENV_NAME = "ASCEND_CUSTOM_OPP_PATH"


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(installed: Path) -> dict[str, Any]:
    raw = os.environ.get(ENV_NAME)
    entries = [] if raw is None else raw.split(":")
    normalized_installed = installed.resolve(strict=True)
    described = []
    for entry in entries:
        candidate = Path(entry)
        absolute = candidate.is_absolute()
        exists = candidate.exists()
        try:
            normalized = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError):
            normalized = candidate.absolute()
        role = "installed" if normalized == normalized_installed else "unknown"
        described.append({
            "role": role,
            "text_sha256": text_sha256(entry),
            "is_absolute": absolute,
            "exists": exists,
        })
    return {
        "is_set": raw is not None,
        "raw_sha256": None if raw is None else text_sha256(raw),
        "entry_count": len(entries),
        "entries": described,
        "roles": [entry["role"] for entry in described],
    }


def set_env_summary(module: Any, installed: Path) -> dict[str, Any]:
    function = getattr(module, "_set_env", None)
    if function is None:
        raise RuntimeError("mx_driving_cloud._set_env is missing")
    source = inspect.getsource(function)
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    module_file = Path(inspect.getsourcefile(module) or "").resolve(strict=True)
    expected_module = installed.parents[2] / "__init__.py"
    if module_file != expected_module.resolve(strict=True):
        raise RuntimeError("mx_driving_cloud source is not the installed package __init__.py")
    compact = "".join(source.split())
    return {
        "module_file_role": "installed_mx_init",
        "module_file_sha256": file_sha256(module_file),
        "function_name": function.__name__,
        "reads_env_name": f'os.environ.get("{ENV_NAME}")' in compact,
        "writes_env_name": f'os.environ["{ENV_NAME}"]=' in compact,
        "existing_value_branch_prepends_installed": (
            'mx_driving_opp_path+":"+ascend_custom_opp_path' in compact
        ),
        "installed_suffix": "packages/vendors/customize",
        "calls_init_op_api_path": any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "_init_op_api_so_path"
            for call in calls
        ),
        "ast_assignment_count": len(assignments),
    }


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed-opp", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    installed = Path(args.installed_opp).resolve(strict=True)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    module_state = {"startup_mx_loaded": "mx_driving_cloud" in sys.modules}
    stages = {"startup": snapshot(installed)}
    import torch
    import torch.distributed  # noqa: F401
    import torch_npu

    stages["after_torch_distributed_torch_npu"] = snapshot(installed)
    module_state["after_torch_imports_mx_loaded"] = "mx_driving_cloud" in sys.modules
    import mx_driving_cloud

    stages["after_mx_driving_cloud"] = snapshot(installed)

    mx_summary = set_env_summary(mx_driving_cloud, installed)
    required_mx_facts = (
        "reads_env_name",
        "writes_env_name",
        "existing_value_branch_prepends_installed",
        "calls_init_op_api_path",
    )
    structural_contract_pass = all(mx_summary[name] is True for name in required_mx_facts)
    stage_contracts = {
        "startup_is_single_installed": stages["startup"]["roles"] == ["installed"],
        "torch_imports_preserve_single_installed": (
            stages["after_torch_distributed_torch_npu"]["roles"] == ["installed"]
        ),
        "mx_import_result_is_double_installed": (
            stages["after_mx_driving_cloud"]["roles"] == ["installed", "installed"]
        ),
        "all_entries_absolute_and_existing": all(
            entry["is_absolute"] and entry["exists"]
            for stage in stages.values()
            for entry in stage["entries"]
        ),
    }

    payload = {
        "schema": "step346-import-env-probe-v1",
        "training": False,
        "npu_api_called": False,
        "device_or_hccl_initialized": False,
        "pid": os.getpid(),
        "versions": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "probe_script_sha256": file_sha256(Path(__file__).resolve(strict=True)),
        "installed_opp_text_sha256": text_sha256(str(installed)),
        "stages": stages,
        "module_state": module_state,
        "stage_contracts": stage_contracts,
        "structural_contract_pass": structural_contract_pass,
        "mx_set_env": mx_summary,
    }
    atomic_json(output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
