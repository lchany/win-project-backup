#!/usr/bin/env python3
"""Fail-closed remote build controller for the STEP376 diagnostic probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shlex
import socket
import sys
from pathlib import Path, PurePosixPath
from typing import Any, List, Optional


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".codex-tools"
LEGACY_PATH = TOOLS / "step357_build_qrv2_release_remote.py"
REMOTE_EXEC_PATH = TOOLS / "remote_exec.py"
OUTER_ZIP = ROOT / (
    "cann-driving-cloud-ops-26.0.7-a3-cann8.3.rc1-torch2.7.1-py3.11-aarch64.zip"
)
ADAPTER = TOOLS / "build_qrv2_diagnostic_probe.py"
BASE_BUILDER = TOOLS / "build_qrv2_release.py"
STEP375_PATCHER = TOOLS / "step375_patch_qr_v2_vtv_direct_qa_legacy_probe_v6.py"
V4_PATCHER = TOOLS / "step338_patch_qr_v2_lifetime.py"

BUILD_READY = False
DIAG_NAME = "step376_attempt3_qrv2_vtv_direct_qa_legacy_probe_v6_build_20260822"
CANDIDATE_IDENTITY = "QrV2_vtv_direct_qa_legacy_probe_v6"
CANDIDATE_SHA256 = "ef5db14e09170806acb7c5227fd619f3f5ffdc7d31f36e49058cc88987fce180"
REVERSE_V4_SHA256 = "2213dbae5027c179f09c8e3b43be51587ba22e3eb2a7546311048efc3844614b"
LEGACY_SHA256 = "bf111e2e7eee407e3af26f0ed4e1aab1f833f0e068e66e463664b115c1879d91"
REMOTE_EXEC_SHA256 = "8dfcdda0630413db6cf3593756b81b6a633bc40fe1c761f8ea9a8c8a4e0ffaab"
EXPECTED_INPUTS = {
    OUTER_ZIP.name: "363fc46e0f3da952ef9c37cdfb67a190f557abc8a879d1438563c2d3eb807da7",
    ADAPTER.name: "fc65fecc58cefb86f64b6e71d64a21e5e4bc1416b42f1cd696aff6bbdedc299e",
    BASE_BUILDER.name: "d6f2aea68574422ba6e28f220c5581adf8b25019494882dea156a80b1b513e90",
    STEP375_PATCHER.name: "98a655f89ac5efedd760067fdda595d9b5fe376b1e51fdc1b12d59c727711768",
    V4_PATCHER.name: "d4b260919440e6298c9caea5ca3a7c8bb7426a30ed84cc68d934250ee0ae9fe2",
}
EXPECTED_TOOL_SHA256 = {
    "diagnostic_adapter_sha256": EXPECTED_INPUTS[ADAPTER.name],
    "base_builder_sha256": EXPECTED_INPUTS[BASE_BUILDER.name],
    "step375_patcher_sha256": EXPECTED_INPUTS[STEP375_PATCHER.name],
    "v4_patcher_sha256": EXPECTED_INPUTS[V4_PATCHER.name],
}
SOCS = ("ascend910_93", "ascend910b")
DRY_RUN_ACTIONS = (
    "upload_new",
    "prepare",
    "canonical_opc",
    "dav2201_alias",
    "seal_diagnostic",
)
FORBIDDEN_ACTIONS = (
    "package",
    "install",
    "NPU",
    "train",
    "torchrun",
    "modify_installed",
)
REMOTE_SHORT_TIMEOUT = 120
REMOTE_BUILD_TIMEOUT = 720
CONTAINER_BUILD_TIMEOUT = 600
CONTAINER_KILL_AFTER = 30
FAILURE_EVIDENCE_SCHEMA_VERSION = 1
FAILURE_TYPE_LIMIT = 128
FAILURE_LABEL_LIMIT = 128
FAILURE_MESSAGE_LIMIT = 4096
FAILURE_CLEANUP_LIMIT = 16
FAILURE_EVIDENCE_PREFIX = "STEP376 failure_evidence: "


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_files() -> tuple[Path, ...]:
    return (OUTER_ZIP, ADAPTER, BASE_BUILDER, STEP375_PATCHER, V4_PATCHER)


def _forbid_legacy_execute(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("legacy.execute is forbidden for STEP376")


def _append_cleanup_error(primary: BaseException, cleanup: BaseException, label: str) -> None:
    detail = (
        _bounded_text(label, FAILURE_LABEL_LIMIT)
        + ": "
        + _bounded_text(type(cleanup).__name__, FAILURE_TYPE_LIMIT)
        + ": "
        + _bounded_text(cleanup, FAILURE_MESSAGE_LIMIT)
    )
    marker = object()
    try:
        first_cleanup = getattr(primary, "cleanup_error", marker)
    except BaseException:
        first_cleanup = marker
    if first_cleanup is marker:
        try:
            setattr(primary, "cleanup_error", cleanup)
        except BaseException:
            pass
    try:
        errors = getattr(primary, "cleanup_errors", None)
        if errors is None:
            errors = []
            setattr(primary, "cleanup_errors", errors)
        errors.append((label, cleanup))
    except BaseException:
        pass
    try:
        add_note = getattr(primary, "add_note", None)
    except BaseException:
        add_note = None
    if add_note is not None:
        try:
            add_note(detail)
        except BaseException:
            pass


def _bounded_text(value: Any, limit: int) -> str:
    try:
        text = str(value)
    except BaseException as formatting_error:
        text = f"<unprintable:{type(value).__name__}:{type(formatting_error).__name__}>"
    if not text:
        text = "<empty>"
    if len(text) <= limit:
        return text
    marker = "...<truncated>"
    return text[: limit - len(marker)] + marker


def _malformed_cleanup_record(message: str) -> dict[str, str]:
    return {
        "label": "evidence normalization",
        "type": "MalformedCleanupEvidence",
        "message": _bounded_text(message, FAILURE_MESSAGE_LIMIT),
    }


def _validate_failure_evidence(evidence: Any) -> dict[str, Any]:
    if not isinstance(evidence, dict) or set(evidence) != {
        "schema_version", "status", "primary", "cleanup_errors"
    }:
        raise RuntimeError("STEP376 failure evidence schema mismatch")
    if (
        type(evidence["schema_version"]) is not int
        or evidence["schema_version"] != FAILURE_EVIDENCE_SCHEMA_VERSION
        or evidence["status"] != "failed"
    ):
        raise RuntimeError("STEP376 failure evidence header mismatch")
    primary = evidence["primary"]
    if not isinstance(primary, dict) or set(primary) != {"type", "message"}:
        raise RuntimeError("STEP376 primary failure evidence schema mismatch")
    for key, limit in (("type", FAILURE_TYPE_LIMIT), ("message", FAILURE_MESSAGE_LIMIT)):
        value = primary[key]
        if not isinstance(value, str) or not value or len(value) > limit:
            raise RuntimeError(f"STEP376 primary failure evidence {key} invalid")
    cleanups = evidence["cleanup_errors"]
    if not isinstance(cleanups, list) or len(cleanups) > FAILURE_CLEANUP_LIMIT:
        raise RuntimeError("STEP376 cleanup failure evidence list invalid")
    for cleanup in cleanups:
        if not isinstance(cleanup, dict) or set(cleanup) != {
            "label", "type", "message"
        }:
            raise RuntimeError("STEP376 cleanup failure evidence schema mismatch")
        for key, limit in (
            ("label", FAILURE_LABEL_LIMIT),
            ("type", FAILURE_TYPE_LIMIT),
            ("message", FAILURE_MESSAGE_LIMIT),
        ):
            value = cleanup[key]
            if not isinstance(value, str) or not value or len(value) > limit:
                raise RuntimeError(f"STEP376 cleanup failure evidence {key} invalid")
    return evidence


def _failure_evidence(error: BaseException) -> dict[str, Any]:
    cleanup_items = []
    observed = set()
    malformed = []
    try:
        raw_cleanups = getattr(error, "cleanup_errors", ())
    except BaseException as access_error:
        raw_cleanups = ()
        malformed.append(
            f"cleanup_errors attribute access failed: {type(access_error).__name__}"
        )
    if not isinstance(raw_cleanups, (list, tuple)):
        malformed.append("cleanup_errors must be a list or tuple")
        raw_cleanups = ()
    try:
        raw_cleanup_list = list(raw_cleanups)
    except BaseException as iteration_error:
        raw_cleanup_list = []
        malformed.append(
            f"cleanup_errors iteration failed: {type(iteration_error).__name__}"
        )
    for item in raw_cleanup_list[:FAILURE_CLEANUP_LIMIT]:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], BaseException)
        ):
            malformed.append("cleanup_errors contained a malformed entry")
            continue
        label, cleanup = item
        observed.add(id(cleanup))
        cleanup_items.append(
            {
                "label": _bounded_text(label, FAILURE_LABEL_LIMIT),
                "type": _bounded_text(type(cleanup).__name__, FAILURE_TYPE_LIMIT),
                "message": _bounded_text(cleanup, FAILURE_MESSAGE_LIMIT),
            }
        )
    if len(raw_cleanup_list) > FAILURE_CLEANUP_LIMIT:
        malformed.append("cleanup_errors exceeded the evidence entry limit")
    try:
        first_cleanup = getattr(error, "cleanup_error", None)
    except BaseException as access_error:
        first_cleanup = None
        malformed.append(
            f"cleanup_error attribute access failed: {type(access_error).__name__}"
        )
    if first_cleanup is not None and not isinstance(first_cleanup, BaseException):
        malformed.append("cleanup_error was not an exception")
        first_cleanup = None
    if first_cleanup is not None and id(first_cleanup) not in observed:
        first_record = {
                "label": "cleanup",
                "type": _bounded_text(
                    type(first_cleanup).__name__, FAILURE_TYPE_LIMIT
                ),
                "message": _bounded_text(first_cleanup, FAILURE_MESSAGE_LIMIT),
            }
        if len(cleanup_items) < FAILURE_CLEANUP_LIMIT:
            cleanup_items.insert(0, first_record)
        else:
            cleanup_items[-1] = _malformed_cleanup_record(
                "cleanup_error could not be represented within the entry limit"
            )
    for message in malformed:
        if len(cleanup_items) >= FAILURE_CLEANUP_LIMIT:
            cleanup_items[-1] = _malformed_cleanup_record(message)
            break
        cleanup_items.append(_malformed_cleanup_record(message))
    evidence = {
        "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
        "status": "failed",
        "primary": {
            "type": _bounded_text(type(error).__name__, FAILURE_TYPE_LIMIT),
            "message": _bounded_text(error, FAILURE_MESSAGE_LIMIT),
        },
        "cleanup_errors": cleanup_items,
    }
    return _validate_failure_evidence(evidence)


def _emit_failure_evidence(error: BaseException, stream: Any = None) -> None:
    destination = sys.stderr if stream is None else stream
    try:
        evidence = _failure_evidence(error)
    except BaseException as normalization_error:
        evidence = {
            "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
            "status": "failed",
            "primary": {
                "type": _bounded_text(type(error).__name__, FAILURE_TYPE_LIMIT),
                "message": _bounded_text(error, FAILURE_MESSAGE_LIMIT),
            },
            "cleanup_errors": [
                _malformed_cleanup_record(
                    "failure evidence normalization failed: "
                    + type(normalization_error).__name__
                )
            ],
        }
    print(
        FAILURE_EVIDENCE_PREFIX
        + json.dumps(evidence, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        file=destination,
    )


def _close_resources_preserving(
    primary: Optional[BaseException], resources: tuple[tuple[str, Any], ...]
) -> Optional[BaseException]:
    current = primary
    for label, resource in resources:
        try:
            resource.close()
        except BaseException as cleanup:
            if current is None:
                current = cleanup
            else:
                _append_cleanup_error(current, cleanup, label)
    return current


def _require_local_build_ready() -> None:
    if not BUILD_READY:
        raise RuntimeError(
            "STEP376 remote build is intentionally disabled; local review must "
            "explicitly set BUILD_READY"
        )
    inputs = input_files()
    if len({path.name for path in inputs}) != len(inputs):
        raise RuntimeError("STEP376 upload inventory contains duplicate names")
    if {path.name for path in inputs} != set(EXPECTED_INPUTS):
        raise RuntimeError("STEP376 upload inventory differs from the fixed contract")
    locked = {
        LEGACY_PATH: LEGACY_SHA256,
        REMOTE_EXEC_PATH: REMOTE_EXEC_SHA256,
        **{path: EXPECTED_INPUTS[path.name] for path in inputs},
    }
    for path, expected in locked.items():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"STEP376 locked input must be a regular file: {path.name}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"STEP376 locked input SHA mismatch: {path.name}: {actual}")


def load_legacy() -> Any:
    _require_local_build_ready()
    spec = importlib.util.spec_from_file_location("step376_legacy_helpers", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load STEP357 helper module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REMOTE_DIAG_NAME = DIAG_NAME
    module.BUILDER = ADAPTER
    module.PATCHER = BASE_BUILDER
    module.PATCHER_DEPENDENCIES = (STEP375_PATCHER, V4_PATCHER)
    module.EXPECTED_INPUTS = dict(EXPECTED_INPUTS)
    module.execute = _forbid_legacy_execute
    return module


def _validated_contract_opc_path(contract: dict[str, Any]) -> str:
    opc = contract.get("opc")
    if not isinstance(opc, dict):
        raise RuntimeError("STEP376 container contract OPC inventory is missing")
    path = opc.get("path")
    if type(path) is not str or not path or "\x00" in path:
        raise RuntimeError("STEP376 container contract OPC path is invalid")
    if not PurePosixPath(path).is_absolute():
        raise RuntimeError("STEP376 container contract OPC path must be absolute")
    return path


def _container_script(legacy: Any, contract: dict[str, Any], remote_diag: str) -> str:
    workdir = remote_diag + "/work"
    contract_path = remote_diag + "/container_contract.json"
    opc_path = _validated_contract_opc_path(contract)
    return f"""
set -eu
cd {shlex.quote(remote_diag)}
export ASCEND_OPP_PATH={shlex.quote(contract['ascend_opp'])}
export PYTHONOPTIMIZE=
export PYTHONDONTWRITEBYTECODE=1
python3 {shlex.quote(ADAPTER.name)} prepare --outer-zip {shlex.quote(OUTER_ZIP.name)} --workdir {shlex.quote(workdir)} --approved-root {shlex.quote(remote_diag)}
python3 {shlex.quote(ADAPTER.name)} build --workdir {shlex.quote(workdir)} --opc {shlex.quote(opc_path)} --container-contract {shlex.quote(contract_path)} --installed-cloud-root {shlex.quote(contract['installed_cloud_root'])} --approved-root {shlex.quote(remote_diag)}
"""


def _upload_gate_code() -> str:
    return r'''
from pathlib import Path
import hashlib,json,sys
def require(condition,message):
 if not condition: raise RuntimeError(message)
root=Path(sys.argv[1]); expected=json.loads(sys.argv[2])
require(root.is_dir() and not root.is_symlink(),'upload root invalid')
entries=list(root.iterdir())
require(len(entries)==len(expected),'upload entry count mismatch')
require({p.name for p in entries}==set(expected),'upload entry names mismatch')
for path in entries:
 require(path.is_file() and not path.is_symlink(),'upload entry not regular: '+path.name)
 require(hashlib.sha256(path.read_bytes()).hexdigest()==expected[path.name],'upload SHA mismatch: '+path.name)
print('uploaded_input_gate=PASS')
'''


def _snapshot_code() -> str:
    return r'''
import json,os,sys
from pathlib import Path
def require(condition,message):
 if not condition: raise RuntimeError(message)
root=Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0,str(root))
import build_qrv2_diagnostic_probe as adapter
base=adapter._load_base()
contract_path=Path(sys.argv[2]); opc=Path(sys.argv[3]); installed=Path(sys.argv[4])
_contract,runtime=base._validate_container_contract(contract_path,opc)
installed_inventory=base.installed_qrv2_inventory(installed)
adapter_name=b'build_qrv2_diagnostic_probe.py'
related=[]
for entry in Path('/proc').iterdir():
 if not entry.name.isdigit(): continue
 if int(entry.name)==os.getpid(): continue
 try: command=(entry/'cmdline').read_bytes()
 except (FileNotFoundError,PermissionError,ProcessLookupError): continue
 arguments=[part for part in command.split(b'\0') if part]
 direct=adapter_name in arguments and b'build' in arguments
 wrapper=adapter_name in command and b' build ' in command
 if str(root).encode() in command and (direct or wrapper):
  related.append(int(entry.name))
require(not related,'related STEP376 build processes remain')
print(json.dumps({'snapshot_schema':1,'runtime':runtime,'installed':installed_inventory,'related_build_processes':related},sort_keys=True))
'''


def _validate_snapshot(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "snapshot_schema", "runtime", "installed", "related_build_processes"
    }:
        raise RuntimeError("STEP376 snapshot schema mismatch")
    if type(snapshot["snapshot_schema"]) is not int or snapshot["snapshot_schema"] != 1:
        raise RuntimeError("STEP376 snapshot version mismatch")
    if not isinstance(snapshot["runtime"], dict) or not isinstance(
        snapshot["installed"], dict
    ):
        raise RuntimeError("STEP376 snapshot inventories must be dicts")
    processes = snapshot["related_build_processes"]
    if not isinstance(processes, list) or processes:
        raise RuntimeError("STEP376 related build process postflight mismatch")
    return snapshot


def _run_snapshot(
    legacy: Any,
    target: Any,
    remote_diag: str,
    contract_path: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    opc_path = _validated_contract_opc_path(contract)
    command = (
        "docker exec " + shlex.quote(legacy.CONTAINER)
        + " env PYTHONOPTIMIZE= PYTHONDONTWRITEBYTECODE=1 python3 -c "
        + shlex.quote(_snapshot_code()) + " "
        + shlex.quote(remote_diag) + " " + shlex.quote(contract_path) + " "
        + shlex.quote(opc_path) + " "
        + shlex.quote(contract["installed_cloud_root"])
    )
    output, _ = legacy.run(target, command, timeout=REMOTE_SHORT_TIMEOUT)
    try:
        snapshot = json.loads(output)
    except (TypeError, ValueError) as error:
        raise RuntimeError("STEP376 snapshot was not a single JSON document") from error
    return _validate_snapshot(snapshot)


def _run_build_transaction(
    legacy: Any,
    target: Any,
    contract: dict[str, Any],
    remote_diag: str,
) -> None:
    contract_path = remote_diag + "/container_contract.json"
    before = _run_snapshot(legacy, target, remote_diag, contract_path, contract)
    primary: Optional[BaseException] = None
    try:
        command = (
            "docker exec " + shlex.quote(legacy.CONTAINER)
            + " env PYTHONOPTIMIZE= PYTHONDONTWRITEBYTECODE=1 "
            + "timeout --signal=TERM --kill-after="
            + str(CONTAINER_KILL_AFTER) + "s " + str(CONTAINER_BUILD_TIMEOUT)
            + "s bash --noprofile --norc -lc "
            + shlex.quote(_container_script(legacy, contract, remote_diag))
        )
        legacy.run(target, command, timeout=REMOTE_BUILD_TIMEOUT)
    except BaseException as error:
        primary = error
    finally:
        try:
            after = _run_snapshot(
                legacy, target, remote_diag, contract_path, contract
            )
            if after["runtime"] != before["runtime"]:
                raise RuntimeError("STEP376 protected runtime snapshot changed")
            if after["installed"] != before["installed"]:
                raise RuntimeError("STEP376 installed QrV2 snapshot changed")
        except BaseException as cleanup:
            if primary is None:
                primary = cleanup
            else:
                _append_cleanup_error(primary, cleanup, "STEP376 postflight")
    if primary is not None:
        raise primary


def _summary_code() -> str:
    return r'''
import hashlib,json,sys
from pathlib import Path
def require(condition,message):
 if not condition: raise RuntimeError(message)
manifest_path=Path(sys.argv[1]); identity=sys.argv[2]; candidate_sha=sys.argv[3]
reverse_v4=sys.argv[4]; expected_tools=json.loads(sys.argv[5]); work=manifest_path.parent
require(work.is_dir() and not work.is_symlink(),'work root invalid')
require(manifest_path.is_file() and not manifest_path.is_symlink(),'manifest path invalid')
work_resolved=work.resolve(strict=True)
require(manifest_path.resolve(strict=True).parent==work_resolved,'manifest escaped work root')
m=json.loads(manifest_path.read_text(encoding='utf-8'))
flags={'artifact_class':'diagnostic_probe','diagnostic_only':True,'release_candidate':False,'package_forbidden':True}
require(m['status']=='diagnostic_built_unvalidated','status mismatch')
for layer in ('policy','candidate'):
 for key,value in flags.items(): require(m[layer][key]==value,layer+'.'+key+' mismatch')
require(m['package']=={'status':'forbidden_diagnostic_probe'},'package mismatch')
c=m['candidate']; require(c['identity']==identity and c['bin_name']==identity,'identity mismatch')
require(c['source_sha256']==candidate_sha and c['reverse_v4_sha256']==reverse_v4,'candidate SHA mismatch')
require(c['structure_assertions']['candidate_identity']==identity,'structure identity mismatch')
require(c['structure_assertions']['reverse_v4_sha256']==reverse_v4,'structure reverse mismatch')
require(m['tools']==expected_tools,'tool SHA mismatch')
require(m['build_runtime']['installed_inventory_closed'] is True,'installed inventory open')
require(m['build_runtime']['runtime_inventory_closed'] is True,'runtime inventory open')
expected_entries=sorted((identity+'_0_mix_aic',identity+'_0_mix_aiv'))
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
audited={}
for soc in ('ascend910_93','ascend910b'):
 a=m['artifacts'][soc]; obj=Path(a['object_path']); meta=Path(a['json_path']); log=Path(a['opc_log_path'])
 build=(work/'build'/soc).resolve(strict=True)
 try: build.relative_to(work_resolved)
 except ValueError: raise RuntimeError('SoC build root escaped work root: '+soc)
 output=(build/'output').resolve(strict=True)
 for path in (obj,meta,log):
  require(path.is_file() and not path.is_symlink(),'artifact not regular: '+str(path))
  try: path.resolve(strict=True).relative_to(build)
  except ValueError: raise RuntimeError('artifact path escaped SoC build root: '+str(path))
 for path in (obj,meta):
  try: path.resolve(strict=True).relative_to(output)
  except ValueError: raise RuntimeError('artifact path escaped SoC output root: '+str(path))
 require(log.resolve(strict=True)==(build/'opc.log').resolve(strict=True),'OPC log path mismatch')
 require(a['object_size']==obj.stat().st_size and a['object_sha256']==sha(obj),'object closure mismatch')
 require(a['json_size']==meta.stat().st_size and a['json_sha256']==sha(meta),'json closure mismatch')
 require(a['opc_log_size']==log.stat().st_size and a['opc_log_sha256']==sha(log),'OPC log closure mismatch')
 require(a['kernel_name']==identity and a['bin_file_name']==identity,'artifact identity mismatch')
 require(sorted(a['concrete_entries'])==expected_entries,'concrete entries mismatch')
 audited[soc]={'object_sha256':sha(obj),'json_sha256':sha(meta),'object_bytes':obj.read_bytes(),'json_bytes':meta.read_bytes(),'opc_log_sha256':sha(log),'concrete_entries':a['concrete_entries']}
left,right=(audited['ascend910_93'],audited['ascend910b'])
for kind in ('object','json'):
 require(left[kind+'_sha256']==right[kind+'_sha256'],'alias SHA mismatch: '+kind)
 require(left[kind+'_bytes']==right[kind+'_bytes'],'alias bytes mismatch: '+kind)
require(not (work/'release').exists() and not (work/'release').is_symlink(),'release directory exists')
allowed_raw=Path(m['paths']['extracted_wheel']); guard_raw=Path(m['immutable_guards']['extracted_original_wheel']['path'])
require(not allowed_raw.is_symlink() and not guard_raw.is_symlink(),'allowed wheel path is symlink')
allowed=allowed_raw.resolve(strict=True); guard=guard_raw.resolve(strict=True)
require(allowed==guard,'allowed wheel differs from immutable guard')
require(allowed.is_file(),'allowed wheel is not regular')
try: allowed.relative_to(work_resolved)
except ValueError: raise RuntimeError('allowed wheel escaped work root')
require(sha(allowed)==m['immutable_guards']['extracted_original_wheel']['sha256'],'allowed wheel SHA mismatch')
for path in work.rglob('*'):
 require(not (path.is_symlink() and path.suffix.lower() in ('.whl','.zip')),'package symlink exists: '+str(path))
 require(not (path.is_file() and path.suffix.lower()=='.zip'),'ZIP exists: '+str(path))
 require(not (path.is_file() and path.suffix.lower()=='.whl' and path.resolve()!=allowed),'new wheel exists: '+str(path))
print(json.dumps({'status':m['status'],'policy':{k:m['policy'][k] for k in flags},'candidate':{'identity':c['identity'],'source_sha256':c['source_sha256'],'reverse_v4_sha256':c['reverse_v4_sha256'],'diagnostic_only':c['diagnostic_only'],'release_candidate':c['release_candidate'],'package_forbidden':c['package_forbidden'],'artifact_class':c['artifact_class']},'package':m['package'],'tools':m['tools'],'artifacts':{soc:{k:audited[soc][k] for k in ('object_sha256','json_sha256','opc_log_sha256','concrete_entries')} for soc in audited},'alias_bytes_equal':True,'installed_inventory_closed':True,'runtime_inventory_closed':True,'release_outputs_absent':True},sort_keys=True))
'''


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _validate_summary(summary: dict[str, Any]) -> None:
    if not isinstance(summary, dict):
        raise RuntimeError("STEP376 summary must be a dict")
    expected_top = {
        "status", "policy", "candidate", "package", "tools", "artifacts",
        "alias_bytes_equal", "installed_inventory_closed",
        "runtime_inventory_closed", "release_outputs_absent",
    }
    if set(summary) != expected_top:
        raise RuntimeError("STEP376 summary schema mismatch")
    if summary["status"] != "diagnostic_built_unvalidated":
        raise RuntimeError("STEP376 summary status mismatch")
    flags = {
        "artifact_class": "diagnostic_probe",
        "diagnostic_only": True,
        "release_candidate": False,
        "package_forbidden": True,
    }
    if not _strict_json_equal(summary["policy"], flags):
        raise RuntimeError("STEP376 summary policy flags mismatch")
    candidate = summary["candidate"]
    if not isinstance(candidate, dict):
        raise RuntimeError("STEP376 summary candidate must be a dict")
    if not _strict_json_equal(candidate, {
        "identity": CANDIDATE_IDENTITY,
        "source_sha256": CANDIDATE_SHA256,
        "reverse_v4_sha256": REVERSE_V4_SHA256,
        **flags,
    }):
        raise RuntimeError("STEP376 summary candidate contract mismatch")
    if not _strict_json_equal(summary["package"], {
        "status": "forbidden_diagnostic_probe"
    }):
        raise RuntimeError("STEP376 summary package contract mismatch")
    if not isinstance(summary["tools"], dict) or summary["tools"] != EXPECTED_TOOL_SHA256:
        raise RuntimeError("STEP376 summary tool SHA mismatch")
    expected_entries = sorted(
        (CANDIDATE_IDENTITY + "_0_mix_aic", CANDIDATE_IDENTITY + "_0_mix_aiv")
    )
    artifacts = summary["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(SOCS):
        raise RuntimeError("STEP376 summary SoC inventory mismatch")
    for soc in SOCS:
        artifact = artifacts[soc]
        if not isinstance(artifact, dict) or set(artifact) != {
            "object_sha256", "json_sha256", "opc_log_sha256", "concrete_entries"
        }:
            raise RuntimeError(f"STEP376 summary {soc} artifact schema mismatch")
        for key in ("object_sha256", "json_sha256", "opc_log_sha256"):
            value = artifact[key]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
            ):
                raise RuntimeError(f"STEP376 summary {soc} {key} mismatch")
        entries = artifact["concrete_entries"]
        if not isinstance(entries, list) or entries != expected_entries:
            raise RuntimeError(f"STEP376 summary {soc} concrete entries mismatch")
    for key in ("object_sha256", "json_sha256"):
        if artifacts[SOCS[0]][key] != artifacts[SOCS[1]][key]:
            raise RuntimeError(f"STEP376 summary SoC alias {key} mismatch")
    for key in (
        "alias_bytes_equal",
        "installed_inventory_closed",
        "runtime_inventory_closed",
        "release_outputs_absent",
    ):
        if summary[key] is not True:
            raise RuntimeError(f"STEP376 summary closure mismatch: {key}")


def _dry_run_payload(legacy: Optional[Any] = None) -> dict[str, Any]:
    legacy = load_legacy() if legacy is None else legacy
    info = legacy.local_preflight(legacy.load_remote_module())
    return {
        "status": "dry_run",
        "diagnostics_name": DIAG_NAME,
        "target_suffix": str(info["target_host"]).split(".")[-1],
        "container": legacy.CONTAINER,
        "candidate_identity": CANDIDATE_IDENTITY,
        "candidate_source_sha256": CANDIDATE_SHA256,
        "input_sha256": dict(EXPECTED_INPUTS),
        "actions": list(DRY_RUN_ACTIONS),
        "forbidden": list(FORBIDDEN_ACTIONS),
    }


def _exclusive_directory_script(path: str) -> str:
    """Return a fail-closed shell script that creates exactly one new directory."""
    quoted = shlex.quote(path)
    return (
        "set -eu\n"
        "[ ! -e " + quoted + " ] || exit 73\n"
        "mkdir -m 700 -- " + quoted + "\n"
    )


def execute() -> dict[str, Any]:
    legacy = load_legacy()
    remote_module = legacy.load_remote_module()
    info = legacy.local_preflight(remote_module)
    jump, target = legacy.connect_target(remote_module, info)
    primary: Optional[BaseException] = None
    result: Optional[dict[str, Any]] = None
    try:
        hostname, _ = legacy.run(
            target, "hostname", timeout=REMOTE_SHORT_TIMEOUT
        )
        if hostname.strip() != legacy.EXPECTED_HOSTNAME:
            raise RuntimeError("second-hop runtime hostname mismatch")
        contract = legacy.container_probe(target)
        remote_diag = legacy.safe_remote_path(str(info["shared"]), DIAG_NAME)
        workdir = remote_diag + "/work"
        legacy.run_host_script(
            target,
            _exclusive_directory_script(remote_diag),
            timeout=REMOTE_SHORT_TIMEOUT,
        )
        contract_payload = {
            key: contract[key]
            for key in (
                "schema_version",
                "container_name",
                "inspect_container_id",
                "inspect_hostname",
                "opc",
                "cann_version_files",
            )
        }
        contract_bytes = (
            json.dumps(contract_payload, sort_keys=True, indent=2) + "\n"
        ).encode()
        sftp = None
        upload_primary: Optional[BaseException] = None
        try:
            sftp = target.open_sftp()
            get_channel = getattr(sftp, "get_channel", None)
            if get_channel is not None:
                get_channel().settimeout(REMOTE_SHORT_TIMEOUT)
            for local in input_files():
                legacy.write_remote_new(
                    sftp, remote_diag + "/" + local.name, local.read_bytes()
                )
            legacy.write_remote_new(
                sftp, remote_diag + "/container_contract.json", contract_bytes
            )
        except BaseException as error:
            upload_primary = error
        finally:
            if sftp is not None:
                upload_primary = _close_resources_preserving(
                    upload_primary, (("SFTP close", sftp),)
                )
        if upload_primary is not None:
            raise upload_primary

        expected = dict(EXPECTED_INPUTS)
        expected["container_contract.json"] = hashlib.sha256(contract_bytes).hexdigest()
        upload_out, _ = legacy.run(
            target,
            "env PYTHONOPTIMIZE= PYTHONDONTWRITEBYTECODE=1 python3 -c "
            + shlex.quote(_upload_gate_code()) + " "
            + shlex.quote(remote_diag) + " "
            + shlex.quote(json.dumps(expected, sort_keys=True)),
            timeout=REMOTE_SHORT_TIMEOUT,
        )
        if "uploaded_input_gate=PASS" not in upload_out:
            raise RuntimeError("STEP376 uploaded input gate did not report PASS")
        _run_build_transaction(legacy, target, contract, remote_diag)
        summary_out, _ = legacy.run(
            target,
            "docker exec " + shlex.quote(legacy.CONTAINER)
            + " env PYTHONOPTIMIZE= PYTHONDONTWRITEBYTECODE=1 python3 -c "
            + shlex.quote(_summary_code()) + " "
            + shlex.quote(workdir + "/release_manifest.json") + " "
            + shlex.quote(CANDIDATE_IDENTITY) + " "
            + shlex.quote(CANDIDATE_SHA256) + " "
            + shlex.quote(REVERSE_V4_SHA256) + " "
            + shlex.quote(json.dumps(EXPECTED_TOOL_SHA256, sort_keys=True)),
            timeout=REMOTE_SHORT_TIMEOUT,
        )
        try:
            summary = json.loads(summary_out)
        except (TypeError, ValueError) as error:
            raise RuntimeError("STEP376 summary was not a single JSON document") from error
        _validate_summary(summary)
        summary["remote_diagnostics_name"] = DIAG_NAME
        summary["uploaded_gate"] = True
        result = summary
    except BaseException as error:
        primary = error
    finally:
        primary = _close_resources_preserving(
            primary, (("target close", target), ("jump close", jump))
        )
    if primary is not None:
        raise primary
    if result is None:
        raise RuntimeError("STEP376 execute completed without a summary")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(_dry_run_payload(), sort_keys=True))
    else:
        print(json.dumps(execute(), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, socket.error) as error:
        _emit_failure_evidence(error)
        raise SystemExit(1)
