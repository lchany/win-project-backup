#!/usr/bin/env python3
"""Read-only in-place audit for the failed STEP369 QrV2 v4 build.

This controller deliberately has no upload, build, package, install, cleanup,
or NPU path.  It reuses the audited two-hop connection guard, inspects the
isolated diagnostics directory in place, and emits only sanitized scalars.
"""

from __future__ import annotations

import importlib.util
import ipaddress
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".codex-tools"
LEGACY_PATH = TOOLS / "step360_rebuild_qrv2_release_remote.py"
DIAG_NAME = "step369_qrv2_lifetime_alpha_sync_v4_release_20260821"
CONTAINER = "mapqr-leicheng"
EXPECTED_HOSTNAME = "yfzy-zhsc-910c-1.novalocal"


def load_legacy() -> Any:
    spec = importlib.util.spec_from_file_location("step369_audit_legacy", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load STEP369 guarded connection controller")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_legacy()


def _process_summary(text: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        pid, command = fields
        role = None
        if re.search(r"(?:^|[ /])opc(?:\s|$)", command):
            role = "opc"
        elif re.search(
            r"build_qrv2_release\.py\s+(?:prepare|build|package)(?:\s|$)", command
        ):
            role = "qrv2_release_builder"
        elif DIAG_NAME in command:
            role = "step369_related"
        if role is not None:
            records.append({"pid": int(pid), "role": role})
    return {"count": len(records), "records": records}


REMOTE_AUDIT_CODE = r'''
import hashlib
import importlib.util
import ipaddress
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

root=Path(sys.argv[1])
installed_root=Path(sys.argv[2])
opc=Path(sys.argv[3])

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def stamp(value):
    return datetime.fromtimestamp(value,timezone.utc).isoformat()

def clean_line(line):
    line=line.replace(str(root),'<diag>').replace(str(installed_root),'<installed>')
    def redact_ip(match):
        value=match.group(0)
        try:
            address=ipaddress.ip_address(value)
        except ValueError:
            return '<redacted-ip>'
        return value if address.is_loopback else '<redacted-ip>'
    line=re.sub(r'(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])',redact_ip,line)
    line=re.sub(r'(?i)(password|passwd|token|secret|access[_-]?key)(\s*[=:]\s*)\S+',r'\1\2<redacted>',line)
    return line.strip()[:1200]

def last_levels(path):
    result={'exists':path.exists(),'is_symlink':path.is_symlink()}
    if not path.exists():
        return result
    st=path.lstat()
    result.update({'mtime_utc':stamp(st.st_mtime),'size':st.st_size})
    return result

def artifact_group(build_dir):
    output=build_dir/'output'
    objects=sorted(p for p in output.rglob('*.o') if p.is_file() and not p.is_symlink()) if output.is_dir() else []
    metadata=sorted(p for p in output.rglob('*.json') if p.is_file() and not p.is_symlink()) if output.is_dir() else []
    records=[]
    for path in objects+metadata:
        item={'kind':path.suffix,'name':path.name,'size':path.stat().st_size,'sha256':sha(path)}
        if path.suffix=='.json':
            try:
                value=json.loads(path.read_text(encoding='utf-8'))
                support=value.get('supportInfo') if isinstance(value,dict) else None
                item['metadata']={
                    'binFileName':value.get('binFileName') if isinstance(value,dict) else None,
                    'kernelName':value.get('kernelName') if isinstance(value,dict) else None,
                    'kernelList_names':[entry.get('kernelName') for entry in value.get('kernelList',[]) if isinstance(entry,dict)] if isinstance(value,dict) and isinstance(value.get('kernelList'),list) else None,
                    'opMode':support.get('opMode') if isinstance(support,dict) else None,
                    'simplifiedKeyMode':support.get('simplifiedKeyMode') if isinstance(support,dict) else None,
                    'simplifiedKey_count':len(support.get('simplifiedKey',[])) if isinstance(support,dict) and isinstance(support.get('simplifiedKey'),list) else None,
                    'input_count':len(support.get('inputs',[])) if isinstance(support,dict) and isinstance(support.get('inputs'),list) else None,
                    'output_count':len(support.get('outputs',[])) if isinstance(support,dict) and isinstance(support.get('outputs'),list) else None,
                }
            except (OSError,UnicodeError,json.JSONDecodeError) as exc:
                item['metadata_error']=type(exc).__name__
        records.append(item)
    return {'object_count':len(objects),'json_count':len(metadata),'files':records}

def log_summary(path):
    result=last_levels(path)
    result.update({'last_error':None,'last_warning':None})
    if not path.is_file() or path.is_symlink():
        return result
    with path.open('r',encoding='utf-8',errors='replace') as stream:
        for line in stream:
            upper=line.upper()
            if 'ERROR' in upper:
                result['last_error']=clean_line(line)
            if 'WARNING' in upper or 'WARN' in upper:
                result['last_warning']=clean_line(line)
    return result

result={
    'diagnostics_name':root.name,
    'directory':last_levels(root),
    'tree':None,
    'manifest':last_levels(root/'work'/'release_manifest.json'),
    'phase_status':None,
    'artifacts':{},
    'opc_logs':{},
    'inventory':{},
}

if root.is_dir() and not root.is_symlink():
    entries=[]
    for path in root.rglob('*'):
        relative=path.relative_to(root)
        entries.append((path,relative))
    files=[pair for pair in entries if pair[0].is_file() and not pair[0].is_symlink()]
    result['tree']={
        'entry_count':len(entries),
        'file_count':len(files),
        'dir_count':sum(1 for path,_ in entries if path.is_dir() and not path.is_symlink()),
        'symlink_count':sum(1 for path,_ in entries if path.is_symlink()),
        'max_depth':max((len(relative.parts) for _,relative in entries),default=0),
        'regular_file_bytes':sum(path.stat().st_size for path,_ in files),
        'top_level_names':sorted(path.name for path in root.iterdir()),
        'newest_mtime_utc':stamp(max((path.lstat().st_mtime for path,_ in entries),default=root.lstat().st_mtime)),
    }

manifest_path=root/'work'/'release_manifest.json'
manifest=None
if manifest_path.is_file() and not manifest_path.is_symlink():
    try:
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        result['manifest']['parse']='PASS'
        result['manifest']['schema_version']=manifest.get('schema_version')
        result['manifest']['status']=manifest.get('status')
        candidate=manifest.get('candidate',{})
        result['manifest']['candidate']={
            'bin_name':candidate.get('bin_name'),
            'source_sha256':candidate.get('source_sha256'),
        }
        build_inputs=manifest.get('build_inputs',{})
        artifacts=manifest.get('artifacts',{})
        package=manifest.get('package',{})
        result['phase_status']={
            'prepare':{
                'manifest_created':True,
                'build_inputs':{key:value.get('status') if isinstance(value,dict) else None for key,value in build_inputs.items()},
            },
            'build':{key:value.get('status') if isinstance(value,dict) else None for key,value in artifacts.items()},
            'package':package.get('status') if isinstance(package,dict) else None,
        }
    except (OSError,UnicodeError,json.JSONDecodeError) as exc:
        result['manifest']['parse']='FAIL:'+type(exc).__name__

for soc in ('ascend910_93','ascend910b'):
    build_dir=root/'work'/'build'/soc
    result['artifacts'][soc]=artifact_group(build_dir)
    result['opc_logs'][soc]=log_summary(build_dir/'opc.log')

canonical=result['artifacts']['ascend910_93']['files']
alias=result['artifacts']['ascend910b']['files']
result['artifact_pair_byte_identical']=(
    len(canonical)==2 and len(alias)==2 and
    sorted((item['kind'],item['sha256']) for item in canonical)==sorted((item['kind'],item['sha256']) for item in alias)
)

builder_path=root/'build_qrv2_release.py'
contract_path=root/'container_contract.json'
try:
    spec=importlib.util.spec_from_file_location('step369_remote_builder_audit',builder_path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _,current_runtime=module._validate_container_contract(contract_path,opc)
    current_installed=module.installed_qrv2_inventory(installed_root)
    original_root=root/'work'/'wheel_original'/'mx_driving_cloud'
    original_inventory=module.installed_qrv2_inventory(original_root)
    result['inventory']['runtime_contract_current']='PASS'
    result['inventory']['installed_matches_original_wheel']=(current_installed['files']==original_inventory['files'])
    marker='QrV2_lifetime_alpha_sync_fix_v4'
    marker_hits=0
    for relative in current_installed['files']:
        path=installed_root/relative
        if marker.encode() in path.read_bytes():
            marker_hits+=1
    result['inventory']['installed_v4_marker_file_count']=marker_hits
    if isinstance(manifest,dict) and isinstance(manifest.get('build_runtime'),dict):
        recorded=manifest['build_runtime']
        result['inventory']['manifest_runtime_inventory_closed']=(
            recorded.get('runtime_inventory_closed') is True and
            recorded.get('runtime_inventory_after')==current_runtime
        )
        result['inventory']['manifest_installed_inventory_closed']=(
            recorded.get('installed_inventory_closed') is True and
            recorded.get('installed_qrv2_after')==current_installed
        )
    else:
        result['inventory']['manifest_runtime_inventory_closed']=None
        result['inventory']['manifest_installed_inventory_closed']=None
except Exception as exc:
    result['inventory']['audit_error']=type(exc).__name__+': '+clean_line(str(exc))

print(json.dumps(result,sort_keys=True))
'''


def execute() -> dict[str, Any]:
    legacy = load_legacy()
    # Required on every access: local_preflight rereads both authority files,
    # validates the private target suffix, and verifies immutable local hashes.
    info = legacy.local_preflight(legacy.load_remote_module())
    target = ipaddress.ip_address(str(info["target_host"]))
    if not target.is_private or str(target).split(".")[-1] != "42":
        raise RuntimeError("target mapping guard failed")

    jump, remote = legacy.connect_target(legacy.load_remote_module(), info)
    try:
        hostname, _ = legacy.run(remote, "hostname")
        if hostname.strip() != EXPECTED_HOSTNAME:
            raise RuntimeError("second-hop hostname guard failed")
        contract = legacy.container_probe(remote)
        if contract.get("container_name") != CONTAINER:
            raise RuntimeError("exact running container guard failed")

        host_processes, _ = legacy.run(remote, "ps -eo pid=,args=")
        container_processes, _ = legacy.run(
            remote, "docker top " + shlex.quote(CONTAINER) + " -eo pid=,args="
        )
        remote_diag = legacy.safe_remote_path(str(info["shared"]), DIAG_NAME)
        command = (
            "docker exec -e PYTHONDONTWRITEBYTECODE=1 "
            + shlex.quote(CONTAINER)
            + " python3 -c "
            + shlex.quote(REMOTE_AUDIT_CODE)
            + " "
            + shlex.quote(remote_diag)
            + " "
            + shlex.quote(str(contract["installed_cloud_root"]))
            + " "
            + shlex.quote(str(contract["opc"]["path"]))
        )
        output, _ = legacy.run(remote, command, timeout=300)
        result = json.loads(output)
        result["guards"] = {
            "target_private": True,
            "target_suffix": "42",
            "second_hop_hostname": "PASS",
            "exact_running_container": CONTAINER,
        }
        result["live_processes"] = {
            "host": _process_summary(host_processes),
            "container": _process_summary(container_processes),
        }
        result["read_only_actions"] = [
            "stat",
            "sha256",
            "json_parse",
            "log_scan",
            "process_list",
            "inventory_compare",
        ]
        result["forbidden_actions_executed"] = []
        return result
    finally:
        remote.close()
        jump.close()


def main() -> int:
    print(json.dumps(execute(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
