#!/usr/bin/env python3
"""Disabled, fail-closed remote OPC controller for the STEP384 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import stat
import types
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".codex-tools"
CONTRACT_PATH = TOOLS / "step376_build_qrv2_delta1_probe_remote.py"
REMOTE_EXEC_PATH = TOOLS / "remote_exec.py"
OUTER_ZIP = ROOT / "cann-driving-cloud-ops-26.0.7-a3-cann8.3.rc1-torch2.7.1-py3.11-aarch64.zip"
ADAPTER = TOOLS / "build_step384_qrv2_delta2_only_diagnostic.py"
ADAPTER_TEST = TOOLS / "test_step384_delta2_only_build_wiring.py"
PATCHER = TOOLS / "step384_patch_qr_v2_delta2_only_diagnostic.py"
AUDITED_ADAPTER = TOOLS / "build_qrv2_diagnostic_probe.py"
BASE_BUILDER = TOOLS / "build_qrv2_release.py"
V4_PATCHER = TOOLS / "step338_patch_qr_v2_lifetime.py"
STEP375_PATCHER = TOOLS / "step375_patch_qr_v2_vtv_direct_qa_legacy_probe_v6.py"
PROCESS_GUARD = TOOLS / "step377_process_guard.py"
WRAPPER_NAME = "step385_owned_wrapper.py"

BUILD_REMOTE_READY = False
ATTEMPT_NAME = "step385_attempt1_qrv2_delta2_only_opc_build_20260822"
SOCS = ("ascend910_93", "ascend910b")
CANDIDATE_IDENTITY = "QrV2_qa_position_delta2_only_diagnostic_v1"
CANDIDATE_SHA256 = "e352ac31f97980bc5c92caac663414782b78fc3004ff36709b6be8000353b003"
REVERSE_V4_SHA256 = "2213dbae5027c179f09c8e3b43be51587ba22e3eb2a7546311048efc3844614b"
CONTRACT_SHA256 = "1bd72bc9e756354431d381f1dd1d9be8206be5998e8f0f72a4bd606e91e2b7a5"
EXPECTED_INPUTS = {
    OUTER_ZIP.name: "363fc46e0f3da952ef9c37cdfb67a190f557abc8a879d1438563c2d3eb807da7",
    ADAPTER.name: "c00af6a2b455c93b35c81e5133af905f460fc0ce61846470f6bb3821509e7083",
    ADAPTER_TEST.name: "7258367c4173910d218f207c45a6a99973c542172b0840c63815de6019713357",
    PATCHER.name: "2bdaf51e3b08388ca5fcb156e0602312b4f1de3dfc533da6e2d7778d10d3820c",
    AUDITED_ADAPTER.name: "fc65fecc58cefb86f64b6e71d64a21e5e4bc1416b42f1cd696aff6bbdedc299e",
    BASE_BUILDER.name: "d6f2aea68574422ba6e28f220c5581adf8b25019494882dea156a80b1b513e90",
    V4_PATCHER.name: "d4b260919440e6298c9caea5ca3a7c8bb7426a30ed84cc68d934250ee0ae9fe2",
    STEP375_PATCHER.name: "98a655f89ac5efedd760067fdda595d9b5fe376b1e51fdc1b12d59c727711768",
    PROCESS_GUARD.name: "8f4886838c39f96e662ff2a5b3d17c79c9ee01d76bfe826f4b19fb63a66e8199",
}
DRY_RUN_ACTIONS = ("upload_new_exact", "readback_exact", "prepare", "build", "validate_manifest_seal", "seal_build_receipt", "owned_cleanup")
FORBIDDEN_ACTIONS = ("package", "install", "train", "torchrun", "modify_installed", "kill_unowned", "use_latest")
FORBIDDEN_OUTPUTS = {"names": ["release", "release_after_npu_smi"], "suffixes": [".whl", ".zip"]}
REMOTE_SHORT_TIMEOUT = 120
REMOTE_BUILD_TIMEOUT = 720


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_files() -> tuple[Path, ...]:
    return (OUTER_ZIP, ADAPTER, ADAPTER_TEST, PATCHER, AUDITED_ADAPTER, BASE_BUILDER, V4_PATCHER, STEP375_PATCHER, PROCESS_GUARD)


def read_local_regular(path: Path, expected_sha256: str) -> bytes:
    """Read a pinned local upload without following aliases or accepting replacement."""
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = -1
    try:
        before_name = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before_name.st_dev, before_name.st_ino):
            raise RuntimeError(f"STEP385 local input identity mismatch: {path.name}")
        chunks: list[bytes] = []
        while block := os.read(fd, 1024 * 1024):
            chunks.append(block)
        closed = os.fstat(fd)
        after_name = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        identity = lambda value: (value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode), value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if not (identity(before_name) == identity(opened) == identity(closed) == identity(after_name)):
            raise RuntimeError(f"STEP385 local input changed while reading: {path.name}")
        data = b"".join(chunks)
        if sha256_bytes(data) != expected_sha256:
            raise RuntimeError(f"STEP385 locked input SHA mismatch: {path.name}")
        return data
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent)


def _local_identity(path: Path) -> tuple[int, int, int, int, int, int]:
    value = path.lstat()
    return (value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode), value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _payload_snapshot() -> tuple[dict[str, bytes], dict[Path, tuple[int, ...]], bytes]:
    """Capture every armed byte once and retain identities for post-upload closure."""
    paths = (CONTRACT_PATH,) + input_files()
    identities = {path: _local_identity(path) for path in paths}
    raw = {path: read_local_regular(path, CONTRACT_SHA256 if path == CONTRACT_PATH else EXPECTED_INPUTS[path.name]) for path in paths}
    if any(_local_identity(path) != identities[path] for path in paths):
        raise RuntimeError("STEP385 local payload snapshot changed")
    payloads = {path.name: raw[path] for path in input_files()}
    payloads[ADAPTER.name] = _remote_adapter_bytes(payloads[ADAPTER.name])
    payloads[WRAPPER_NAME] = _wrapper_code()
    return payloads, identities, raw[CONTRACT_PATH]


def _revalidate_payload_identities(identities: dict[Path, tuple[int, ...]]) -> None:
    if any(_local_identity(path) != identity for path, identity in identities.items()):
        raise RuntimeError("STEP385 local payload identity changed after upload")


def _require_remote_ready() -> None:
    if BUILD_REMOTE_READY is not True:
        raise RuntimeError("STEP385 remote build is intentionally disabled: BUILD_REMOTE_READY is false")
    if "latest" in ATTEMPT_NAME.lower() or not ATTEMPT_NAME.startswith("step385_attempt"):
        raise RuntimeError("STEP385 attempt identity is not immutable")
    files = input_files()
    if len(files) != len(EXPECTED_INPUTS) or {p.name for p in files} != set(EXPECTED_INPUTS):
        raise RuntimeError("STEP385 locked upload closure mismatch")
    locked = {CONTRACT_PATH: CONTRACT_SHA256, **{p: EXPECTED_INPUTS[p.name] for p in files}}
    for path, expected in locked.items():
        read_local_regular(path, expected)


def _remote_adapter_bytes(source: bytes) -> bytes:
    """Arm only the already pinned payload; never reopen ADAPTER by path."""
    old, new = b"BUILD_READY = False", b"BUILD_READY = True"
    if source.count(old) != 1 or new in source:
        raise RuntimeError("STEP385 adapter readiness token is not uniquely patchable")
    result = source.replace(old, new, 1)
    if len(result) != len(source) - len(old) + len(new):
        raise RuntimeError("STEP385 adapter patch changed unexpected bytes")
    return result


def load_contract(contract_source: Optional[bytes] = None) -> Any:
    _require_remote_ready()
    if contract_source is None:
        contract_source = read_local_regular(CONTRACT_PATH, CONTRACT_SHA256)
    if sha256_bytes(contract_source) != CONTRACT_SHA256:
        raise RuntimeError("STEP385 locked contract bytes SHA mismatch")
    module = types.ModuleType("step385_fixed_step376_contract")
    module.__file__ = str(CONTRACT_PATH)
    exec(compile(contract_source, str(CONTRACT_PATH), "exec"), module.__dict__)
    # STEP376 is a pinned contract factory.  Its own gate is enabled only in
    # this private module instance so it can verify and return the non-latest
    # STEP357 transport helpers; no source file is modified.
    module.BUILD_READY = True
    return module.load_legacy()


def _validated_opc(contract: dict[str, Any]) -> str:
    opc = contract.get("opc")
    path = opc.get("path") if isinstance(opc, dict) else None
    if not isinstance(opc,dict) or set(opc)!={"path","sha256"} or type(path) is not str or not path or "\0" in path or not PurePosixPath(path).is_absolute() or not _lower_hex64(opc.get("sha256")):
        raise RuntimeError("STEP385 locked OPC path invalid")
    return path


def _validate_contract(contract: Any) -> dict[str, Any]:
    required = {"schema_version", "container_name", "inspect_container_id", "inspect_hostname", "opc", "cann_version_files", "ascend_opp", "installed_cloud_root"}
    if not isinstance(contract, dict) or set(contract) != required:
        raise RuntimeError("STEP385 container contract schema mismatch")
    if contract["container_name"] != "mapqr-leicheng" or any(type(contract[key]) is not str or not PurePosixPath(contract[key]).is_absolute() for key in ("ascend_opp", "installed_cloud_root")):
        raise RuntimeError("STEP385 container contract identity mismatch")
    versions = contract["cann_version_files"]
    hostname=contract["inspect_hostname"]
    if type(contract["schema_version"]) is not int or contract["schema_version"] != 1 or type(contract["inspect_container_id"]) is not str or len(contract["inspect_container_id"]) != 64 or any(c not in "0123456789abcdef" for c in contract["inspect_container_id"]) or type(hostname) is not str or not 1<=len(hostname)<=64 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in hostname) or not isinstance(versions, list) or not versions or any(not isinstance(row, dict) or set(row) != {"path", "sha256"} or type(row["path"]) is not str or not PurePosixPath(row["path"]).is_absolute() or not _lower_hex64(row["sha256"]) for row in versions):
        raise RuntimeError("STEP385 container contract field mismatch")
    if len({row["path"] for row in versions}) != len(versions):
        raise RuntimeError("STEP385 duplicate CANN version path")
    _validated_opc(contract)
    return contract


def _container_script(contract_module: Any, contract: dict[str, Any], remote_attempt: str, phase: str = "all") -> str:
    work = remote_attempt + "/work"
    staging = remote_attempt + "/staging"
    opc = _validated_opc(contract)
    adapter = ADAPTER.name
    if phase not in ("prepare", "build", "all"):
        raise ValueError("invalid STEP385 phase")
    prepare = f"python3 {shlex.quote(adapter)} prepare --outer-zip {shlex.quote(OUTER_ZIP.name)} --workdir {shlex.quote(work)} --approved-root {shlex.quote(remote_attempt)}"
    build = f"python3 {shlex.quote(adapter)} build --workdir {shlex.quote(work)} --opc {shlex.quote(opc)} --container-contract {shlex.quote(staging + '/container_contract.json')} --installed-cloud-root {shlex.quote(contract['installed_cloud_root'])} --approved-root {shlex.quote(remote_attempt)}"
    selected = "\n".join((prepare, build) if phase == "all" else ((prepare,) if phase == "prepare" else (build,)))
    return f"""set -eu
cd {shlex.quote(staging)}
export ASCEND_OPP_PATH={shlex.quote(contract['ascend_opp'])}
export PYTHONOPTIMIZE=
export PYTHONDONTWRITEBYTECODE=1
{selected}
"""


def _exclusive_attempt_script(path: str) -> str:
    quoted = shlex.quote(path)
    return "set -eu\n[ ! -e " + quoted + " ] || exit 73\nmkdir -m 700 -- " + quoted + "\nmkdir -m 700 -- " + quoted + "/staging\n"


def _upload_gate_code() -> str:
    return r'''import hashlib,json,os,stat,sys
root=sys.argv[1]; expected=json.loads(sys.argv[2]); locked=json.loads(sys.argv[3]) if len(sys.argv)>3 else None; d=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
 ds0=os.fstat(d); root_id={'dev':ds0.st_dev,'ino':ds0.st_ino,'mode':stat.S_IMODE(ds0.st_mode)}
 assert stat.S_ISDIR(ds0.st_mode) and (locked is None or root_id==locked)
 names=set(os.listdir(d)); assert names==set(expected)
 for name in names:
  before=os.stat(name,dir_fd=d,follow_symlinks=False)
  f=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=d)
  try:
   opened=os.fstat(f); assert stat.S_ISREG(opened.st_mode); h=hashlib.sha256()
   while chunk:=os.read(f,1048576): h.update(chunk)
   closed=os.fstat(f); after=os.stat(name,dir_fd=d,follow_symlinks=False)
   ident=lambda s:(s.st_dev,s.st_ino,stat.S_IFMT(s.st_mode),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
   assert ident(before)==ident(opened)==ident(closed)==ident(after) and stat.S_IMODE(opened.st_mode)==384 and h.hexdigest()==expected[name]
  finally: os.close(f)
 assert set(os.listdir(d))==names
 ds1=os.fstat(d); assert (ds0.st_dev,ds0.st_ino,stat.S_IFMT(ds0.st_mode))==(ds1.st_dev,ds1.st_ino,stat.S_IFMT(ds1.st_mode))
finally: os.close(d)
print(json.dumps({'schema':'step385-staging-gate-v1','root':root_id},sort_keys=True))'''


def _snapshot_code() -> str:
    return r'''import hashlib,importlib.util,json,os,stat,subprocess,sys
from pathlib import Path
installed=Path(sys.argv[1]); attempt=Path(sys.argv[2]); opc=Path(sys.argv[3]); guard=Path(sys.argv[4]); expected_receipt=json.loads(sys.argv[5]) if len(sys.argv)>5 else None
guard_name='_locked_step377_guard_'+hashlib.sha256(os.fsencode(str(guard))).hexdigest(); sentinel=object(); prior=sys.modules.get(guard_name,sentinel); spec=importlib.util.spec_from_file_location(guard_name,guard); g=importlib.util.module_from_spec(spec); sys.modules[guard_name]=g
try: spec.loader.exec_module(g)
finally:
 if prior is sentinel: del sys.modules[guard_name]
 else: sys.modules[guard_name]=prior
def inventory(root):
 root_state=root.lstat(); assert stat.S_ISDIR(root_state.st_mode) and not root.is_symlink(); out={}; paths=sorted(root.rglob('*')); names=[str(p.relative_to(root)) for p in paths]
 for p in paths:
  s=p.lstat(); rel=str(p.relative_to(root)); mode=stat.S_IMODE(s.st_mode)
  if stat.S_ISDIR(s.st_mode): out[rel]={'type':'DIR','mode':mode,'size':s.st_size,'sha256':None,'target':None}
  elif stat.S_ISREG(s.st_mode):
   f=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
   try:
    opened=os.fstat(f); h=hashlib.sha256()
    while chunk:=os.read(f,1048576): h.update(chunk)
    closed=os.fstat(f); after=p.lstat(); ident=lambda x:(x.st_dev,x.st_ino,stat.S_IFMT(x.st_mode),x.st_size,x.st_mtime_ns,x.st_ctime_ns); assert ident(s)==ident(opened)==ident(closed)==ident(after)
    out[rel]={'type':'REG','mode':mode,'size':s.st_size,'sha256':h.hexdigest(),'target':None}
   finally: os.close(f)
  elif stat.S_ISLNK(s.st_mode): out[rel]={'type':'SYMLINK','mode':mode,'size':s.st_size,'sha256':None,'target':os.readlink(p)}
  else: raise AssertionError(('special installed entry',rel))
 root_after=root.lstat(); after_names=[str(p.relative_to(root)) for p in sorted(root.rglob('*'))]; rid=lambda s:(s.st_dev,s.st_ino,stat.S_IFMT(s.st_mode),s.st_mtime_ns,s.st_ctime_ns); assert rid(root_state)==rid(root_after) and names==after_names and set(out)==set(names)
 return {'root':{'dev':root_state.st_dev,'ino':root_state.st_ino,'mode':stat.S_IMODE(root_state.st_mode)},'entries':out}
def regular(path):
 real=str(path.resolve(strict=True)); parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); f=-1
 try:
  before=os.stat(path.name,dir_fd=parent,follow_symlinks=False); f=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent); opened=os.fstat(f); h=hashlib.sha256()
  while chunk:=os.read(f,1048576): h.update(chunk)
  closed=os.fstat(f); after=os.stat(path.name,dir_fd=parent,follow_symlinks=False); ident=lambda s:(s.st_dev,s.st_ino,stat.S_IFMT(s.st_mode),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
  assert stat.S_ISREG(opened.st_mode) and ident(before)==ident(opened)==ident(closed)==ident(after)
  return {'path':str(path),'realpath':real,'dev':opened.st_dev,'ino':opened.st_ino,'mode':stat.S_IMODE(opened.st_mode),'size':opened.st_size,'sha256':h.hexdigest()}
 finally:
  if f>=0: os.close(f)
  os.close(parent)
attempt_state=attempt.lstat(); assert stat.S_ISDIR(attempt_state.st_mode) and not attempt.is_symlink()
if expected_receipt is not None:
 assert set(expected_receipt)=={'dev','ino','mode','size','sha256'}
 work=attempt/'work'; work_state=work.lstat(); assert stat.S_ISDIR(work_state.st_mode) and not work.is_symlink()
 receipt_path=attempt/'build_receipt.json'; receipt_state=regular(receipt_path); assert {k:receipt_state[k] for k in ('dev','ino','mode','size','sha256')}==expected_receipt
 receipt_parent=os.open(attempt,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); receipt_fd=os.open(receipt_path.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=receipt_parent)
 try:
  opened=os.fstat(receipt_fd); chunks=[]
  while chunk:=os.read(receipt_fd,4096): chunks.append(chunk)
  named=os.stat(receipt_path.name,dir_fd=receipt_parent,follow_symlinks=False); assert (opened.st_dev,opened.st_ino)==(named.st_dev,named.st_ino)==(expected_receipt['dev'],expected_receipt['ino']); receipt_value=json.loads(b''.join(chunks))
 finally: os.close(receipt_fd); os.close(receipt_parent)
 assert receipt_value['attempt']=={'path':str(attempt),'dev':attempt_state.st_dev,'ino':attempt_state.st_ino} and receipt_value['work']=={'path':str(work),'dev':work_state.st_dev,'ino':work_state.st_ino}
def npu():
 text=subprocess.run(['npu-smi','info'],check=True,text=True,capture_output=True).stdout
 rows=g.parse_back8_idle(text)
 return {'rows':[{'physical':r.physical,'chip':r.chip,'device_id':r.device_id,'host_pid':r.host_pid} for r in rows],'device_ids':[],'host_pids':[]}
ia=inventory(installed); ib=inventory(installed); assert ia==ib
oa=regular(opc); ob=regular(opc); assert oa==ob
na=npu(); nb=npu(); assert na==nb
print(json.dumps({'schema':'step385-closure-v2','installed_samples':[ia,ib],'opc_samples':[oa,ob],'npu_samples':[na,nb]},sort_keys=True))'''


def _receipt_code() -> str:
    return r'''import hashlib,json,os,stat,sys,types
from pathlib import Path
root=Path(sys.argv[1]); adapter_path=root/'staging'/sys.argv[2]; expected_adapter=sys.argv[3]; container_id=sys.argv[4]; init_starttime=int(sys.argv[5]); expected_mnt=int(sys.argv[6]); work=root/'work'; receipt_path=root/'build_receipt.json'
def read_regular(path):
 parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); fd=-1
 try:
  before=os.stat(path.name,dir_fd=parent,follow_symlinks=False); fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent); opened=os.fstat(fd); chunks=[]
  while chunk:=os.read(fd,1048576): chunks.append(chunk)
  closed=os.fstat(fd); after=os.stat(path.name,dir_fd=parent,follow_symlinks=False); ident=lambda s:(s.st_dev,s.st_ino,stat.S_IFMT(s.st_mode),s.st_size,s.st_mtime_ns,s.st_ctime_ns); assert stat.S_ISREG(opened.st_mode) and ident(before)==ident(opened)==ident(closed)==ident(after); payload=b''.join(chunks)
  return payload,{'path':str(path),'dev':opened.st_dev,'ino':opened.st_ino,'mode':stat.S_IMODE(opened.st_mode),'size':opened.st_size,'sha256':hashlib.sha256(payload).hexdigest()}
 finally:
  if fd>=0: os.close(fd)
  os.close(parent)
def load_locked(name,path,expected):
 payload,_state=read_regular(path); assert hashlib.sha256(payload).hexdigest()==expected and name not in sys.modules; module=types.ModuleType(name); module.__file__=str(path); sys.modules[name]=module; exec(compile(payload,str(path),'exec'),module.__dict__); return module
load_locked('step338_patch_qr_v2_lifetime',root/'staging'/'step338_patch_qr_v2_lifetime.py',sys.argv[7]); load_locked('step375_patch_qr_v2_vtv_direct_qa_legacy_probe_v6',root/'staging'/'step375_patch_qr_v2_vtv_direct_qa_legacy_probe_v6.py',sys.argv[8])
adapter_payload,_=read_regular(adapter_path); assert hashlib.sha256(adapter_payload).hexdigest()==expected_adapter
m=types.ModuleType('_step387_receipt_adapter'); m.__file__=str(adapter_path); exec(compile(adapter_payload,str(adapter_path),'exec'),m.__dict__)
manifest_payload,manifest_state=read_regular(work/m.MANIFEST_NAME); manifest=json.loads(manifest_payload); base=m._load_base(); m._validate_manifest(base,work,manifest,expected_status=m.DIAGNOSTIC_BUILT_STATUS); m._require_completed_attempt(base,work,manifest); m._validate_built_artifact_closure(base,work,manifest,enrich=False); m._assert_no_release_outputs(work,manifest); assert set(manifest['artifacts'])=={'ascend910_93','ascend910b'}
seal_payload,seal_state=read_regular(work/m.ATTEMPT_MARKER_NAME); assert json.loads(seal_payload)['status']=='completed_consumable'
artifact_states={}
for soc in sorted(manifest['artifacts']):
 row=manifest['artifacts'][soc]; artifact_states[soc]={}
 for key in ('object_path','json_path','opc_log_path'):
  _payload,state=read_regular(Path(row[key])); artifact_states[soc][key]=state
attempt_stat=root.lstat(); work_stat=work.lstat(); assert stat.S_ISDIR(attempt_stat.st_mode) and stat.S_ISDIR(work_stat.st_mode) and not root.is_symlink() and not work.is_symlink(); actual_mnt=os.stat('/proc/self/ns/mnt').st_ino; assert actual_mnt==expected_mnt
receipt={'schema':'step387-build-receipt-v1','container':{'id':container_id,'init_starttime':init_starttime,'mnt_ns':actual_mnt},'attempt':{'path':str(root),'dev':attempt_stat.st_dev,'ino':attempt_stat.st_ino},'work':{'path':str(work),'dev':work_stat.st_dev,'ino':work_stat.st_ino},'manifest':manifest_state,'completion':seal_state,'artifacts':artifact_states}
parent=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); fd=-1
try:
 fd=os.open(receipt_path.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent); payload=(json.dumps(receipt,sort_keys=True)+'\n').encode(); view=memoryview(payload)
 while view:
  count=os.write(fd,view); assert count>0; view=view[count:]
 os.fsync(fd); opened=os.fstat(fd); os.fsync(parent); named=os.stat(receipt_path.name,dir_fd=parent,follow_symlinks=False); assert stat.S_ISREG(opened.st_mode) and stat.S_IMODE(opened.st_mode)==0o600 and (opened.st_dev,opened.st_ino)==(named.st_dev,named.st_ino)
finally:
 if fd>=0: os.close(fd)
 os.close(parent)'''


def _wrapper_code() -> bytes:
    """Trusted, SHA-pinned wrapper: publish ownership before spawning a child."""
    return r'''import json,os,signal,stat,subprocess,sys,time
from pathlib import Path
manifest=Path(sys.argv[1]); token=sys.argv[2]; container_id=sys.argv[3]; expected_hostname=sys.argv[4]; command=sys.argv[5]; nonce=os.urandom(32).hex(); decision_path=manifest.with_name('ownership_start_decision.json'); precommit_path=manifest.with_name('ownership_precommit_seal.json'); lock_path=manifest.with_name('ownership_host_lock.json')
actual_hostname=os.uname().nodename; assert actual_hostname==expected_hostname
def write_all(fd,data):
 view=memoryview(data)
 while view:
  count=os.write(fd,view)
  if count<=0: raise OSError('short write')
  view=view[count:]
def persist_new(path,value):
 parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); fd=-1
 try:
  fd=os.open(path.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent); opened=os.fstat(fd); assert stat.S_ISREG(opened.st_mode) and stat.S_IMODE(opened.st_mode)==0o600
  write_all(fd,(json.dumps(value,sort_keys=True)+'\n').encode()); os.fsync(fd); closed=os.fstat(fd); os.fsync(parent); named=os.stat(path.name,dir_fd=parent,follow_symlinks=False)
  assert stat.S_ISREG(named.st_mode) and (opened.st_dev,opened.st_ino)==(closed.st_dev,closed.st_ino)==(named.st_dev,named.st_ino)
 finally:
  if fd>=0: os.close(fd)
  os.close(parent)
def stat_fields(pid):
 fields=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split(); return int(fields[19]),int(fields[1]),int(fields[2]),int(fields[3])
status=Path('/proc/self/status').read_text().splitlines(); nspid=[int(x) for x in next(x for x in status if x.startswith('NSpid:')).split()[1:]]
starttime,ppid,pgid,sid=stat_fields(os.getpid()); argv=Path('/proc/self/cmdline').read_bytes().split(b'\0'); argv=[os.fsdecode(x) for x in argv if x]; cgroup=Path('/proc/self/cgroup').read_text(); namespaces={name:os.stat('/proc/self/ns/'+name).st_ino for name in ('pid','mnt','cgroup')}
record={'schema':'step385-owned-wrapper-v2','token':token,'nonce':nonce,'container_id':container_id,'container_hostname':actual_hostname,'wrapper':{'container_pid':os.getpid(),'container_nspid':nspid,'container_starttime':starttime,'container_ppid':ppid,'container_pgid':pgid,'container_sid':sid,'argv':argv,'cgroup':cgroup,'cgroup_sha256':__import__('hashlib').sha256(cgroup.encode()).hexdigest(),'namespaces':namespaces}}
stop_requested=False; child=None
def stop(_sig,_frame):
 global stop_requested; stop_requested=True
signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGHUP,stop)
persist_new(manifest,record)
if stop_requested:
 persist_new(manifest.with_name('build_result.json'),{'schema':'step385-build-result-v1','returncode':143}); raise SystemExit(0)
def read_strict(path,limit=65536):
 parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); fd=-1
 try:
  before=os.stat(path.name,dir_fd=parent,follow_symlinks=False); fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent); opened=os.fstat(fd); chunks=[]; total=0
  while True:
   chunk=os.read(fd,4096)
   if not chunk: break
   total+=len(chunk); assert total<=limit; chunks.append(chunk)
  closed=os.fstat(fd); after=os.stat(path.name,dir_fd=parent,follow_symlinks=False); identity=lambda s:(s.st_dev,s.st_ino,stat.S_IFMT(s.st_mode),s.st_size,s.st_mtime_ns,s.st_ctime_ns); assert stat.S_ISREG(opened.st_mode) and stat.S_IMODE(opened.st_mode)==0o600 and identity(before)==identity(opened)==identity(closed)==identity(after); return b''.join(chunks),opened
 finally:
  if fd>=0: os.close(fd)
  os.close(parent)
deadline=time.monotonic()+120
while not decision_path.exists() and time.monotonic()<deadline and not stop_requested: time.sleep(.05)
if stop_requested or not decision_path.exists(): raise SystemExit(0)
decision_payload,_=read_strict(decision_path); decision=json.loads(decision_payload); assert type(decision) is dict and decision.get('schema')=='step385-start-decision-v1' and decision.get('status') in ('committed','aborted') and decision.get('token')==token and decision.get('wrapper_nonce')==nonce
if decision['status']=='aborted':
 assert set(decision)=={'schema','status','token','wrapper_nonce'}; raise SystemExit(0)
assert set(decision)=={'schema','status','token','wrapper_nonce','host_lock_sha256','host_lock_dev','host_lock_ino'} and all(type(decision[k]) is int and decision[k]>0 for k in ('host_lock_dev','host_lock_ino'))
lock_payload,lock_stat=read_strict(lock_path); precommit_payload,_=read_strict(precommit_path); precommit=json.loads(precommit_payload); lock_state={'dev':lock_stat.st_dev,'ino':lock_stat.st_ino,'mode':stat.S_IMODE(lock_stat.st_mode),'sha256':__import__('hashlib').sha256(lock_payload).hexdigest()}; assert set(precommit)=={'schema','token','wrapper_nonce','lock','start_decision_sha256'} and precommit['schema']=='step385-precommit-seal-v1' and precommit['token']==token and precommit['wrapper_nonce']==nonce and precommit['lock']==lock_state and precommit['start_decision_sha256']==__import__('hashlib').sha256(decision_payload).hexdigest(); assert lock_state['sha256']==decision['host_lock_sha256'] and (lock_state['dev'],lock_state['ino'])==(decision['host_lock_dev'],decision['host_lock_ino'])
stdout_path=manifest.with_name('child_stdout.log'); stderr_path=manifest.with_name('child_stderr.log'); log_parent=os.open(manifest.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); stdout_fd=stderr_fd=-1
try:
 stdout_fd=os.open(stdout_path.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=log_parent); stderr_fd=os.open(stderr_path.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=log_parent); stdout_open=os.fstat(stdout_fd); stderr_open=os.fstat(stderr_fd); assert stat.S_ISREG(stdout_open.st_mode) and stat.S_ISREG(stderr_open.st_mode) and stat.S_IMODE(stdout_open.st_mode)==stat.S_IMODE(stderr_open.st_mode)==0o600; os.fsync(log_parent)
 child=subprocess.Popen(['bash','--noprofile','--norc','-lc',command,'STEP385_CHILD_'+token],stdout=stdout_fd,stderr=stderr_fd)
 while child.poll() is None: time.sleep(.1)
 rc=child.wait(); os.fsync(stdout_fd); os.fsync(stderr_fd); stdout_closed=os.fstat(stdout_fd); stderr_closed=os.fstat(stderr_fd); stdout_named=os.stat(stdout_path.name,dir_fd=log_parent,follow_symlinks=False); stderr_named=os.stat(stderr_path.name,dir_fd=log_parent,follow_symlinks=False); assert (stdout_open.st_dev,stdout_open.st_ino)==(stdout_closed.st_dev,stdout_closed.st_ino)==(stdout_named.st_dev,stdout_named.st_ino) and (stderr_open.st_dev,stderr_open.st_ino)==(stderr_closed.st_dev,stderr_closed.st_ino)==(stderr_named.st_dev,stderr_named.st_ino) and stdout_closed.st_size<=1048576 and stderr_closed.st_size<=1048576
finally:
 if stdout_fd>=0: os.close(stdout_fd)
 if stderr_fd>=0: os.close(stderr_fd)
 os.close(log_parent)
stdout_payload,stdout_stat=read_strict(stdout_path,1048576); stderr_payload,stderr_stat=read_strict(stderr_path,1048576); log_state=lambda path,payload,s:{'path':str(path),'dev':s.st_dev,'ino':s.st_ino,'mode':stat.S_IMODE(s.st_mode),'size':s.st_size,'sha256':__import__('hashlib').sha256(payload).hexdigest()}; result={'schema':'step385-build-result-v3','returncode':rc,'logs':{'stdout':log_state(stdout_path,stdout_payload,stdout_stat),'stderr':log_state(stderr_path,stderr_payload,stderr_stat)}}
if rc==0:
 receipt_payload,receipt_stat=read_strict(manifest.with_name('build_receipt.json')); receipt=json.loads(receipt_payload); assert receipt.get('schema')=='step387-build-receipt-v1'; result['receipt']={'dev':receipt_stat.st_dev,'ino':receipt_stat.st_ino,'mode':stat.S_IMODE(receipt_stat.st_mode),'size':receipt_stat.st_size,'sha256':__import__('hashlib').sha256(receipt_payload).hexdigest()}
persist_new(manifest.with_name('build_result.json'),result)
while not stop_requested: time.sleep(.1)
raise SystemExit(0)'''.encode()


def _result_wait_code() -> str:
    return r'''import hashlib,json,os,stat,subprocess,sys,time
from pathlib import Path
p=Path(sys.argv[1]); deadline=time.monotonic()+int(sys.argv[2]); attempt=Path(sys.argv[3]); container=sys.argv[4]; expected_id=sys.argv[5]; expected_start=int(sys.argv[6]); expected_mnt=int(sys.argv[7])
def read_state(path,limit=10485760):
 parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); fd=-1
 try:
  before=os.stat(path.name,dir_fd=parent,follow_symlinks=False); fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent); opened=os.fstat(fd); chunks=[]; total=0
  while True:
   chunk=os.read(fd,4096)
   if not chunk: break
   total+=len(chunk); assert total<=limit; chunks.append(chunk)
  closed=os.fstat(fd); after=os.stat(path.name,dir_fd=parent,follow_symlinks=False); ident=lambda s:(s.st_dev,s.st_ino,stat.S_IFMT(s.st_mode),s.st_size,s.st_mtime_ns,s.st_ctime_ns); assert stat.S_ISREG(opened.st_mode) and ident(before)==ident(opened)==ident(closed)==ident(after); data=b''.join(chunks); return data,{'path':str(path),'dev':opened.st_dev,'ino':opened.st_ino,'mode':stat.S_IMODE(opened.st_mode),'size':opened.st_size,'sha256':hashlib.sha256(data).hexdigest()}
 finally:
  if fd>=0: os.close(fd)
  os.close(parent)
def verify_receipt(expected):
 data,state=read_state(attempt/'build_receipt.json'); assert {k:state[k] for k in ('dev','ino','mode','size','sha256')}==expected and state['mode']==0o600; receipt=json.loads(data); assert set(receipt)=={'schema','container','attempt','work','manifest','completion','artifacts'} and receipt['schema']=='step387-build-receipt-v1' and receipt['container']=={'id':expected_id,'init_starttime':expected_start,'mnt_ns':expected_mnt}
 for key,path in (('attempt',attempt),('work',attempt/'work')):
  s=path.lstat(); assert stat.S_ISDIR(s.st_mode) and not path.is_symlink() and receipt[key]=={'path':str(path),'dev':s.st_dev,'ino':s.st_ino}
 for key in ('manifest','completion'):
  bound=receipt[key]; current=read_state(Path(bound['path']))[1]; assert current==bound
 assert set(receipt['artifacts'])=={'ascend910_93','ascend910b'}
 for soc,row in receipt['artifacts'].items():
  assert set(row)=={'object_path','json_path','opc_log_path'}
  for bound in row.values(): assert read_state(Path(bound['path']))[1]==bound
 raw=subprocess.run(['docker','inspect',container],check=True,capture_output=True,text=True).stdout; rows=json.loads(raw); assert type(rows) is list and len(rows)==1 and rows[0]['Id']==expected_id; init_pid=rows[0]['State']['Pid']; fields=Path('/proc/%d/stat'%init_pid).read_text().rsplit(')',1)[1].split(); assert int(fields[19])==expected_start and os.stat('/proc/%d/ns/mnt'%init_pid).st_ino==expected_mnt
 return state
while time.monotonic()<deadline:
 try:
  data,_state=read_state(p); assert _state['mode']==0o600; value=json.loads(data); assert type(value) is dict and value.get('schema')=='step385-build-result-v3' and type(value.get('returncode')) is int and type(value.get('logs')) is dict and set(value['logs'])=={'stdout','stderr'}
  for name in ('stdout','stderr'):
   bound=value['logs'][name]; assert type(bound) is dict and set(bound)=={'path','dev','ino','mode','size','sha256'} and bound['path']==str(attempt/('child_'+name+'.log')) and bound['mode']==0o600 and 0<=bound['size']<=1048576 and read_state(Path(bound['path']),1048576)[1]==bound
  if value['returncode']==0:
   assert set(value)=={'schema','returncode','logs','receipt'} and set(value['receipt'])=={'dev','ino','mode','size','sha256'}; verify_receipt(value['receipt'])
  else: assert set(value)=={'schema','returncode','logs'}
  print(json.dumps(value,sort_keys=True)); raise SystemExit(0)
 except (FileNotFoundError,json.JSONDecodeError,AssertionError): time.sleep(.2)
raise SystemExit(124)'''


def _container_identity_code() -> str:
    return r'''import hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
container=sys.argv[1]; expected_id=sys.argv[2]; expected_hostname=sys.argv[3]; guard=Path(sys.argv[4])
raw=subprocess.run(['docker','inspect',container],check=True,capture_output=True,text=True).stdout; rows=json.loads(raw); assert type(rows) is list and len(rows)==1; row=rows[0]; init_pid=row['State']['Pid']; assert row['Id']==expected_id and row['Config']['Hostname']==expected_hostname and type(init_pid) is int and init_pid>1
name='_locked_step377_guard_'+hashlib.sha256(os.fsencode(str(guard))).hexdigest(); sentinel=object(); prior=sys.modules.get(name,sentinel); spec=importlib.util.spec_from_file_location(name,guard); g=importlib.util.module_from_spec(spec); sys.modules[name]=g
try: spec.loader.exec_module(g)
finally:
 if prior is sentinel: del sys.modules[name]
 else: sys.modules[name]=prior
proc=os.open('/proc/%d'%init_pid,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
 start_a,pgid_a=g.parse_stat_identity(g._read_at(proc,'stat',65536)); lines=[line for line in g._read_at(proc,'status',1048576).splitlines() if line.startswith(b'NSpid:')]; assert len(lines)==1; nspid=[int(x) for x in lines[0].split()[1:]]; assert nspid and nspid[0]==init_pid and nspid[-1]==1 and all(x>=1 for x in nspid); argv=[os.fsdecode(x) for x in g._read_at(proc,'cmdline',1048576).split(b'\0') if x]; start_b,pgid_b=g.parse_stat_identity(g._read_at(proc,'stat',65536)); assert (start_a,pgid_a)==(start_b,pgid_b)
finally: os.close(proc)
cgroup=Path('/proc/%d/cgroup'%init_pid).read_text(); assert expected_id in cgroup or expected_id[:12] in cgroup; namespaces={key:os.stat('/proc/%d/ns/%s'%(init_pid,key)).st_ino for key in ('pid','mnt','cgroup')}
print(json.dumps({'schema':'step385-host-container-v1','container_id':expected_id,'hostname':expected_hostname,'init':{'host_pid':init_pid,'starttime':start_a,'nspid':nspid,'pgid':pgid_a,'argv':argv},'cgroup':cgroup,'cgroup_sha256':hashlib.sha256(cgroup.encode()).hexdigest(),'namespaces':namespaces},sort_keys=True))'''


def _validate_host_container_identity(value: Any, contract: dict[str, Any]) -> dict[str, Any]:
    top = {"schema", "container_id", "hostname", "init", "cgroup", "cgroup_sha256", "namespaces"}
    if type(value) is not dict or set(value) != top or value["schema"] != "step385-host-container-v1" or value["container_id"] != contract["inspect_container_id"] or value["hostname"] != contract["inspect_hostname"]:
        raise RuntimeError("STEP385 host container identity mismatch")
    init = value["init"]
    if type(init) is not dict or set(init) != {"host_pid", "starttime", "nspid", "pgid", "argv"} or any(type(init[k]) is not int or init[k] <= 1 for k in ("host_pid", "starttime", "pgid")) or type(init["nspid"]) is not list or not init["nspid"] or init["nspid"][0] != init["host_pid"] or init["nspid"][-1] != 1 or any(type(x) is not int or x < 1 for x in init["nspid"]) or type(init["argv"]) is not list or any(type(x) is not str for x in init["argv"]):
        raise RuntimeError("STEP385 container init identity malformed")
    if type(value["cgroup"]) is not str or not value["cgroup"] or sha256_bytes(value["cgroup"].encode()) != value["cgroup_sha256"] or not _lower_hex64(value["cgroup_sha256"]) or type(value["namespaces"]) is not dict or set(value["namespaces"]) != {"pid", "mnt", "cgroup"} or any(type(x) is not int or x <= 0 for x in value["namespaces"].values()):
        raise RuntimeError("STEP385 container namespace/cgroup identity malformed")
    return value


def _owned_code() -> str:
    return r'''import hashlib,importlib.util,json,os,stat,sys,time
from pathlib import Path
root=Path(sys.argv[1]); manifest=root/'ownership.json'; expected_client=json.loads(sys.argv[2]); expected_token=sys.argv[3]; expected_container=sys.argv[4]; expected_hostname=sys.argv[5]; host_contract=json.loads(sys.argv[6]); action=sys.argv[7]; expected_commit_seal=json.loads(sys.argv[8]) if len(sys.argv)>8 else None
guard=root/'staging'/'step377_process_guard.py'; guard_name='_locked_step377_guard_'+hashlib.sha256(os.fsencode(str(guard))).hexdigest(); sentinel=object(); prior=sys.modules.get(guard_name,sentinel); spec=importlib.util.spec_from_file_location(guard_name,guard); g=importlib.util.module_from_spec(spec); sys.modules[guard_name]=g
try: spec.loader.exec_module(g)
finally:
 if prior is sentinel: del sys.modules[guard_name]
 else: sys.modules[guard_name]=prior
def allproc():
 out={}
 for name in os.listdir('/proc'):
  if not name.isdecimal(): continue
  d=-1
  try:
   pid=int(name); d=os.open('/proc/'+name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); ident=g._identity_from_open_dir(pid,d); raw=g._read_at(d,'stat',1048576).rsplit(b')',1)[1].split(); cgroup=g._read_at(d,'cgroup',1048576).decode(); namespaces={key:os.stat('/proc/'+name+'/ns/'+key).st_ino for key in ('pid','mnt','cgroup')}; out[pid]=(ident,int(raw[1]),int(raw[3]),cgroup,namespaces)
  except (FileNotFoundError,ProcessLookupError,PermissionError,RuntimeError): pass
  finally:
   if d>=0: os.close(d)
 return out
def read_regular_state(path):
 parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); fd=-1
 try:
  before=os.stat(path.name,dir_fd=parent,follow_symlinks=False); fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent); opened=os.fstat(fd); chunks=[]; total=0
  while True:
   chunk=os.read(fd,4096)
   if not chunk: break
   total+=len(chunk); assert total<=65536; chunks.append(chunk)
  closed=os.fstat(fd); after=os.stat(path.name,dir_fd=parent,follow_symlinks=False); identity=lambda s:(s.st_dev,s.st_ino,stat.S_IFMT(s.st_mode),s.st_size,s.st_mtime_ns,s.st_ctime_ns); assert stat.S_ISREG(opened.st_mode) and stat.S_IMODE(opened.st_mode)==0o600 and identity(before)==identity(opened)==identity(closed)==identity(after); payload=b''.join(chunks); return payload,{'dev':opened.st_dev,'ino':opened.st_ino,'mode':stat.S_IMODE(opened.st_mode),'sha256':hashlib.sha256(payload).hexdigest()}
 finally:
  if fd>=0: os.close(fd)
  os.close(parent)
def read_regular(path): return read_regular_state(path)[0]
def persist_new(path,value):
 parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); fd=-1
 try:
  fd=os.open(path.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent); payload=(json.dumps(value,sort_keys=True)+'\n').encode(); view=memoryview(payload)
  while view:
   count=os.write(fd,view); assert count>0; view=view[count:]
  os.fsync(fd); opened=os.fstat(fd); os.fsync(parent); named=os.stat(path.name,dir_fd=parent,follow_symlinks=False); assert stat.S_ISREG(opened.st_mode) and stat.S_IMODE(opened.st_mode)==0o600 and (opened.st_dev,opened.st_ino)==(named.st_dev,named.st_ino); return {'dev':opened.st_dev,'ino':opened.st_ino,'sha256':hashlib.sha256(payload).hexdigest()}
 finally:
  if fd>=0: os.close(fd)
  os.close(parent)
def publish_terminal_decision(path,value,nonce):
 parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); fd=-1; committed=False; payload=(json.dumps(value,sort_keys=True)+'\n').encode(); temp='.step385-decision-'+nonce+'-'+hashlib.sha256(payload).hexdigest()
 try:
  fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent); view=memoryview(payload)
  while view:
   count=os.write(fd,view); assert count>0; view=view[count:]
  os.fsync(fd); opened=os.fstat(fd); assert stat.S_ISREG(opened.st_mode) and stat.S_IMODE(opened.st_mode)==0o600; os.fsync(parent); state={'dev':opened.st_dev,'ino':opened.st_ino,'mode':0o600,'sha256':hashlib.sha256(payload).hexdigest()}; os.close(fd); fd=-1
  try: os.link(temp,path.name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
  except FileExistsError:
   try: os.unlink(temp,dir_fd=parent); os.fsync(parent)
   except OSError: pass
   raise
  committed=True
  try: os.fsync(parent); os.unlink(temp,dir_fd=parent); os.fsync(parent)
  except OSError: pass
  return state
 finally:
  try:
   if fd>=0: os.close(fd)
   os.close(parent)
  except OSError:
   if not committed: raise
def verify_commit_chain(data):
 lock_payload,lock_state=read_regular_state(root/'ownership_host_lock.json'); precommit_payload,precommit_state=read_regular_state(root/'ownership_precommit_seal.json'); decision_payload,decision_state=read_regular_state(root/'ownership_start_decision.json'); lock=json.loads(lock_payload); precommit=json.loads(precommit_payload); decision=json.loads(decision_payload)
 assert set(decision)=={'schema','status','token','wrapper_nonce','host_lock_sha256','host_lock_dev','host_lock_ino'} and decision['schema']=='step385-start-decision-v1' and decision['status']=='committed' and decision['token']==expected_token and decision['wrapper_nonce']==data['nonce'] and all(type(decision[k]) is int and decision[k]>0 for k in ('host_lock_dev','host_lock_ino')) and type(decision['host_lock_sha256']) is str
 assert (lock_state['dev'],lock_state['ino'],lock_state['sha256'])==(decision['host_lock_dev'],decision['host_lock_ino'],decision['host_lock_sha256'])
 assert set(precommit)=={'schema','token','wrapper_nonce','lock','start_decision_sha256'} and precommit['schema']=='step385-precommit-seal-v1' and precommit['token']==expected_token and precommit['wrapper_nonce']==data['nonce'] and set(precommit['lock'])=={'dev','ino','mode','sha256'} and precommit['lock']==lock_state and precommit['start_decision_sha256']==hashlib.sha256(decision_payload).hexdigest() and lock_state['mode']==precommit_state['mode']==decision_state['mode']==0o600
 commit_state={'precommit':precommit_state,'start_decision':decision_state}
 if expected_commit_seal is not None: assert expected_commit_seal==commit_state
 return lock,precommit,decision,commit_state
def approved():
 data=json.loads(read_regular(manifest)); assert set(data)=={'schema','token','nonce','container_id','container_hostname','wrapper'} and data['schema']=='step385-owned-wrapper-v2' and data['token']==expected_token and type(data['nonce']) is str and data['nonce']!=expected_token and all(len(value)==64 and all(c in '0123456789abcdef' for c in value) for value in (expected_token,data['nonce'])) and data['container_id']==expected_container and data['container_hostname']==expected_hostname; w=data['wrapper']; required={'container_pid','container_nspid','container_starttime','container_ppid','container_pgid','container_sid','argv','cgroup','cgroup_sha256','namespaces'}; assert type(w) is dict and set(w)==required and all(type(w[k]) is int and w[k]>0 for k in ('container_pid','container_starttime','container_ppid','container_pgid','container_sid')) and type(w['container_nspid']) is list and w['container_nspid'] and all(type(x) is int and x>0 for x in w['container_nspid']) and w['container_nspid'][-1]==w['container_pid'] and type(w['argv']) is list and w['argv'] and all(type(x) is str for x in w['argv']) and any(expected_token in x for x in w['argv']) and type(w['cgroup']) is str and type(w['cgroup_sha256']) is str and hashlib.sha256(w['cgroup'].encode()).hexdigest()==w['cgroup_sha256'] and type(w['namespaces']) is dict and set(w['namespaces'])=={'pid','mnt','cgroup'} and all(type(x) is int and x>0 for x in w['namespaces'].values()); assert host_contract['container_id']==expected_container and host_contract['hostname']==expected_hostname; host_cgroup=host_contract['cgroup']; host_ns=host_contract['namespaces']; procs=allproc()
 member=lambda row: row[3]==host_cgroup and row[4]==host_ns
 matches=[pid for pid,row in procs.items() if member(row) and row[0].nspid[-1]==w['container_pid'] and row[0].starttime==w['container_starttime'] and [os.fsdecode(x) for x in row[0].argv]==w['argv']]
 assert len(matches)<=1; wrapper=matches[0] if matches else None; lock_path=root/'ownership_host_lock.json'; new_lock=False; commit_state=None
 if lock_path.exists() and action not in ('commit','recover_commit'): assert expected_commit_seal is not None, 'sealed recovery token missing'
 if wrapper is not None:
  item,ppid,host_sid,cg,nss=procs[wrapper]; assert host_sid==wrapper==item.host_pid==item.pgid, 'wrapper is not the mapped host session leader'; lock={'schema':'step385-host-ownership-v2','token':expected_token,'wrapper_nonce':data['nonce'],'host_sid':host_sid,'wrapper':{'host_pid':item.host_pid,'starttime':item.starttime,'nspid':list(item.nspid),'pgid':item.pgid,'argv':[os.fsdecode(x) for x in item.argv]},'host_cgroup':host_cgroup,'host_cgroup_sha256':hashlib.sha256(host_cgroup.encode()).hexdigest(),'host_namespaces':host_ns}
  if not lock_path.exists():
   assert action=='commit', 'host lock may only be created by commit'; existing={pid for pid,row in procs.items() if member(row) and row[2]==host_sid}; assert existing=={wrapper}, 'pre-ACK SID domain is not exclusive'; new_lock=True
  else:
   sealed_lock,_precommit,_ack,commit_state=verify_commit_chain(data); assert sealed_lock==lock
 else:
  assert lock_path.exists(); lock,_precommit,_ack,commit_state=verify_commit_chain(data); assert set(lock)=={'schema','token','wrapper_nonce','host_sid','wrapper','host_cgroup','host_cgroup_sha256','host_namespaces'} and lock['schema']=='step385-host-ownership-v2' and lock['token']==expected_token and lock['wrapper_nonce']==data['nonce'] and lock['host_cgroup']==host_cgroup and lock['host_cgroup_sha256']==hashlib.sha256(host_cgroup.encode()).hexdigest() and lock['host_namespaces']==host_ns
 assert type(lock) is dict and set(lock)=={'schema','token','wrapper_nonce','host_sid','wrapper','host_cgroup','host_cgroup_sha256','host_namespaces'} and type(lock['host_sid']) is int and lock['host_sid']>1 and type(lock['host_cgroup']) is str and type(lock['host_namespaces']) is dict and set(lock['host_namespaces'])=={'pid','mnt','cgroup'} and all(type(x) is int and x>0 for x in lock['host_namespaces'].values()); locked_wrapper=lock['wrapper']; assert type(locked_wrapper) is dict and set(locked_wrapper)=={'host_pid','starttime','nspid','pgid','argv'} and all(type(locked_wrapper[k]) is int and locked_wrapper[k]>1 for k in ('host_pid','starttime','pgid')) and type(locked_wrapper['nspid']) is list and locked_wrapper['nspid'] and locked_wrapper['nspid'][0]==locked_wrapper['host_pid'] and all(type(x) is int and x>0 for x in locked_wrapper['nspid']) and type(locked_wrapper['argv']) is list and locked_wrapper['argv'] and all(type(x) is str for x in locked_wrapper['argv'])
 host_sid=lock['host_sid']; drift=[pid for pid,row in procs.items() if member(row) and row[2]!=host_sid and any(expected_token in os.fsdecode(arg) for arg in row[0].argv)]; assert not drift, 'owned token setsid drift'; roots={wrapper} if wrapper is not None else {pid for pid,row in procs.items() if member(row) and row[2]==host_sid and any(expected_token in os.fsdecode(arg) for arg in row[0].argv)}; chosen=set(roots); changed=True
 while changed:
  changed=False
  for pid,row in procs.items():
   if pid not in chosen and row[1] in chosen and member(row):
    if row[2]!=host_sid: raise RuntimeError('owned descendant setsid drift')
    chosen.add(pid); changed=True
 residual={pid for pid,row in procs.items() if member(row) and row[2]==host_sid}; chosen|=residual
 again=allproc()
 for pid in chosen: assert pid in again and again[pid]==procs[pid]
 if new_lock:
  final=allproc(); assert wrapper in final and final[wrapper]==procs[wrapper] and {pid for pid,row in final.items() if member(row) and row[2]==host_sid}=={wrapper}; lock_seal=persist_new(lock_path,lock); decision={'schema':'step385-start-decision-v1','status':'committed','token':expected_token,'wrapper_nonce':data['nonce'],'host_lock_sha256':lock_seal['sha256'],'host_lock_dev':lock_seal['dev'],'host_lock_ino':lock_seal['ino']}; decision_payload=(json.dumps(decision,sort_keys=True)+'\n').encode(); precommit={'schema':'step385-precommit-seal-v1','token':expected_token,'wrapper_nonce':data['nonce'],'lock':{**lock_seal,'mode':0o600},'start_decision_sha256':hashlib.sha256(decision_payload).hexdigest()}; precommit_state=persist_new(root/'ownership_precommit_seal.json',precommit); final2=allproc(); assert wrapper in final2 and final2[wrapper]==procs[wrapper] and {pid for pid,row in final2.items() if member(row) and row[2]==host_sid}=={wrapper}; decision_state=publish_terminal_decision(root/'ownership_start_decision.json',decision,data['nonce']); commit_state={'precommit':{**precommit_state,'mode':0o600},'start_decision':decision_state}
 return data,procs,wrapper,host_sid,tuple(sorted(chosen-({wrapper} if wrapper is not None else set()))),commit_state
def sample():
 procs=allproc(); clients=[pid for pid,(i,_,_,_,_) in procs.items() if [os.fsdecode(x) for x in i.argv]==expected_client]
 if not manifest.exists(): return {'host_client':clients,'container_wrapper':[],'descendant_opc':[]}
 decision_path=root/'ownership_start_decision.json'
 if decision_path.exists():
  terminal=json.loads(read_regular(decision_path))
  if terminal.get('status')=='aborted':
   assert set(terminal)=={'schema','status','token','wrapper_nonce'} and terminal['schema']=='step385-start-decision-v1' and terminal['token']==expected_token; data=json.loads(read_regular(manifest)); assert set(data)=={'schema','token','nonce','container_id','container_hostname','wrapper'} and data['schema']=='step385-owned-wrapper-v2' and data['token']==expected_token and data['nonce']==terminal['wrapper_nonce']; w=data['wrapper']; assert type(w) is dict and set(w)=={'container_pid','container_nspid','container_starttime','container_ppid','container_pgid','container_sid','argv','cgroup','cgroup_sha256','namespaces'}; member=lambda row: row[3]==host_contract['cgroup'] and row[4]==host_contract['namespaces']; matches=[pid for pid,row in procs.items() if member(row) and row[0].nspid[-1]==w['container_pid'] and row[0].starttime==w['container_starttime'] and [os.fsdecode(x) for x in row[0].argv]==w['argv']]; assert len(matches)<=1
   if matches:
    wrapper=matches[0]; row=procs[wrapper]; assert row[2]==wrapper==row[0].host_pid==row[0].pgid and {pid for pid,item in procs.items() if member(item) and item[2]==wrapper}=={wrapper}; return {'host_client':clients,'container_wrapper':[wrapper],'descendant_opc':[]}
   return {'host_client':clients,'container_wrapper':[],'descendant_opc':[]}
 data,procs,wrapper,_host_sid,kids,_commit_state=approved(); return {'host_client':clients,'container_wrapper':[] if wrapper is None else [wrapper],'descendant_opc':list(kids)}
if action=='recover_commit':
 deadline=time.monotonic()+5
 while not manifest.exists() and time.monotonic()<deadline: time.sleep(.05)
 assert manifest.exists(); recovery_data=json.loads(read_regular(manifest)); assert recovery_data['schema']=='step385-owned-wrapper-v2' and recovery_data['token']==expected_token and type(recovery_data['nonce']) is str
 decision_path=root/'ownership_start_decision.json'
 if not decision_path.exists():
  aborted={'schema':'step385-start-decision-v1','status':'aborted','token':expected_token,'wrapper_nonce':recovery_data['nonce']}
  try: publish_terminal_decision(decision_path,aborted,recovery_data['nonce'])
  except FileExistsError: pass
 decision=json.loads(read_regular(decision_path)); assert decision.get('schema')=='step385-start-decision-v1' and decision.get('status') in ('committed','aborted') and decision.get('token')==expected_token and decision.get('wrapper_nonce')==recovery_data['nonce']
 if decision['status']=='aborted': assert set(decision)=={'schema','status','token','wrapper_nonce'}; print(json.dumps({'schema':'step385-ownership-recovery-v2','status':'aborted'},sort_keys=True)); raise SystemExit(0)
 _data,_procs,_wrapper,_sid,_kids,commit_state=approved(); assert commit_state is not None; print(json.dumps({'schema':'step385-ownership-recovery-v2','status':'committed','commit_state':commit_state},sort_keys=True))
elif action=='commit':
 deadline=time.monotonic()+120
 while not manifest.exists() and time.monotonic()<deadline: time.sleep(.05)
 assert manifest.exists(); _data,_procs,_wrapper,_sid,_kids,commit_state=approved(); assert commit_state is not None; print(json.dumps({'schema':'step385-ownership-commit-v3','committed':True,'commit_state':commit_state},sort_keys=True))
elif action=='snapshot':
 for _attempt in range(4):
  a=sample(); b=sample()
  if a==b: print(json.dumps(a,sort_keys=True)); raise SystemExit(0)
 raise RuntimeError('ownership snapshot did not stabilize')
elif action=='cleanup':
 deadline=time.monotonic()+30; rounds=0; zero={'host_client':[],'container_wrapper':[],'descendant_opc':[]}
 while rounds<16 and time.monotonic()<deadline:
  rounds+=1; current=sample()
  if current==zero:
   time.sleep(.1)
   if sample()==zero: print(json.dumps({'schema':'step385-owned-clean-v1','remaining':0},sort_keys=True)); raise SystemExit(0)
  procs=allproc(); identities=[]
  for pid in current['host_client']+current['container_wrapper']+current['descendant_opc']:
   if pid in procs: identities.append(procs[pid][0])
  remaining=deadline-time.monotonic(); assert remaining>0; grace=min(1.0,remaining/2); g.terminate_owned(tuple(identities),g.owned_identity_alive,grace_seconds=grace); assert time.monotonic()<=deadline
 raise RuntimeError('owned cleanup deadline/max-rounds exceeded')
else: raise AssertionError('bad action')'''


def _summary_code() -> str:
    return r'''import hashlib,json,os,stat,sys,types
from pathlib import Path
root=Path(sys.argv[1]); adapter_path=root/'staging'/sys.argv[2]; work=root/'work'
parent=os.open(adapter_path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); fd=-1
try:
 before=os.stat(adapter_path.name,dir_fd=parent,follow_symlinks=False); fd=os.open(adapter_path.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent); opened=os.fstat(fd); payload=b''
 while chunk:=os.read(fd,1048576): payload+=chunk
 closed=os.fstat(fd); after=os.stat(adapter_path.name,dir_fd=parent,follow_symlinks=False); ident=lambda s:(s.st_dev,s.st_ino,stat.S_IFMT(s.st_mode),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 assert stat.S_ISREG(opened.st_mode) and ident(before)==ident(opened)==ident(closed)==ident(after) and hashlib.sha256(payload).hexdigest()==sys.argv[4]
finally:
 if fd>=0: os.close(fd)
 os.close(parent)
def load_locked(name,path,expected):
 parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); f=-1
 try:
  before=os.stat(path.name,dir_fd=parent,follow_symlinks=False); f=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent); opened=os.fstat(f); chunks=[]
  while chunk:=os.read(f,1048576): chunks.append(chunk)
  closed=os.fstat(f); after=os.stat(path.name,dir_fd=parent,follow_symlinks=False); identity=lambda s:(s.st_dev,s.st_ino,stat.S_IFMT(s.st_mode),s.st_size,s.st_mtime_ns,s.st_ctime_ns); data=b''.join(chunks); assert stat.S_ISREG(opened.st_mode) and identity(before)==identity(opened)==identity(closed)==identity(after) and hashlib.sha256(data).hexdigest()==expected and name not in sys.modules
 finally:
  if f>=0: os.close(f)
  os.close(parent)
 module=types.ModuleType(name); module.__file__=str(path); sys.modules[name]=module; exec(compile(data,str(path),'exec'),module.__dict__); return module
load_locked('step338_patch_qr_v2_lifetime',root/'staging'/'step338_patch_qr_v2_lifetime.py',sys.argv[9]); load_locked('step375_patch_qr_v2_vtv_direct_qa_legacy_probe_v6',root/'staging'/'step375_patch_qr_v2_vtv_direct_qa_legacy_probe_v6.py',sys.argv[10])
m=types.ModuleType('_step385_remote_adapter'); m.__file__=str(adapter_path); exec(compile(payload,str(adapter_path),'exec'),m.__dict__)
manifest=json.loads((work/m.MANIFEST_NAME).read_text()); base=m._load_base()
m._validate_manifest(base,work,manifest,expected_status=m.DIAGNOSTIC_BUILT_STATUS)
m._require_completed_attempt(base,work,manifest)
m._validate_built_artifact_closure(base,work,manifest,enrich=False)
m._assert_no_release_outputs(work,manifest)
assert set(manifest['artifacts'])=={'ascend910_93','ascend910b'}
assert manifest['status']=='diagnostic_built_unvalidated'
def regular_state(path):
 parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); f=-1
 try:
  before=os.stat(path.name,dir_fd=parent,follow_symlinks=False); f=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent); opened=os.fstat(f); chunks=[]
  while chunk:=os.read(f,1048576): chunks.append(chunk)
  closed=os.fstat(f); after=os.stat(path.name,dir_fd=parent,follow_symlinks=False); identity=lambda s:(s.st_dev,s.st_ino,stat.S_IFMT(s.st_mode),s.st_size,s.st_mtime_ns,s.st_ctime_ns); assert stat.S_ISREG(opened.st_mode) and identity(before)==identity(opened)==identity(closed)==identity(after); data=b''.join(chunks); return data,{'path':str(path),'dev':opened.st_dev,'ino':opened.st_ino,'mode':stat.S_IMODE(opened.st_mode),'size':opened.st_size,'sha256':hashlib.sha256(data).hexdigest()}
 finally:
  if f>=0: os.close(f)
  os.close(parent)
expected_receipt=json.loads(sys.argv[5]); receipt_payload,receipt_state=regular_state(root/'build_receipt.json'); assert {k:receipt_state[k] for k in ('dev','ino','mode','size','sha256')}==expected_receipt; receipt=json.loads(receipt_payload); root_state=root.lstat(); work_state=work.lstat(); assert receipt['schema']=='step387-build-receipt-v1' and receipt['container']=={'id':sys.argv[6],'init_starttime':int(sys.argv[7]),'mnt_ns':int(sys.argv[8])} and os.stat('/proc/self/ns/mnt').st_ino==int(sys.argv[8]) and receipt['attempt']=={'path':str(root),'dev':root_state.st_dev,'ino':root_state.st_ino} and receipt['work']=={'path':str(work),'dev':work_state.st_dev,'ino':work_state.st_ino}; assert regular_state(Path(receipt['manifest']['path']))[1]==receipt['manifest'] and regular_state(Path(receipt['completion']['path']))[1]==receipt['completion']
for soc,row in receipt['artifacts'].items():
 for bound in row.values(): assert regular_state(Path(bound['path']))[1]==bound
allowed=Path(manifest['immutable_guards']['extracted_original_wheel']['path']).resolve(strict=True); forbidden=json.loads(sys.argv[3])
for p in work.rglob('*'):
 assert p.name not in forbidden['names']; assert p.suffix.lower() not in forbidden['suffixes'] or p.resolve(strict=True)==allowed
flags={k:manifest['policy'][k] for k in ('artifact_class','diagnostic_only','release_candidate','package_forbidden')}; c=manifest['candidate']
artifacts={}
for soc,a in manifest['artifacts'].items():
 artifacts[soc]={k:a[k] for k in ('object_path','object_size','object_sha256','json_path','json_size','json_sha256','opc_log_path','opc_log_size','opc_log_sha256','kernel_name','bin_file_name','concrete_entries')}
print(json.dumps({'schema':'step385-summary-v2','status':manifest['status'],'seal_valid':True,'policy':flags,'candidate':{k:c[k] for k in ('identity','source_sha256','reverse_v4_sha256','artifact_class','diagnostic_only','release_candidate','package_forbidden')},'package':manifest['package'],'tools':manifest['tools'],'artifacts':artifacts,'installed_inventory_closed':manifest['build_runtime']['installed_inventory_closed'],'runtime_inventory_closed':manifest['build_runtime']['runtime_inventory_closed'],'alias_bytes_equal':True,'forbidden_outputs_absent':True},sort_keys=True))'''


def _json_run(contract_module: Any, target: Any, command: str) -> dict[str, Any]:
    output, _ = contract_module.run(target, command, timeout=REMOTE_SHORT_TIMEOUT)
    value = json.loads(output)
    if not isinstance(value, dict):
        raise RuntimeError("STEP385 remote JSON result is not an object")
    return value


def _summary_run(contract_module: Any, target: Any, command: str, expected_gate: dict[str, Any], expected_adapter_sha256: str) -> dict[str, Any]:
    output, _ = contract_module.run(target, command, timeout=REMOTE_SHORT_TIMEOUT)
    if type(output) is not str:
        raise RuntimeError("STEP391 summary stdout is not text")
    lines = output.splitlines()
    if len(lines) != 2 or any(not line.strip() for line in lines):
        raise RuntimeError("STEP391 summary stdout must contain exactly two non-empty JSON lines")
    try:
        gate_value, summary_value = (json.loads(line) for line in lines)
    except json.JSONDecodeError as error:
        raise RuntimeError("STEP391 summary stdout contains invalid JSON") from error
    if gate_value != expected_gate:
        raise RuntimeError("STEP391 final staging gate mismatch")
    if type(summary_value) is not dict:
        raise RuntimeError("STEP391 summary JSON is not an object")
    return _validate_summary(summary_value, expected_adapter_sha256)


def _owned_run(contract_module: Any, target: Any, command: str) -> dict[str, list[int]]:
    output, _ = contract_module.run(target, command, timeout=REMOTE_SHORT_TIMEOUT)
    value = json.loads(output)
    keys = {"host_client", "container_wrapper", "descendant_opc"}
    if not isinstance(value, dict) or set(value) != keys or any(not isinstance(value[key], list) or any(type(pid) is not int or pid <= 1 for pid in value[key]) for key in keys):
        raise RuntimeError("STEP385 owned identity result mismatch")
    return value


def _owned_cleanup_run(contract_module: Any, target: Any, command: str) -> dict[str, Any]:
    output, _ = contract_module.run(target, command, timeout=REMOTE_SHORT_TIMEOUT)
    value = json.loads(output)
    if value != {"schema":"step385-owned-clean-v1","remaining":0}:
        raise RuntimeError("STEP385 owned cleanup result mismatch")
    return value


def _validate_commit_state(state: Any) -> bool:
    return type(state) is dict and set(state) == {"precommit", "start_decision"} and not any(type(state[name]) is not dict or set(state[name]) != {"dev", "ino", "mode", "sha256"} or any(type(state[name][key]) is not int or state[name][key] <= 0 for key in ("dev", "ino", "mode")) or not _lower_hex64(state[name]["sha256"]) for name in ("precommit", "start_decision"))


def _owned_commit_run(contract_module: Any, target: Any, command: str) -> dict[str, Any]:
    output, _ = contract_module.run(target, command, timeout=REMOTE_SHORT_TIMEOUT)
    value = json.loads(output)
    state = value.get("commit_state") if type(value) is dict else None
    if type(value) is not dict or set(value) != {"schema", "committed", "commit_state"} or value["schema"] != "step385-ownership-commit-v3" or value["committed"] is not True or not _validate_commit_state(state):
        raise RuntimeError("STEP385 ownership commit result mismatch")
    return value


def _owned_recover_commit_run(contract_module: Any, target: Any, command: str) -> dict[str, Any]:
    output, _ = contract_module.run(target, command, timeout=REMOTE_SHORT_TIMEOUT)
    value = json.loads(output)
    if value == {"schema": "step385-ownership-recovery-v2", "status": "aborted"}:
        return value
    if type(value) is not dict or set(value) != {"schema", "status", "commit_state"} or value["schema"] != "step385-ownership-recovery-v2" or value["status"] != "committed" or not _validate_commit_state(value["commit_state"]):
        raise RuntimeError("STEP385 ownership recovery result mismatch")
    return value


def _append_secondary(primary: BaseException, label: str, secondary: BaseException) -> None:
    note = f"{label}: {secondary!r}"
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(note)
    else:
        errors = list(getattr(primary, "cleanup_errors", ()))
        errors.append(note)
        primary.cleanup_errors = tuple(errors)  # type: ignore[attr-defined]


def _run_build_transaction(helper: Any, target: Any, command: str, snapshot_command: str, before: dict[str, Any], owned_snapshot: str, owned_cleanup: str) -> dict[str, Any]:
    primary: Optional[BaseException] = None
    receipt_state: Optional[dict[str, Any]] = None
    try:
        output, _ = helper.run(target, command, timeout=REMOTE_BUILD_TIMEOUT)
        if type(output) is not str or not output.strip():
            raise RuntimeError("STEP388 build wait produced empty stdout")
        result = json.loads(output)
        if type(result) is not dict or result.get("schema") != "step385-build-result-v3" or type(result.get("returncode")) is not int or type(result.get("logs")) is not dict or set(result["logs"]) != {"stdout", "stderr"}:
            raise RuntimeError("STEP387 build receipt result mismatch")
        for name in ("stdout", "stderr"):
            item = result["logs"][name]
            if type(item) is not dict or set(item) != {"path", "dev", "ino", "mode", "size", "sha256"} or type(item["path"]) is not str or any(type(item[k]) is not int or item[k] <= 0 for k in ("dev", "ino")) or item["mode"] != 0o600 or type(item["size"]) is not int or not 0 <= item["size"] <= 1048576 or not _lower_hex64(item["sha256"]):
                raise RuntimeError("STEP389 build diagnostic log reference mismatch")
        if result["returncode"] != 0:
            if set(result) != {"schema", "returncode", "logs"}:
                raise RuntimeError("STEP389 failed build result schema mismatch")
            raise RuntimeError(f"STEP389 build child failed rc={result['returncode']} logs=child_stdout.log,child_stderr.log")
        receipt_state = result.get("receipt")
        if set(result) != {"schema", "returncode", "logs", "receipt"} or type(receipt_state) is not dict or set(receipt_state) != {"dev", "ino", "mode", "size", "sha256"} or any(type(receipt_state[k]) is not int or receipt_state[k] <= 0 for k in ("dev", "ino", "mode", "size")) or not _lower_hex64(receipt_state["sha256"]):
            raise RuntimeError("STEP387 build receipt result mismatch")
    except BaseException as error:
        primary = error
    try:
        owned = _owned_run(helper, target, owned_snapshot)
        if any(owned.values()):
            _owned_cleanup_run(helper, target, owned_cleanup)
        zero = {"host_client": [], "container_wrapper": [], "descendant_opc": []}
        if _owned_run(helper, target, owned_snapshot) != zero or _owned_run(helper, target, owned_snapshot) != zero:
            raise RuntimeError("STEP385 controller ownership closure is not stably zero")
    except BaseException as cleanup_error:
        if primary is None: primary = cleanup_error
        else: _append_secondary(primary, "STEP385 owned cleanup failed", cleanup_error)
    try:
        postflight = snapshot_command if receipt_state is None else snapshot_command + " " + shlex.quote(json.dumps(receipt_state, sort_keys=True))
        after = _json_run(helper, target, postflight)
        if after != before:
            raise RuntimeError("STEP385 installed/process/OPC closure changed")
    except BaseException as postflight_error:
        if primary is None:
            primary = postflight_error
        else:
            _append_secondary(primary, "STEP385 postflight failed", postflight_error)
    if primary is not None:
        raise primary
    assert receipt_state is not None
    return receipt_state


def _recover_handshake_failure(helper: Any, target: Any, primary: BaseException, snapshot_command: str, before: dict[str, Any], owned_snapshot: str, owned_cleanup: str) -> None:
    try:
        owned = _owned_run(helper, target, owned_snapshot)
        if any(owned.values()):
            _owned_cleanup_run(helper, target, owned_cleanup)
    except BaseException as cleanup_error:
        _append_secondary(primary, "STEP385 handshake cleanup refused/failed", cleanup_error)
    try:
        after = _json_run(helper, target, snapshot_command)
        if after != before:
            raise RuntimeError("STEP385 handshake failure changed closure")
    except BaseException as postflight_error:
        _append_secondary(primary, "STEP385 handshake postflight failed", postflight_error)
    raise primary


def _dry_run_payload() -> dict[str, Any]:
    _require_remote_ready()
    payloads, identities, _ = _payload_snapshot()
    _revalidate_payload_identities(identities)
    return {"status": "dry_run", "attempt_name": ATTEMPT_NAME, "socs": list(SOCS), "input_sha256": dict(EXPECTED_INPUTS), "remote_adapter_sha256": sha256_bytes(payloads[ADAPTER.name]), "actions": list(DRY_RUN_ACTIONS), "forbidden": list(FORBIDDEN_ACTIONS)}


def execute() -> dict[str, Any]:
    _require_remote_ready()
    payloads, local_identities, contract_source = _payload_snapshot()
    helper = load_contract(contract_source)
    remote_module = helper.load_remote_module()
    info = helper.local_preflight(remote_module)
    jump, target = helper.connect_target(remote_module, info)
    primary: Optional[BaseException] = None
    try:
        hostname, _ = helper.run(target, "hostname", timeout=REMOTE_SHORT_TIMEOUT)
        if hostname.strip() != helper.EXPECTED_HOSTNAME:
            raise RuntimeError("STEP385 second-hop hostname mismatch")
        contract = _validate_contract(helper.container_probe(target))
        remote_attempt = helper.safe_remote_path(str(info["shared"]), ATTEMPT_NAME)
        helper.run_host_script(target, _exclusive_attempt_script(remote_attempt), timeout=REMOTE_SHORT_TIMEOUT)
        contract_payload = {k: contract[k] for k in ("schema_version", "container_name", "inspect_container_id", "inspect_hostname", "opc", "cann_version_files")}
        payloads["container_contract.json"] = (json.dumps(contract_payload, sort_keys=True, indent=2) + "\n").encode()
        sftp = target.open_sftp()
        try:
            for name, data in payloads.items():
                helper.write_remote_new(sftp, remote_attempt + "/staging/" + name, data)
        except BaseException as upload_error:
            try: sftp.close()
            except BaseException as close_error: _append_secondary(upload_error, "STEP385 SFTP close failed", close_error)
            raise
        else:
            sftp.close()
        _revalidate_payload_identities(local_identities)
        expected = {name: sha256_bytes(data) for name, data in payloads.items()}
        staging = remote_attempt + "/staging"
        gate_base = "env PYTHONDONTWRITEBYTECODE=1 python3 -c " + shlex.quote(_upload_gate_code()) + " " + shlex.quote(staging) + " " + shlex.quote(json.dumps(expected, sort_keys=True))
        gate = _json_run(helper, target, gate_base)
        if set(gate) != {"schema", "root"} or gate["schema"] != "step385-staging-gate-v1" or set(gate["root"]) != {"dev", "ino", "mode"}:
            raise RuntimeError("STEP385 upload/readback gate failed")
        locked_root = json.dumps(gate["root"], sort_keys=True)
        identity_command = "env PYTHONDONTWRITEBYTECODE=1 python3 -c " + shlex.quote(_container_identity_code()) + " " + shlex.quote(helper.CONTAINER) + " " + shlex.quote(contract["inspect_container_id"]) + " " + shlex.quote(contract["inspect_hostname"]) + " " + shlex.quote(staging + '/' + PROCESS_GUARD.name)
        host_container = _validate_host_container_identity(_json_run(helper, target, identity_command), contract)
        snapshot_command = "docker exec " + shlex.quote(helper.CONTAINER) + " env PYTHONDONTWRITEBYTECODE=1 python3 -c " + shlex.quote(_snapshot_code()) + " " + shlex.quote(contract["installed_cloud_root"]) + " " + shlex.quote(remote_attempt) + " " + shlex.quote(_validated_opc(contract)) + " " + shlex.quote(staging + '/' + PROCESS_GUARD.name)
        before = _json_run(helper, target, snapshot_command)
        _validate_snapshot(before)
        if before["opc_samples"][0]["path"] != contract["opc"]["path"] or before["opc_samples"][0]["realpath"] != contract["opc"]["path"] or before["opc_samples"][0]["sha256"] != contract["opc"]["sha256"]:
            raise RuntimeError("STEP385 OPC snapshot differs from contract")
        phase_gate = gate_base + " " + shlex.quote(locked_root)
        adapter_sha256 = sha256_bytes(payloads[ADAPTER.name])
        receipt_command = "env PYTHONDONTWRITEBYTECODE=1 python3 -c " + shlex.quote(_receipt_code()) + " " + shlex.quote(remote_attempt) + " " + shlex.quote(ADAPTER.name) + " " + shlex.quote(adapter_sha256) + " " + shlex.quote(contract['inspect_container_id']) + " " + shlex.quote(str(host_container['init']['starttime'])) + " " + shlex.quote(str(host_container['namespaces']['mnt'])) + " " + shlex.quote(EXPECTED_INPUTS[V4_PATCHER.name]) + " " + shlex.quote(EXPECTED_INPUTS[STEP375_PATCHER.name])
        phase_script = phase_gate + "\n" + _container_script(helper, contract, remote_attempt, "prepare") + "\n" + phase_gate + "\n" + _container_script(helper, contract, remote_attempt, "build") + "\n" + phase_gate + "\n" + receipt_command
        attempt_token = sha256_bytes(os.urandom(32))
        wrapper_call = "python3 " + shlex.quote(staging + '/' + WRAPPER_NAME) + " " + shlex.quote(remote_attempt + '/ownership.json') + " " + shlex.quote(attempt_token) + " " + shlex.quote(contract['inspect_container_id']) + " " + shlex.quote(contract['inspect_hostname']) + " " + shlex.quote(phase_script)
        container_body = phase_gate + " && exec setsid " + wrapper_call
        detached = "docker exec -d " + shlex.quote(helper.CONTAINER) + " env PYTHONDONTWRITEBYTECODE=1 bash --noprofile --norc -lc " + shlex.quote(container_body)
        wait_result = "env PYTHONDONTWRITEBYTECODE=1 python3 -c " + shlex.quote(_result_wait_code()) + " " + shlex.quote(remote_attempt + '/build_result.json') + " 650 " + shlex.quote(remote_attempt) + " " + shlex.quote(helper.CONTAINER) + " " + shlex.quote(contract['inspect_container_id']) + " " + shlex.quote(str(host_container['init']['starttime'])) + " " + shlex.quote(str(host_container['namespaces']['mnt']))
        build_command = "setsid --wait bash --noprofile --norc -lc " + shlex.quote(wait_result)
        expected_client_argv = shlex.split(build_command)
        owned_base = "env PYTHONDONTWRITEBYTECODE=1 python3 -c " + shlex.quote(_owned_code()) + " " + shlex.quote(remote_attempt) + " " + shlex.quote(json.dumps(expected_client_argv)) + " " + shlex.quote(attempt_token) + " " + shlex.quote(contract['inspect_container_id']) + " " + shlex.quote(contract['inspect_hostname']) + " " + shlex.quote(json.dumps(host_container,sort_keys=True))
        owned_snapshot = owned_base + " snapshot"
        owned_cleanup = owned_base + " cleanup"
        if any(_owned_run(helper, target, owned_snapshot).values()): raise RuntimeError("STEP385 owned process set not initially empty")
        try:
            helper.run(target, detached, timeout=REMOTE_SHORT_TIMEOUT)
            commit_result = _owned_commit_run(helper, target, owned_base + " commit")
        except BaseException as handshake_error:
            try:
                recovered = _owned_recover_commit_run(helper, target, owned_base + " recover_commit")
                if recovered["status"] == "committed":
                    recovered_suffix = " " + shlex.quote(json.dumps(recovered["commit_state"], sort_keys=True))
                    owned_snapshot = owned_base + " snapshot" + recovered_suffix
                    owned_cleanup = owned_base + " cleanup" + recovered_suffix
            except BaseException as recovery_error:
                _append_secondary(handshake_error, "STEP385 read-only commit recovery failed", recovery_error)
            _recover_handshake_failure(helper, target, handshake_error, snapshot_command, before, owned_snapshot, owned_cleanup)
        sealed_suffix = " " + shlex.quote(json.dumps(commit_result["commit_state"], sort_keys=True))
        owned_snapshot = owned_base + " snapshot" + sealed_suffix
        owned_cleanup = owned_base + " cleanup" + sealed_suffix
        receipt_state = _run_build_transaction(helper, target, build_command, snapshot_command, before, owned_snapshot, owned_cleanup)
        summary_body = phase_gate + " && env PYTHONDONTWRITEBYTECODE=1 python3 -c " + shlex.quote(_summary_code()) + " " + shlex.quote(remote_attempt) + " " + shlex.quote(ADAPTER.name) + " " + shlex.quote(json.dumps(FORBIDDEN_OUTPUTS, sort_keys=True)) + " " + shlex.quote(adapter_sha256) + " " + shlex.quote(json.dumps(receipt_state, sort_keys=True)) + " " + shlex.quote(contract['inspect_container_id']) + " " + shlex.quote(str(host_container['init']['starttime'])) + " " + shlex.quote(str(host_container['namespaces']['mnt'])) + " " + shlex.quote(EXPECTED_INPUTS[V4_PATCHER.name]) + " " + shlex.quote(EXPECTED_INPUTS[STEP375_PATCHER.name])
        summary_command = "docker exec " + shlex.quote(helper.CONTAINER) + " bash --noprofile --norc -lc " + shlex.quote(summary_body)
        summary = _summary_run(helper, target, summary_command, gate, adapter_sha256)
        return {**summary, "attempt_name": ATTEMPT_NAME, "uploaded_readback": True}
    except BaseException as error:
        primary = error
        raise
    finally:
        for resource in (target, jump):
            try:
                resource.close()
            except BaseException as close_error:
                if primary is None: primary = close_error
                else: _append_secondary(primary, "STEP385 close failed", close_error)
        if primary is not None:
            raise primary


def _lower_hex64(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema", "installed_samples", "opc_samples", "npu_samples"} or value.get("schema") != "step385-closure-v2":
        raise RuntimeError("STEP385 snapshot schema mismatch")
    installed=value["installed_samples"]
    if not isinstance(installed,list) or len(installed)!=2 or installed[0]!=installed[1]:
        raise RuntimeError("STEP385 installed inventory mismatch")
    for sample in installed:
        if not isinstance(sample,dict) or set(sample)!={"root","entries"} or set(sample["root"])!={"dev","ino","mode"} or any(type(sample["root"][k]) is not int for k in sample["root"]): raise RuntimeError("STEP385 installed root mismatch")
        if not isinstance(sample["entries"],dict): raise RuntimeError("STEP385 installed entries mismatch")
        for name,row in sample["entries"].items():
            if type(name) is not str or not isinstance(row,dict) or set(row)!={"type","mode","size","sha256","target"} or row["type"] not in {"DIR","REG","SYMLINK"} or type(row["mode"]) is not int or type(row["size"]) is not int: raise RuntimeError("STEP385 installed entry mismatch")
            if row["type"]=="REG" and (not _lower_hex64(row["sha256"]) or row["target"] is not None): raise RuntimeError("STEP385 installed regular mismatch")
            if row["type"]=="SYMLINK" and (type(row["target"]) is not str or row["sha256"] is not None): raise RuntimeError("STEP385 installed symlink mismatch")
            if row["type"]=="DIR" and (row["sha256"] is not None or row["target"] is not None): raise RuntimeError("STEP385 installed directory mismatch")
    if not isinstance(value["opc_samples"], list) or len(value["opc_samples"]) != 2 or value["opc_samples"][0] != value["opc_samples"][1] or not isinstance(value["opc_samples"][0],dict):
        raise RuntimeError("STEP385 OPC samples mismatch")
    opc=value["opc_samples"][0]
    if set(opc)!={"path","realpath","dev","ino","mode","size","sha256"} or any(type(opc[k]) is not int for k in ("dev","ino","mode","size")) or any(type(opc[k]) is not str or not PurePosixPath(opc[k]).is_absolute() for k in ("path","realpath")) or not _lower_hex64(opc["sha256"]): raise RuntimeError("STEP385 OPC samples mismatch")
    if not isinstance(value["npu_samples"], list) or len(value["npu_samples"]) != 2 or value["npu_samples"][0] != value["npu_samples"][1]:
        raise RuntimeError("STEP385 NPU samples mismatch")
    npu=value["npu_samples"][0]
    if not isinstance(npu,dict) or set(npu)!={"rows","device_ids","host_pids"} or npu!={"rows":[],"device_ids":[],"host_pids":[]}: raise RuntimeError("STEP385 NPU idle mismatch")
    return value


def _validate_summary(value: Any, expected_adapter_sha256: str) -> dict[str, Any]:
    top={"schema","status","seal_valid","policy","candidate","package","tools","artifacts","installed_inventory_closed","runtime_inventory_closed","alias_bytes_equal","forbidden_outputs_absent"}
    if type(value) is not dict or set(value)!=top or value["schema"]!="step385-summary-v2" or value["status"]!="diagnostic_built_unvalidated" or any(value[k] is not True for k in ("seal_valid","installed_inventory_closed","runtime_inventory_closed","alias_bytes_equal","forbidden_outputs_absent")):
        raise RuntimeError("STEP385 completion summary mismatch")
    flags={"artifact_class":"diagnostic_probe","diagnostic_only":True,"release_candidate":False,"package_forbidden":True}
    if value["policy"]!=flags or value["package"]!={"status":"forbidden_diagnostic_probe"}: raise RuntimeError("STEP385 summary policy mismatch")
    if value["candidate"]!={"identity":CANDIDATE_IDENTITY,"source_sha256":CANDIDATE_SHA256,"reverse_v4_sha256":REVERSE_V4_SHA256,**flags}: raise RuntimeError("STEP385 summary candidate mismatch")
    tools={"diagnostic_adapter_sha256":expected_adapter_sha256,"audited_adapter_sha256":EXPECTED_INPUTS[AUDITED_ADAPTER.name],"base_builder_sha256":EXPECTED_INPUTS[BASE_BUILDER.name],"step384_patcher_sha256":EXPECTED_INPUTS[PATCHER.name],"v4_patcher_sha256":EXPECTED_INPUTS[V4_PATCHER.name]}
    if value["tools"]!=tools: raise RuntimeError("STEP385 summary tools mismatch")
    if not isinstance(value["artifacts"],dict) or set(value["artifacts"])!=set(SOCS): raise RuntimeError("STEP385 summary SoC mismatch")
    keys={"object_path","object_size","object_sha256","json_path","json_size","json_sha256","opc_log_path","opc_log_size","opc_log_sha256","kernel_name","bin_file_name","concrete_entries"}; entries=sorted((CANDIDATE_IDENTITY+"_0_mix_aic",CANDIDATE_IDENTITY+"_0_mix_aiv"))
    for soc,row in value["artifacts"].items():
        if not isinstance(row,dict) or set(row)!=keys or any(type(row[k]) is not str or not PurePosixPath(row[k]).is_absolute() for k in ("object_path","json_path","opc_log_path")) or any(type(row[k]) is not int or row[k]<=0 for k in ("object_size","json_size","opc_log_size")) or any(not _lower_hex64(row[k]) for k in ("object_sha256","json_sha256","opc_log_sha256")) or row["kernel_name"]!=CANDIDATE_IDENTITY or row["bin_file_name"]!=CANDIDATE_IDENTITY or row["concrete_entries"]!=entries: raise RuntimeError(f"STEP385 summary artifact mismatch: {soc}")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    _require_remote_ready()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = _dry_run_payload() if args.dry_run else execute()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
