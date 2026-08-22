#!/usr/bin/env python3
"""Disarmed remote controller skeleton for the STEP377 diagnostic shadow run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import ipaddress
import json
import os
import shlex
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".codex-tools"
REMOTE_EXEC = TOOLS / "remote_exec.py"
STEP357 = TOOLS / "step357_build_qrv2_release_remote.py"
PROCESS_GUARD = TOOLS / "step377_process_guard.py"
AUTHORITY_MAP = Path("/home/l30002999/import-md/hw-import-ip.md")
CONTAINER = "mapqr-leicheng"
NPU_READY = False
REMOTE_DIAG_NAME = "step377_attempt10_qrv2_vtv_direct_qa_legacy_probe_v6_shadow_world8_20260822"
ATTEMPT3_DIR = "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step376_attempt3_qrv2_vtv_direct_qa_legacy_probe_v6_build_20260822"
ATTEMPT3_MANIFEST = ATTEMPT3_DIR + "/work/release_manifest.json"
ATTEMPT3_MANIFEST_SHA256: str | None = "18f7434836014f012f9308bfdf95f2f4b9f9a846cf3eb99942e0e22cfda8c6a1"
IMMUTABLE_ORIGINAL_WHEEL: str | None = "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step376_attempt3_qrv2_vtv_direct_qa_legacy_probe_v6_build_20260822/work/outer_original/mx_driving_cloud-26.0.7+CANN8.3.RC1.A3-cp311-cp311-linux_aarch64.whl"
IMMUTABLE_ORIGINAL_WHEEL_SHA256: str | None = "23253f7fa2b9bfb1b6ff3c77df6620f6c559f68be154f6333246d73178eb5da9"
REMOTE_EXEC_SHA256 = "8dfcdda0630413db6cf3593756b81b6a633bc40fe1c761f8ea9a8c8a4e0ffaab"
STEP357_SHA256 = "bf111e2e7eee407e3af26f0ed4e1aab1f833f0e068e66e463664b115c1879d91"
PROCESS_GUARD_SHA256 = "7b4dcb578fd5227f51cf54b2acaa0591840261794b3296eeafa5731e76ad27c5"
IDLE_SENTINEL_PGID = 2147483647
PORT = 34377

TOOLS_TO_UPLOAD = (
    TOOLS / "step377_prepare_diagnostic_shadow.py",
    TOOLS / "step377_diagnostic_math_worker.py",
    TOOLS / "step377_diagnostic_host_case.py",
    TOOLS / "step358_qrv2_release_math_worker.py",
    TOOLS / "step358_host_case.py",
    TOOLS / "step343_world8_controller.py",
    TOOLS / "step343_qrv2_cold_case.py",
    TOOLS / "qrv2_release_oracle.py",
    PROCESS_GUARD,
    STEP357,
)
INPUTS = tuple(
    ROOT / "step260_qr_bad_tensors" / f"rank{rank}_step10_ind0_192x192_BAD.pt"
    for rank in range(8)
)
FILES = TOOLS_TO_UPLOAD + INPUTS
EXPECTED_SHA256 = {
    "step377_prepare_diagnostic_shadow.py": "bb080a8209f55327bd5774ac14ae99d5d58ba9ff8147140a447b9f37ab6356ff",
    "step377_diagnostic_math_worker.py": "f363ac8bd85bb6e56e0de9f1cc6eb8b321b9d6db296af61e3c12384bb4ce4c3d",
    "step377_diagnostic_host_case.py": "91dd54cf26183861d2e389944b7232337b10e28a1544a66584a826dd1d7bc704",
    "step358_qrv2_release_math_worker.py": "f5e3bc0b4e333109c8c3c0003e3467b995fe6a3c061e911704ab06a29bfe10c7",
    "step358_host_case.py": "94e5a46059c0a57bb883999f2648f755158c879568ed3c33b6d4fde8cf1c7070",
    "step343_world8_controller.py": "ea0e587cd0b6c1b31fe753e3239a63c91597cee8f4ec917ad08ab7999bb82ce6",
    "step343_qrv2_cold_case.py": "8a5abcd6e9654fc943847d6695bec1bd71fe2b2558a3ec7b903fe13a4eeb6508",
    "qrv2_release_oracle.py": "d92e02c3df761ddcc94580836615daa661c0e31c23eb8dc32a25dbd806bf6492",
    "step377_process_guard.py": PROCESS_GUARD_SHA256,
    "step357_build_qrv2_release_remote.py": STEP357_SHA256,
    "rank0_step10_ind0_192x192_BAD.pt": "23ad9198223159fc6aa67f79642c299fd86e0aaa2b7ae72bdea297fcb023ab55",
    "rank1_step10_ind0_192x192_BAD.pt": "2cb99d06aa9c96d61f0b615cf41fa579bd6779f7f97c97fa84693180c32adb5b",
    "rank2_step10_ind0_192x192_BAD.pt": "61dcbad02578e60ce7bb82b837f0b33fff2e0071fbde530a339dcad1ce2a692d",
    "rank3_step10_ind0_192x192_BAD.pt": "89266a246497f51d1c6db5e698ee1442abc91bd48c7dc539a09d2373c21b3ac1",
    "rank4_step10_ind0_192x192_BAD.pt": "e750ddcc8dd892ece49d04873910752c657f6d853f8e698daf03fa3fce3a73ca",
    "rank5_step10_ind0_192x192_BAD.pt": "bbceebf84c574e21e9262774c41e0c8bb5eb7f5add0d0cf123e4efbd6a95dc68",
    "rank6_step10_ind0_192x192_BAD.pt": "f2091ec0c618721ba95452fcca82288a2fc8148f40718f945a9e80646dd1d766",
    "rank7_step10_ind0_192x192_BAD.pt": "3dcc3f2bdb7945eaac7ce246128804dfecd89d381e27dc108e99d90d2df2121c",
}
EXPECTED_INPUT_SHA256 = {
    str(rank): EXPECTED_SHA256[f"rank{rank}_step10_ind0_192x192_BAD.pt"]
    for rank in range(8)
}
DRY_RUN_ACTIONS = (
    "exclusive_remote_directory", "upload_exact_inventory", "snapshot_pre",
    "prepare_diagnostic_shadow", "validate_shadow", "world8_back8_once",
    "validate_diagnostic_summary", "snapshot_post", "ownership_cleanup",
)
FORBIDDEN_ACTIONS = (
    "package", "wheel_write", "release", "install", "modify_installed",
    "train", "download_remote_artifacts",
)
EMBEDDED_SAFE_READER = r'''
import hashlib,json,os,stat
from pathlib import Path
def safe_file(path_text,limit=None,expected_sha256=None,after_open=None,capture=False):
 p=Path(path_text); before=p.lstat()
 if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode): raise RuntimeError('unsafe regular file')
 if limit is not None and before.st_size>limit: raise RuntimeError('file exceeds limit')
 fd=os.open(str(p),os.O_RDONLY|os.O_NOFOLLOW)
 try:
  opened=os.fstat(fd)
  if after_open is not None: after_open()
  h=hashlib.sha256(); chunks=[]; total=0
  while True:
   data=os.read(fd,1048576)
   if not data: break
   total+=len(data)
   if limit is not None and total>limit: raise RuntimeError('file exceeds limit')
   h.update(data)
   if capture: chunks.append(data)
  closed=os.fstat(fd)
 finally: os.close(fd)
 after=p.lstat()
 identity=lambda s:(s.st_dev,s.st_ino,stat.S_IFMT(s.st_mode),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if identity(before)!=identity(opened) or identity(opened)!=identity(closed) or identity(opened)!=identity(after): raise RuntimeError('file identity changed')
 digest=h.hexdigest()
 if expected_sha256 is not None and digest!=expected_sha256: raise RuntimeError('file SHA mismatch')
 return (b''.join(chunks) if capture else None),{'path':str(p.absolute()),'sha256':digest,'size':total,'mode':oct(stat.S_IMODE(opened.st_mode)),'device':opened.st_dev,'inode':opened.st_ino,'mtime_ns':opened.st_mtime_ns,'ctime_ns':opened.st_ctime_ns,'type':'file'}
def safe_json(path_text,limit):
 data,_=safe_file(path_text,limit,capture=True)
 return json.loads(data.decode('utf-8'))
'''


def embedded_script(body: str) -> str:
    return EMBEDDED_SAFE_READER + "\n" + body


def upload_embedded_script() -> str:
    return r'''import hashlib,json,os,stat,sys
from pathlib import Path
r=Path(sys.argv[1]); e=json.loads(sys.argv[2]); mode=(sys.argv[3] if len(sys.argv)>3 else 'initial'); prehost=(mode=='prehost'); root0=r.lstat(); assert stat.S_ISDIR(root0.st_mode) and not r.is_symlink(); rfd=os.open(r,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
 root_open=os.fstat(rfd); assert (root0.st_dev,root0.st_ino)==(root_open.st_dev,root_open.st_ino)
 expected_root={'inputs','run'}|{x for x in e if '/' not in x}; expected_root|=({'shadow_work'} if prehost else set()); assert set(os.listdir(rfd))==expected_root
 inputs0=os.stat('inputs',dir_fd=rfd,follow_symlinks=False); run0=os.stat('run',dir_fd=rfd,follow_symlinks=False); assert stat.S_ISDIR(inputs0.st_mode) and stat.S_ISDIR(run0.st_mode)
 ifd=os.open('inputs',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=rfd); runfd=os.open('run',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=rfd)
 try:
  inputs_open=os.fstat(ifd); run_open=os.fstat(runfd); assert (inputs0.st_dev,inputs0.st_ino)==(inputs_open.st_dev,inputs_open.st_ino) and (run0.st_dev,run0.st_ino)==(run_open.st_dev,run_open.st_ino)
  if mode=='race-inputs': os.rename(r/'inputs',r/'inputs.old'); os.rename(sys.argv[4],r/'inputs')
  assert not os.listdir(runfd); assert set(os.listdir(ifd))=={x.split('/',1)[1] for x in e if x.startswith('inputs/')}
  for rel,d in e.items():
   parent,name=(ifd,rel.split('/',1)[1]) if rel.startswith('inputs/') else (rfd,rel)
   s=os.stat(name,dir_fd=parent,follow_symlinks=False); assert stat.S_ISREG(s.st_mode)
   fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent)
   try:
    opened=os.fstat(fd); h=hashlib.sha256()
    while True:
     b=os.read(fd,1048576)
     if not b: break
     h.update(b)
   finally: os.close(fd)
   after=os.stat(name,dir_fd=parent,follow_symlinks=False); assert (s.st_dev,s.st_ino)==(opened.st_dev,opened.st_ino)==(after.st_dev,after.st_ino) and h.hexdigest()==d
  assert set(os.listdir(rfd))==expected_root and not os.listdir(runfd) and set(os.listdir(ifd))=={x.split('/',1)[1] for x in e if x.startswith('inputs/')}
  inputs1=os.stat('inputs',dir_fd=rfd,follow_symlinks=False); run1=os.stat('run',dir_fd=rfd,follow_symlinks=False)
  assert (inputs_open.st_dev,inputs_open.st_ino)==(inputs1.st_dev,inputs1.st_ino) and (run_open.st_dev,run_open.st_ino)==(run1.st_dev,run1.st_ino)
 finally: os.close(runfd); os.close(ifd)
 root1=os.fstat(rfd); root2=r.lstat(); assert (root0.st_dev,root0.st_ino)==(root1.st_dev,root1.st_ino)==(root2.st_dev,root2.st_ino)
finally: os.close(rfd)
print(json.dumps({'count':len(e)}))'''


def artifact_embedded_script() -> str:
    return embedded_script(r'''import json,sys
result={}
for label,path_text,digest in (("manifest",sys.argv[1],sys.argv[2]),("wheel",sys.argv[3],sys.argv[4])):
 _,row=safe_file(path_text,None,digest); result[label]=row
print(json.dumps(result,sort_keys=True))''')


def snapshot_embedded_scripts() -> tuple[str, str, str]:
    artifacts = embedded_script(r'''import json,sys
a=json.loads(sys.argv[1]); out={}
for label,row in a.items():
 _,out[label]=safe_file(row["path"])
print(json.dumps(out,sort_keys=True))''')
    installed = embedded_script(r'''import json,os,stat,sys
from pathlib import Path
r0=Path(sys.argv[1]); rs=r0.lstat(); assert stat.S_ISDIR(rs.st_mode) and not stat.S_ISLNK(rs.st_mode); r=r0.resolve(strict=True); out={}
def tree_state():
 state={'.':(r.lstat().st_dev,r.lstat().st_ino,r.lstat().st_mtime_ns,r.lstat().st_ctime_ns,stat.S_IFMT(r.lstat().st_mode))}
 for p in r.rglob('*'):
  s=p.lstat(); kind=stat.S_IFMT(s.st_mode); assert kind in (stat.S_IFREG,stat.S_IFDIR); state[p.relative_to(r).as_posix()]=(s.st_dev,s.st_ino,s.st_mtime_ns,s.st_ctime_ns,kind)
 return state
tree0=tree_state()
def parts(relative):
 p=Path(relative); assert not p.is_absolute() and all(x not in ('','.','..') for x in p.parts); return p.parts
def bind_dir(relative):
 fd=os.open(r,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for name in parts(relative):
   nxt=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd); os.close(fd); fd=nxt
  s=os.fstat(fd); q=(r/relative).lstat(); assert (s.st_dev,s.st_ino)==(q.st_dev,q.st_ino)
 finally: os.close(fd)
def add_dir(p):
 rel=p.relative_to(r).as_posix(); bind_dir(rel); s=p.lstat(); assert stat.S_ISDIR(s.st_mode) and not stat.S_ISLNK(s.st_mode); out[rel]={'type':'directory','mode':oct(stat.S_IMODE(s.st_mode)),'device':s.st_dev,'inode':s.st_ino}
def add_file(p,capture=False):
 rel=p.relative_to(r).as_posix(); names=parts(rel); fd=os.open(r,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for name in names[:-1]:
   nxt=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd); os.close(fd); fd=nxt
  before=os.stat(names[-1],dir_fd=fd,follow_symlinks=False); f=os.open(names[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
  try:
   opened=os.fstat(f); h=hashlib.sha256(); total=0; chunks=[]
   while True:
    chunk=os.read(f,1048576)
    if not chunk: break
    total+=len(chunk); h.update(chunk)
    if capture: chunks.append(chunk)
   closed=os.fstat(f)
  finally: os.close(f)
  after=os.stat(names[-1],dir_fd=fd,follow_symlinks=False)
 finally: os.close(fd)
 ident=lambda s:(s.st_dev,s.st_ino,stat.S_IFMT(s.st_mode),s.st_size,s.st_mtime_ns,s.st_ctime_ns); assert stat.S_ISREG(before.st_mode) and ident(before)==ident(opened)==ident(closed)==ident(after)
 out[rel]={'path':str(p.absolute()),'sha256':h.hexdigest(),'size':total,'mode':oct(stat.S_IMODE(opened.st_mode)),'device':opened.st_dev,'inode':opened.st_ino,'type':'file'}
 return b''.join(chunks) if capture else None
k=r/'op_impl/ai_core/tbe/kernel'
for d in (r/'op_impl',r/'op_impl/ai_core',r/'op_impl/ai_core/tbe',k,k/'config'): add_dir(d)
dynamic=r/'op_impl/ai_core/tbe/customize_impl/dynamic'
for d in (dynamic.parent,dynamic): add_dir(d)
for name in ('qr_v2.cpp','qr_v2.py'):
 p=dynamic/name; assert p.is_file() and not p.is_symlink(); add_file(p)
cache=dynamic/'__pycache__'; add_dir(cache)
pyc=cache/'qr_v2.cpython-311.pyc'; assert pyc.is_file() and not pyc.is_symlink(); add_file(pyc)
api=r/'op_api'; include=api/'include'
for d in (api,include): add_dir(d)
header=include/'aclnn_qr_v2.h'; assert header.is_file() and not header.is_symlink(); add_file(header)
ops_config=r/'op_impl/ai_core/tbe/config'; add_dir(ops_config)
for soc in ('ascend910_93','ascend910b'):
 d=ops_config/soc; add_dir(d)
 p=d/('aic-'+soc+'-ops-info.json'); assert p.is_file() and not p.is_symlink(); add_file(p)
op_info=r/'op_impl/ai_core/tbe/op_info_cfg'; op_info_ai=op_info/'ai_core'
for d in (op_info,op_info_ai): add_dir(d)
supported=op_info_ai/'npu_supported_ops.json'; assert supported.is_file() and not supported.is_symlink(); add_file(supported)
for soc in ('ascend910_93','ascend910b'):
 add_dir(k/soc); d=k/soc/'qr_v2'; add_dir(d)
 children=sorted(d.iterdir()); assert children and all(not p.is_symlink() for p in children)
 for p in children:
  assert p.is_file(); add_file(p)
 c=k/'config'/soc; add_dir(c)
 q=c/'qr_v2.json'; b=c/'binary_info_config.json'; qdata=add_file(q,True); bdata=add_file(b,True); assert len(qdata)<=4194304 and len(bdata)<=4194304
 qv=json.loads(qdata); rows=qv.get('binList'); assert isinstance(rows,list) and len(rows)==1
 route=rows[0].get('binInfo',{}).get('jsonFilePath'); assert isinstance(route,str) and '\\' not in route and all(x not in ('','.','..') for x in route.split('/')) and route.startswith(soc+'/qr_v2/') and route.endswith('.json'); assert (k/route).relative_to(d)
 bv=json.loads(bdata); binary=bv.get('QrV2',{}).get('binaryList'); assert isinstance(binary,list) and len(binary)==2
 routes={x.get('binPath') for x in binary if isinstance(x,dict)}; assert len(routes)==1
 route_o=next(iter(routes)); assert isinstance(route_o,str) and '\\' not in route_o and all(x not in ('','.','..') for x in route_o.split('/')) and route_o.startswith(soc+'/qr_v2/') and route_o.endswith('.o'); assert (k/route_o).relative_to(d)
 assert (k/route).is_file() and (k/route_o).is_file()
 assert {p.name for p in children}=={Path(route).name,Path(route_o).name}
expected=set(out)
for p in r.rglob('*'):
 rel=p.relative_to(r).as_posix(); low=rel.lower()
 if p.is_symlink(): raise RuntimeError('customize inventory rejects symlink: '+rel)
 if ('qr_v2' in low or 'qrv2' in low or p.name=='binary_info_config.json') and rel not in expected: raise RuntimeError('unexpected QrV2 customize route: '+rel)
 if p.is_file() and (p.suffix.lower()=='.json' or 'manifest' in p.name.lower()):
  payload,_=safe_file(p,16777216,capture=True)
  try: value=json.loads(payload)
  except Exception as exc: raise RuntimeError('invalid customize JSON/manifest: '+rel) from exc
  strings=[]
  def walk(x):
   if isinstance(x,str): strings.append(x)
   elif isinstance(x,dict):
    for key,item in x.items(): walk(key); walk(item)
   elif isinstance(x,list):
    for item in x: walk(item)
  walk(value)
  semantic=any('qr_v2' in x.lower() or 'qrv2' in x.lower() for x in strings)
  if semantic and rel not in expected: raise RuntimeError('unexpected QrV2 semantic manifest route: '+rel)
if len(sys.argv)>2 and sys.argv[2]=='race-add': (r/'race-added').write_bytes(b'race')
assert tree_state()==tree0
print(json.dumps({'root':str(r),'entries':out},sort_keys=True))''')
    processes = r'''import json,os,sys
from pathlib import Path
needle=sys.argv[1].encode(); port=sys.argv[2].encode(); rows=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 if int(p.name)==os.getpid(): continue
 try: cmd=(p/'cmdline').read_bytes()
 except (FileNotFoundError,PermissionError,ProcessLookupError): continue
 if needle in cmd or port in cmd: rows.append(int(p.name))
print(json.dumps(sorted(rows)))'''
    return artifacts, installed, processes


def shadow_embedded_script() -> str:
    return embedded_script(r'''import hashlib,json,os,stat,sys,zipfile
from pathlib import Path
r=Path(sys.argv[1]); p=r/'shadow_manifest.json'; m=safe_json(p,4194304); source_data,_=safe_file(sys.argv[5],4194304,sys.argv[3],capture=True); source=json.loads(source_data.decode()); assert set(m)=={'schema','status','diagnostic_only','package_forbidden','source_overlay','record_unchanged','attempt3_manifest','original_wheel','shadow_root','package_root','candidate_identity','record','attempt3_artifact_inputs','artifacts'}
def bound(root,path,expected=None,capture=False):
 root=Path(root); path=Path(path); rel=path.relative_to(root); assert rel.parts and all(x not in ('','.','..') for x in rel.parts)
 root0=root.lstat(); d=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for name in rel.parts[:-1]:
   nxt=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=d); os.close(d); d=nxt
  before=os.stat(rel.parts[-1],dir_fd=d,follow_symlinks=False); f=os.open(rel.parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=d)
  try:
   opened=os.fstat(f); h=hashlib.sha256(); chunks=[]
   while True:
    chunk=os.read(f,1048576)
    if not chunk: break
    h.update(chunk)
    if capture: chunks.append(chunk)
   closed=os.fstat(f)
  finally: os.close(f)
  after=os.stat(rel.parts[-1],dir_fd=d,follow_symlinks=False)
 finally: os.close(d)
 root1=root.lstat(); ident=lambda s:(s.st_dev,s.st_ino,stat.S_IFMT(s.st_mode),s.st_size,s.st_mtime_ns,s.st_ctime_ns); assert ident(root0)==ident(root1) and stat.S_ISREG(before.st_mode) and ident(before)==ident(opened)==ident(closed)==ident(after)
 digest=h.hexdigest(); assert expected is None or digest==expected; return b''.join(chunks) if capture else None
assert m['schema']=='step377-diagnostic-shadow-v1' and m['status']==sys.argv[2] and m['diagnostic_only'] is True and m['package_forbidden'] is True and m['source_overlay'] is False and m['record_unchanged'] is True
assert m['candidate_identity']=='QrV2_vtv_direct_qa_legacy_probe_v6' and m['attempt3_manifest']['sha256']==sys.argv[3] and m['original_wheel']['sha256']==sys.argv[4]
assert isinstance(m['original_wheel']['path'],str) and m['original_wheel']['path']==sys.argv[6] and Path(sys.argv[6]).is_absolute()
assert Path(m['shadow_root']).resolve()==(r/'shadow').resolve() and Path(m['package_root']).resolve()==(r/'shadow/mx_driving_cloud').resolve()
assert m['attempt3_manifest']['path']==str(Path(sys.argv[5]).resolve(strict=True))
assert set(m['attempt3_artifact_inputs'])=={'ascend910_93','ascend910b'} and set(source.get('artifacts',{}))=={'ascend910_93','ascend910b'}
assert set(m['artifacts'])=={'ascend910_93','ascend910b'}
def shadow_tree_state():
 root=r/'shadow'; s=root.lstat(); assert stat.S_ISDIR(s.st_mode); state={'.':(s.st_dev,s.st_ino,s.st_mtime_ns,s.st_ctime_ns,stat.S_IFMT(s.st_mode))}
 for entry in root.rglob('*'):
  value=entry.lstat(); kind=stat.S_IFMT(value.st_mode); assert kind in (stat.S_IFREG,stat.S_IFDIR); state[entry.relative_to(root).as_posix()]=(value.st_dev,value.st_ino,value.st_mtime_ns,value.st_ctime_ns,kind)
 return state
shadow_initial=shadow_tree_state()
shadow_configs={}
for soc,v in m['artifacts'].items():
 assert set(v)=={'json_path','json_sha256','object_path','object_sha256','config_sha256','binary_info_config_sha256'}
 src=source['artifacts'][soc]; inp=m['attempt3_artifact_inputs'][soc]; assert set(inp)=={'object_path','object_sha256','json_path','json_sha256'}
 assert inp['object_path']==str(Path(src['object_path']).resolve(strict=True)) and inp['json_path']==str(Path(src['json_path']).resolve(strict=True)) and inp['object_sha256']==src['object_sha256'] and inp['json_sha256']==src['json_sha256']
 assert v['object_sha256']==inp['object_sha256'] and v['json_sha256']==inp['json_sha256']
 safe_file(inp['object_path'],None,inp['object_sha256']); safe_file(inp['json_path'],None,inp['json_sha256'])
 for key in ('json_path','object_path'):
  q=Path(v[key]); expected=r/'shadow/mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/kernel'/soc/'qr_v2'/('QrV2_vtv_direct_qa_legacy_probe_v6'+('.json' if key=='json_path' else '.o')); assert q.absolute()==expected.absolute(); bound(r/'shadow',q,v[key.replace('_path','_sha256')])
 config=r/'shadow/mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/kernel/config'/soc
 qcfg=json.loads(bound(r/'shadow',config/'qr_v2.json',v['config_sha256'],True)); rows=qcfg.get('binList'); assert isinstance(rows,list) and len(rows)==1 and rows[0].get('binInfo')=={'jsonFilePath':soc+'/qr_v2/QrV2_vtv_direct_qa_legacy_probe_v6.json'}
 bcfg=json.loads(bound(r/'shadow',config/'binary_info_config.json',v['binary_info_config_sha256'],True)); binary=bcfg.get('QrV2',{}).get('binaryList'); assert isinstance(binary,list) and len(binary)==2 and all(x.get('binPath')==soc+'/qr_v2/QrV2_vtv_direct_qa_legacy_probe_v6.o' for x in binary)
 shadow_configs[soc]=(qcfg,bcfg)
record=m['record']; assert set(record)=={'relative_path','sha256'}; rp=r/'shadow'/record['relative_path']; bound(r/'shadow',rp,record['sha256'])
assert record['relative_path'].endswith('.dist-info/RECORD') and '..' not in record['relative_path'].split('/') and '.' not in record['relative_path'].split('/')
wheel=Path(m['original_wheel']['path']); before=wheel.lstat(); fd=os.open(wheel,os.O_RDONLY|os.O_NOFOLLOW)
try:
 opened=os.fstat(fd); wheel_files={}; wheel_dirs=set(); wheel_payloads={}
 with zipfile.ZipFile(os.fdopen(os.dup(fd),'rb')) as z:
  infos=z.infolist(); assert len({x.filename.rstrip('/') for x in infos})==len(infos)
  for info in infos:
   name=info.filename.rstrip('/'); assert name and not name.startswith('/') and '\\' not in name and all(x not in ('','.','..') for x in name.split('/'))
   file_type=stat.S_IFMT(info.external_attr>>16)
   if info.is_dir(): assert file_type in (0,stat.S_IFDIR); wheel_dirs.add(name)
   else:
    assert file_type in (0,stat.S_IFREG)
    h=hashlib.sha256(); chunks=[]; total=0
    with z.open(info) as stream:
     while True:
      chunk=stream.read(1048576)
      if not chunk: break
      total+=len(chunk); assert total<=2147483648; h.update(chunk)
      if name.endswith('.json'): assert total<=16777216; chunks.append(chunk)
    wheel_files[name]=h.hexdigest()
    if name.endswith('.json'): wheel_payloads[name]=b''.join(chunks)
 os.lseek(fd,0,os.SEEK_SET); whole=hashlib.sha256()
 while True:
  chunk=os.read(fd,1048576)
  if not chunk: break
  whole.update(chunk)
 closed=os.fstat(fd)
finally: os.close(fd)
after=wheel.lstat(); ident=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns); assert ident(before)==ident(opened)==ident(closed)==ident(after) and whole.hexdigest()==sys.argv[4]==m['original_wheel']['sha256']
entries=list((r/'shadow').rglob('*'))
for entry in entries:
 kind=stat.S_IFMT(entry.lstat().st_mode); assert kind in (stat.S_IFREG,stat.S_IFDIR)
shadow_state={p.relative_to(r/'shadow').as_posix():(p.lstat().st_dev,p.lstat().st_ino,p.lstat().st_mtime_ns,p.lstat().st_ctime_ns,stat.S_IFMT(p.lstat().st_mode)) for p in entries}; shadow_root_state=(r/'shadow').lstat(); shadow_root_identity=(shadow_root_state.st_dev,shadow_root_state.st_ino,shadow_root_state.st_mtime_ns,shadow_root_state.st_ctime_ns)
shadow_files={p.relative_to(r/'shadow').as_posix() for p in entries if p.is_file()}; shadow_dirs={p.relative_to(r/'shadow').as_posix() for p in entries if p.is_dir()}
required_dirs={str(Path(name).parent).replace('\\','/') for name in shadow_files}; required_dirs.discard('.'); required_dirs|={str(parent).replace('\\','/') for name in shadow_files for parent in Path(name).parents if str(parent)!='.'}; assert shadow_dirs==required_dirs
wheel_required={str(parent).replace('\\','/') for name in wheel_files for parent in Path(name).parents if str(parent)!='.'}; assert wheel_dirs<=wheel_required and wheel_required==shadow_dirs
records=[name for name in wheel_files if name.endswith('.dist-info/RECORD')]; assert records==[record['relative_path']] and wheel_files[record['relative_path']]==record['sha256']
mutable=set(); prefix='mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/kernel/'
for soc in ('ascend910_93','ascend910b'):
 qname=prefix+'config/'+soc+'/qr_v2.json'; bname=prefix+'config/'+soc+'/binary_info_config.json'; assert qname in wheel_files and bname in wheel_files
 qold=json.loads(wheel_payloads[qname]); bold=json.loads(wheel_payloads[bname]); rows=qold.get('binList'); assert isinstance(rows,list) and len(rows)==1
 old_json=rows[0].get('binInfo',{}).get('jsonFilePath'); binaries=bold.get('QrV2',{}).get('binaryList'); assert isinstance(old_json,str) and isinstance(binaries,list) and len(binaries)==2
 old_objects={x.get('binPath') for x in binaries if isinstance(x,dict)}; assert len(old_objects)==1; old_object=next(iter(old_objects))
 for route,suffix in ((old_json,'.json'),(old_object,'.o')): assert route.startswith(soc+'/qr_v2/') and route.endswith(suffix) and all(x not in ('','.','..') for x in route.split('/'))
 newbase=prefix+soc+'/qr_v2/QrV2_vtv_direct_qa_legacy_probe_v6'; mutable|={prefix+old_json,prefix+old_object,newbase+'.json',newbase+'.o',qname,bname}
 assert {x for x in wheel_files if x.startswith(prefix+soc+'/qr_v2/')}=={prefix+old_json,prefix+old_object}
 assert {x for x in shadow_files if x.startswith(prefix+soc+'/qr_v2/')}=={newbase+'.json',newbase+'.o'}
 def transform(value,old,new):
  count=0
  def visit(x):
   nonlocal count
   if isinstance(x,str) and x==old: count+=1; return new
   if isinstance(x,list): return [visit(v) for v in x]
   if isinstance(x,dict): return {k:visit(v) for k,v in x.items()}
   return x
  return visit(value),count
 expected_q,qcount=transform(qold,old_json,soc+'/qr_v2/QrV2_vtv_direct_qa_legacy_probe_v6.json'); expected_b,bcount=transform(bold,old_object,soc+'/qr_v2/QrV2_vtv_direct_qa_legacy_probe_v6.o')
 assert qcount==1 and bcount==2 and shadow_configs[soc]==(expected_q,expected_b)
assert (wheel_files.keys()-mutable)==(shadow_files-mutable)
for name,digest in wheel_files.items():
 if name not in mutable: bound(r/'shadow',r/'shadow'/name,digest)
if len(sys.argv)>7 and sys.argv[7]=='race-add': (r/'shadow/race-added').write_bytes(b'race')
entries_after=list((r/'shadow').rglob('*')); state_after={p.relative_to(r/'shadow').as_posix():(p.lstat().st_dev,p.lstat().st_ino,p.lstat().st_mtime_ns,p.lstat().st_ctime_ns,stat.S_IFMT(p.lstat().st_mode)) for p in entries_after}; root_after=(r/'shadow').lstat(); assert state_after==shadow_state and (root_after.st_dev,root_after.st_ino,root_after.st_mtime_ns,root_after.st_ctime_ns)==shadow_root_identity and shadow_tree_state()==shadow_initial
assert not list(r.rglob('*.whl')) and not list(r.rglob('*.zip')) and not (r/'release').exists()
print(json.dumps({'status':m['status']}))''')


def summary_embedded_script() -> str:
    return embedded_script(r'''import json,sys
from pathlib import Path
p=Path(sys.argv[1]); print(json.dumps(safe_json(p,1048576)))''')


def forbidden_embedded_script() -> str:
    return r'''import json,sys
from pathlib import Path
r=Path(sys.argv[1]); bad=[str(p) for p in r.rglob('*') if p.name in {'release','release_after_npu_smi'} or p.suffix in {'.whl','.zip'}]; assert not bad; print(json.dumps({'bad':bad}))'''


def ownership_embedded_script(*, optional: bool) -> str:
    if optional:
        return embedded_script(r'''import json,sys
from pathlib import Path
p=Path(sys.argv[1]);
if not p.is_file() or p.is_symlink(): print('null')
else:
 data,row=safe_file(p,65536,capture=True); v=json.loads(data); assert set(v)=={'schema','port','launcher_host_pid','launcher_starttime','launcher_pgid'} and v['schema']=='step358-launcher-ownership-v1' and v['port']==int(sys.argv[2]); print(json.dumps({'manifest':v,'sha256':row['sha256'],'device':row['device'],'inode':row['inode'],'size':row['size'],'mtime_ns':row['mtime_ns'],'ctime_ns':row['ctime_ns']},sort_keys=True))''')
    return embedded_script(r'''import json,sys
from pathlib import Path
p=Path(sys.argv[1]); data,row=safe_file(p,65536,capture=True); v=json.loads(data); keys={'schema','port','launcher_host_pid','launcher_starttime','launcher_pgid'}; assert set(v)==keys and v['schema']=='step358-launcher-ownership-v1' and type(v['port']) is int and v['port']==int(sys.argv[2]); assert all(type(v[k]) is int and v[k]>1 for k in ('launcher_host_pid','launcher_starttime','launcher_pgid')); print(json.dumps({'manifest':v,'sha256':row['sha256'],'device':row['device'],'inode':row['inode'],'size':row['size'],'mtime_ns':row['mtime_ns'],'ctime_ns':row['ctime_ns']},sort_keys=True))''')


def rank_ownership_embedded_script() -> str:
    return embedded_script(r'''import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file() or p.is_symlink(): print('null')
else:
 data,row=safe_file(p,4194304,capture=True); v=json.loads(data); print(json.dumps({'manifest':v,'sha256':row['sha256'],'device':row['device'],'inode':row['inode'],'size':row['size'],'mtime_ns':row['mtime_ns'],'ctime_ns':row['ctime_ns']},sort_keys=True))''')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_local_regular(path: Path, expected_sha256: str | None = None) -> tuple[bytes, str]:
    """Read one local upload through an identity-checked, non-following descriptor."""
    before = path.lstat()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        digest = hashlib.sha256()
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    identity = lambda value: (
        value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode), value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )
    if not stat.S_ISREG(before.st_mode) or not (
        identity(before) == identity(opened) == identity(closed) == identity(after)
    ):
        raise RuntimeError(f"local file identity changed: {path.name}")
    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(f"local file SHA mismatch: {path.name}")
    return b"".join(chunks), actual


def _lower_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_ownership_evidence(value: Any) -> dict[str, Any]:
    keys = {"manifest", "sha256", "device", "inode", "size", "mtime_ns", "ctime_ns"}
    manifest_keys = {"schema", "port", "launcher_host_pid", "launcher_starttime", "launcher_pgid"}
    if not isinstance(value, dict) or set(value) != keys or not _lower_hex64(value.get("sha256")):
        raise RuntimeError("STEP377 ownership evidence mismatch")
    manifest = value.get("manifest")
    if not isinstance(manifest, dict) or set(manifest) != manifest_keys or manifest.get("schema") != "step358-launcher-ownership-v1" or manifest.get("port") != PORT:
        raise RuntimeError("STEP377 ownership manifest mismatch")
    if type(value["device"]) is not int or value["device"] < 0 or any(
        type(value[key]) is not int or value[key] <= 0
        for key in ("inode", "size", "mtime_ns", "ctime_ns")
    ):
        raise RuntimeError("STEP377 ownership identity mismatch")
    if any(type(manifest[key]) is not int or manifest[key] <= 1 for key in ("launcher_host_pid", "launcher_starttime", "launcher_pgid")):
        raise RuntimeError("STEP377 ownership process identity mismatch")
    return value


def _validate_rank_ownership_evidence(value: Any, launcher_sha256: str, case_path: str) -> dict[str, Any]:
    keys = {"manifest", "sha256", "device", "inode", "size", "mtime_ns", "ctime_ns"}
    if not isinstance(value, dict) or set(value) != keys or not _lower_hex64(value.get("sha256")):
        raise RuntimeError("STEP377 rank ownership evidence mismatch")
    if type(value["device"]) is not int or value["device"] < 0 or any(
        type(value[key]) is not int or value[key] <= 0 for key in ("inode", "size", "mtime_ns", "ctime_ns")
    ):
        raise RuntimeError("STEP377 rank ownership file identity mismatch")
    manifest = value.get("manifest")
    top = {"schema", "launcher_ownership_sha256", "gate_token_sha256", "case_path", "port", "ranks"}
    row_keys = {"rank", "local_rank", "host_pid", "container_pid", "physical", "chip", "device_id", "starttime", "pgid", "nspid", "argv"}
    if (
        not isinstance(manifest, dict) or set(manifest) != top
        or manifest.get("schema") != "step377-rank-ownership-v1"
        or manifest.get("launcher_ownership_sha256") != launcher_sha256
        or not _lower_hex64(manifest.get("gate_token_sha256"))
        or manifest.get("case_path") != case_path or manifest.get("port") != PORT
        or not isinstance(manifest.get("ranks"), list) or len(manifest["ranks"]) != 8
    ):
        raise RuntimeError("STEP377 rank ownership manifest mismatch")
    ranks = manifest["ranks"]
    for row in ranks:
        if not isinstance(row, dict) or set(row) != row_keys:
            raise RuntimeError("STEP377 rank ownership row mismatch")
        if any(type(row[key]) is not int for key in row_keys - {"nspid", "argv"}):
            raise RuntimeError("STEP377 rank ownership integer mismatch")
        if not isinstance(row["nspid"], list) or not row["nspid"] or any(type(item) is not int for item in row["nspid"]):
            raise RuntimeError("STEP377 rank ownership nspid mismatch")
        if not isinstance(row["argv"], list) or not row["argv"] or any(type(item) is not str for item in row["argv"]):
            raise RuntimeError("STEP377 rank ownership argv mismatch")
        if row["rank"] != row["local_rank"] or row["device_id"] != 8 + row["rank"] or row["physical"] != 4 + row["rank"] // 2 or row["chip"] != row["rank"] % 2 or row["host_pid"] <= 1 or row["container_pid"] <= 1 or row["starttime"] <= 0 or row["pgid"] <= 1 or any(item <= 0 for item in row["nspid"]) or row["nspid"][0] != row["host_pid"] or row["nspid"][-1] != row["container_pid"]:
            raise RuntimeError("STEP377 rank ownership identity mismatch")
    if {row["rank"] for row in ranks} != set(range(8)) or len({(row["host_pid"], row["starttime"]) for row in ranks}) != 8 or len({row["container_pid"] for row in ranks}) != 8:
        raise RuntimeError("STEP377 rank ownership bijection mismatch")
    return value


def _validate_stable_clear(value: Any) -> dict[str, Any]:
    keys = {"schema", "back8_process_count", "case_process_count", "sample_sha256"}
    if (
        not isinstance(value, dict) or set(value) != keys
        or value.get("schema") != "step377-stable-clear-v1"
        or type(value.get("back8_process_count")) is not int or value.get("back8_process_count") != 0
        or type(value.get("case_process_count")) is not int or value.get("case_process_count") != 0
        or not isinstance(value.get("sample_sha256"), list)
        or len(value["sample_sha256"]) != 2
        or not all(_lower_hex64(item) for item in value["sample_sha256"])
    ):
        raise RuntimeError("STEP377 stable-clear schema mismatch")
    return value


def _validate_guard_result(value: Any, *, cleanup: bool, expect_rank: bool = False) -> dict[str, Any]:
    expected = {"schema", "stable_clear", "port_free"} | ({"launcher_cleanup", "rank_cleanup"} if cleanup else set())
    schema = "step377-cleanup-owned-v1" if cleanup else "step377-snapshot-idle-v1"
    if not isinstance(value, dict) or set(value) != expected or value.get("schema") != schema or value.get("port_free") is not True:
        raise RuntimeError("STEP377 process guard result mismatch")
    _validate_stable_clear(value.get("stable_clear"))
    if cleanup:
        clean = value.get("launcher_cleanup")
        if (
            not isinstance(clean, dict)
            or set(clean) != {"schema", "member_count", "consecutive_empty_group_scans", "external_stable_clear_required"}
            or clean.get("schema") != "step377-owned-group-clean-v1"
            or type(clean.get("member_count")) is not int or clean.get("member_count") != 0
            or type(clean.get("consecutive_empty_group_scans")) is not int or clean.get("consecutive_empty_group_scans") != 2
            or clean.get("external_stable_clear_required") is not True
        ):
            raise RuntimeError("STEP377 cleanup guard schema mismatch")
        rank = value.get("rank_cleanup")
        if expect_rank:
            if not isinstance(rank, dict) or set(rank) != {"schema", "rank_count"} or rank.get("schema") != "step377-fixed-ranks-clean-v1" or type(rank.get("rank_count")) is not int or rank.get("rank_count") != 8:
                raise RuntimeError("STEP377 rank cleanup guard schema mismatch")
        elif rank is not None:
            raise RuntimeError("STEP377 unexpected rank cleanup evidence")
    return value


def local_preflight() -> None:
    """Fail before reading credentials, mapping, or loading an SSH helper."""
    if NPU_READY is not True:
        raise RuntimeError("STEP377 NPU controller is intentionally disarmed")
    if not all(_lower_hex64(value) for value in (
        ATTEMPT3_MANIFEST_SHA256, IMMUTABLE_ORIGINAL_WHEEL_SHA256
    )) or not isinstance(IMMUTABLE_ORIGINAL_WHEEL, str):
        raise RuntimeError("STEP377 remote artifact contract is not armed")
    if {path.name for path in FILES} != set(EXPECTED_SHA256):
        raise RuntimeError("STEP377 upload inventory mismatch")
    for path in FILES:
        if path.is_symlink() or not path.is_file() or sha256_file(path) != EXPECTED_SHA256[path.name]:
            raise RuntimeError(f"STEP377 local input mismatch: {path.name}")
    if REMOTE_EXEC.is_symlink() or sha256_file(REMOTE_EXEC) != REMOTE_EXEC_SHA256:
        raise RuntimeError("STEP377 remote helper mismatch")


def load_backend() -> Any:
    spec = importlib.util.spec_from_file_location("_step377_step357", STEP357)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load STEP377 remote backend")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return RealBackend(module)


class RealBackend:
    """Thin adapter over the SHA-locked STEP357 helper; no invented remote primitives."""

    def __init__(self, legacy: Any):
        self.legacy = legacy
        self.remote_module: Any | None = None
        self.sftp: Any | None = None
        self.remote_root: str | None = None

    def parse_machine_info(self, _path: Path) -> dict[str, object]:
        self.remote_module = self.legacy.load_remote_module()
        AUTHORITY_MAP.read_text(encoding="utf-8")
        info = self.remote_module.parse_machine_info()
        target = ipaddress.ip_address(str(info["target_host"]))
        if not target.is_private or str(target).split(".")[-1] != "42":
            raise RuntimeError("STEP377 target mapping must be private and end in 42")
        if info["jump_host"] == info["target_host"]:
            raise RuntimeError("STEP377 mapping must contain two distinct hops")
        self.remote_root = self.legacy.safe_remote_path(str(info["shared"]), REMOTE_DIAG_NAME)
        return info

    def connect_target(self, info: dict[str, object]):
        if self.remote_module is None:
            raise RuntimeError("STEP377 remote module was not loaded")
        jump, target = self.legacy.connect_target(self.remote_module, info)
        try:
            self.sftp = target.open_sftp()
        except BaseException as primary:
            for resource in (target, jump):
                try:
                    resource.close()
                except BaseException as cleanup:
                    _append_cleanup(primary, cleanup)
            raise
        return jump, target

    def require_hostname(self, target: Any) -> None:
        out, _ = self.legacy.run_host_script(target, "hostname", timeout=30)
        if out.strip() != self.legacy.EXPECTED_HOSTNAME:
            raise RuntimeError("STEP377 target hostname mismatch")

    def require_container(self, target: Any, name: str) -> None:
        if name != CONTAINER:
            raise RuntimeError("STEP377 container mismatch")
        self.legacy.container_probe(target)

    def exclusive_directory(self, target: Any, name: str, mode: int) -> None:
        if name != REMOTE_DIAG_NAME or mode != 0o700 or self.remote_root is None:
            raise RuntimeError("STEP377 remote directory contract mismatch")
        script = (
            "set -eu\numask 077\n[ ! -e " + shlex.quote(self.remote_root) + " ]\n"
            "mkdir -m 700 -- " + shlex.quote(self.remote_root) + "\n"
            "mkdir -m 700 -- " + shlex.quote(self.remote_root + "/inputs") + " "
            + shlex.quote(self.remote_root + "/run")
        )
        self.legacy.run_host_script(target, script, timeout=30)

    def upload_new(self, _target: Any, path: Path, destination: str, digest: str) -> None:
        if self.sftp is None:
            raise RuntimeError("STEP377 upload precondition mismatch")
        payload, actual = read_local_regular(path, digest)
        if actual != digest:
            raise RuntimeError("STEP377 upload precondition mismatch")
        relative = ("inputs/" if path.suffix == ".pt" else "") + path.name
        if self.remote_root is None or destination != REMOTE_DIAG_NAME + "/" + relative:
            raise RuntimeError("STEP377 upload destination mismatch")
        self.legacy.write_remote_new(
            self.sftp, self.remote_root + "/" + relative, payload, mode=0o600
        )

    def verify_uploads(self, target: Any, *, pre_host: bool = False) -> None:
        if self.remote_root is None:
            raise RuntimeError("STEP377 remote root unavailable")
        expected = {
            (("inputs/" if path.suffix == ".pt" else "") + path.name): EXPECTED_SHA256[path.name]
            for path in FILES
        }
        code = upload_embedded_script()
        arguments = [self.remote_root, json.dumps(expected, sort_keys=True)]
        if pre_host:
            arguments.append("prehost")
        value = self._run_json(target, code, *arguments)
        if value != {"count": len(expected)}:
            raise RuntimeError("STEP377 remote upload inventory mismatch")

    def _run_json(self, target: Any, code: str, *arguments: str, timeout: int = 120) -> Any:
        command = "docker exec " + shlex.quote(CONTAINER) + " python3 -c " + shlex.quote(code)
        command += "".join(" " + shlex.quote(item) for item in arguments)
        out, _ = self.legacy.run(target, command, timeout=timeout)
        return json.loads(out)

    def validate_readonly_artifacts(self, target: Any, *arguments: Any) -> dict[str, Any]:
        code = artifact_embedded_script()
        return self._run_json(target, code, *(str(item) for item in arguments))

    def snapshot(self, target: Any, artifact: dict[str, Any]) -> dict[str, Any]:
        code, installed_code, _process_code = snapshot_embedded_scripts()
        artifacts = self._run_json(target, code, json.dumps(artifact, sort_keys=True))
        runtime = self.legacy.container_probe(target)
        installed_root = _installed_root_from_runtime(runtime)
        installed = self._run_json(target, installed_code, installed_root)
        clear = self.require_stable_clear(target)
        processes: list[int] = []
        rows: list[dict[str, int]] = []
        if (not isinstance(artifacts, dict) or not isinstance(installed, dict)
                or installed.get("root") != installed_root or clear.get("port_free") is not True):
            raise RuntimeError("STEP377 snapshot schema mismatch")
        return {
            "schema": "step377-protected-snapshot-v1",
            "artifacts": artifacts,
            "runtime": runtime,
            "installed_root": installed_root,
            "installed_qrv2": installed,
            "related_processes": processes,
            "back8": {"rows": rows, "device_ids": [], "host_pids": []},
        }

    def require_stable_clear(self, target: Any) -> dict[str, Any]:
        if self.remote_root is None:
            raise RuntimeError("STEP377 remote root unavailable")
        command = (
            "python3 " + shlex.quote(self.remote_root + "/step377_process_guard.py")
            + " snapshot-idle --case-path "
            + shlex.quote(self.remote_root + "/step377_diagnostic_host_case.py")
            + " --port " + str(PORT) + " --expected-pgid " + str(IDLE_SENTINEL_PGID)
        )
        out, _ = self.legacy.run_host_script(target, command, timeout=120)
        value = json.loads(out)
        return _validate_guard_result(value, cleanup=False)

    def prepare_shadow(self, target: Any, plan: dict[str, Any], timeout: int) -> None:
        if self.remote_root is None:
            raise RuntimeError("STEP377 remote root unavailable")
        approved = self.remote_root + "/shadow_work"
        self.legacy.run_host_script(target, "mkdir -m 700 -- " + shlex.quote(approved), timeout=30)
        command = (
            "docker exec -e PYTHONDONTWRITEBYTECODE=1 " + shlex.quote(CONTAINER)
            + " python3 " + shlex.quote(self.remote_root + "/step377_prepare_diagnostic_shadow.py")
            + " --attempt3-manifest " + shlex.quote(ATTEMPT3_MANIFEST)
            + " --wheel " + shlex.quote(str(IMMUTABLE_ORIGINAL_WHEEL))
            + " --approved-root " + shlex.quote(approved)
            + " --output-manifest " + shlex.quote(approved + "/shadow_manifest.json")
        )
        self.legacy.run(target, command, timeout=timeout)

    def validate_shadow(self, target: Any, **contract: Any) -> None:
        if self.remote_root is None:
            raise RuntimeError("STEP377 remote root unavailable")
        code = shadow_embedded_script()
        self._run_json(target, code, self.remote_root + "/shadow_work", contract["status"], str(ATTEMPT3_MANIFEST_SHA256), str(IMMUTABLE_ORIGINAL_WHEEL_SHA256), ATTEMPT3_MANIFEST, str(IMMUTABLE_ORIGINAL_WHEEL))

    def pre_host_closure(self, target: Any, artifact: dict[str, Any]) -> None:
        """Re-open every immutable launch input immediately before host execution."""
        self.verify_uploads(target, pre_host=True)
        current = self.validate_readonly_artifacts(
            target, ATTEMPT3_MANIFEST, ATTEMPT3_MANIFEST_SHA256,
            IMMUTABLE_ORIGINAL_WHEEL, IMMUTABLE_ORIGINAL_WHEEL_SHA256,
        )
        if current != artifact:
            raise RuntimeError("STEP377 pre-host artifact identity changed")
        self.validate_shadow(
            target, status="diagnostic_shadow_unvalidated", forbidden=FORBIDDEN_ACTIONS
        )
        self.require_stable_clear(target)

    def run_host_once(self, target: Any, plan: dict[str, Any], **contract: Any) -> dict[str, Any]:
        installed_root = contract.get("installed_custom_opp")
        if (self.remote_root is None
                or contract != {"devices": list(range(8, 16)), "world_size": 8,
                                "port": PORT, "timeout": 1800,
                                "installed_custom_opp": installed_root}
                or _strict_absolute_path(installed_root, "installed custom OPP") != installed_root):
            raise RuntimeError("STEP377 host launch contract mismatch")
        command = (
            "cd " + shlex.quote(self.remote_root) + " && PYTHONDONTWRITEBYTECODE=1 python3 step377_diagnostic_host_case.py"
            + " --port " + str(PORT) + " --output-dir " + shlex.quote(self.remote_root + "/run")
            + " --input-dir " + shlex.quote(self.remote_root + "/inputs")
            + " --shadow-root " + shlex.quote(self.remote_root + "/shadow_work/shadow")
            + " --installed-custom-opp " + shlex.quote(installed_root)
        )
        self.legacy.run_host_script(target, command, timeout=contract["timeout"])
        code = ownership_embedded_script(optional=False)
        return _validate_ownership_evidence(self._run_json(target, code, self.remote_root + "/run/launcher_ownership.json", str(PORT)))

    def read_ownership(self, target: Any) -> dict[str, Any] | None:
        if self.remote_root is None:
            raise RuntimeError("STEP377 remote root unavailable")
        code = ownership_embedded_script(optional=True)
        value = self._run_json(target, code, self.remote_root + "/run/launcher_ownership.json", str(PORT))
        return None if value is None else _validate_ownership_evidence(value)

    def read_rank_ownership(self, target: Any, launcher_sha256: str) -> dict[str, Any] | None:
        if self.remote_root is None or not _lower_hex64(launcher_sha256):
            raise RuntimeError("STEP377 rank ownership precondition mismatch")
        value = self._run_json(
            target, rank_ownership_embedded_script(),
            self.remote_root + "/run/rank_ownership.json",
        )
        if value is None:
            return None
        return _validate_rank_ownership_evidence(
            value, launcher_sha256,
            self.remote_root + "/step377_diagnostic_host_case.py",
        )

    def read_summary(self, target: Any, name: str) -> dict[str, Any]:
        if self.remote_root is None or name != REMOTE_DIAG_NAME:
            raise RuntimeError("STEP377 summary path mismatch")
        code = summary_embedded_script()
        return self._run_json(target, code, self.remote_root + "/run/step377_diagnostic_summary.json")

    def scan_forbidden_outputs(self, target: Any, name: str, forbidden: tuple[str, ...]) -> None:
        if self.remote_root is None or name != REMOTE_DIAG_NAME:
            raise RuntimeError("STEP377 forbidden scan path mismatch")
        code = forbidden_embedded_script()
        self._run_json(target, code, self.remote_root)

    def cleanup_owned(self, target: Any, port: int, ownership: dict[str, Any], rank_ownership: dict[str, Any] | None = None) -> dict[str, Any]:
        evidence = _validate_ownership_evidence(ownership)
        manifest = evidence["manifest"]
        if (
            port != PORT
            or manifest.get("schema") != "step358-launcher-ownership-v1"
            or manifest.get("port") != PORT
        ):
            raise RuntimeError("STEP377 cleanup ownership contract mismatch")
        if self.remote_root is None:
            raise RuntimeError("STEP377 remote root unavailable")
        if rank_ownership is not None:
            rank_ownership = _validate_rank_ownership_evidence(
                rank_ownership, evidence["sha256"],
                self.remote_root + "/step377_diagnostic_host_case.py",
            )
        command = (
            "python3 " + shlex.quote(self.remote_root + "/step377_process_guard.py")
            + " cleanup-owned --ownership "
            + shlex.quote(self.remote_root + "/run/launcher_ownership.json")
            + " --expected-ownership-sha256 " + evidence["sha256"]
            + " --case-path " + shlex.quote(self.remote_root + "/step377_diagnostic_host_case.py")
            + " --port " + str(port)
        )
        if rank_ownership is not None:
            command += (
                " --rank-ownership " + shlex.quote(self.remote_root + "/run/rank_ownership.json")
                + " --expected-rank-ownership-sha256 " + rank_ownership["sha256"]
                + " --expected-gate-token-sha256 " + rank_ownership["manifest"]["gate_token_sha256"]
            )
        out, _ = self.legacy.run_host_script(target, command, timeout=120)
        value = json.loads(out)
        return _validate_guard_result(value, cleanup=True, expect_rank=rank_ownership is not None)


def dry_run_plan() -> dict[str, Any]:
    local_preflight()
    return {
        "schema": "step377-diagnostic-remote-plan-v1",
        "diagnostic_only": True,
        "container": CONTAINER,
        "remote_directory": REMOTE_DIAG_NAME,
        "attempt3_manifest": ATTEMPT3_MANIFEST,
        "uploads": [path.name for path in FILES],
        "actions": list(DRY_RUN_ACTIONS),
        "forbidden": list(FORBIDDEN_ACTIONS),
        "npu_devices": list(range(8, 16)),
        "world_size": 8,
    }


def validate_summary(value: Any) -> dict[str, Any]:
    keys = {
        "schema", "status", "diagnostic_only", "release_candidate", "rank_count",
        "raw_profiles_retained", "input_sha256", "module_file_sha256",
        "gate_token_sha256", "launcher_ownership_sha256", "rank_ownership_sha256",
        "ranks", "raw_profiles",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeError("STEP377 remote summary schema mismatch")
    if (
        value["schema"] != "step377-diagnostic-host-summary-v1"
        or value["status"] != "diagnostic_world8_pass"
        or value["diagnostic_only"] is not True or value["release_candidate"] is not False
        or type(value["rank_count"]) is not int or value["rank_count"] != 8
        or value["raw_profiles_retained"] is not True
        or not _lower_hex64(value["gate_token_sha256"])
        or not _lower_hex64(value["launcher_ownership_sha256"])
        or not _lower_hex64(value["rank_ownership_sha256"])
        or not isinstance(value["ranks"], list) or len(value["ranks"]) != 8
        or not isinstance(value["raw_profiles"], list) or len(value["raw_profiles"]) != 8
    ):
        raise RuntimeError("STEP377 remote summary contract mismatch")
    if value["input_sha256"] != EXPECTED_INPUT_SHA256:
        raise RuntimeError("STEP377 remote summary input closure mismatch")
    if set(value["module_file_sha256"]) != {"cloud_init", "cloud_extension", "cloud_linalg"} or not all(
        _lower_hex64(item) for item in value["module_file_sha256"].values()
    ):
        raise RuntimeError("STEP377 remote summary module closure mismatch")
    ranks = value["ranks"]
    if any(
        not isinstance(row, dict) or set(row) != {"rank", "call_count", "identity_pass"}
        or type(row["rank"]) is not int or type(row["call_count"]) is not int
        or row["call_count"] != 1 or row["identity_pass"] is not True
        for row in ranks
    ) or {row["rank"] for row in ranks} != set(range(8)):
        raise RuntimeError("STEP377 remote summary rank closure mismatch")
    profiles = value["raw_profiles"]
    if any(
        not isinstance(row, dict) or set(row) != {"rank", "file_count", "total_bytes"}
        or type(row["rank"]) is not int or type(row["file_count"]) is not int
        or type(row["total_bytes"]) is not int or row["file_count"] <= 0 or row["total_bytes"] <= 0
        for row in profiles
    ) or {row["rank"] for row in profiles} != set(range(8)):
        raise RuntimeError("STEP377 remote raw profile closure mismatch")
    return value


def _append_cleanup(primary: BaseException, cleanup: BaseException) -> None:
    try:
        errors = getattr(primary, "cleanup_errors", None)
        if errors is None:
            errors = []
            setattr(primary, "cleanup_errors", errors)
        errors.append(cleanup)
    except BaseException:
        pass


def _strict_absolute_path(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not value.startswith("/") or value == "/"
            or value.endswith("/") or "\\" in value or "\0" in value
            or any(part in ("", ".", "..") for part in value.split("/")[1:])):
        raise RuntimeError(f"STEP377 {label} path contract mismatch")
    return value


def _installed_root_from_runtime(runtime: Any) -> str:
    if not isinstance(runtime, dict):
        raise RuntimeError("STEP377 runtime schema mismatch")
    cloud_root = _strict_absolute_path(runtime.get("installed_cloud_root"), "installed cloud root")
    return _strict_absolute_path(
        cloud_root + "/packages/vendors/customize", "installed custom OPP"
    )


def _validate_snapshot(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "artifacts", "runtime", "installed_root", "installed_qrv2", "related_processes", "back8"}
        or value.get("schema") != "step377-protected-snapshot-v1"
        or not isinstance(value.get("artifacts"), dict)
        or not isinstance(value.get("runtime"), dict)
        or value.get("installed_root") != _installed_root_from_runtime(value.get("runtime"))
        or not isinstance(value.get("installed_qrv2"), dict)
        or value["installed_qrv2"].get("root") != value.get("installed_root")
        or value.get("related_processes") != []
        or value.get("back8") != {"rows": [], "device_ids": [], "host_pids": []}
    ):
        raise RuntimeError("STEP377 protected snapshot mismatch")
    return value


def execute(backend: Any | None = None) -> dict[str, Any]:
    """Run the reviewed transaction; currently unreachable without patched readiness."""
    plan = dry_run_plan()
    remote = backend if backend is not None else load_backend()
    resources = []
    primary: BaseException | None = None
    result: dict[str, Any] | None = None
    owned = False
    target: Any | None = None
    artifact: dict[str, Any] | None = None
    before: dict[str, Any] | None = None
    ownership: dict[str, Any] | None = None
    rank_ownership: dict[str, Any] | None = None
    cleanup_postflight: dict[str, Any] | None = None
    try:
        info = remote.parse_machine_info(AUTHORITY_MAP)
        jump, target = remote.connect_target(info)
        resources.extend((target, jump))
        if getattr(remote, "sftp", None) is not None:
            resources.insert(0, remote.sftp)
        remote.require_hostname(target)
        remote.require_container(target, CONTAINER)
        remote.exclusive_directory(target, REMOTE_DIAG_NAME, mode=0o700)
        owned = True
        for path in FILES:
            relative = ("inputs/" if path.suffix == ".pt" else "") + path.name
            remote.upload_new(target, path, REMOTE_DIAG_NAME + "/" + relative, EXPECTED_SHA256[path.name])
        remote.verify_uploads(target)
        artifact = remote.validate_readonly_artifacts(
            target, ATTEMPT3_MANIFEST, ATTEMPT3_MANIFEST_SHA256,
            IMMUTABLE_ORIGINAL_WHEEL, IMMUTABLE_ORIGINAL_WHEEL_SHA256,
        )
        before = _validate_snapshot(remote.snapshot(target, artifact))
        remote.prepare_shadow(target, plan, timeout=900)
        remote.validate_shadow(target, status="diagnostic_shadow_unvalidated", forbidden=FORBIDDEN_ACTIONS)
        remote.pre_host_closure(target, artifact)
        ownership = remote.run_host_once(
            target, plan, devices=list(range(8, 16)), world_size=8, port=PORT,
            timeout=1800, installed_custom_opp=before["installed_root"],
        )
        _validate_ownership_evidence(ownership)
        rank_ownership = remote.read_rank_ownership(target, ownership["sha256"])
        result = validate_summary(remote.read_summary(target, REMOTE_DIAG_NAME))
        if result["launcher_ownership_sha256"] != ownership["sha256"] or rank_ownership is None or result["rank_ownership_sha256"] != rank_ownership["sha256"] or result["gate_token_sha256"] != rank_ownership["manifest"]["gate_token_sha256"]:
            raise RuntimeError("STEP377 summary ownership closure mismatch")
    except BaseException as error:
        primary = error
    finally:
        if owned and target is not None and ownership is None:
            try:
                ownership = remote.read_ownership(target)
            except BaseException as ownership_error:
                if primary is None:
                    primary = ownership_error
                else:
                    _append_cleanup(primary, ownership_error)
        if owned and target is not None and ownership is not None and rank_ownership is None:
            try:
                rank_ownership = remote.read_rank_ownership(target, ownership["sha256"])
            except BaseException as rank_error:
                if primary is None:
                    primary = rank_error
                else:
                    _append_cleanup(primary, rank_error)
        if owned and target is not None and ownership is not None:
            try:
                cleanup_postflight = remote.cleanup_owned(target, PORT, ownership, rank_ownership)
            except BaseException as cleanup:
                if primary is None:
                    primary = cleanup
                else:
                    _append_cleanup(primary, cleanup)
        if owned and target is not None and artifact is not None and before is not None:
            try:
                remote.scan_forbidden_outputs(target, REMOTE_DIAG_NAME, FORBIDDEN_ACTIONS)
                after = _validate_snapshot(remote.snapshot(target, artifact))
                if before != after:
                    raise RuntimeError("STEP377 installed/runtime/artifact closure changed")
            except BaseException as postflight:
                if primary is None:
                    primary = postflight
                else:
                    _append_cleanup(primary, postflight)
        for resource in resources:
            try:
                resource.close()
            except BaseException as cleanup:
                if primary is None:
                    primary = cleanup
                else:
                    _append_cleanup(primary, cleanup)
    if primary is not None:
        raise primary
    if result is None:
        raise RuntimeError("STEP377 transaction returned no summary")
    if cleanup_postflight is None:
        raise RuntimeError("STEP377 cleanup returned no postflight")
    result = dict(result)
    result["cleanup_postflight"] = {
        "schema": "step377-controller-cleanup-postflight-v1",
        "rank_evidence_present": rank_ownership is not None,
        "rank_cleanup_count": 8 if rank_ownership is not None else 0,
        "rank_identities_dead": rank_ownership is not None,
        "launcher_member_count": cleanup_postflight["launcher_cleanup"]["member_count"],
        "stable_clear_samples": len(cleanup_postflight["stable_clear"]["sample_sha256"]),
        "port_free": cleanup_postflight["port_free"],
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_plan(), sort_keys=True))
    else:
        execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
