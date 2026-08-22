#!/usr/bin/env python3
"""Build and package the QrV2 Matmul-position v5 candidate exactly once."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shlex
import socket
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".codex-tools"
LEGACY_PATH = TOOLS / "step357_build_qrv2_release_remote.py"
BUILDER = TOOLS / "build_qrv2_release.py"
V5_PATCHER = TOOLS / "step372_patch_qr_v2_matmul_position_v5.py"
V4_PATCHER = TOOLS / "step338_patch_qr_v2_lifetime.py"
DIAG_NAME = "step373_qrv2_matmul_position_v5_release_20260821"
EXPECTED_KERNEL = "QrV2_matmul_position_fix_v5"
EXPECTED_SOURCE_SHA256 = "e6ccbb84b0e0dbdc026ecdc6b6e07936fbd659401e35c38f7e9eb974d99bc3b7"
EXPECTED_OUTER_SUFFIX = "-qrv2-matmul-position-fix-v5.zip"
OUTER_ZIP_SHA256 = "363fc46e0f3da952ef9c37cdfb67a190f557abc8a879d1438563c2d3eb807da7"
BUILD_READY = True
BUILDER_SHA256 = "d6f2aea68574422ba6e28f220c5581adf8b25019494882dea156a80b1b513e90"
LEGACY_SHA256 = "bf111e2e7eee407e3af26f0ed4e1aab1f833f0e068e66e463664b115c1879d91"
V5_PATCHER_SHA256 = "82c418490925c0a02e3cd5bd7573eb8070777f83607fe5782f86b13f67c2612d"
V4_PATCHER_SHA256 = "d4b260919440e6298c9caea5ca3a7c8bb7426a30ed84cc68d934250ee0ae9fe2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_local_release_ready() -> None:
    if not BUILD_READY:
        raise RuntimeError(
            "QrV2 v5 build inputs are not reviewed; lock builder/legacy SHA values "
            "and set BUILD_READY only after local tests pass"
        )
    expected = {
        LEGACY_PATH: LEGACY_SHA256,
        BUILDER: BUILDER_SHA256,
        V5_PATCHER: V5_PATCHER_SHA256,
        V4_PATCHER: V4_PATCHER_SHA256,
    }
    for path, digest in expected.items():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"v5 build input must be a regular file: {path.name}")
        if sha256_file(path) != digest:
            raise RuntimeError(f"v5 build input SHA mismatch: {path.name}")


def load_legacy() -> Any:
    _require_local_release_ready()
    spec = importlib.util.spec_from_file_location("step373_legacy_build", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load STEP357 build controller")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REMOTE_DIAG_NAME = DIAG_NAME
    module.PATCHER = V5_PATCHER
    module.PATCHER_DEPENDENCIES = (V4_PATCHER,)
    module.EXPECTED_INPUTS = {
        module.OUTER_ZIP.name: OUTER_ZIP_SHA256,
        BUILDER.name: BUILDER_SHA256,
        V5_PATCHER.name: V5_PATCHER_SHA256,
        V4_PATCHER.name: V4_PATCHER_SHA256,
    }
    return module


def execute() -> dict[str, Any]:
    legacy = load_legacy()
    build_summary = legacy.execute()

    # Packaging is a second access, so all mapping, SHA and identity gates run again.
    remote_module = legacy.load_remote_module()
    info = legacy.local_preflight(remote_module)
    jump, target = legacy.connect_target(remote_module, info)
    try:
        hostname, _ = legacy.run(target, "hostname")
        if hostname.strip() != legacy.EXPECTED_HOSTNAME:
            raise RuntimeError("second-hop runtime hostname mismatch before package")
        contract = legacy.container_probe(target)
        remote_diag = legacy.safe_remote_path(str(info["shared"]), DIAG_NAME)
        workdir = remote_diag + "/work"
        package_script = f"""
set -eu
cd {shlex.quote(remote_diag)}
export ASCEND_OPP_PATH={shlex.quote(contract['ascend_opp'])}
python3 {shlex.quote(legacy.BUILDER.name)} package {shlex.quote(workdir)}
"""
        legacy.run(
            target,
            "docker exec " + shlex.quote(legacy.CONTAINER)
            + " bash --noprofile --norc -lc " + shlex.quote(package_script),
            timeout=300,
        )
        summary_code = r'''
import hashlib,json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
kernel=sys.argv[2]; source_sha=sys.argv[3]; suffix=sys.argv[4]
assert m['status']=='packaged_unvalidated'
assert m['candidate']['bin_name']==kernel
assert m['candidate']['source_sha256']==source_sha
assert m['candidate']['structure_assertions']['candidate_identity']==kernel
assert m['candidate']['structure_assertions']['reverse_v4_sha256']==m['candidate']['structure_assertions']['v4_candidate_sha256']
p=m['package']; wheel=Path(p['wheel_path']); outer=Path(p['outer_zip_path'])
assert wheel.is_file() and not wheel.is_symlink()
assert outer.is_file() and not outer.is_symlink() and outer.name.endswith(suffix)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
assert sha(wheel)==p['wheel_sha256'] and sha(outer)==p['outer_zip_sha256']
for soc in ('ascend910_93','ascend910b'):
 a=m['artifacts'][soc]; assert a['kernel_name']==kernel
 b=p['packaged_files'][soc]['binary_info_config']
 assert len(b['sha256'])==64 and b['path'].endswith('/binary_info_config.json')
print(json.dumps({
 'status':m['status'],'kernel':kernel,'candidate_source_sha256':source_sha,
 'wheel_path':str(wheel),'wheel_sha256':sha(wheel),'wheel_size':wheel.stat().st_size,
 'outer_zip_path':str(outer),'outer_zip_sha256':sha(outer),
 'artifact_sha256':{soc:{'object':m['artifacts'][soc]['object_sha256'],'json':m['artifacts'][soc]['json_sha256']} for soc in ('ascend910_93','ascend910b')},
 'binary_info_sha256':{soc:p['packaged_files'][soc]['binary_info_config']['sha256'] for soc in ('ascend910_93','ascend910b')},
 'tool_sha256':m['tools'],
 'installed_inventory_closed':m['build_runtime']['installed_inventory_closed'],
 'runtime_inventory_closed':m['build_runtime']['runtime_inventory_closed'],
},sort_keys=True))
'''
        summary, _ = legacy.run(
            target,
            "docker exec " + shlex.quote(legacy.CONTAINER) + " python3 -c "
            + shlex.quote(summary_code) + " "
            + shlex.quote(workdir + "/release_manifest.json") + " "
            + shlex.quote(EXPECTED_KERNEL) + " "
            + shlex.quote(EXPECTED_SOURCE_SHA256) + " "
            + shlex.quote(EXPECTED_OUTER_SUFFIX),
        )
        result = json.loads(summary)
        result["remote_diagnostics_name"] = DIAG_NAME
        result["build_status"] = build_summary["status"]
        result["uploaded_gate"] = build_summary["uploaded_gate"]
        return result
    finally:
        target.close()
        jump.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    legacy = load_legacy()
    if args.dry_run:
        info = legacy.local_preflight(legacy.load_remote_module())
        print(json.dumps({
            "status": "dry_run",
            "diagnostics_name": DIAG_NAME,
            "target_suffix": str(info["target_host"]).split(".")[-1],
            "container": legacy.CONTAINER,
            "kernel": EXPECTED_KERNEL,
            "candidate_source_sha256": EXPECTED_SOURCE_SHA256,
            "input_sha256": legacy.EXPECTED_INPUTS,
            "actions": ["prepare", "canonical_opc", "dav2201_alias", "package"],
            "forbidden": ["install", "train", "modify_installed"],
        }, sort_keys=True))
        return 0
    print(json.dumps(execute(), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, socket.error) as error:
        print(f"STEP373 failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
