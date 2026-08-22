#!/usr/bin/env python3
"""Rebuild/package QrV2 after the audited binary-info metadata fix."""

from __future__ import annotations

import argparse
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
DIAG_NAME = "step370_qrv2_lifetime_alpha_sync_v4_release_20260821"
BUILDER_SHA256 = "e3d56078915ed4ee2c8724bdcaa2580eebae00797fac7c915eae097d30682da9"
PATCHER_SHA256 = "d4b260919440e6298c9caea5ca3a7c8bb7426a30ed84cc68d934250ee0ae9fe2"


def load_legacy() -> Any:
    spec = importlib.util.spec_from_file_location("step360_legacy_build", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load STEP357 build controller")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REMOTE_DIAG_NAME = DIAG_NAME
    module.EXPECTED_INPUTS[module.BUILDER.name] = BUILDER_SHA256
    module.EXPECTED_INPUTS[module.PATCHER.name] = PATCHER_SHA256
    return module


def execute() -> dict[str, Any]:
    legacy = load_legacy()
    build_summary = legacy.execute()
    # A second connection is a separate access: reread both mappings and all local SHA gates.
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
import json,sys,hashlib
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
assert m['status']=='packaged_unvalidated'
p=m['package']; wheel=Path(p['wheel_path']); outer=Path(p['outer_zip_path'])
assert wheel.is_file() and not wheel.is_symlink() and outer.is_file() and not outer.is_symlink()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(wheel)==p['wheel_sha256'] and sha(outer)==p['outer_zip_sha256']
for soc in ('ascend910_93','ascend910b'):
 b=p['packaged_files'][soc]['binary_info_config']
 assert len(b['sha256'])==64 and b['path'].endswith('/binary_info_config.json')
print(json.dumps({
 'status':m['status'],'wheel_path':str(wheel),'wheel_sha256':sha(wheel),
 'wheel_size':wheel.stat().st_size,'outer_zip_sha256':sha(outer),
 'binary_info_sha256':{soc:p['packaged_files'][soc]['binary_info_config']['sha256'] for soc in ('ascend910_93','ascend910b')},
 'installed_inventory_closed':m['build_runtime']['installed_inventory_closed'],
 'runtime_inventory_closed':m['build_runtime']['runtime_inventory_closed'],
},sort_keys=True))
'''
        summary, _ = legacy.run(
            target,
            "docker exec " + shlex.quote(legacy.CONTAINER) + " python3 -c "
            + shlex.quote(summary_code) + " "
            + shlex.quote(workdir + "/release_manifest.json"),
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
            "status": "dry_run", "diagnostics_name": DIAG_NAME,
            "target_suffix": str(info["target_host"]).split(".")[-1],
            "container": legacy.CONTAINER, "builder_sha256": BUILDER_SHA256,
            "patcher_sha256": PATCHER_SHA256,
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
        print(f"STEP360 failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
