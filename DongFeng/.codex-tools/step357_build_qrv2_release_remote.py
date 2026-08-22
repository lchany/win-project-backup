#!/usr/bin/env python3
"""Build the audited QrV2 candidate in a new remote diagnostics directory.

The controller never installs or packages the candidate.  It validates the
two-hop target, the exact running container, the container/CANN/OPC identity,
uploads an exact immutable input inventory, and invokes ``prepare`` followed by ``build``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import ipaddress
import json
import shlex
import socket
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".codex-tools"
REMOTE_EXEC = TOOLS / "remote_exec.py"
AUTHORITY_MAP = Path("/home/l30002999/import-md/hw-import-ip.md")
OUTER_ZIP = ROOT / "cann-driving-cloud-ops-26.0.7-a3-cann8.3.rc1-torch2.7.1-py3.11-aarch64.zip"
BUILDER = TOOLS / "build_qrv2_release.py"
PATCHER = TOOLS / "step338_patch_qr_v2_lifetime.py"
PATCHER_DEPENDENCIES: tuple[Path, ...] = ()
EXPECTED_INPUTS = {
    OUTER_ZIP.name: "363fc46e0f3da952ef9c37cdfb67a190f557abc8a879d1438563c2d3eb807da7",
    BUILDER.name: "eed04a12da088b5950877c50030118e2d0d0d2207cec263e2eaf6acdbc6215f6",
    PATCHER.name: "0436cd481c297bc95aac936bce74ab7d1d641243a8f09793818f470d1c561b87",
}
REMOTE_DIAG_NAME = "step357_qrv2_release_build_retry2_20260821"
CONTAINER = "mapqr-leicheng"
EXPECTED_HOSTNAME = "yfzy-zhsc-910c-1.novalocal"
OPC = "/usr/local/Ascend/ascend-toolkit/latest/bin/opc"
ASCEND_OPP = "/usr/local/Ascend/ascend-toolkit/latest/opp"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_files() -> tuple[Path, ...]:
    """Return the exact upload inventory for the selected release patcher."""

    return (OUTER_ZIP, BUILDER, PATCHER, *PATCHER_DEPENDENCIES)


def load_remote_module():
    spec = importlib.util.spec_from_file_location("qrv2_remote_exec", REMOTE_EXEC)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load remote helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_preflight(module: Any) -> dict[str, object]:
    AUTHORITY_MAP.read_text(encoding="utf-8")
    info = module.parse_machine_info()
    target = ipaddress.ip_address(str(info["target_host"]))
    if not target.is_private or str(target).split(".")[-1] != "42":
        raise RuntimeError("target mapping must be private and end in 42")
    if info["jump_host"] == info["target_host"]:
        raise RuntimeError("project mapping must describe two distinct SSH hops")
    inputs = input_files()
    if len({path.name for path in inputs}) != len(inputs):
        raise RuntimeError("local input filenames must be unique")
    if {path.name for path in inputs} != set(EXPECTED_INPUTS):
        raise RuntimeError("local upload inventory differs from fixed hash contract")
    for path in inputs:
        argument = path.absolute()
        if argument.is_symlink() or not argument.is_file():
            raise RuntimeError(f"local input must be a regular non-symlink file: {path.name}")
        actual = sha256_file(argument)
        if actual != EXPECTED_INPUTS[path.name]:
            raise RuntimeError(f"local input SHA mismatch: {path.name}: {actual}")
    return info


def run(client: Any, command: str, *, timeout: int = 300) -> tuple[str, str]:
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    status = stdout.channel.recv_exit_status()
    if status != 0:
        raise RuntimeError(f"remote command rc={status}: {err.strip() or out.strip()}")
    return out, err


def run_host_script(client: Any, script: str, *, timeout: int = 300) -> tuple[str, str]:
    return run(client, "bash --noprofile --norc -lc " + shlex.quote(script), timeout=timeout)


def safe_remote_path(base: str, child: str) -> str:
    base_path = PurePosixPath(base)
    result = base_path / "diagnostics" / child
    if not result.is_absolute() or ".." in result.parts:
        raise RuntimeError("unsafe remote diagnostics path")
    return result.as_posix()


def container_probe(client: Any) -> dict[str, Any]:
    host_script = f"""
set -eu
rows=$(docker ps -a --filter 'name=^/{CONTAINER}$' --format '{{{{.Names}}}} {{{{.State}}}}')
[ "$rows" = '{CONTAINER} running' ]
docker inspect --format '{{{{.Id}}}}' {CONTAINER}
docker inspect --format '{{{{.Config.Hostname}}}}' {CONTAINER}
"""
    out, _ = run_host_script(client, host_script)
    lines = out.splitlines()
    if len(lines) != 2:
        raise RuntimeError("docker inspect preflight returned an unexpected shape")
    container_id, inspect_hostname = lines
    if len(container_id) != 64 or any(ch not in "0123456789abcdef" for ch in container_id):
        raise RuntimeError("docker inspect returned an invalid container ID")

    probe_code = r'''
import hashlib, importlib.util, json, os, socket
from pathlib import Path

def inv(path):
    path=Path(path).resolve(strict=True)
    assert path.is_file() and not path.is_symlink()
    return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

opc=Path("/usr/local/Ascend/ascend-toolkit/latest/bin/opc")
versions=[p for p in (
    Path("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info"),
    Path("/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info"),
) if p.is_file()]
spec=importlib.util.find_spec("mx_driving_cloud")
assert spec and spec.origin
cloud=Path(spec.origin).resolve(strict=True).parent
opp=Path("/usr/local/Ascend/ascend-toolkit/latest/opp").resolve(strict=True)
assert (opp/"built-in/op_impl/ai_core/tbe/impl/util/platform_adapter.py").is_file()
print(json.dumps({
    "hostname":socket.gethostname(),
    "opc":inv(opc),
    "cann_version_files":[inv(p) for p in versions],
    "installed_cloud_root":str(cloud),
    "ascend_opp":str(opp),
},sort_keys=True))
'''
    out, _ = run(
        client,
        "docker exec "
        + shlex.quote(CONTAINER)
        + " python3 -c "
        + shlex.quote(probe_code),
    )
    probe = json.loads(out)
    if probe.get("hostname") != inspect_hostname:
        raise RuntimeError("container inspect hostname differs from runtime hostname")
    ascend_opp = probe.get("ascend_opp")
    if not isinstance(ascend_opp, str) or not PurePosixPath(ascend_opp).is_absolute():
        raise RuntimeError("container returned an invalid resolved ASCEND_OPP_PATH")
    versions = probe.get("cann_version_files")
    if not isinstance(versions, list) or not versions:
        raise RuntimeError("no CANN version file found")
    return {
        "schema_version": 1,
        "container_name": CONTAINER,
        "inspect_container_id": container_id,
        "inspect_hostname": inspect_hostname,
        "opc": probe["opc"],
        "cann_version_files": versions,
        "installed_cloud_root": probe["installed_cloud_root"],
        "ascend_opp": ascend_opp,
    }


def connect_target(module: Any, info: dict[str, object]):
    jump = module.connect(
        str(info["jump_host"]),
        int(info["jump_port"]),
        str(info["jump_user"]),
        str(info["jump_password"]),
    )
    try:
        channel = jump.get_transport().open_channel(
            "direct-tcpip",
            (str(info["target_host"]), int(info["target_port"])),
            ("127.0.0.1", 0),
        )
        target = module.connect(
            str(info["target_host"]),
            int(info["target_port"]),
            str(info["target_user"]),
            str(info["target_password"]),
            sock=channel,
        )
    except Exception:
        jump.close()
        raise
    return jump, target


def write_remote_new(sftp: Any, path: str, payload: bytes, mode: int = 0o600) -> None:
    try:
        sftp.lstat(path)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(path)
    with sftp.open(path, "wx") as stream:
        stream.write(payload)
    sftp.chmod(path, mode)


def execute() -> dict[str, Any]:
    module = load_remote_module()
    info = local_preflight(module)
    jump, target = connect_target(module, info)
    try:
        out, _ = run(target, "hostname")
        if out.strip() != EXPECTED_HOSTNAME:
            raise RuntimeError("second-hop runtime hostname mismatch")
        contract = container_probe(target)
        remote_diag = safe_remote_path(str(info["shared"]), REMOTE_DIAG_NAME)
        workdir = remote_diag + "/work"
        run_host_script(target, f"set -eu\n[ ! -e {shlex.quote(remote_diag)} ]\nmkdir -m 700 {shlex.quote(remote_diag)}")
        sftp = target.open_sftp()
        try:
            for local in input_files():
                remote = remote_diag + "/" + local.name
                write_remote_new(sftp, remote, local.read_bytes())
            contract_path = remote_diag + "/container_contract.json"
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
            write_remote_new(
                sftp,
                contract_path,
                (json.dumps(contract_payload, sort_keys=True, indent=2) + "\n").encode(),
            )
        finally:
            sftp.close()

        expected = {name: digest for name, digest in EXPECTED_INPUTS.items()}
        expected["container_contract.json"] = hashlib.sha256(
            (json.dumps(contract_payload, sort_keys=True, indent=2) + "\n").encode()
        ).hexdigest()
        verify_code = """
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]); expected=json.loads(sys.argv[2])
assert root.is_dir() and not root.is_symlink()
for name,digest in expected.items():
 p=root/name; assert p.is_file() and not p.is_symlink()
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
print('uploaded_input_gate=PASS')
"""
        upload_out, _ = run(
            target,
            "python3 -c "
            + shlex.quote(verify_code)
            + " "
            + shlex.quote(remote_diag)
            + " "
            + shlex.quote(json.dumps(expected, sort_keys=True)),
        )

        container_script = f"""
set -eu
cd {shlex.quote(remote_diag)}
export ASCEND_OPP_PATH={shlex.quote(contract["ascend_opp"])}
python3 {shlex.quote(BUILDER.name)} prepare {shlex.quote(OUTER_ZIP.name)} {shlex.quote(workdir)}
python3 {shlex.quote(BUILDER.name)} build {shlex.quote(workdir)} --opc {shlex.quote(OPC)} --container-contract {shlex.quote(contract_path)} --installed-cloud-root {shlex.quote(contract['installed_cloud_root'])}
"""
        out, _ = run(
            target,
            "docker exec "
            + shlex.quote(CONTAINER)
            + " bash --noprofile --norc -lc "
            + shlex.quote(container_script),
            timeout=600,
        )
        summary_code = """
from pathlib import Path
import json,sys
m=json.loads(Path(sys.argv[1]).read_text())
assert m['status']=='built'
print(json.dumps({'status':m['status'],'candidate_source_sha256':m['candidate']['source_sha256'],'artifacts':{k:{'object_sha256':v['object_sha256'],'object_size':v['object_size'],'json_sha256':v['json_sha256'],'kernel_name':v['kernel_name']} for k,v in m['artifacts'].items()},'installed_inventory_closed':m['build_runtime']['installed_inventory_closed'],'runtime_inventory_closed':m['build_runtime']['runtime_inventory_closed']},sort_keys=True))
"""
        summary, _ = run(
            target,
            "docker exec "
            + shlex.quote(CONTAINER)
            + " python3 -c "
            + shlex.quote(summary_code)
            + " "
            + shlex.quote(workdir + "/release_manifest.json"),
        )
        result = json.loads(summary)
        result["remote_diagnostics_name"] = REMOTE_DIAG_NAME
        result["uploaded_gate"] = "uploaded_input_gate=PASS" in upload_out
        return result
    finally:
        target.close()
        jump.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module = load_remote_module()
    info = local_preflight(module)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "target_private": True,
                    "target_suffix": str(info["target_host"]).split(".")[-1],
                    "container": CONTAINER,
                    "diagnostics_name": REMOTE_DIAG_NAME,
                    "input_sha256": EXPECTED_INPUTS,
                    "actions": ["upload_new", "prepare", "build"],
                    "forbidden": ["package", "install", "train"],
                },
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps(execute(), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, socket.error) as error:
        print(f"STEP357 failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
