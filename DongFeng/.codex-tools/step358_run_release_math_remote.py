#!/usr/bin/env python3
"""Run the audited QrV2 Matmul-position v5 math/identity gate exactly once."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import ipaddress
import json
import shlex
import socket
import sys
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".codex-tools"
REMOTE_EXEC = TOOLS / "remote_exec.py"
AUTHORITY_MAP = Path("/home/l30002999/import-md/hw-import-ip.md")
CONTAINER = "mapqr-leicheng"
EXPECTED_HOSTNAME = "yfzy-zhsc-910c-1.novalocal"
EXPECTED_RELEASE_KERNEL = "QrV2_matmul_position_fix_v5"
V5_RELEASE_READY = True
REMOTE_DIAG_NAME = "step374_qrv2_matmul_position_v5_first192_20260821"
PORT = 34359
REMOTE_WHEEL: str | None = (
    "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/"
    "step373_qrv2_matmul_position_v5_release_20260821/work/release/"
    "mx_driving_cloud-26.0.7+CANN8.3.RC1.A3-cp311-cp311-linux_aarch64.whl"
)
REMOTE_WHEEL_SHA256: str | None = (
    "f20c3db839b669ef6919b2a40df80b475676a9c0149910e0c73eb65064b8c11b"
)
FORBIDDEN_V4_WHEEL_SHA256 = (
    "4c158915bd5ae3fad4834a4f88028702d2d6fb534d69da45cd06f0b536f8dead"
)
FILES = (
    TOOLS / "step358_prepare_release_shadow.py",
    TOOLS / "step358_qrv2_release_math_worker.py",
    TOOLS / "step358_host_case.py",
    TOOLS / "qrv2_release_oracle.py",
    TOOLS / "step343_qrv2_cold_case.py",
    TOOLS / "step343_world8_controller.py",
    *(ROOT / "step260_qr_bad_tensors" / f"rank{rank}_step10_ind0_192x192_BAD.pt" for rank in range(8)),
)
EXPECTED_SHA256 = {
    "step358_prepare_release_shadow.py": "b36107e97c7c2456052d827c66c9da90208d4a3c3e605a0166bd0f657c782f85",
    "step358_qrv2_release_math_worker.py": "f5e3bc0b4e333109c8c3c0003e3467b995fe6a3c061e911704ab06a29bfe10c7",
    "step358_host_case.py": "94e5a46059c0a57bb883999f2648f755158c879568ed3c33b6d4fde8cf1c7070",
    "qrv2_release_oracle.py": "d92e02c3df761ddcc94580836615daa661c0e31c23eb8dc32a25dbd806bf6492",
    "step343_qrv2_cold_case.py": "8a5abcd6e9654fc943847d6695bec1bd71fe2b2558a3ec7b903fe13a4eeb6508",
    "step343_world8_controller.py": "ea0e587cd0b6c1b31fe753e3239a63c91597cee8f4ec917ad08ab7999bb82ce6",
    "rank0_step10_ind0_192x192_BAD.pt": "23ad9198223159fc6aa67f79642c299fd86e0aaa2b7ae72bdea297fcb023ab55",
    "rank1_step10_ind0_192x192_BAD.pt": "2cb99d06aa9c96d61f0b615cf41fa579bd6779f7f97c97fa84693180c32adb5b",
    "rank2_step10_ind0_192x192_BAD.pt": "61dcbad02578e60ce7bb82b837f0b33fff2e0071fbde530a339dcad1ce2a692d",
    "rank3_step10_ind0_192x192_BAD.pt": "89266a246497f51d1c6db5e698ee1442abc91bd48c7dc539a09d2373c21b3ac1",
    "rank4_step10_ind0_192x192_BAD.pt": "e750ddcc8dd892ece49d04873910752c657f6d853f8e698daf03fa3fce3a73ca",
    "rank5_step10_ind0_192x192_BAD.pt": "bbceebf84c574e21e9262774c41e0c8bb5eb7f5add0d0cf123e4efbd6a95dc68",
    "rank6_step10_ind0_192x192_BAD.pt": "f2091ec0c618721ba95452fcca82288a2fc8148f40718f945a9e80646dd1d766",
    "rank7_step10_ind0_192x192_BAD.pt": "3dcc3f2bdb7945eaac7ce246128804dfecd89d381e27dc108e99d90d2df2121c",
}
EXPECTED_INPUT_FILE_SHA256_BY_RANK = [
    EXPECTED_SHA256[f"rank{rank}_step10_ind0_192x192_BAD.pt"]
    for rank in range(8)
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_remote_module() -> Any:
    spec = importlib.util.spec_from_file_location("step358_remote_exec", REMOTE_EXEC)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load remote helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_preflight(module: Any) -> dict[str, object]:
    if not V5_RELEASE_READY:
        raise RuntimeError(
            "QrV2 v5 release wheel has not been built; set the unique diagnostics "
            "directory, wheel path/SHA and helper SHAs only after local build review"
        )
    if EXPECTED_RELEASE_KERNEL != "QrV2_matmul_position_fix_v5":
        raise RuntimeError("QrV2 v5 release kernel identity mismatch")
    if not isinstance(REMOTE_WHEEL, str) or not REMOTE_WHEEL.startswith("/"):
        raise RuntimeError("QrV2 v5 release wheel path is not armed")
    if "step370" in REMOTE_WHEEL or "sync_v4" in REMOTE_WHEEL:
        raise RuntimeError("known QrV2 v4 wheel path is forbidden")
    if (
        not isinstance(REMOTE_WHEEL_SHA256, str)
        or len(REMOTE_WHEEL_SHA256) != 64
        or any(char not in "0123456789abcdef" for char in REMOTE_WHEEL_SHA256)
    ):
        raise RuntimeError("QrV2 v5 release wheel SHA is not armed")
    if REMOTE_WHEEL_SHA256 == FORBIDDEN_V4_WHEEL_SHA256:
        raise RuntimeError("known QrV2 v4 wheel SHA is forbidden")
    AUTHORITY_MAP.read_text(encoding="utf-8")
    info = module.parse_machine_info()
    target = ipaddress.ip_address(str(info["target_host"]))
    if not target.is_private or str(target).split(".")[-1] != "42":
        raise RuntimeError("target mapping must be private and end in 42")
    if info["jump_host"] == info["target_host"]:
        raise RuntimeError("project mapping must contain two distinct hops")
    if {path.name for path in FILES} != set(EXPECTED_SHA256):
        raise RuntimeError("local upload inventory differs from fixed contract")
    for path in FILES:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"local input is not a regular file: {path.name}")
        if sha256_file(path) != EXPECTED_SHA256[path.name]:
            raise RuntimeError(f"local input SHA mismatch: {path.name}")
    return info


def connect_target(module: Any, info: dict[str, object]):
    jump = module.connect(
        str(info["jump_host"]), int(info["jump_port"]),
        str(info["jump_user"]), str(info["jump_password"]),
    )
    try:
        channel = jump.get_transport().open_channel(
            "direct-tcpip",
            (str(info["target_host"]), int(info["target_port"])),
            ("127.0.0.1", 0),
        )
        target = module.connect(
            str(info["target_host"]), int(info["target_port"]),
            str(info["target_user"]), str(info["target_password"]), sock=channel,
        )
    except Exception:
        jump.close()
        raise
    return jump, target


def run(client: Any, command: str, *, timeout: int = 300, check: bool = True):
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError(f"remote command rc={rc}: {err.strip() or out.strip()}")
    return rc, out, err


def host_script(client: Any, script: str, *, timeout: int = 300, check: bool = True):
    return run(
        client,
        "bash --noprofile --norc -lc " + shlex.quote(script),
        timeout=timeout,
        check=check,
    )


def write_new(sftp: Any, path: str, payload: bytes) -> None:
    try:
        sftp.lstat(path)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(path)
    with sftp.open(path, "wx") as stream:
        stream.write(payload)
    sftp.chmod(path, 0o600)


def container_python(client: Any, code: str, *arguments: str, timeout: int = 300):
    command = (
        "docker exec " + shlex.quote(CONTAINER) + " python3 -c " + shlex.quote(code)
        + "".join(" " + shlex.quote(argument) for argument in arguments)
    )
    return run(client, command, timeout=timeout)


INVENTORY_CODE = r'''
import hashlib, importlib.util, json, sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(strict=True)
base=root/'packages/vendors/customize/op_impl/ai_core/tbe/kernel'
result={}
for soc in ('ascend910_93','ascend910b'):
 config=base/'config'/soc/'qr_v2.json'
 payload=json.loads(config.read_text())
 rows=payload['binList']; assert len(rows)==1
 rel=rows[0]['binInfo']['jsonFilePath']
 kernel_json=base/rel
 kernel_o=kernel_json.with_suffix('.o')
 for label,path in (('config',config),('json',kernel_json),('object',kernel_o)):
  assert path.is_file() and not path.is_symlink()
  result[f'{soc}:{label}']={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'size':path.stat().st_size}
print(json.dumps(result,sort_keys=True))
'''


SUMMARY_CODE = r'''
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
expected_input_sha=json.loads(sys.argv[2])
assert isinstance(expected_input_sha,list) and len(expected_input_sha)==8
assert not list((root/'failure').glob('rank*.txt'))
done=[json.loads((root/'done'/f'rank{r}.json').read_text()) for r in range(8)]
identity=[json.loads((root/f'profiler_identity_rank{r}.json').read_text()) for r in range(8)]
controller=json.loads((root/'controller_status.json').read_text())
post=json.loads((root/'postflight_status.json').read_text())
cleanup=json.loads((root/'finally_cleanup_status.json').read_text())
assert controller['status']=='PASS' and post['status']=='PASS' and cleanup['status']=='PASS'
assert controller['physical_device_ids']==list(range(8,16))
assert all(row['rank']==r and row['all_contract_pass'] and row['profiler_identity_pass'] for r,row in enumerate(done))
assert all(row['world_size']==8 and row['local_rank']==r for r,row in enumerate(done))
assert all(row['input_file_sha256']==expected_input_sha[r] for r,row in enumerate(done))
assert all(
 row['first_profiled_only'] is True
 and row['state_diagnostic_only'] is True
 and row['call_count']==1
 and row['eligible_call_count']==1
 and row['mx_qr_call_count']==1
 for row in done
)
assert all(
 row['pass'] is True
 and row['candidate_aic_reference_count']==1
 and row['expected_aic_reference_count']==1
 and row['raw_profile_retained'] is True
 for row in identity
)
assert all(row['eligible_fallback_count']==0 and row['eligible_call_count']==row['mx_qr_call_count'] for row in done)
calls=[row['calls'][0] for row in done]
assert all(
 row['shape']==[192,192]
 and row['dtype']=='torch.float32'
 and row['eligible_mx_branch'] is True
 and row['wrapper_branch']=='mx_fixed'
 and row['expected_padded_shape']==[192,192]
 and row['mx_qr_call_delta']==1
 and row['mx_qr_input']=={'shape':[192,192],'dtype':'torch.float32','contiguous':True}
 and row['input_unmodified'] is True
 and row['shape_pass'] is True
 and row['finite_pass'] is True
 and row['reconstruction']['violation_count']==0
 and row['orthogonality']['violation_count']==0
 for row in calls
)
print(json.dumps({
 'status':'PASS','rank_count':8,
 'call_count_by_rank':[row['call_count'] for row in done],
 'eligible_call_count_by_rank':[row['eligible_call_count'] for row in done],
 'candidate_aic_reference_count_by_rank':[row['candidate_aic_reference_count'] for row in identity],
 'input_file_sha256_by_rank':[row['input_file_sha256'] for row in done],
 'shape_by_rank':[row['shape'] for row in calls],
 'dtype_by_rank':[row['dtype'] for row in calls],
 'eligible_mx_branch_by_rank':[row['eligible_mx_branch'] for row in calls],
 'wrapper_branch_by_rank':[row['wrapper_branch'] for row in calls],
 'mx_qr_input_by_rank':[row['mx_qr_input'] for row in calls],
 'input_unmodified_by_rank':[row['input_unmodified'] for row in calls],
 'shape_pass_by_rank':[row['shape_pass'] for row in calls],
 'finite_pass_by_rank':[row['finite_pass'] for row in calls],
 'reconstruction_violation_count_by_rank':[row['reconstruction']['violation_count'] for row in calls],
 'orthogonality_violation_count_by_rank':[row['orthogonality']['violation_count'] for row in calls],
 'physical_device_ids':controller['physical_device_ids'],
 'postflight':post['status'],'cleanup':cleanup['status'],
 'raw_profile_retained_by_rank':[row['raw_profile_retained'] for row in identity],
 'raw_profiles_retained':all(row['raw_profile_retained'] for row in identity),
},sort_keys=True))
'''


def _raise_primary_with_secondary(
    primary: Exception, secondary: list[str]
) -> None:
    """Preserve the primary failure first and append bounded recovery failures."""

    if not secondary:
        raise primary
    details = "\n".join(f"secondary {item}" for item in secondary)
    raise RuntimeError(f"{type(primary).__name__}: {primary}\n{details}") from primary


def _cleanup_owned_after_outer_failure(
    client: Any, remote_diag: str, *, timeout: int = 120
) -> None:
    """Invoke only the recorded PID/starttime/PGID cleanup protocol."""

    command = (
        "python3 " + shlex.quote(remote_diag + "/step343_world8_controller.py")
        + " --output-dir " + shlex.quote(remote_diag + "/run")
        + f" --port {PORT} --cleanup-owned"
    )
    rc, out, err = host_script(client, command, timeout=timeout, check=False)
    if rc != 0:
        raise RuntimeError(
            f"owned cleanup rc={rc}: {(err.strip() or out.strip())}"
        )


def _run_case_with_recovery(
    client: Any,
    case_command: str,
    installed_cloud: str,
    installed_before: dict[str, Any],
    remote_diag: str,
) -> None:
    """Run once, then retain the first error while performing bounded recovery."""

    primary: Exception | None = None
    secondary: list[str] = []
    try:
        rc, _out, err = host_script(client, case_command, timeout=1000, check=False)
        if rc != 0:
            raise RuntimeError(f"STEP358 host gate rc={rc}: {err.strip()}")
    except Exception as error:
        primary = error

    try:
        _, after_out, _ = container_python(client, INVENTORY_CODE, installed_cloud)
        installed_after = json.loads(after_out)
        if installed_after != installed_before:
            raise RuntimeError("installed QrV2 inventory changed during STEP358")
    except Exception as error:
        if primary is None:
            primary = error
        else:
            secondary.append(
                f"installed inventory check: {type(error).__name__}: {error}"
            )

    if primary is not None:
        try:
            _cleanup_owned_after_outer_failure(client, remote_diag)
        except Exception as error:
            secondary.append(f"owned cleanup: {type(error).__name__}: {error}")
        _raise_primary_with_secondary(primary, secondary)


def execute() -> dict[str, Any]:
    module = load_remote_module()
    info = local_preflight(module)
    jump, target = connect_target(module, info)
    remote_diag = f"{info['shared']}/diagnostics/{REMOTE_DIAG_NAME}"
    remote_diag_path = PurePosixPath(remote_diag)
    if not remote_diag_path.is_absolute() or ".." in remote_diag_path.parts:
        target.close()
        jump.close()
        raise RuntimeError("unsafe remote diagnostics path")
    try:
        _, hostname, _ = run(target, "hostname")
        if hostname.strip() != EXPECTED_HOSTNAME:
            raise RuntimeError("second-hop runtime hostname mismatch")
        probe = f"""
set -eu
rows=$(docker ps -a --filter 'name=^/{CONTAINER}$' --format '{{{{.Names}}}} {{{{.State}}}}')
[ "$rows" = '{CONTAINER} running' ]
[ ! -e {shlex.quote(remote_diag)} ]
[ "$(sha256sum {shlex.quote(REMOTE_WHEEL)} | cut -d' ' -f1)" = '{REMOTE_WHEEL_SHA256}' ]
mkdir -m 700 {shlex.quote(remote_diag)}
mkdir -m 700 {shlex.quote(remote_diag + '/inputs')}
mkdir -m 700 {shlex.quote(remote_diag + '/run')}
"""
        host_script(target, probe)
        sftp = target.open_sftp()
        try:
            for path in FILES:
                directory = "/inputs/" if path.suffix == ".pt" else "/"
                write_new(sftp, remote_diag + directory + path.name, path.read_bytes())
        finally:
            sftp.close()
        expected_remote = {
            ("inputs/" if path.suffix == ".pt" else "") + path.name: EXPECTED_SHA256[path.name]
            for path in FILES
        }
        verify = r'''
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); expected=json.loads(sys.argv[2])
for rel,digest in expected.items():
 p=root/rel; assert p.is_file() and not p.is_symlink()
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
print('PASS')
'''
        _, uploaded, _ = run(
            target,
            "python3 -c " + shlex.quote(verify) + " " + shlex.quote(remote_diag)
            + " " + shlex.quote(json.dumps(expected_remote, sort_keys=True)),
        )
        if uploaded.strip() != "PASS":
            raise RuntimeError("uploaded input gate failed")
        cloud_probe = r'''
import importlib.util
from pathlib import Path
s=importlib.util.find_spec('mx_driving_cloud'); assert s and s.origin
print(Path(s.origin).resolve(strict=True).parent)
'''
        _, cloud_out, _ = container_python(target, cloud_probe)
        installed_cloud = cloud_out.strip()
        if not installed_cloud.startswith("/"):
            raise RuntimeError("installed cloud root probe is invalid")
        _, before_out, _ = container_python(target, INVENTORY_CODE, installed_cloud)
        installed_before = json.loads(before_out)
        prepare_command = (
            "docker exec -e PYTHONPATH=" + shlex.quote(remote_diag) + " " + shlex.quote(CONTAINER)
            + " python3 " + shlex.quote(remote_diag + "/step358_prepare_release_shadow.py")
            + " --wheel " + shlex.quote(REMOTE_WHEEL)
            + " --shadow-root " + shlex.quote(remote_diag + "/shadow")
            + " --manifest " + shlex.quote(remote_diag + "/shadow_manifest.json")
            + " --expected-wheel-sha256 " + REMOTE_WHEEL_SHA256
        )
        run(target, prepare_command, timeout=300)
        installed_custom = installed_cloud + "/packages/vendors/customize"
        case_command = (
            "cd " + shlex.quote(remote_diag) + " && python3 step358_host_case.py"
            + f" --port {PORT} --output-dir " + shlex.quote(remote_diag + "/run")
            + " --input-dir " + shlex.quote(remote_diag + "/inputs")
            + " --worker " + shlex.quote(remote_diag + "/step358_qrv2_release_math_worker.py")
            + " --shadow-root " + shlex.quote(remote_diag + "/shadow")
            + " --installed-custom-opp " + shlex.quote(installed_custom)
            + " --state-diagnostic-only --first-profiled-only"
        )
        _run_case_with_recovery(
            target,
            case_command,
            installed_cloud,
            installed_before,
            remote_diag,
        )
        summary_command = (
            "python3 -c " + shlex.quote(SUMMARY_CODE)
            + " " + shlex.quote(remote_diag + "/run")
            + " " + shlex.quote(json.dumps(EXPECTED_INPUT_FILE_SHA256_BY_RANK))
        )
        _, summary_out, _ = run(target, summary_command)
        summary = json.loads(summary_out)
        summary.update(
            remote_diagnostics_name=REMOTE_DIAG_NAME,
            wheel_sha256=REMOTE_WHEEL_SHA256,
            installed_inventory_unchanged=True,
            uploaded_input_gate=True,
        )
        return summary
    finally:
        target.close()
        jump.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    module = load_remote_module()
    info = local_preflight(module)
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run", "target_private": True,
            "target_suffix": str(info["target_host"]).split(".")[-1],
            "container": CONTAINER, "port": PORT,
            "diagnostics_name": REMOTE_DIAG_NAME,
            "wheel_sha256": REMOTE_WHEEL_SHA256,
            "upload_count": len(FILES),
            "actions": ["prepare_shadow", "rear8_world8_math_identity", "postflight"],
            "forbidden": ["install", "modify_installed", "train", "download_remote_artifacts"],
        }, sort_keys=True))
        return 0
    print(json.dumps(execute(), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, socket.error) as error:
        print(f"STEP358 failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
