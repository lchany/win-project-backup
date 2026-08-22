#!/usr/bin/env python3
"""Prepare and (unless --dry-run) execute isolated STEP347 QrV2 cold A/B remotely."""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import shlex
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paramiko

from remote_exec import connect, parse_machine_info, redact


LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parent
TOOLS = (
    "step343_prepare_overlay.py",
    "step343_world8_controller.py",
    "step343_host_case.py",
    "step343_qrv2_cold_case.py",
)
INPUT_NAMES = tuple(f"rank{rank}_step10_ind0_192x192_BAD.pt" for rank in range(8))
INPUT_DIR = ROOT / "step260_qr_bad_tensors"
EXPECTED_INPUT_SHA256 = {
    "rank0_step10_ind0_192x192_BAD.pt": "23ad9198223159fc6aa67f79642c299fd86e0aaa2b7ae72bdea297fcb023ab55",
    "rank1_step10_ind0_192x192_BAD.pt": "2cb99d06aa9c96d61f0b615cf41fa579bd6779f7f97c97fa84693180c32adb5b",
    "rank2_step10_ind0_192x192_BAD.pt": "61dcbad02578e60ce7bb82b837f0b33fff2e0071fbde530a339dcad1ce2a692d",
    "rank3_step10_ind0_192x192_BAD.pt": "89266a246497f51d1c6db5e698ee1442abc91bd48c7dc539a09d2373c21b3ac1",
    "rank4_step10_ind0_192x192_BAD.pt": "e750ddcc8dd892ece49d04873910752c657f6d853f8e698daf03fa3fce3a73ca",
    "rank5_step10_ind0_192x192_BAD.pt": "bbceebf84c574e21e9262774c41e0c8bb5eb7f5add0d0cf123e4efbd6a95dc68",
    "rank6_step10_ind0_192x192_BAD.pt": "f2091ec0c618721ba95452fcca82288a2fc8148f40718f945a9e80646dd1d766",
    "rank7_step10_ind0_192x192_BAD.pt": "3dcc3f2bdb7945eaac7ce246128804dfecd89d381e27dc108e99d90d2df2121c",
}
RELATIVE_DIAG = "diagnostics/step347_qrv2_cloud_base_restore_cold_ab_20260821"
RELATIVE_STEP260 = "diagnostics/step260_qr_tensor_dump_30step_20260818T194457/qr_tensors"
RELATIVE_CANDIDATE = "diagnostics/step338_mx_qr_lifetime_fix_opc_retry4_20260821/output"
INSTALLED_CUSTOM = (
    "/home/ma-user/anaconda3/envs/PyTorch-2.7.1/lib/python3.11/site-packages/"
    "mx_driving_cloud/packages/vendors/customize"
)
VISIBLE = "8,9,10,11,12,13,14,15"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_contract() -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    for name in INPUT_NAMES:
        path = INPUT_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != EXPECTED_INPUT_SHA256[name]:
            raise RuntimeError(f"local STEP260 SHA mismatch: {name} {actual}")
        files[name] = {"sha256": actual, "bytes": path.stat().st_size}
    for name in TOOLS:
        path = LOCAL / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return {
        "schema": "step347-dry-run-v1",
        "remote_execution": False,
        "training": False,
        "world_size": 8,
        "visible_devices": VISIBLE,
        "modes": ["original", "candidate"],
        "ports": [34341, 34342],
        "candidate_kernel": "QrV2_step338_lifetime_fix",
        "opp_kernel_names": {
            "original": "QrV2_566c2e1c0e6c8c92152ad84416d77006",
            "candidate": "QrV2_step338_lifetime_fix",
        },
        "profiler_identity_gate": (
            "generic QrV2 kernel_details evidence plus hash_dic concrete-name mapping "
            "that is referenced by little-endian uint64 in task_track"
        ),
        "profiler_expected_concrete_names": {
            "original": "QrV2_566c2e1c0e6c8c92152ad84416d77006_0_mix_aic",
            "candidate": "QrV2_step338_lifetime_fix_mix_aic",
        },
        "custom_opp_import_gate": (
            "snapshot requested roles before torch/torch_npu/mx imports; require exact cloud,base prefix; "
            "restore stable-dedup(requested + actual) before any torch.npu API"
        ),
        "candidate_artifact_contract": {
            "o_bytes": 136872, "json_bytes": 2019,
            "sha256": "required exact 64-hex values at --execute; no truncated hash accepted",
        },
        "overlay_file_contract": (
            "exactly six files: two SOC configs plus candidate .o/.json in both SOC artifact dirs"
        ),
        "math_tolerance_basis": "FP32 gamma_n forward-error bound plus same-A torch.linalg.qr oracle",
        "failure_only_remote_bundle": ["A", "Q", "R", "actual", "expected", "diff", "metadata"],
        "timing_excludes_reconstruction_and_dump": True,
        "files": files,
    }


def run(client: paramiko.SSHClient, command: str, timeout: int) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def remote_sha256(client: paramiko.SSHClient, path: str) -> str:
    inner = "sha256sum -- " + shlex.quote(path) + " | awk '{print $1}'"
    command = "bash --noprofile --norc -o pipefail -c " + shlex.quote(inner)
    status, out, err = run(client, command, 60)
    fields = out.strip().splitlines()
    if status != 0 or err.strip() or len(fields) != 1:
        raise RuntimeError("remote sha256sum failed its output contract")
    digest = fields[0]
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError("remote sha256sum returned an invalid digest")
    return digest


def finally_cleanup_command(diag: str, mode: str, port: int) -> str:
    manifest = posixpath.join(diag, "overlay_manifest.json")
    case = posixpath.join(diag, mode)
    controller = posixpath.join(diag, "step343_world8_controller.py")
    return (
        "if test -L " + shlex.quote(manifest) + "; then exit 91; "
        + "elif test -f " + shlex.quote(manifest) + " && test -d " + shlex.quote(case)
        + "; then python3 " + shlex.quote(controller) + " --cleanup-owned --output-dir "
        + shlex.quote(case) + f" --port {port}; fi"
    )


def finalize_remote(
    target: paramiko.SSHClient,
    info: dict[str, object],
    diag: str,
    prepare_succeeded: bool,
    attempted_modes: tuple[str, ...] = (),
) -> list[str]:
    errors: list[str] = []
    ports = {"original": 34341, "candidate": 34342}
    ordered_modes = list(reversed(dict.fromkeys(attempted_modes)))
    ordered_modes.extend(mode for mode in ("candidate", "original") if mode not in ordered_modes)
    for mode in ordered_modes:
        port = ports[mode]
        try:
            status, out, err = run(target, finally_cleanup_command(diag, mode, port), 90)
            print(redact(out + err, info), end="")
            if status != 0:
                errors.append(f"{mode}_owned_cleanup_rc={status}")
        except BaseException as exc:
            errors.append(f"{mode}_owned_cleanup={type(exc).__name__}:{exc}")
    if prepare_succeeded:
        verify_installed_command = (
            "docker exec mapqr-leicheng python3 "
            + shlex.quote(posixpath.join(diag, "step343_prepare_overlay.py"))
            + " --manifest " + shlex.quote(posixpath.join(diag, "overlay_manifest.json"))
            + " --verify-installed"
        )
        try:
            status, out, err = run(target, verify_installed_command, 60)
            print(redact(out + err, info), end="")
            if status != 0:
                errors.append(f"installed_tracked_artifacts_verify_rc={status}")
        except BaseException as exc:
            errors.append(f"installed_tracked_artifacts_verify={type(exc).__name__}:{exc}")
    return errors


def shell_case(diag: str, mode: str, port: int, custom_opp: str) -> str:
    """Build the effective host-side case command; the container only receives torchrun."""
    case = posixpath.join(diag, mode)
    manifest = posixpath.join(diag, "overlay_manifest.json")
    original_substitution = (
        "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))"
        "[\"original_kernel_name\"])' " + shlex.quote(manifest) + ")"
    )
    original = '"' + original_substitution + '"'
    expected = original if mode == "original" else "QrV2_step338_lifetime_fix"
    return (
        "mkdir -p " + " ".join(shlex.quote(posixpath.join(case, name)) for name in ("ready", "done", "failure"))
        + " && python3 " + shlex.quote(posixpath.join(diag, "step343_host_case.py"))
        + f" --mode {mode} --port {port} --output-dir " + shlex.quote(case)
        + " --input-dir " + shlex.quote(posixpath.join(diag, "inputs"))
        + " --worker " + shlex.quote(posixpath.join(diag, "step343_qrv2_cold_case.py"))
        + " --expected-kernel " + expected + " --original-kernel " + original
        + " --custom-opp " + shlex.quote(custom_opp)
        + " --installed-custom-opp " + shlex.quote(INSTALLED_CUSTOM)
        + (
            " --overlay-custom-opp " + shlex.quote(posixpath.join(diag, "overlay"))
            if mode == "candidate" else ""
        )
    )


def execute(args: argparse.Namespace) -> int:
    # parse_machine_info reads 机器IP.md on every invocation and rejects a target not ending in .42.
    info = parse_machine_info()
    shared = str(info["shared"]).rstrip("/")
    diag = posixpath.join(shared, RELATIVE_DIAG)
    step260 = posixpath.join(shared, RELATIVE_STEP260)
    candidate = posixpath.join(shared, RELATIVE_CANDIDATE)
    jump = connect(
        str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"])
    )
    target = None
    prepare_succeeded = False
    attempted_modes: list[str] = []
    try:
        transport = jump.get_transport()
        if transport is None or not transport.is_active():
            raise RuntimeError("jump transport is inactive")
        channel = transport.open_channel(
            "direct-tcpip",
            (str(info["target_host"]), int(info["target_port"])),
            ("127.0.0.1", 0),
        )
        target = connect(
            str(info["target_host"]), int(info["target_port"]), str(info["target_user"]),
            str(info["target_password"]), sock=channel,
        )
        quoted_diag = shlex.quote(diag)
        preflight = f"""set -eu
test "$(hostname)" = yfzy-zhsc-910c-1.novalocal
count=$(docker ps --format '{{{{.Names}}}}' | awk '$0 == "mapqr-leicheng" {{n++}} END {{print n+0}}')
test "$count" -eq 1
test ! -e {quoted_diag}
test "$(stat -c %s {shlex.quote(posixpath.join(candidate, 'QrV2_step338_lifetime_fix.o'))})" -eq 136872
test "$(stat -c %s {shlex.quote(posixpath.join(candidate, 'QrV2_step338_lifetime_fix.json'))})" -eq 2019
mkdir -p {quoted_diag}/inputs
"""
        status, out, err = run(target, preflight, 60)
        print(redact(out + err, info), end="")
        if status != 0:
            raise RuntimeError(f"remote preflight failed rc={status}")

        sftp = target.open_sftp()
        try:
            for name in TOOLS:
                sftp.put(str(LOCAL / name), posixpath.join(diag, name))
            input_manifest: dict[str, dict[str, str]] = {}
            for name in INPUT_NAMES:
                existing = posixpath.join(step260, name)
                destination = posixpath.join(diag, "inputs", name)
                try:
                    attrs = sftp.stat(existing)
                except FileNotFoundError:
                    attrs = None
                if attrs is not None:
                    remote_sha = remote_sha256(target, existing)
                    if remote_sha != EXPECTED_INPUT_SHA256[name]:
                        raise RuntimeError(f"existing remote STEP260 SHA mismatch: {name}")
                    link_cmd = f"ln -s {shlex.quote(existing)} {shlex.quote(destination)}"
                    link_status, _, link_err = run(target, link_cmd, 30)
                    if link_status != 0:
                        raise RuntimeError(f"failed to link existing STEP260 input {name}: {link_err}")
                    source = "existing_remote"
                else:
                    local_path = INPUT_DIR / name
                    if sha256(local_path) != EXPECTED_INPUT_SHA256[name]:
                        raise RuntimeError(f"local STEP260 SHA changed: {name}")
                    sftp.put(str(local_path), destination)
                    source = "uploaded_local_fallback"
                input_manifest[name] = {"sha256": EXPECTED_INPUT_SHA256[name], "source": source}
            with sftp.open(posixpath.join(diag, "input_manifest.json"), "w") as stream:
                stream.write(json.dumps(input_manifest, indent=2, sort_keys=True))
        finally:
            sftp.close()

        expected_lines = "\n".join(
            f"{digest}  {posixpath.join(diag, 'inputs', name)}"
            for name, digest in EXPECTED_INPUT_SHA256.items()
        )
        verify_inputs = "set -eu\nprintf '%s\\n' " + shlex.quote(expected_lines) + " | sha256sum -c -\n"
        status, out, err = run(target, verify_inputs, 60)
        print(redact(out + err, info), end="")
        if status != 0:
            raise RuntimeError(f"remote input SHA gate failed rc={status}")

        prepare = "docker exec mapqr-leicheng bash --noprofile --norc -lc " + shlex.quote(
            'python3 ' + posixpath.join(diag, 'step343_prepare_overlay.py')
            + ' --installed-custom-opp ' + INSTALLED_CUSTOM
            + ' --candidate-dir ' + candidate
            + ' --overlay ' + posixpath.join(diag, 'overlay')
            + ' --manifest ' + posixpath.join(diag, 'overlay_manifest.json')
            + ' --candidate-o-sha256 ' + args.candidate_o_sha256
            + ' --candidate-json-sha256 ' + args.candidate_json_sha256
        )
        status, out, err = run(target, prepare, 120)
        print(redact(out + err, info), end="")
        if status != 0:
            raise RuntimeError(f"remote overlay prepare failed rc={status}")
        prepare_succeeded = True
        harness_manifest = (
            f"mkdir -p {quoted_diag}/original {quoted_diag}/candidate && "
            f"sha256sum {quoted_diag}/step343_*.py {quoted_diag}/input_manifest.json "
            f"{quoted_diag}/overlay_manifest.json >{quoted_diag}/harness_manifest.sha256"
        )
        status, out, err = run(target, harness_manifest, 60)
        print(redact(out + err, info), end="")
        if status != 0:
            raise RuntimeError(f"remote harness manifest failed rc={status}")

        original_opp = INSTALLED_CUSTOM
        candidate_opp = posixpath.join(diag, "overlay") + ":" + INSTALLED_CUSTOM
        for mode, port, opp in (("original", 34341, original_opp), ("candidate", 34342, candidate_opp)):
            attempted_modes.append(mode)
            command = shell_case(diag, mode, port, opp)
            status, out, err = run(target, command, 360)
            print(redact(out + err, info), end="")
            if status != 0:
                raise RuntimeError(f"remote {mode} case failed rc={status}")
            summarize_one = (
                "docker exec mapqr-leicheng python3 "
                + shlex.quote(posixpath.join(diag, "step343_qrv2_cold_case.py"))
                + " --summarize " + shlex.quote(posixpath.join(diag, mode))
            )
            status, out, err = run(target, summarize_one, 60)
            print(redact(out + err, info), end="")
            if status != 0:
                raise RuntimeError(f"remote {mode} summarize failed rc={status}")
        summarize_command = (
            "docker exec mapqr-leicheng python3 "
            + shlex.quote(posixpath.join(diag, "step343_qrv2_cold_case.py"))
            + " --summarize-ab "
            + shlex.quote(diag)
        )
        status, out, err = run(target, summarize_command, 60)
        print(redact(out + err, info), end="")
        if status != 0:
            raise RuntimeError(f"remote A/B summarize failed rc={status}")
        return 0
    finally:
        active_error = sys.exc_info()[1]
        finalization_errors: list[str] = []
        if target is not None:
            finalization_errors = finalize_remote(
                target, info, diag, prepare_succeeded, tuple(attempted_modes)
            )
        if target is not None:
            target.close()
        jump.close()
        if finalization_errors:
            primary = "none" if active_error is None else f"{type(active_error).__name__}:{active_error}"
            raise RuntimeError(
                f"STEP347 primary_error={primary}; finalization_errors={finalization_errors}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--candidate-o-sha256")
    parser.add_argument("--candidate-json-sha256")
    args = parser.parse_args()
    if args.dry_run == args.execute:
        parser.error("choose exactly one of --dry-run or --execute")
    contract = local_contract()
    if args.dry_run:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    for value in (args.candidate_o_sha256, args.candidate_json_sha256):
        if value is None or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            parser.error("--execute requires both exact lowercase 64-hex candidate SHA256 values")
    result = execute(args)
    if result == 0:
        print(
            "step347_remote_result=COLD_VALIDATION_PASS causal_fix_proven=False "
            "raw_profiles_retained_remote=True training_started=False"
        )
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, socket.error, paramiko.SSHException) as exc:
        print(f"step347 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
