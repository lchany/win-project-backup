#!/usr/bin/env python3
"""Local-only structural tests for STEP347; never imports torch_npu or opens sockets."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

from step343_prepare_overlay import (
    CANDIDATE_STEM,
    discover_qrv2_configs,
    prepare,
    verify_installed,
)
from step343_host_case import SUPERVISOR_SIGNALS, install_signal_handlers, restore_signal_handlers
from step343_qrv2_cold_case import (
    CANDIDATE_AIC,
    ORIGINAL_AIC,
    ORIGINAL_KERNEL,
    evaluate_math_contract,
    requested_opp_contract,
    summarize_ab,
    validate_opp_import_transition,
    validate_opp_restoration,
    validate_kernel_argument_contract,
    vendor_root_from_module_file,
    verify_profile_hit,
)
from step343_world8_controller import (
    BACK8_DEVICE_IDS,
    BACK8_PAIRS,
    parse_back8,
    process_starttime,
    same_process_alive,
    terminate_owned,
    validate_ready_opp_transition,
    validate_rank_device_mapping,
)
import step343_remote_cold_ab as remote_cold_ab


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_installed_fixture(root: Path, socs: tuple[str, ...]) -> tuple[Path, dict[str, dict]]:
    installed = root / "installed"
    originals: dict[str, dict] = {}
    kernel_text = json.dumps({
        "kernelName": "QrV2_original", "binFileName": "QrV2_original", "binFileSuffix": ".o"
    })
    for soc in socs:
        config = installed / f"op_impl/ai_core/tbe/config/{soc}/qr_v2.json"
        kernel = installed / f"op_impl/ai_core/tbe/kernel/{soc}/QrV2_original.json"
        config.parent.mkdir(parents=True)
        kernel.parent.mkdir(parents=True)
        original = {
            "binList": [
                {
                    "binInfo": {"jsonFilePath": f"{soc}/Other.json"},
                    "simplifiedKey": "Other/key",
                },
                {
                    "binInfo": {"jsonFilePath": f"{soc}/QrV2_original.json"},
                    "simplifiedKey": "QrV2/key/192",
                },
            ],
        }
        config.write_text(json.dumps(original), encoding="utf-8")
        kernel.write_text(kernel_text, encoding="utf-8")
        kernel.with_name("QrV2_original.o").write_bytes(b"installed-object")
        originals[soc] = original
    return installed, originals


def build_candidate(root: Path) -> tuple[Path, Path, Path]:
    candidate = root / "candidate"
    candidate.mkdir()
    candidate_json = candidate / f"{CANDIDATE_STEM}.json"
    candidate_o = candidate / f"{CANDIDATE_STEM}.o"
    candidate_text = json.dumps({
        "kernelName": CANDIDATE_STEM,
        "binFileName": CANDIDATE_STEM,
        "binFileSuffix": ".o",
    })
    candidate_json.write_text(candidate_text + " " * (2019 - len(candidate_text)), encoding="utf-8")
    candidate_o.write_bytes(b"x" * 136872)
    return candidate, candidate_json, candidate_o


def test_overlay_exact_files_and_config_diff() -> None:
    with tempfile.TemporaryDirectory(prefix="step343_static_") as raw:
        root = Path(raw)
        installed, originals = build_installed_fixture(root, ("ascend910_93", "ascend910b"))
        candidate, candidate_json, candidate_o = build_candidate(root)
        overlay = root / "overlay"
        manifest_path = root / "manifest.json"

        manifest = prepare(
            installed, candidate, overlay, manifest_path, digest(candidate_o), digest(candidate_json)
        )
        assert manifest["schema"] == "step347-overlay-v1"
        files = sorted(path for path in overlay.rglob("*") if path.is_file())
        assert len(files) == 6
        assert sum(path.name == "qr_v2.json" for path in files) == 2
        assert sum(path.name == candidate_json.name for path in files) == 2
        assert sum(path.name == candidate_o.name for path in files) == 2
        assert len(manifest["configs"]) == 2
        assert len(manifest["installed_source_sha256"]) == 6
        assert len(manifest["installed_artifacts"]) == 6
        assert digest(candidate_json) == manifest["candidate_source_files"][candidate_json.name]
        assert digest(candidate_o) == manifest["candidate_source_files"][candidate_o.name]
        for config_rel, record in manifest["configs"].items():
            soc = Path(config_rel).parent.name
            assert record["changed_leaf_paths"][0][-1] == "jsonFilePath"
            changed = json.loads((overlay / config_rel).read_text(encoding="utf-8"))
            assert changed["binList"][0] == originals[soc]["binList"][0]
            assert changed["binList"][1]["simplifiedKey"] == "QrV2/key/192"
            assert changed["binList"][1]["binInfo"]["jsonFilePath"].endswith(candidate_json.name)
            assert record["original_json_before_sha256"] == record["original_json_expected_after_sha256"]
            assert record["original_o_before_sha256"] == record["original_o_expected_after_sha256"]
        verify_installed(manifest_path)
        cli = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "step343_prepare_overlay.py"),
                "--manifest",
                str(manifest_path),
                "--verify-installed",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert cli.returncode == 0, cli.stderr
        assert json.loads(cli.stdout)["installed_tracked_artifacts_gate"] == "PASS"
        missing_cli = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "step343_prepare_overlay.py")],
            check=False,
            capture_output=True,
            text=True,
        )
        assert missing_cli.returncode == 2
        assert "overlay creation requires" in missing_cli.stderr
        tracked_config = installed / next(iter(manifest["configs"]))
        original_mode = tracked_config.stat().st_mode & 0o7777
        tracked_config.chmod(original_mode ^ 0o100)
        try:
            verify_installed(manifest_path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("installed artifact mode mutation must fail")
        tracked_config.chmod(original_mode)
        first_record = next(iter(manifest["configs"].values()))
        tracked_json = installed / first_record["original_json_relative_path"]
        tracked_json_bytes = tracked_json.read_bytes()
        tracked_json_mode = tracked_json.stat().st_mode & 0o7777
        symlink_target = root / "symlink_target.json"
        symlink_target.write_bytes(tracked_json_bytes)
        tracked_json.unlink()
        tracked_json.symlink_to(symlink_target)
        try:
            verify_installed(manifest_path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("installed artifact symlink replacement must fail")
        tracked_json.unlink()
        tracked_json.write_bytes(tracked_json_bytes)
        tracked_json.chmod(tracked_json_mode)
        verify_installed(manifest_path)
        bad_text = json.dumps({
            "kernelName": "wrong", "binFileName": CANDIDATE_STEM, "binFileSuffix": ".o"
        })
        candidate_json.write_text(bad_text + " " * (2019 - len(bad_text)), encoding="utf-8")
        try:
            prepare(
                installed, candidate, root / "bad_overlay", root / "bad_manifest.json",
                digest(candidate_o), digest(candidate_json),
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("candidate kernelName mismatch must fail")
        changed_object = installed / "op_impl/ai_core/tbe/kernel/ascend910b/QrV2_original.o"
        changed_object.write_bytes(b"changed")
        try:
            verify_installed(manifest_path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("installed package mutation must fail the SHA gate")


def test_overlay_requires_exact_two_soc_configs() -> None:
    for socs in (("ascend910_93",), ("ascend910_93", "ascend910b", "ascend910x")):
        with tempfile.TemporaryDirectory(prefix="step343_soc_count_") as raw:
            installed, _ = build_installed_fixture(Path(raw), socs)
            try:
                discover_qrv2_configs(installed)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"SOC config set {socs!r} must fail closed")
    with tempfile.TemporaryDirectory(prefix="step343_soc_duplicate_") as raw:
        root = Path(raw)
        installed, originals = build_installed_fixture(root, ("ascend910_93", "ascend910b"))
        duplicate = installed / "duplicate/config/ascend910_93/qr_v2.json"
        duplicate.parent.mkdir(parents=True)
        duplicate.write_text(json.dumps(originals["ascend910_93"]), encoding="utf-8")
        try:
            discover_qrv2_configs(installed)
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate SOC config suffix must fail closed")


def test_remote_sha_is_summary_only() -> None:
    digest_value = "a" * 64
    with mock.patch.object(remote_cold_ab, "run", return_value=(0, digest_value + "\n", "")) as called:
        assert remote_cold_ab.remote_sha256(object(), "/remote/input.pt") == digest_value
        command = called.call_args.args[1]
        assert "awk" in command and "sha256sum" in command
    for response in ((1, "", "bad"), (0, "", ""), (0, "bad\n", ""), (0, digest_value + "\nextra\n", "")):
        with mock.patch.object(remote_cold_ab, "run", return_value=response):
            try:
                remote_cold_ab.remote_sha256(object(), "/remote/input.pt")
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"invalid remote SHA response must fail: {response!r}")
    remote_source = (Path(__file__).parent / "step343_remote_cold_ab.py").read_text(encoding="utf-8")
    assert 'sftp.open(existing, "rb")' not in remote_source

    responses = [RuntimeError("candidate cleanup transport"), (9, "", ""), (7, "", "")]
    with mock.patch.object(remote_cold_ab, "run", side_effect=responses) as called, mock.patch.object(
        remote_cold_ab, "redact", side_effect=lambda text, _info: text
    ):
        errors = remote_cold_ab.finalize_remote(object(), {}, "/remote/diag", True)
    assert called.call_count == 3
    assert len(errors) == 3
    assert any("candidate_owned_cleanup" in item for item in errors)
    assert any("original_owned_cleanup_rc=9" in item for item in errors)
    assert any("installed_tracked_artifacts_verify_rc=7" in item for item in errors)
    with mock.patch.object(remote_cold_ab, "run", return_value=(0, "", "")) as called, mock.patch.object(
        remote_cold_ab, "redact", side_effect=lambda text, _info: text
    ):
        assert remote_cold_ab.finalize_remote(object(), {}, "/remote/diag", False) == []
    assert called.call_count == 2


def test_owned_cleanup_and_rank_device_fail_closed() -> None:
    rows = [
        {"rank": rank, "local_rank": rank, "container_pid": 1000 + rank}
        for rank in range(8)
    ]
    mapping = validate_rank_device_mapping(rows, {1000 + rank: 8 + rank for rank in range(8)})
    assert [row["physical_device"] for row in mapping] == list(range(8, 16))
    bad = {1000 + rank: 8 + rank for rank in range(8)}
    bad[1003] = 12
    try:
        validate_rank_device_mapping(rows, bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("rank/local_rank to physical device mismatch must fail")

    with tempfile.TemporaryDirectory(prefix="step343_owned_cleanup_") as raw:
        root = Path(raw)
        ownership = {
            "port": 34341,
            "launcher_host_pid": os.getpid(),
            "launcher_starttime": process_starttime(os.getpid()) + 1,
            "launcher_pgid": os.getpid(),
        }
        (root / "launcher_ownership.json").write_text(json.dumps(ownership), encoding="utf-8")
        with mock.patch("step343_world8_controller.os.kill") as kill, mock.patch(
            "step343_world8_controller.os.killpg"
        ) as killpg:
            assert terminate_owned(root, 34341) == os.getpid()
            kill.assert_not_called()
            killpg.assert_not_called()
        owned_pid = 424242
        ownership.update(
            launcher_host_pid=owned_pid,
            launcher_starttime=777,
            launcher_pgid=owned_pid,
        )
        (root / "launcher_ownership.json").write_text(json.dumps(ownership), encoding="utf-8")
        with mock.patch(
            "step343_world8_controller.same_process_alive", return_value=True
        ), mock.patch(
            "step343_world8_controller.os.getpgid", return_value=owned_pid
        ), mock.patch(
            "step343_world8_controller.time.monotonic", side_effect=(0.0, 6.0)
        ), mock.patch("step343_world8_controller.os.kill") as kill, mock.patch(
            "step343_world8_controller.os.killpg"
        ) as killpg:
            assert terminate_owned(root, 34341) == owned_pid
            assert killpg.call_args_list == [
                mock.call(owned_pid, signal.SIGTERM),
                mock.call(owned_pid, signal.SIGKILL),
            ]
            kill.assert_not_called()
        (root / "launcher_ownership.json").unlink()
        (root / "real.json").write_text("{}", encoding="utf-8")
        (root / "launcher_ownership.json").symlink_to(root / "real.json")
        try:
            terminate_owned(root, 34341)
        except RuntimeError:
            pass
        else:
            raise AssertionError("symlink ownership manifest must fail without killing")


def test_custom_opp_import_rewrite_and_restore_gate() -> None:
    with tempfile.TemporaryDirectory(prefix="step347_vendor_root_") as raw:
        package = Path(raw) / "mx_package"
        module_file = package / "__init__.py"
        vendor = package / "packages/vendors/customize"
        vendor.mkdir(parents=True)
        module_file.write_text("", encoding="utf-8")
        assert vendor_root_from_module_file(str(module_file)) == str(vendor)

    cases = (
        ("original", "/vendor/cloud", "/vendor/cloud:/vendor/base:/vendor/cloud", "/vendor/cloud:/vendor/base"),
        (
            "candidate",
            "/diag/overlay:/vendor/cloud",
            "/vendor/cloud:/vendor/base:/diag/overlay:/vendor/cloud",
            "/diag/overlay:/vendor/cloud:/vendor/base",
        ),
    )
    for mode, requested, after_import, desired in cases:
        overlay = "/diag/overlay" if mode == "candidate" else None
        contract = requested_opp_contract(mode, requested, "/vendor/cloud", overlay)
        imported = validate_opp_import_transition(
            contract, after_import, "/vendor/cloud", "/vendor/base"
        )
        transition = validate_opp_restoration(contract, imported, desired)
        row = {"mode": mode, "custom_opp_transition": transition}
        assert validate_ready_opp_transition(row)
        serialized = json.dumps(transition, sort_keys=True)
        assert all(path not in serialized for path in ("/vendor/cloud", "/vendor/base", "/diag/overlay"))

    for mode, requested, installed, overlay in (
        ("original", "/overlay:/installed", "/installed", None),
        ("candidate", "/same:/same", "/same", "/same"),
        ("candidate", "/overlay::/installed", "/installed", "/overlay"),
        ("candidate", "/wrong:/installed", "/installed", "/overlay"),
    ):
        try:
            requested_opp_contract(mode, requested, installed, overlay)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid requested role sequence must fail: {mode} {requested}")

    contract = requested_opp_contract(
        "candidate", "/overlay:/installed", "/installed", "/overlay"
    )
    invalid_imports = (
        ("/installed:/base:/overlay:/installed", "/installed", ""),
        ("/base:/installed:/overlay:/installed", "/installed", "/base"),
        ("/installed:/base:/overlay:/installed", "/installed", "/installed"),
        ("/installed:/base:/overlay:/installed", "/installed", "/overlay"),
        ("/installed:/base:/unknown:/installed", "/installed", "/base"),
    )
    for after_import, cloud, base in invalid_imports:
        try:
            validate_opp_import_transition(contract, after_import, cloud, base)
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing/wrong/duplicate/unknown import rewrite must fail")
    imported = validate_opp_import_transition(
        contract, "/installed:/base:/overlay:/installed", "/installed", "/base"
    )
    try:
        validate_opp_restoration(contract, imported, "/installed:/overlay:/installed")
    except RuntimeError:
        pass
    else:
        raise AssertionError("failure to restore exact requested OPP must fail")

    worker_source = (Path(__file__).parent / "step343_qrv2_cold_case.py").read_text(encoding="utf-8")
    try_index = worker_source.index("    try:\n", worker_source.index("def run(args:"))
    static_gate_index = worker_source.index("world_size == 8", try_index)
    snapshot_index = worker_source.index('requested_raw = os.environ.get("ASCEND_CUSTOM_OPP_PATH")')
    import_index = worker_source.index("        import torch\n", snapshot_index)
    restore_index = worker_source.index('os.environ["ASCEND_CUSTOM_OPP_PATH"] =', import_index)
    first_npu_api_index = worker_source.index("torch.npu.is_available()", restore_index)
    assert try_index < static_gate_index < snapshot_index < import_index < restore_index < first_npu_api_index
    generated = remote_cold_ab.shell_case(
        "/remote/step347", "original", 34341, "/installed"
    )
    assert '--original-kernel "$(' in generated
    assert "--installed-custom-opp" in generated
    assert ORIGINAL_AIC == ORIGINAL_KERNEL + "_0_mix_aic"
    assert CANDIDATE_AIC == CANDIDATE_STEM + "_mix_aic"
    validate_kernel_argument_contract("original", ORIGINAL_KERNEL, ORIGINAL_KERNEL)
    validate_kernel_argument_contract(
        "candidate", CANDIDATE_STEM, ORIGINAL_KERNEL
    )
    for mode, expected, original in (
        ("original", CANDIDATE_STEM, ORIGINAL_KERNEL),
        ("candidate", ORIGINAL_KERNEL, ORIGINAL_KERNEL),
        ("original", ORIGINAL_KERNEL + "_0", ORIGINAL_KERNEL + "_0"),
        ("candidate", CANDIDATE_STEM, "wrong_original"),
    ):
        try:
            validate_kernel_argument_contract(mode, expected, original)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid kernel argument contract must fail")


def hash_record(hash_value: int, name: str) -> bytes:
    return f"{hash_value}:{name}\n".encode("utf-8")


def task_record(hash_value: int) -> bytes:
    record = bytearray(64)
    record[40:48] = hash_value.to_bytes(8, "little")
    return bytes(record)


def build_profile_fixture(
    root: Path, mappings: list[tuple[int, str]], references: list[int]
) -> tuple[Path, Path]:
    (root / "kernel_details.csv").write_text(
        "Name,Duration(us)\nQrV2,1.0\nOtherKernel,2.0\n", encoding="utf-8"
    )
    data = root / "rank0/ASCEND_PROFILER_OUTPUT/host/data"
    data.mkdir(parents=True)
    dictionary = data / "unaging.additional.hash_dic.slice_0"
    task = data / "aging.compact.task_track.slice_0"
    dictionary.write_bytes(b"".join(hash_record(*entry) for entry in mappings))
    task.write_bytes(b"".join(task_record(value) for value in references))
    return dictionary, task


def assert_profile_gate_fails(root: Path, mode: str) -> None:
    try:
        verify_profile_hit(root, mode)
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"profiler runtime identity gate must fail for {mode}")


def test_profile_hit_and_npu_smi_parsers() -> None:
    original_hash = 0x1111222233334444
    candidate_hash = 0xAAAABBBBCCCCDDDD
    original_aiv_hash = 0x1212121212121212
    original_aiv = ORIGINAL_AIC.removesuffix("aic") + "aiv"
    with tempfile.TemporaryDirectory(prefix="step343_profile_original_") as raw:
        root = Path(raw)
        build_profile_fixture(root, [(original_hash, ORIGINAL_AIC)], [original_hash])
        hit = verify_profile_hit(root, "original")
        assert hit["pass"] is True and hit["reference_count"] == 1
        assert hit["matched_entry"]["name"] == ORIGINAL_AIC
        assert hit["hash_dictionary_sources"] == [
            "rank0/ASCEND_PROFILER_OUTPUT/host/data/unaging.additional.hash_dic.slice_0"
        ]

    with tempfile.TemporaryDirectory(prefix="step343_profile_candidate_") as raw:
        root = Path(raw)
        dictionary, task = build_profile_fixture(
            root,
            [(candidate_hash, CANDIDATE_AIC), (original_hash, ORIGINAL_AIC), (original_aiv_hash, original_aiv)],
            [candidate_hash],
        )
        dictionary.with_name(dictionary.name + ".done").write_bytes(b"malformed")
        task.with_name(task.name + ".done").write_bytes(b"malformed")
        assert verify_profile_hit(root, "candidate")["reference_count"] == 1

    with tempfile.TemporaryDirectory(prefix="step343_profile_original_forbidden_") as raw:
        root = Path(raw)
        build_profile_fixture(
            root,
            [(original_hash, ORIGINAL_AIC), (candidate_hash, CANDIDATE_AIC)],
            [original_hash, candidate_hash],
        )
        assert_profile_gate_fails(root, "original")

    with tempfile.TemporaryDirectory(prefix="step343_profile_unreferenced_") as raw:
        root = Path(raw)
        build_profile_fixture(root, [(candidate_hash, CANDIDATE_AIC)], [0x9999999999999999])
        assert_profile_gate_fails(root, "candidate")

    with tempfile.TemporaryDirectory(prefix="step343_profile_forbidden_") as raw:
        root = Path(raw)
        build_profile_fixture(
            root,
            [
                (candidate_hash, CANDIDATE_AIC),
                (original_hash, ORIGINAL_AIC),
                (original_aiv_hash, original_aiv),
            ],
            [candidate_hash, original_aiv_hash],
        )
        assert_profile_gate_fails(root, "candidate")

    with tempfile.TemporaryDirectory(prefix="step343_profile_conflict_") as raw:
        root = Path(raw)
        dictionary, _ = build_profile_fixture(root, [(candidate_hash, CANDIDATE_AIC)], [candidate_hash])
        dictionary.with_name("unaging.additional.hash_dic.slice_1").write_bytes(
            hash_record(candidate_hash, ORIGINAL_AIC)
        )
        assert_profile_gate_fails(root, "candidate")

    with tempfile.TemporaryDirectory(prefix="step343_profile_truncated_") as raw:
        root = Path(raw)
        dictionary, _ = build_profile_fixture(root, [(candidate_hash, CANDIDATE_AIC)], [candidate_hash])
        dictionary.write_bytes(f"{candidate_hash}:{CANDIDATE_AIC}".encode("utf-8"))
        assert_profile_gate_fails(root, "candidate")

    with tempfile.TemporaryDirectory(prefix="step343_task_truncated_") as raw:
        root = Path(raw)
        _, task = build_profile_fixture(root, [(candidate_hash, CANDIDATE_AIC)], [candidate_hash])
        task.write_bytes(task.read_bytes() + b"x")
        assert_profile_gate_fails(root, "candidate")

    with tempfile.TemporaryDirectory(prefix="step343_profile_missing_") as raw:
        root = Path(raw)
        (root / "kernel_details.csv").write_text("Name\nQrV2\n", encoding="utf-8")
        assert_profile_gate_fails(root, "original")

    synthetic = "\n".join(
        f"| {physical}       {chip}                 | {20000 + physical * 2 + chip}       | rank | 133 |"
        for physical, chip in sorted(BACK8_PAIRS)
    )
    assert {(a, b) for a, b, _ in parse_back8(synthetic)} == BACK8_PAIRS
    assert {a * 2 + b for a, b, _ in parse_back8(synthetic)} == BACK8_DEVICE_IDS == set(range(8, 16))
    starttime = process_starttime(os.getpid())
    assert starttime > 0 and same_process_alive(os.getpid(), starttime)
    assert not same_process_alive(os.getpid(), starttime + 1)


def test_ab_summary_inventory() -> None:
    with tempfile.TemporaryDirectory(prefix="step343_ab_") as raw:
        root = Path(raw)
        for mode, elapsed in (("original", 2.0), ("candidate", 1.5)):
            directory = root / mode
            directory.mkdir()
            rows = []
            for rank in range(8):
                rows.append({
                    "rank": rank,
                    "input_file": f"rank{rank}_step10_ind0_192x192_BAD.pt",
                    "qr_elapsed_ms": elapsed,
                    "A": {"nan": 0},
                    "Q": {"nan": 0},
                    "R": {"nan": 0},
                    "tensor_stats": {},
                    "math_contract_pass": mode == "candidate",
                    "input_a_content_sha256_before": f"sha-rank-{rank}",
                    "input_a_content_sha256_after": f"sha-rank-{rank}",
                    "input_a_unmodified": True,
                    "tolerance_basis": "oracle", "fp32_unit_roundoff": 1.0e-8,
                    "matrix_n": 192, "gamma_n": 1.0e-5,
                    "tolerance_mismatch_counts": {"reconstruction": 0, "orthogonality": 0, "lower_triangle": 0},
                    "first_anomaly": None,
                    "recon_max_abs": 0.0,
                    "recon_max_scaled_relative": 0.0,
                    "recon_l2_relative": 0.0,
                    "orthogonality_max_abs": 0.0,
                    "r_lower_max_abs": 0.0,
                    "failure_bundle": None,
                    "full_tensor_log": None,
                })
            payload = {
                "mode": mode,
                "rank_count": 8,
                "profiler_hit": {"pass": True, "expected_kernel": mode},
                "all_a_finite": True,
                "all_q_finite": mode == "candidate",
                "all_r_finite": mode == "candidate",
                "all_math_contract_pass": mode == "candidate",
                "all_input_a_unmodified": True,
                "rows": rows,
            }
            (directory / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
        assert summarize_ab(root) == 0
        comparison = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
        assert len(comparison["rows"]) == 8
        assert comparison["decision"] == "COLD_VALIDATION_PASS"
        assert comparison["observed_improvement"] is True
        assert comparison["causal_fix_proven"] is False
        assert all(row["candidate_minus_original_ms"] == -0.5 for row in comparison["rows"])
        original_summary_path = root / "original" / "summary.json"
        original_summary = json.loads(original_summary_path.read_text(encoding="utf-8"))
        original_summary.update(
            all_q_finite=True, all_r_finite=True, all_math_contract_pass=True
        )
        original_summary_path.write_text(json.dumps(original_summary), encoding="utf-8")
        assert summarize_ab(root) == 0
        comparison = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
        assert comparison["decision"] == "COLD_VALIDATION_PASS"
        assert comparison["observed_improvement"] is False
        environment = os.environ.copy()
        environment["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        summarize_cli = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "step343_qrv2_cold_case.py"),
                "--summarize-ab",
                str(root),
            ],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert summarize_cli.returncode == 0, summarize_cli.stderr
        assert json.loads(summarize_cli.stdout)["ab_summary"] == "COLD_VALIDATION_PASS"


def test_ab_summary_rejects_candidate_math_failure() -> None:
    with tempfile.TemporaryDirectory(prefix="step343_ab_fail_") as raw:
        root = Path(raw)
        for mode in ("original", "candidate"):
            directory = root / mode
            directory.mkdir()
            rows = []
            for rank in range(8):
                rows.append({
                    "rank": rank,
                    "input_file": f"rank{rank}_step10_ind0_192x192_BAD.pt",
                    "qr_elapsed_ms": 1.0,
                    "A": {"nan": 0}, "Q": {"nan": 0}, "R": {"nan": 0},
                    "tensor_stats": {}, "math_contract_pass": mode == "original",
                    "input_a_content_sha256_before": f"sha-rank-{rank}",
                    "input_a_content_sha256_after": f"sha-rank-{rank}",
                    "input_a_unmodified": True,
                    "tolerance_basis": "oracle", "fp32_unit_roundoff": 1.0e-8,
                    "matrix_n": 192, "gamma_n": 1.0e-5,
                    "tolerance_mismatch_counts": {"reconstruction": 0 if mode == "original" else 1, "orthogonality": 0, "lower_triangle": 0},
                    "first_anomaly": None, "recon_max_abs": 0.0,
                    "recon_max_scaled_relative": 0.0, "recon_l2_relative": 0.0,
                    "orthogonality_max_abs": 0.0, "r_lower_max_abs": 0.0,
                    "failure_bundle": None, "full_tensor_log": None,
                })
            payload = {
                "mode": mode, "rank_count": 8, "profiler_hit": {"pass": True},
                "all_a_finite": True, "all_q_finite": True, "all_r_finite": True,
                "all_math_contract_pass": mode == "original", "rows": rows,
                "all_input_a_unmodified": True,
            }
            (directory / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
        assert summarize_ab(root) != 0
        comparison = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
        assert comparison["decision"] == "COLD_VALIDATION_FAIL"
        assert comparison["failure_reasons"] == ["candidate_math_contract_failed"]
        candidate_path = root / "candidate" / "summary.json"
        candidate_summary = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate_summary["rows"][0]["input_a_content_sha256_before"] = "different"
        candidate_path.write_text(json.dumps(candidate_summary), encoding="utf-8")
        try:
            summarize_ab(root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("A/B loaded-A SHA mismatch must fail")


def test_failure_bundle_is_only_written_on_contract_failure() -> None:
    import torch

    with tempfile.TemporaryDirectory(prefix="step343_bundle_") as raw:
        root = Path(raw)
        identity = torch.eye(2, dtype=torch.float32)
        metadata = {"rank": 0, "mode": "candidate"}
        passed = evaluate_math_contract(identity, identity, identity, identity, root, metadata)
        assert passed["math_contract_pass"] is True
        assert not (root / "failure_bundles").exists()

        bad_q = identity.clone()
        bad_q[0, 0] = torch.nan
        actual = bad_q @ identity
        failed = evaluate_math_contract(identity, bad_q, identity, actual, root, metadata)
        assert failed["math_contract_pass"] is False
        assert failed["first_anomaly"] == {"tensor": "Q", "coordinate": [0, 0], "value": "nan"}
        bundle = Path(failed["failure_bundle"])
        log = Path(failed["full_tensor_log"])
        assert bundle.is_file() and log.is_file()
        text = log.read_text(encoding="utf-8")
        for label in ("INPUT_A", "Q", "R", "ACTUAL_QR", "EXPECTED_A", "DIFF"):
            assert f"\n{label}\n" in text
        saved = torch.load(bundle, map_location="cpu", weights_only=False)
        assert set(saved) == {"INPUT_A", "Q", "R", "ACTUAL_QR", "EXPECTED_A", "DIFF", "metadata"}
        assert saved["metadata"]["failure_bundle"] == str(bundle)
        assert saved["metadata"]["full_tensor_log"] == str(log)

    with tempfile.TemporaryDirectory(prefix="step343_orth_") as raw:
        root = Path(raw)
        identity = torch.eye(2, dtype=torch.float32)
        q = torch.diag(torch.tensor([2.0, 1.0]))
        r = torch.diag(torch.tensor([0.5, 1.0]))
        failed = evaluate_math_contract(identity, q, r, q @ r, root, {"rank": 0, "mode": "candidate"})
        assert failed["math_contract_pass"] is False
        assert failed["tolerance_mismatch_counts"]["orthogonality"] > 0

    with tempfile.TemporaryDirectory(prefix="step343_lower_") as raw:
        root = Path(raw)
        q = torch.eye(2, dtype=torch.float32)
        r = torch.tensor([[1.0, 0.0], [0.25, 1.0]], dtype=torch.float32)
        failed = evaluate_math_contract(r, q, r, q @ r, root, {"rank": 0, "mode": "candidate"})
        assert failed["math_contract_pass"] is False
        assert failed["tolerance_mismatch_counts"]["lower_triangle"] > 0


def main() -> int:
    test_overlay_exact_files_and_config_diff()
    test_overlay_requires_exact_two_soc_configs()
    test_remote_sha_is_summary_only()
    test_owned_cleanup_and_rank_device_fail_closed()
    test_custom_opp_import_rewrite_and_restore_gate()
    test_profile_hit_and_npu_smi_parsers()
    test_ab_summary_inventory()
    test_ab_summary_rejects_candidate_math_failure()
    test_failure_bundle_is_only_written_on_contract_failure()
    host_source = (Path(__file__).parent / "step343_host_case.py").read_text(encoding="utf-8")
    assert '"timeout", "--signal=TERM"' in host_source
    assert "start_new_session=True" in host_source
    assert '"docker", "exec"' in host_source
    assert "launcher_ownership.json" in host_source
    assert "SUPERVISOR_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)" in host_source
    assert "previous_handlers = install_signal_handlers()" in host_source
    assert "restore_signal_handlers(previous_handlers)" in host_source
    assert "finally:" in host_source and "terminate_group(process)" in host_source
    original_handlers = {signum: signal.getsignal(signum) for signum in SUPERVISOR_SIGNALS}
    installed_previous = install_signal_handlers()
    try:
        assert installed_previous == original_handlers
        assert all(signal.getsignal(signum) != original_handlers[signum] for signum in SUPERVISOR_SIGNALS)
    finally:
        restore_signal_handlers(installed_previous)
    assert all(signal.getsignal(signum) == original_handlers[signum] for signum in SUPERVISOR_SIGNALS)
    print("step347_static_tests=PASS remote_execution=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
