#!/usr/bin/env python3
"""Offline tests for the STEP377 diagnostic-only shadow builder."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import step377_prepare_diagnostic_shadow as shadow


class Step377ShadowTests(unittest.TestCase):
    @staticmethod
    def _replace_wheel_member(wheel: Path, name: str, payload: bytes) -> None:
        replacement = wheel.with_suffix(".replacement")
        with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(replacement, "w") as target:
            found = False
            for info in source.infolist():
                if info.filename == name:
                    target.writestr(info, payload)
                    found = True
                else:
                    target.writestr(info, source.read(info))
        if not found:
            raise AssertionError(f"fixture member missing: {name}")
        os.replace(replacement, wheel)

    def _fixture(self, root: Path) -> tuple[Path, Path, dict]:
        work = root / "work"
        wheel = work / "wheel_original/original.whl"
        wheel.parent.mkdir(parents=True)
        files = {
            "mx_driving_cloud/__init__.py": b"",
            "mx_driving_cloud/ops/linalg.py": b"old",
            "mx_driving_cloud-1.dist-info/RECORD": b"immutable-record\n",
        }
        for soc in shadow.SOCS:
            base = "mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/kernel/"
            files[base + f"{soc}/qr_v2/QrV2_old.json"] = b'{"kernelName":"old"}'
            files[base + f"{soc}/qr_v2/QrV2_old.o"] = b"old-object"
            files[base + f"config/{soc}/qr_v2.json"] = json.dumps(
                {"binList": [{"binInfo": {"jsonFilePath": f"{soc}/qr_v2/QrV2_old.json"}}]}
            ).encode()
            files[base + f"config/{soc}/binary_info_config.json"] = json.dumps(
                {"Other": {"keep": True}, "QrV2": {"binaryList": [
                    {"coreType": 0, "binPath": f"{soc}/qr_v2/QrV2_old.o"},
                    {"coreType": 0, "binPath": f"{soc}/qr_v2/QrV2_old.o"},
                ]}}
            ).encode()
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, payload in files.items():
                archive.writestr(name, payload)
        artifacts = {}
        build = work / "build"
        identity = shadow.EXPECTED_IDENTITY
        for soc in shadow.SOCS:
            output = build / soc
            output.mkdir(parents=True)
            meta = output / f"{identity}.json"
            obj = output / f"{identity}.o"
            meta.write_text(
                json.dumps({"kernelName": identity, "binFileName": identity}),
                encoding="utf-8",
            )
            obj.write_bytes(b"candidate-object")
            artifacts[soc] = {
                "json_path": str(meta.resolve()),
                "json_size": meta.stat().st_size,
                "json_sha256": shadow.sha256_file(meta),
                "object_path": str(obj.resolve()),
                "object_size": obj.stat().st_size,
                "object_sha256": shadow.sha256_file(obj),
                "kernel_name": identity,
                "bin_file_name": identity,
                "concrete_entries": sorted(
                    (identity + "_0_mix_aic", identity + "_0_mix_aiv")
                ),
            }
        flags = {
            "artifact_class": "diagnostic_probe",
            "diagnostic_only": True,
            "release_candidate": False,
            "package_forbidden": True,
        }
        full_policy = {
            "original_archives_read_only": True,
            "installed_package_inventory": "required_at_build",
            "required_container": "mapqr-leicheng",
            "soc_artifacts_built_independently": False,
            "soc_alias_contract": {
                "npu_arch": "DAV_2201",
                "canonical": shadow.SOCS[0],
                "alias": shadow.SOCS[1],
                "byte_identical_required": True,
            },
            **flags,
        }
        manifest = {
            "status": "diagnostic_built_unvalidated",
            "policy": full_policy,
            "package": {"status": "forbidden_diagnostic_probe"},
            "candidate": {
                "identity": identity,
                "source_sha256": shadow.EXPECTED_SOURCE_SHA256,
                "reverse_v4_sha256": shadow.EXPECTED_REVERSE_V4_SHA256,
                **flags,
            },
            "tools": dict(shadow.EXPECTED_TOOLS),
            "immutable_guards": {"extracted_original_wheel": {
                "path": str(wheel.resolve()), "sha256": shadow.sha256_file(wheel)
            }},
            "artifacts": artifacts,
        }
        manifest_path = work / "release_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, wheel, manifest

    def _run(self, root: Path):
        manifest, wheel, _ = self._fixture(root)
        approved = root / "approved"
        approved.mkdir()
        output = approved / "shadow_manifest.json"
        return shadow.prepare(manifest, wheel, approved, output), approved, output

    def test_success_replaces_only_qrv2_artifacts_and_configs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, approved, output = self._run(Path(directory))
            self.assertEqual(result["status"], shadow.STATUS)
            self.assertTrue(result["diagnostic_only"])
            self.assertTrue(result["package_forbidden"])
            self.assertFalse(result["source_overlay"])
            self.assertTrue(result["record_unchanged"])
            self.assertEqual(json.loads(output.read_text()), result)
            package = approved / "shadow/mx_driving_cloud"
            self.assertEqual((package / "ops/linalg.py").read_bytes(), b"old")
            self.assertEqual(
                (approved / "shadow/mx_driving_cloud-1.dist-info/RECORD").read_bytes(),
                b"immutable-record\n",
            )
            for soc in shadow.SOCS:
                artifact = result["artifacts"][soc]
                self.assertEqual(Path(artifact["object_path"]).read_bytes(), b"candidate-object")
                kernel_dir = Path(artifact["object_path"]).parent
                self.assertEqual(
                    sorted(path.name for path in kernel_dir.iterdir()),
                    [result["candidate_identity"] + ".json", result["candidate_identity"] + ".o"],
                )
                config = package / f"packages/vendors/customize/op_impl/ai_core/tbe/kernel/config/{soc}/qr_v2.json"
                self.assertEqual(
                    json.loads(config.read_text())["binList"][0]["binInfo"]["jsonFilePath"],
                    f"{soc}/qr_v2/{result['candidate_identity']}.json",
                )
                binary = config.with_name("binary_info_config.json")
                value = json.loads(binary.read_text())
                self.assertEqual(value["Other"], {"keep": True})
                self.assertEqual(
                    {row["binPath"] for row in value["QrV2"]["binaryList"]},
                    {f"{soc}/qr_v2/{result['candidate_identity']}.o"},
                )
            self.assertFalse((approved / "release").exists())
            self.assertFalse(any(approved.rglob("*.whl")))
            self.assertFalse(any(approved.rglob("*.zip")))

    def test_rejects_input_drift_and_bad_manifest_contracts(self) -> None:
        mutations = (
            lambda manifest: manifest.update({"status": "built"}),
            lambda manifest: manifest["immutable_guards"]["extracted_original_wheel"].update({"sha256": "0" * 64}),
            lambda manifest: manifest["candidate"].update({"identity": ""}),
            lambda manifest: manifest["candidate"].update({"reverse_v4_sha256": "0" * 64}),
            lambda manifest: manifest["tools"].update({"v4_patcher_sha256": "0" * 64}),
            lambda manifest: manifest["artifacts"][shadow.SOCS[0]].update({"object_sha256": "0" * 64}),
            lambda manifest: manifest["artifacts"][shadow.SOCS[1]].update({"json_sha256": "0" * 64}),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path, wheel, manifest = self._fixture(root)
                mutate(manifest)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                approved = root / "approved"
                approved.mkdir()
                with self.assertRaises(RuntimeError):
                    shadow.prepare(manifest_path, wheel, approved, approved / "out.json")

    def test_rejects_symlink_inputs_roots_outputs_and_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, wheel, _ = self._fixture(root)
            approved = root / "approved"
            approved.mkdir()
            wheel_link = root / "wheel-link"
            wheel_link.symlink_to(wheel)
            with self.assertRaises(RuntimeError):
                shadow.prepare(manifest, wheel_link, approved, approved / "out.json")
            root_link = root / "approved-link"
            root_link.symlink_to(approved, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                shadow.prepare(manifest, wheel, root_link, root_link / "out.json")
            output_link = approved / "out.json"
            output_link.symlink_to(root / "missing")
            with self.assertRaises(RuntimeError):
                shadow.prepare(manifest, wheel, approved, output_link)
            output_link.unlink()
            (approved / "shadow").mkdir()
            with self.assertRaises(FileExistsError):
                shadow.prepare(manifest, wheel, approved, approved / "out.json")

    def test_safe_extraction_rejects_escape_duplicate_symlink_and_o_excl_collision(self) -> None:
        cases = ("escape", "duplicate", "symlink")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / "bad.whl"
                with zipfile.ZipFile(archive, "w") as wheel:
                    if case == "escape":
                        wheel.writestr("../escape", b"x")
                    elif case == "duplicate":
                        wheel.writestr("same", b"x")
                        wheel.writestr("same", b"y")
                    else:
                        info = zipfile.ZipInfo("link")
                        info.external_attr = (stat.S_IFLNK | 0o777) << 16
                        wheel.writestr(info, "target")
                with self.assertRaises(RuntimeError):
                    shadow._extract_wheel(archive, root / "shadow")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing"
            path.write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                shadow._write_new(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"original")

    def test_pre_and_post_input_hashes_are_locked(self) -> None:
        for target_kind in ("wheel", "artifact"):
            with self.subTest(target_kind=target_kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, wheel, source = self._fixture(root)
                approved = root / "approved"
                approved.mkdir()
                real_update = shadow._update_soc
                calls = 0

                def tamper(*args, **kwargs):
                    nonlocal calls
                    result = real_update(*args, **kwargs)
                    calls += 1
                    if calls == 2:
                        target = wheel if target_kind == "wheel" else Path(
                            source["artifacts"][shadow.SOCS[0]]["object_path"]
                        )
                        target.write_bytes(b"tampered")
                    return result

                with mock.patch.object(shadow, "_update_soc", side_effect=tamper):
                    with self.assertRaisesRegex(RuntimeError, "immutable STEP376 input changed"):
                        shadow.prepare(manifest, wheel, approved, approved / "out.json")
                self.assertFalse((approved / "out.json").exists())

    def test_rejects_artifact_escape_and_nonunique_target_contracts(self) -> None:
        mutations = ("escape", "extra-old", "config-rows", "binary-rows")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path, wheel, manifest = self._fixture(root)
                if mutation == "escape":
                    external = root / "external.o"
                    external.write_bytes(b"candidate-object")
                    value = manifest["artifacts"][shadow.SOCS[0]]
                    value.update({
                        "object_path": str(external),
                        "object_size": external.stat().st_size,
                        "object_sha256": shadow.sha256_file(external),
                    })
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                else:
                    with zipfile.ZipFile(wheel, "a") as archive:
                        base = "mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/kernel/"
                        soc = shadow.SOCS[0]
                        if mutation == "extra-old":
                            archive.writestr(base + f"{soc}/qr_v2/extra.o", b"extra")
                    if mutation == "config-rows":
                        name = base + f"config/{soc}/qr_v2.json"
                        self._replace_wheel_member(
                            wheel, name, json.dumps({"binList": []}).encode()
                        )
                    elif mutation == "binary-rows":
                        name = base + f"config/{soc}/binary_info_config.json"
                        self._replace_wheel_member(
                            wheel,
                            name,
                            json.dumps({"QrV2": {"binaryList": []}}).encode(),
                        )
                    manifest["immutable_guards"]["extracted_original_wheel"]["sha256"] = shadow.sha256_file(wheel)
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                approved = root / "approved"
                approved.mkdir()
                with self.assertRaises(RuntimeError):
                    shadow.prepare(manifest_path, wheel, approved, approved / "out.json")

    def test_never_mentions_release_packaging_or_record_write(self) -> None:
        source = Path(shadow.__file__).read_text(encoding="utf-8")
        self.assertNotIn("package_release", source)
        self.assertNotIn("ZipFile(.*, \"w\"", source)
        self.assertNotIn("BUILD_READY", source)
        self.assertNotIn("NPU_READY", source)

    def test_failure_keeps_only_marked_partial_and_never_final_shadow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, wheel, source = self._fixture(root)
            base = "mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/kernel/"
            name = base + f"config/{shadow.SOCS[0]}/qr_v2.json"
            self._replace_wheel_member(wheel, name, b'{"binList":[]}')
            source["immutable_guards"]["extracted_original_wheel"]["sha256"] = shadow.sha256_file(wheel)
            manifest.write_text(json.dumps(source), encoding="utf-8")
            approved = root / "approved"
            approved.mkdir()
            output = approved / "out.json"
            with self.assertRaises(RuntimeError):
                shadow.prepare(manifest, wheel, approved, output)
            self.assertFalse((approved / "shadow").exists())
            self.assertFalse(output.exists())
            marker = approved / "shadow.partial" / shadow.INCOMPLETE_MARKER
            self.assertEqual(json.loads(marker.read_text())["consumable"], False)

    def test_manifest_write_failure_has_no_completion_signal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, wheel, _ = self._fixture(root)
            approved = root / "approved"
            approved.mkdir()
            output = approved / "out.json"
            real_write = shadow._write_new_at

            def fail_output(parent_fd, name, payload):
                if name == output.name:
                    raise OSError("manifest write failed")
                return real_write(parent_fd, name, payload)

            with mock.patch.object(shadow, "_write_new_at", side_effect=fail_output):
                with self.assertRaisesRegex(OSError, "manifest write failed"):
                    shadow.prepare(manifest, wheel, approved, output)
            self.assertFalse((approved / "shadow.partial").exists())
            self.assertTrue((approved / "shadow").is_dir())
            self.assertFalse((approved / "shadow" / shadow.INCOMPLETE_MARKER).exists())
            self.assertFalse(output.exists())

    def test_marker_removal_failure_cannot_publish_completion_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, wheel, _ = self._fixture(root)
            approved = root / "approved"
            approved.mkdir()
            output = approved / "out.json"
            real_unlink = os.unlink

            def fail_marker(path, *args, **kwargs):
                if path == shadow.INCOMPLETE_MARKER:
                    raise OSError("marker removal failed")
                return real_unlink(path, *args, **kwargs)

            with mock.patch.object(shadow.os, "unlink", side_effect=fail_marker):
                with self.assertRaisesRegex(OSError, "marker removal failed"):
                    shadow.prepare(manifest, wheel, approved, output)
            self.assertTrue((approved / "shadow" / shadow.INCOMPLETE_MARKER).is_file())
            self.assertFalse(output.exists())

    def test_reserved_output_manifest_names_are_rejected(self) -> None:
        for name in ("shadow", "shadow.partial", shadow.INCOMPLETE_MARKER):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, wheel, _ = self._fixture(root)
                approved = root / "approved"
                approved.mkdir()
                with self.assertRaisesRegex(RuntimeError, "reserved"):
                    shadow.prepare(manifest, wheel, approved, approved / name)

    def test_tree_delta_and_single_record_are_fail_closed(self) -> None:
        for mutation in ("unexpected-tree-write", "second-record"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, wheel, source = self._fixture(root)
                approved = root / "approved"
                approved.mkdir()
                patcher = mock.patch.object(shadow, "_update_soc", wraps=shadow._update_soc)
                if mutation == "second-record":
                    with zipfile.ZipFile(wheel, "a") as archive:
                        archive.writestr("another.dist-info/RECORD", b"second")
                    source["immutable_guards"]["extracted_original_wheel"]["sha256"] = shadow.sha256_file(wheel)
                    manifest.write_text(json.dumps(source), encoding="utf-8")
                else:
                    real_update = shadow._update_soc

                    def inject(shadow_fd, package, *args, **kwargs):
                        value = real_update(shadow_fd, package, *args, **kwargs)
                        (package / "unexpected.txt").write_text("escape", encoding="utf-8")
                        return value

                    patcher = mock.patch.object(shadow, "_update_soc", side_effect=inject)
                with patcher, self.assertRaises(RuntimeError):
                    shadow.prepare(manifest, wheel, approved, approved / "out.json")
                self.assertFalse((approved / "shadow").exists())

    def test_streaming_extraction_rejects_extreme_compression_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "ratio.whl"
            with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("large-zero-file", b"\0" * (1024 * 1024))
            with self.assertRaisesRegex(RuntimeError, "compression ratio"):
                shadow._extract_wheel(wheel, root / "shadow")

    def test_parent_directory_swap_is_detected_before_consumable_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, wheel, _ = self._fixture(root)
            approved = root / "approved"
            approved.mkdir()
            moved = root / "approved-original"
            attacker = root / "attacker"
            attacker.mkdir()
            real_extract = shadow._extract_wheel

            def swap_parent(*args, **kwargs):
                result = real_extract(*args, **kwargs)
                approved.rename(moved)
                approved.symlink_to(attacker, target_is_directory=True)
                return result

            try:
                with mock.patch.object(shadow, "_extract_wheel", side_effect=swap_parent):
                    with self.assertRaisesRegex(RuntimeError, "approved root changed"):
                        shadow.prepare(manifest, wheel, approved, approved / "out.json")
                self.assertFalse((attacker / "shadow").exists())
                self.assertFalse((attacker / "out.json").exists())
            finally:
                if approved.is_symlink():
                    approved.unlink()

    def test_internal_kernel_and_config_parent_swaps_cannot_redirect_overlay(self) -> None:
        for target_kind in ("kernel", "config"):
            with self.subTest(target_kind=target_kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, wheel, _ = self._fixture(root)
                approved = root / "approved"
                approved.mkdir()
                attacker = root / "attacker"
                attacker.mkdir()
                real_open_chain = shadow._open_existing_chain
                swapped = False

                def swap_internal(root_fd, parts):
                    nonlocal swapped
                    matches = (
                        target_kind == "kernel" and parts[-2:] == (shadow.SOCS[0], "qr_v2")
                    ) or (
                        target_kind == "config" and parts[-2:] == ("config", shadow.SOCS[0])
                    )
                    if matches and not swapped:
                        internal = approved / "shadow.partial" / Path(*parts)
                        saved = internal.with_name(internal.name + ".saved")
                        internal.rename(saved)
                        internal.symlink_to(attacker, target_is_directory=True)
                        swapped = True
                    return real_open_chain(root_fd, parts)

                with mock.patch.object(shadow, "_open_existing_chain", side_effect=swap_internal):
                    with self.assertRaises(OSError):
                        shadow.prepare(manifest, wheel, approved, approved / "out.json")
                self.assertTrue(swapped)
                self.assertEqual(list(attacker.iterdir()), [])
                self.assertFalse((approved / "shadow").exists())
                self.assertFalse((approved / "out.json").exists())

    def test_outer_finally_closes_root_fd_for_publish_stage_exceptions(self) -> None:
        for failure in ("root-identity", "tree-inventory", "open-shadow"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, wheel, _ = self._fixture(root)
                approved = root / "approved"
                approved.mkdir()
                captured = []
                stack = []
                if failure == "root-identity":
                    real_identity = shadow._assert_root_identity
                    calls = 0

                    def fail_identity(root_fd, path):
                        nonlocal calls
                        calls += 1
                        if calls == 3:
                            captured.append(root_fd)
                            raise OSError("identity injection")
                        return real_identity(root_fd, path)

                    stack.append(mock.patch.object(shadow, "_assert_root_identity", side_effect=fail_identity))
                elif failure == "tree-inventory":
                    real_inventory = shadow._tree_inventory
                    calls = 0

                    def fail_inventory(*args, **kwargs):
                        nonlocal calls
                        calls += 1
                        if calls == 3:
                            raise OSError("inventory injection")
                        return real_inventory(*args, **kwargs)

                    stack.append(mock.patch.object(shadow, "_tree_inventory", side_effect=fail_inventory))
                    real_open = shadow.os.open

                    def capture_root(path, *args, **kwargs):
                        descriptor = real_open(path, *args, **kwargs)
                        if path == approved:
                            captured.append(descriptor)
                        return descriptor

                    stack.append(mock.patch.object(shadow.os, "open", side_effect=capture_root))
                else:
                    real_open = shadow.os.open

                    def fail_open(path, *args, **kwargs):
                        if path == "shadow" and kwargs.get("dir_fd") is not None:
                            captured.append(kwargs["dir_fd"])
                            raise OSError("open shadow injection")
                        return real_open(path, *args, **kwargs)

                    stack.append(mock.patch.object(shadow.os, "open", side_effect=fail_open))
                for patcher in stack:
                    patcher.start()
                try:
                    with self.assertRaises(OSError):
                        shadow.prepare(manifest, wheel, approved, approved / "out.json")
                finally:
                    for patcher in reversed(stack):
                        patcher.stop()
                self.assertTrue(captured)
                with self.assertRaises(OSError):
                    os.fstat(captured[-1])


if __name__ == "__main__":
    unittest.main()
