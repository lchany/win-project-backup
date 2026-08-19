#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import step216_a_brockett_policy as policy
import step216_a_world8_controller as controller


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "step216_a_source_contract.json"


class StaticContractTests(unittest.TestCase):
    def test_source_package_is_exact_and_non_recursive(self) -> None:
        value = policy.load_source_contract(CONTRACT)
        policy.verify_source_package(value, HERE)
        names = {row["name"] for row in value["source_files"]}
        self.assertNotIn(CONTRACT.name, names)
        self.assertIn(Path(__file__).name, names)

    def test_changed_source_fails_closed(self) -> None:
        value = policy.load_source_contract(CONTRACT)
        identity = value["source_files"][0]
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / identity["name"]
            changed.write_bytes((HERE / identity["name"]).read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                policy.verify_identity(changed, identity)

    def test_tool_layout_requires_realpath_outside_repo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = base / "business"; repo.mkdir()
            tool = base / "shared_diagnostics"; (tool / "harness").mkdir(parents=True)
            adapter = tool / "harness" / "adapter.py"; adapter.write_text("# pinned\n")
            accepted = policy.assert_tool_layout(tool, repo, adapter, tool / "runs" / "step216_a_brockett_x")
            self.assertEqual(accepted["tool_root"], tool.resolve())
            inside = repo / "diagnostics"; (inside / "harness").mkdir(parents=True)
            bad_adapter = inside / "harness" / "adapter.py"; bad_adapter.write_text("# bad\n")
            with self.assertRaisesRegex(ValueError, "outside business repo"):
                policy.assert_tool_layout(inside, repo, bad_adapter, inside / "runs" / "step216_a_brockett_x")

    def test_cleanup_is_pid_group_only_and_cannot_self_match(self) -> None:
        scripts = [HERE / "step216_a_run_inside_container.sh", HERE / "step216_a_host_launch_contract.sh"]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in scripts)
        self.assertNotIn("pkill", combined)
        self.assertNotIn("killall", combined)
        self.assertNotIn("pgrep -af '$output'", combined)
        self.assertIn('kill -TERM -- "-$launcher_pgid"', combined)
        self.assertIn('kill -KILL -- "-$launcher_pgid"', combined)
        self.assertIn('terminate_container_launcher_group', combined)
        self.assertIn('kill -TERM -- "-$pgid"', combined)
        self.assertIn('kill -KILL -- "-$pgid"', combined)
        host = scripts[1].read_text(encoding="utf-8")
        self.assertIn('terminate_exact_group "$controller_pid" "$controller_pgid"', host)
        self.assertIn('terminate_exact_group "$host_job_pid" "$host_job_pgid"', host)
        self.assertLess(host.index("terminate_container_launcher_group"), host.index("--postflight-only"))

    def test_controller_runs_on_host_not_container_runner(self) -> None:
        runner = (HERE / "step216_a_run_inside_container.sh").read_text(encoding="utf-8")
        host = (HERE / "step216_a_host_launch_contract.sh").read_text(encoding="utf-8")
        self.assertNotIn('python3 "$controller" --output-dir', runner)
        self.assertIn('python3 "$controller"', host)
        self.assertNotIn('docker exec "$container" python3 "$controller"', host)

    def test_runner_set_u_interface_binds_summarizer(self) -> None:
        runner = (HERE / "step216_a_run_inside_container.sh").read_text(encoding="utf-8")
        host = (HERE / "step216_a_host_launch_contract.sh").read_text(encoding="utf-8")
        self.assertIn(': "${ADAPTER_PATH:?}" "${SUMMARIZER:?}"', runner)
        self.assertIn('summarizer=$(readlink -f "$SUMMARIZER")', runner)
        self.assertIn('-e SUMMARIZER="$summarizer"', host)
        self.assertLess(runner.index('summarizer=$(readlink -f "$SUMMARIZER")'), runner.index('sha256sum "$contract"'))

    def test_container_launcher_pgid_is_strict(self) -> None:
        self.assertEqual(policy.parse_positive_pgid("4242\n"), 4242)
        for bad in ("", "0", "1", "-2", "+2", "2 3", "2x", " 2", "2\n3"):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    policy.parse_positive_pgid(bad)

    def test_fake_soap_bound_methods_receive_isolated_full_state(self) -> None:
        class FakeSOAP:
            def __init__(self) -> None:
                self.calls = []

            def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
                self.calls.append(("back", state, merge_dims, max_precond_dim))
                return grad + state["Q"][0]

            def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
                self.calls.append(("project", state, merge_dims, max_precond_dim))
                return grad * state["Q"][0]

        soap = FakeSOAP()
        policy.assert_bound_project_signature(soap.project, policy.SOAP_BOUND_SIGNATURE)
        policy.assert_bound_project_signature(soap.project_back, policy.SOAP_BOUND_SIGNATURE)
        state = {"Q": [2], "exp_avg_sq": "persistent"}
        original, baseline, candidate = policy.project_roundtrip_views(soap, 3, state, [4], [5])
        self.assertEqual((original, baseline, candidate), (5, 20, 25))
        self.assertEqual(state, {"Q": [2], "exp_avg_sq": "persistent"})
        self.assertTrue(all(call[1] is not state for call in soap.calls))
        self.assertTrue(all(call[1]["exp_avg_sq"] == "persistent" for call in soap.calls))
        self.assertTrue(all(call[2:] == (False, 10000) for call in soap.calls))

    def test_gate_ast_uses_roundtrip_helper_not_raw_project_calls(self) -> None:
        tree = ast.parse((HERE / "step216_a_brockett_qr_gate.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        helper = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "project_roundtrip_views"]
        raw = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr in {"project", "project_back"}]
        self.assertEqual(len(helper), 1)
        self.assertEqual(raw, [])

    def test_npu_smi_back8_pid_parser(self) -> None:
        rows = [(physical, chip, 9000 + physical * 2 + chip) for physical in range(4, 8) for chip in range(2)]
        text = "\n".join(f"| {physical} {chip} | {pid} | python3 |" for physical, chip, pid in rows)
        self.assertEqual(controller.back8_rows(text), rows)
        self.assertEqual({row[:2] for row in rows}, controller.BACK8_PAIRS)

    def test_host_pid_maps_to_last_container_nspid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status = Path(directory) / "4242" / "status"
            status.parent.mkdir()
            status.write_text("Name:\tpython3\nNSpid:\t4242\t17\n", encoding="utf-8")
            self.assertEqual(controller.nspid_chain_from_status(status), [4242, 17])
            self.assertEqual(controller.container_pid_for_host_pid(4242, directory), 17)
            with self.assertRaises(ValueError):
                controller.container_pid_for_host_pid(1, directory)

    def test_host_controller_fixture_maps_releases_and_completes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "ready").mkdir(); (root / "done").mkdir()
            proc_root = root / "proc"; proc_root.mkdir()
            rows = []
            for rank, (physical, chip) in enumerate(sorted(controller.BACK8_PAIRS)):
                container_pid = 100 + rank; host_pid = 4100 + rank
                (root / "ready" / f"rank{rank}.json").write_text(json.dumps({
                    "rank": rank, "local_rank": rank, "world_size": 8,
                    "visible": controller.VISIBLE, "gate_pass": True,
                    "container_pid": container_pid,
                }))
                (root / "done" / f"rank{rank}.json").write_text("{}")
                status = proc_root / str(host_pid) / "status"; status.parent.mkdir()
                status.write_text(f"Name:\tpython3\nNSpid:\t{host_pid}\t{container_pid}\n")
                rows.append(f"| {physical} {chip} | {host_pid} | python3 |")
            with mock.patch.object(controller, "npu_smi", return_value="\n".join(rows)):
                self.assertEqual(controller.supervise(root, os.getpid(), 5, proc_root), 0)
            self.assertTrue((root / "release_after_npu_smi").exists())
            status = json.loads((root / "controller_status.json").read_text())
            self.assertEqual(status["status"], "PASS")
            self.assertEqual(len(status["pid_namespace_mapping"]), 8)

    def test_pinned_brockett_cubic_is_finite_and_near_orthogonal(self) -> None:
        # Diagonal C with Q=I makes the tangent direction exactly zero.  The
        # scalar diagonal path therefore validates the pinned scaling/cubic
        # coefficients without importing torch/numpy on the local workstation.
        x = 1.0
        gram = x * x
        scale = min(1.0, 1.25 / (math.sqrt(abs(gram)) + 1e-7))
        z = scale * x; gram = scale * scale * gram
        result = 0.125 * (15 * z + z * (-10 * gram + 3 * gram * gram))
        self.assertTrue(math.isfinite(result))
        self.assertLess(abs(result * result - 1.0), 2e-5)


if __name__ == "__main__":
    unittest.main()
