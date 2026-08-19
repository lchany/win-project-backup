#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import step216_a_brockett_policy as policy


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "step216_a_turbosoap_brockett_core_config.json"


class BrockettPolicyTests(unittest.TestCase):
    def test_pinned_policy_loads(self) -> None:
        value = policy.load_verified_policy(CONFIG)
        self.assertEqual(value["authority"]["revision"], policy.REVISION)

    def test_missing_policy_fails_closed(self) -> None:
        with self.assertRaises(FileNotFoundError):
            policy.load_verified_policy(HERE / "does-not-exist.json")

    def test_parameter_mutation_is_rejected(self) -> None:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        value["algorithm"]["eta"] = 0.02
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parameters differ"):
                policy.load_verified_policy(path)

    def test_unverified_source_is_rejected(self) -> None:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        value["authority"]["status"] = "unverified"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unverified.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not verified"):
                policy.load_verified_policy(path)

    def test_dispatch_guard(self) -> None:
        good = dict(square=True, dtype="torch.float32", contiguous=True, requires_grad=False)
        self.assertTrue(policy.dispatch_candidate(2560, **good))
        self.assertFalse(policy.dispatch_candidate(5120, **good))
        self.assertFalse(policy.dispatch_candidate(257, **good))
        self.assertFalse(policy.dispatch_candidate(256, **{**good, "dtype": "torch.float16"}))
        self.assertFalse(policy.dispatch_candidate(256, **{**good, "requires_grad": True}))

    def test_inventory_is_exact(self) -> None:
        policy.assert_inventory(copy.deepcopy(policy.APPROVED_COUNTS))
        bad = copy.deepcopy(policy.APPROVED_COUNTS)
        bad[5120] = 1
        with self.assertRaisesRegex(ValueError, "23-shape"):
            policy.assert_inventory(bad)


if __name__ == "__main__":
    unittest.main()
