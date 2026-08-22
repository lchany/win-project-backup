#!/usr/bin/env python3
"""Focused offline contract tests for the STEP392 shadow layer."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import step392_prepare_delta2_shadow as shadow


class Step392ShadowContractTests(unittest.TestCase):
    def test_locked_attempt5_evidence_and_strict_negative_table(self) -> None:
        evidence = shadow.load_locked_evidence()
        self.assertEqual(evidence["manifest"]["sha256"], "0221f5b64fe682d230f834554b3b8d977673f807c6a890c5279c835ebe173de8")
        self.assertEqual(evidence["receipt"]["sha256"], "2d5845c7c2dd74b5d50c7689e3c89f0c1144e472e804f34d8bbd421770d26f9c")
        self.assertEqual(evidence["completion"]["sha256"], "d37f0bf9754fe9e67a5428b133d396792f9237975b2e120258095730c4ba8dda")
        self.assertEqual(evidence["original_wheel"]["sha256"], "23253f7fa2b9bfb1b6ff3c77df6620f6c559f68be154f6333246d73178eb5da9")
        self.assertEqual(set(evidence["inputs"]), {str(rank) for rank in range(8)})
        self.assertEqual({evidence["artifacts"][soc]["object"]["sha256"] for soc in shadow.SOCS}, {"fe4ec6f0330d0c7b2d2dc91770b40ae7b4ec20e7b046766584c0fef0887bd361"})
        self.assertEqual({evidence["artifacts"][soc]["json"]["sha256"] for soc in shadow.SOCS}, {"b1c30a14c5084e2e786677f246b08fb0581e9d144b3c034de8741a3910ddc703"})
        for bad_mode in (True, -1, 0o10000, 0o644):
            row = copy.deepcopy(evidence["receipt"])
            row["mode"] = bad_mode
            with self.subTest(mode=bad_mode), self.assertRaises(RuntimeError):
                shadow._file_contract(row, "receipt", 0o600)
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / shadow.EVIDENCE_NAME
            tampered.write_bytes((Path(__file__).resolve().parents[1] / shadow.EVIDENCE_NAME).read_bytes() + b"\n")
            with self.assertRaisesRegex(RuntimeError, "evidence SHA"):
                shadow.load_locked_evidence(tampered)


if __name__ == "__main__":
    unittest.main()
