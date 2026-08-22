#!/usr/bin/env python3
"""Prepare a two-SoC overlay for the SHA-locked STEP350 diagnostic QrV2."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import step343_prepare_overlay as base


CANDIDATE_STEM = "QrV2_step350_context_capture_r1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed-custom-opp", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate-o-sha256", required=True)
    parser.add_argument("--candidate-json-sha256", required=True)
    args = parser.parse_args()

    candidate_dir = Path(args.candidate_dir).resolve(strict=True)
    candidate_o = candidate_dir / f"{CANDIDATE_STEM}.o"
    candidate_json = candidate_dir / f"{CANDIDATE_STEM}.json"
    for path, expected in (
        (candidate_o, args.candidate_o_sha256),
        (candidate_json, args.candidate_json_sha256),
    ):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if sha256(path) != expected:
            raise RuntimeError(f"candidate SHA256 mismatch: {path.name}")

    base.CANDIDATE_STEM = CANDIDATE_STEM
    base.CANDIDATE_O_SIZE = candidate_o.stat().st_size
    base.CANDIDATE_JSON_SIZE = candidate_json.stat().st_size
    manifest = base.prepare(
        Path(args.installed_custom_opp),
        candidate_dir,
        Path(args.overlay),
        Path(args.manifest),
        args.candidate_o_sha256,
        args.candidate_json_sha256,
    )
    if manifest.get("candidate_kernel_name") != CANDIDATE_STEM:
        raise RuntimeError("prepared manifest candidate identity mismatch")
    print("STEP351_DIAGNOSTIC_OVERLAY_PREPARE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
