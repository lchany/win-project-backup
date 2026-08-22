#!/usr/bin/env python3
"""Generate the fail-closed STEP375 delta1-only QrV2 diagnostic probe.

The probe starts from the deterministic v4 candidate, applies only the direct
VECIN ``vLocal`` input for the second LARFB vtv multiply, and deliberately
retains the legacy qa declaration.  It is diagnostic evidence for isolating
the v5 qa delta; it is not a correctness fix or a release candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

import step338_patch_qr_v2_lifetime as release_v4


CANDIDATE_IDENTITY = "QrV2_vtv_direct_qa_legacy_probe_v6"
EXPECTED_SOURCE_SHA256 = release_v4.EXPECTED_SOURCE_SHA256
EXPECTED_V4_CANDIDATE_SHA256 = release_v4.EXPECTED_CANDIDATE_SHA256
EXPECTED_CANDIDATE_SHA256 = "ef5db14e09170806acb7c5227fd619f3f5ffdc7d31f36e49058cc88987fce180"
EXPECTED_DELTA2_ONLY_SHA256 = "e352ac31f97980bc5c92caac663414782b78fc3004ff36709b6be8000353b003"
EXPECTED_V5_SHA256 = "e6ccbb84b0e0dbdc026ecdc6b6e07936fbd659401e35c38f7e9eb974d99bc3b7"

V4_QA_DECLARATION = """    Matmul<MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t>,
        MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t>, MatmulType<TPosition::GM, CubeFormat::ND, float32_t>>
        qaMatmulObj;
"""
V5_QA_DECLARATION = """    Matmul<MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t>,
        MatmulType<TPosition::GM, CubeFormat::ND, float32_t>, MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t>>
        qaMatmulObj;
"""
DIRECT_VTV_A = "        this->vtvMatmulObj.SetTensorA(this->vLocal);\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def _locked_v4(source: bytes) -> bytes:
    source_sha = sha256_bytes(source)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"source SHA-256 mismatch: {source_sha}")
    v4 = release_v4.build_candidate(source)
    v4_sha = sha256_bytes(v4)
    if v4_sha != EXPECTED_V4_CANDIDATE_SHA256:
        raise RuntimeError(f"v4 candidate SHA-256 mismatch: {v4_sha}")
    return v4


def build_unverified(source: bytes) -> bytes:
    text = _locked_v4(source).decode("utf-8")
    return replace_once(
        text,
        release_v4.V4_CALC_Q_SCRATCH_BLOCK,
        DIRECT_VTV_A,
        "delta1 CalcQForLARFB direct vLocal",
    ).encode("utf-8")


def candidate_matrix(source: bytes) -> dict[str, bytes]:
    v4 = _locked_v4(source)
    v4_text = v4.decode("utf-8")
    delta1_only = replace_once(
        v4_text,
        release_v4.V4_CALC_Q_SCRATCH_BLOCK,
        DIRECT_VTV_A,
        "matrix delta1",
    ).encode("utf-8")
    delta2_only = replace_once(
        v4_text,
        V4_QA_DECLARATION,
        V5_QA_DECLARATION,
        "matrix delta2",
    ).encode("utf-8")
    v5 = replace_once(
        delta1_only.decode("utf-8"),
        V4_QA_DECLARATION,
        V5_QA_DECLARATION,
        "matrix delta1+delta2",
    ).encode("utf-8")
    return {
        "v4": v4,
        "delta1_only": delta1_only,
        "delta2_only": delta2_only,
        "delta1_and_delta2_v5": v5,
    }


def verify_candidate_structure(source: bytes, candidate: bytes) -> dict[str, Any]:
    expected = build_unverified(source)
    if candidate != expected:
        raise RuntimeError("candidate bytes differ from deterministic delta1-only build")
    candidate_sha = sha256_bytes(candidate)
    if candidate_sha != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError(f"candidate SHA-256 mismatch: {candidate_sha}")

    text = candidate.decode("utf-8")
    if text.count(V4_QA_DECLARATION) != 1 or V5_QA_DECLARATION in text:
        raise RuntimeError("probe must retain exactly one legacy qa declaration")
    calc_larfb = release_v4.extract_block(
        text, "__aicore__ inline void CalcQForLARFB("
    )
    if calc_larfb.count(DIRECT_VTV_A) != 1:
        raise RuntimeError("probe must apply exactly one delta1 direct vLocal input")
    for forbidden in (
        "calcQScratchOffset",
        "DataCopy(workspaceInGm[calcQScratchOffset]",
        "eventIDMTE3_MTE2",
        "SetFlag<HardEvent::MTE3_MTE2>",
        "WaitFlag<HardEvent::MTE3_MTE2>",
    ):
        if forbidden in calc_larfb:
            raise RuntimeError(f"delta1 retains removed scratch/event fragment: {forbidden}")

    restored = replace_once(
        text,
        DIRECT_VTV_A,
        release_v4.V4_CALC_Q_SCRATCH_BLOCK,
        "reverse delta1",
    ).encode("utf-8")
    restored_sha = sha256_bytes(restored)
    if restored_sha != EXPECTED_V4_CANDIDATE_SHA256:
        raise RuntimeError(
            "probe delta exceeds delta1: "
            f"restored_v4_sha256={restored_sha}"
        )

    matrix = candidate_matrix(source)
    matrix_hashes = {name: sha256_bytes(payload) for name, payload in matrix.items()}
    expected_matrix_hashes = {
        "v4": EXPECTED_V4_CANDIDATE_SHA256,
        "delta1_only": EXPECTED_CANDIDATE_SHA256,
        "delta2_only": EXPECTED_DELTA2_ONLY_SHA256,
        "delta1_and_delta2_v5": EXPECTED_V5_SHA256,
    }
    if matrix_hashes != expected_matrix_hashes:
        raise RuntimeError(f"four-cell candidate matrix SHA mismatch: {matrix_hashes}")
    if len(set(matrix_hashes.values())) != 4:
        raise RuntimeError("four-cell candidate matrix identities are not unique")
    if matrix["delta1_only"] != candidate:
        raise RuntimeError("candidate is not the matrix delta1-only cell")

    return {
        "candidate_identity": CANDIDATE_IDENTITY,
        "diagnostic_only": True,
        "diagnostic_question": "whether the v5 qa delta2 causes the runtime trap",
        "source_sha256": sha256_bytes(source),
        "candidate_sha256": candidate_sha,
        "reverse_v4_sha256": restored_sha,
        "delta1_direct_vlocal": True,
        "delta2_qa_position_change": False,
        "qa_positions_retained": ["VECIN", "VECIN", "GM"],
        "four_cell_matrix_sha256": matrix_hashes,
        "release_candidate": False,
    }


def build_candidate(source: bytes) -> bytes:
    candidate = build_unverified(source)
    verify_candidate_structure(source, candidate)
    return candidate


def _append_cleanup_error(primary: BaseException, cleanup: BaseException) -> None:
    detail = f"candidate cleanup failed: {type(cleanup).__name__}: {cleanup}"
    primary.cleanup_error = detail  # type: ignore[attr-defined]
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        try:
            add_note(detail)
        except BaseException:
            pass


def write_new_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as primary:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except BaseException as cleanup:
            _append_cleanup_error(primary, cleanup)
        raise


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    source_argument = args.source.absolute()
    if source_argument.is_symlink():
        raise ValueError("source must not be a symlink")
    source_path = source_argument.resolve(strict=True)
    if not source_path.is_file():
        raise ValueError("source must be a non-symlink regular file")
    source = source_path.read_bytes()
    candidate = build_candidate(source)
    report = verify_candidate_structure(source, candidate)
    for key in (
        "candidate_identity",
        "candidate_sha256",
        "reverse_v4_sha256",
        "diagnostic_only",
    ):
        print(f"{key}={report[key]}")
    if args.check:
        if args.output is not None:
            raise ValueError("output must be omitted with --check")
        return 0
    if args.output is None:
        raise ValueError("output is required unless --check is used")
    output = args.output.absolute()
    if output == source_path or output.exists() or output.is_symlink():
        raise ValueError("output must be a new path different from source")
    write_new_file(output, candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
