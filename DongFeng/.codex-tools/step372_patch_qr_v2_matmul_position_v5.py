#!/usr/bin/env python3
"""Create the fail-closed QrV2 Matmul tensor-position v5 candidate.

The original vendor source and the v4 patcher are read-only.  This patch adds
exactly two contract fixes on top of v4: the second LARFB vtv multiply consumes
V directly from VECIN, and qa declares the positions its existing arguments
actually use (VECIN, GM, VECIN).
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

import step338_patch_qr_v2_lifetime as release_v4


CANDIDATE_IDENTITY = "QrV2_matmul_position_fix_v5"
EXPECTED_SOURCE_SHA256 = release_v4.EXPECTED_SOURCE_SHA256
EXPECTED_V4_CANDIDATE_SHA256 = release_v4.EXPECTED_CANDIDATE_SHA256
EXPECTED_CANDIDATE_SHA256 = "e6ccbb84b0e0dbdc026ecdc6b6e07936fbd659401e35c38f7e9eb974d99bc3b7"

V4_QA_DECLARATION = """    Matmul<MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t>,
        MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t>, MatmulType<TPosition::GM, CubeFormat::ND, float32_t>>
        qaMatmulObj;
"""
V5_QA_DECLARATION = """    Matmul<MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t>,
        MatmulType<TPosition::GM, CubeFormat::ND, float32_t>, MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t>>
        qaMatmulObj;
"""
V5_DIRECT_VTV_A = "        this->vtvMatmulObj.SetTensorA(this->vLocal);\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def build_unverified(source: bytes) -> bytes:
    source_sha = sha256_bytes(source)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"source SHA-256 mismatch: {source_sha}")
    v4 = release_v4.build_candidate(source)
    v4_sha = sha256_bytes(v4)
    if v4_sha != EXPECTED_V4_CANDIDATE_SHA256:
        raise RuntimeError(f"v4 candidate SHA-256 mismatch: {v4_sha}")
    text = v4.decode("utf-8")
    text = replace_once(
        text,
        release_v4.V4_CALC_Q_SCRATCH_BLOCK,
        V5_DIRECT_VTV_A,
        "CalcQForLARFB VECIN tensor-A contract",
    )
    text = replace_once(
        text,
        V4_QA_DECLARATION,
        V5_QA_DECLARATION,
        "qa Matmul tensor-position contract",
    )
    return text.encode("utf-8")


def _verify_set_tensor_positions(text: str) -> dict[str, int]:
    calc_larfb = release_v4.extract_block(text, "__aicore__ inline void CalcQForLARFB(")
    calc_ssrfb = release_v4.extract_block(text, "__aicore__ inline void CalcQForSSRFB(")
    update_larfb = release_v4.extract_block(text, "__aicore__ inline void UpdateAForLARFB(")
    update_ssrfb = release_v4.extract_block(text, "__aicore__ inline void UpdateAForSSRFB(")
    update_colq = release_v4.extract_block(text, "__aicore__ inline void UpdateColQ(")
    calc_currentq = release_v4.extract_block(text, "__aicore__ inline void CalcCurrentQ(")

    expected_counts = {
        "vtv_a_vecin": sum(
            block.count("vtvMatmulObj.SetTensorA(this->tLocal")
            + block.count("vtvMatmulObj.SetTensorA(this->vLocal")
            for block in (calc_larfb, calc_ssrfb)
        ),
        "vtv_b_vecin": sum(block.count("vtvMatmulObj.SetTensorB(this->") for block in (calc_larfb, calc_ssrfb)),
        "vtv_c_vecin": sum(block.count("vtvMatmulObj.IterateAll(this->qLocal") for block in (calc_larfb, calc_ssrfb)),
        "qa_a_vecin": sum(block.count("qaMatmulObj.SetTensorA(qLocal)") for block in (update_larfb, update_ssrfb)),
        "qa_b_gm": sum(block.count("qaMatmulObj.SetTensorB(workspaceInGm[") for block in (update_larfb, update_ssrfb)),
        "qa_c_vecin": sum(block.count("qaMatmulObj.IterateAll(aLocal)") for block in (update_larfb, update_ssrfb)),
        "blockq_a_gm": update_colq.count("blockQMatmulObj.SetTensorA(colQGm["),
        "blockq_b_gm": update_colq.count("blockQMatmulObj.SetTensorB(colQGm["),
        "blockq_c_gm": update_colq.count("blockQMatmulObj.IterateAll(colQGm["),
        "currentq_a_gm": calc_currentq.count("currentQMatmulObj.SetTensorA(matrixA["),
        "currentq_b_gm": calc_currentq.count("currentQMatmulObj.SetTensorB(matrixB["),
        "currentq_c_gm": calc_currentq.count("currentQMatmulObj.IterateAll(matrixC["),
    }
    required = {
        "vtv_a_vecin": 4,
        "vtv_b_vecin": 4,
        "vtv_c_vecin": 4,
        "qa_a_vecin": 2,
        "qa_b_gm": 2,
        "qa_c_vecin": 2,
        "blockq_a_gm": 2,
        "blockq_b_gm": 2,
        "blockq_c_gm": 2,
        "currentq_a_gm": 1,
        "currentq_b_gm": 1,
        "currentq_c_gm": 1,
    }
    if expected_counts != required:
        raise RuntimeError(f"SetTensor position audit failed: {expected_counts}")
    if "vtvMatmulObj.SetTensorA(workspaceInGm[" in text:
        raise RuntimeError("vtv VECIN tensor-A still receives GM")
    return expected_counts


def verify_candidate_structure(source: bytes, candidate: bytes) -> dict[str, Any]:
    expected = build_unverified(source)
    if candidate != expected:
        raise RuntimeError("candidate bytes differ from deterministic v5 build")
    candidate_sha = sha256_bytes(candidate)
    if candidate_sha != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError(f"candidate SHA-256 mismatch: {candidate_sha}")
    text = candidate.decode("utf-8")
    if text.count(V5_QA_DECLARATION) != 1 or V4_QA_DECLARATION in text:
        raise RuntimeError("qa position declaration contract failed")
    calc_larfb = release_v4.extract_block(text, "__aicore__ inline void CalcQForLARFB(")
    if calc_larfb.count(V5_DIRECT_VTV_A) != 1:
        raise RuntimeError("CalcQForLARFB must contain one direct vLocal SetTensorA")
    for forbidden in (
        "calcQScratchOffset",
        "DataCopy(workspaceInGm[calcQScratchOffset]",
        "eventIDMTE3_MTE2",
        "SetFlag<HardEvent::MTE3_MTE2>",
        "WaitFlag<HardEvent::MTE3_MTE2>",
    ):
        if forbidden in calc_larfb:
            raise RuntimeError(f"CalcQForLARFB retains removed v4 scratch/event fragment: {forbidden}")

    position_counts = _verify_set_tensor_positions(text)

    restored = replace_once(text, V5_DIRECT_VTV_A, release_v4.V4_CALC_Q_SCRATCH_BLOCK, "reverse CalcQ")
    restored = replace_once(restored, V5_QA_DECLARATION, V4_QA_DECLARATION, "reverse qa declaration")
    restored_sha = sha256_bytes(restored.encode("utf-8"))
    if restored_sha != EXPECTED_V4_CANDIDATE_SHA256:
        raise RuntimeError(f"v5 delta exceeds the two approved position fixes: restored_sha256={restored_sha}")

    return {
        "candidate_identity": CANDIDATE_IDENTITY,
        "source_sha256": sha256_bytes(source),
        "v4_candidate_sha256": EXPECTED_V4_CANDIDATE_SHA256,
        "candidate_sha256": candidate_sha,
        "reverse_v4_sha256": restored_sha,
        "delta_exactly_two_position_fixes": True,
        "calc_q_direct_vecin_a": True,
        "calc_q_scratch_removed": True,
        "calc_q_scratch_event_removed": True,
        "qa_positions": ["VECIN", "GM", "VECIN"],
        "set_tensor_position_counts": position_counts,
        "math_shape_dtype_workspace_output_init_unchanged_from_v4": True,
    }


def build_candidate(source: bytes) -> bytes:
    candidate = build_unverified(source)
    verify_candidate_structure(source, candidate)
    return candidate


def write_new_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
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
        "delta_exactly_two_position_fixes",
    ):
        print(f"{key}={report[key]}")
    if args.check:
        if args.output is not None:
            raise ValueError("output must be omitted with --check")
        return 0
    if args.output is None:
        raise ValueError("output is required unless --check is used")
    output = args.output.absolute()
    if output == source_path or output.exists():
        raise ValueError("output must be a new path different from source")
    write_new_file(output, candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
