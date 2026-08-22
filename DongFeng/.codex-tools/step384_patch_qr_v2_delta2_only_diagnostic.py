#!/usr/bin/env python3
"""Generate the fail-closed STEP384 qa-position delta2-only diagnostic.

This probe starts from the deterministic v4 candidate and changes only the
qa Matmul position declaration.  It retains the v4 per-core GM scratch and
MTE3_MTE2 hand-off in CalcQForLARFB and is not a release candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
from pathlib import Path
from typing import Any, Iterable

import step338_patch_qr_v2_lifetime as release_v4


CANDIDATE_IDENTITY = "QrV2_qa_position_delta2_only_diagnostic_v1"
EXPECTED_SOURCE_SHA256 = release_v4.EXPECTED_SOURCE_SHA256
EXPECTED_V4_CANDIDATE_SHA256 = release_v4.EXPECTED_CANDIDATE_SHA256
EXPECTED_CANDIDATE_SHA256 = "e352ac31f97980bc5c92caac663414782b78fc3004ff36709b6be8000353b003"
DIAGNOSTIC_QUESTION = (
    "with delta1 absent and v4 per-core GM scratch restored, whether delta2-only "
    "completes normally and changes v4 finite behavior"
)

V4_QA_DECLARATION = """    Matmul<MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t>,
        MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t>, MatmulType<TPosition::GM, CubeFormat::ND, float32_t>>
        qaMatmulObj;
"""
DELTA2_QA_DECLARATION = """    Matmul<MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t>,
        MatmulType<TPosition::GM, CubeFormat::ND, float32_t>, MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t>>
        qaMatmulObj;
"""
DIRECT_VLOCAL = "        this->vtvMatmulObj.SetTensorA(this->vLocal);\n"


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
    return replace_once(
        _locked_v4(source).decode("utf-8"),
        V4_QA_DECLARATION,
        DELTA2_QA_DECLARATION,
        "delta2 qa Matmul declaration",
    ).encode("utf-8")


def verify_candidate_structure(source: bytes, candidate: bytes) -> dict[str, Any]:
    expected = build_unverified(source)
    if candidate != expected:
        raise RuntimeError("candidate bytes differ from deterministic delta2-only build")
    candidate_sha = sha256_bytes(candidate)
    if candidate_sha != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError(f"candidate SHA-256 mismatch: {candidate_sha}")

    text = candidate.decode("utf-8")
    if text.count(DELTA2_QA_DECLARATION) != 1 or V4_QA_DECLARATION in text:
        raise RuntimeError("probe must contain exactly one delta2 qa declaration")
    calc_q = release_v4.extract_block(text, "__aicore__ inline void CalcQForLARFB(")
    if calc_q.count(release_v4.V4_CALC_Q_SCRATCH_BLOCK) != 1:
        raise RuntimeError("probe must retain the v4 CalcQ per-core GM scratch block")
    if calc_q.count(release_v4.V3_MTE3_MTE2_SEQUENCE) != 1:
        raise RuntimeError("probe must retain the v4 CalcQ MTE3_MTE2 hand-off")
    if DIRECT_VLOCAL in calc_q:
        raise RuntimeError("delta2-only probe must reject direct vLocal Matmul input")

    restored = replace_once(
        text,
        DELTA2_QA_DECLARATION,
        V4_QA_DECLARATION,
        "reverse delta2",
    ).encode("utf-8")
    restored_sha = sha256_bytes(restored)
    if restored_sha != EXPECTED_V4_CANDIDATE_SHA256:
        raise RuntimeError(
            "probe delta exceeds delta2: " f"reverse_v4_sha256={restored_sha}"
        )

    return {
        "candidate_identity": CANDIDATE_IDENTITY,
        "diagnostic_only": True,
        "diagnostic_question": DIAGNOSTIC_QUESTION,
        "source_sha256": sha256_bytes(source),
        "candidate_sha256": candidate_sha,
        "reverse_v4_sha256": restored_sha,
        "delta1_direct_vlocal": False,
        "delta2_qa_position_change": True,
        "qa_positions": ["VECIN", "GM", "VECIN"],
        "calc_q_per_core_gm_scratch_retained": True,
        "calc_q_mte3_mte2_retained": True,
        "release_candidate": False,
        "package_forbidden": True,
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
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    parent_fd = os.open(path.parent, directory_flags)
    parent_identity: os.stat_result | None = None
    descriptor: int | None = None
    created_identity: os.stat_result | None = None
    primary: BaseException | None = None
    try:
        parent_identity = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_identity.st_mode):
            raise NotADirectoryError(f"output parent is not a directory: {path.parent}")
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        descriptor = os.open(path.name, file_flags, 0o644, dir_fd=parent_fd)
        opened_identity = os.fstat(descriptor)
        if not stat.S_ISREG(opened_identity.st_mode):
            raise RuntimeError("created output is not a regular file")
        created_identity = opened_identity
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as error:
        primary = error
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as cleanup:
                _append_cleanup_error(primary, cleanup)
        if created_identity is not None:
            try:
                current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
                same_file = (
                    stat.S_ISREG(current.st_mode)
                    and current.st_dev == created_identity.st_dev
                    and current.st_ino == created_identity.st_ino
                )
                if same_file:
                    os.unlink(path.name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except BaseException as cleanup:
                _append_cleanup_error(primary, cleanup)
        raise
    finally:
        try:
            if parent_identity is not None:
                current_parent = os.fstat(parent_fd)
                if (
                    current_parent.st_dev != parent_identity.st_dev
                    or current_parent.st_ino != parent_identity.st_ino
                ):
                    raise RuntimeError("output parent directory identity changed")
        except BaseException as cleanup:
            if primary is None:
                raise
            _append_cleanup_error(primary, cleanup)
        finally:
            try:
                os.close(parent_fd)
            except BaseException as cleanup:
                if primary is None:
                    raise
                _append_cleanup_error(primary, cleanup)


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
