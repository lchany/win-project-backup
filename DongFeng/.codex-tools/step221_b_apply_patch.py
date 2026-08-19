#!/usr/bin/env python3
"""Apply the STEP-221 stale-Q patch to a copy of soap.py, outside the repo.

Fail-closed: the source identity, every anchor count, the appended block and the
resulting syntax are all verified. The business repository is never written to.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from pathlib import Path

import step221_b_stale_q_candidate as candidate

SOURCE_SHA256 = "0e49429dbca9d9a2546c29f54e79639265f7468703ba4b36fa3b3796861a1077"
SOURCE_BYTES = 19169


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL_CLOSED: {message}")


def apply_patch(source: Path, destination: Path, repo: Path) -> dict[str, object]:
    source = source.resolve(strict=True)
    repo = repo.resolve(strict=True)
    destination = destination.resolve(strict=False)
    require(not destination.is_relative_to(repo), "destination must be outside the business repo")

    original = source.read_bytes()
    require(len(original) == SOURCE_BYTES, f"source byte size changed: {len(original)}")
    require(sha256_bytes(original) == SOURCE_SHA256, "source SHA256 is not the authoritative soap.py")

    patched = original
    applied = []
    for replacement in candidate.REPLACEMENTS:
        old = replacement["old"]
        new = replacement["new"]
        found = patched.count(old)
        require(found == replacement["count"], f"anchor {replacement['id']} count={found}")
        require(patched.count(new) == 0, f"anchor {replacement['id']} already applied")
        patched = patched.replace(old, new, replacement["count"])
        applied.append(replacement["id"])

    require(candidate.APPEND_AT_END_OF_CLASS, "candidate no longer appends at end of class")
    tail = candidate.STALE_Q_METHODS.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
    # The authoritative file ends with `        return final` and no newline.
    separator = b"" if patched.endswith(b"\n") else b"\r\n"
    patched = patched + separator + tail

    tree = ast.parse(patched.decode("utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SOAP"]
    require(len(classes) == 1, "patched file does not define exactly one SOAP class")
    methods = {node.name for node in classes[0].body if isinstance(node, ast.FunctionDef)}
    for required in (
        "_stale_q_k", "_stale_q_side_stream", "_stale_q_eligible",
        "_qr_plan", "_qr_finish", "_qr_install",
        "_stale_q_submit", "_stale_q_install_if_due", "state_dict",
        "get_orthogonal_matrix_QR", "step", "update_preconditioner",
    ):
        require(required in methods, f"patched SOAP class is missing {required}")

    # The baseline entry point must survive byte-untouched.
    def function_bytes(blob: bytes, name: str) -> bytes:
        text = blob.decode("utf-8")
        soap = next(
            node for node in ast.parse(text).body
            if isinstance(node, ast.ClassDef) and node.name == "SOAP"
        )
        target = next(
            node for node in soap.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        lines = text.splitlines(keepends=True)
        body = "".join(lines[target.lineno - 1:target.end_lineno]).encode("utf-8")
        # The authoritative file has no final newline, so appending the trio
        # terminates this function's last line. That is the only tolerated delta.
        return body.rstrip(b"\r\n")

    require(
        function_bytes(original, "get_orthogonal_matrix_QR")
        == function_bytes(patched, "get_orthogonal_matrix_QR"),
        "get_orthogonal_matrix_QR was modified; the k=0 path must stay untouched",
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(patched)
    return {
        "applied": applied,
        "source_sha256": SOURCE_SHA256,
        "patched_sha256": sha256_bytes(patched),
        "patched_bytes": len(patched),
        "added_bytes": len(patched) - len(original),
        "destination": str(destination),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    report = apply_patch(Path(args.source), Path(args.dest), Path(args.repo))
    for key, value in report.items():
        print(f"{key}={value}")
    print("PATCH_OK")
    sys.exit(0)
