#!/usr/bin/env python3
"""Patch an isolated pre-3a1d763 SOAP source for training-time QR capture."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import uuid
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, got {count}")
    return text.replace(old, new, 1)


def patch_source(source: str) -> str:
    if source.count("mx_driving_cloud.linalg.qr(") != 2:
        raise RuntimeError("expected exactly two production MX QR calls")
    if "qrv2_training_capture" in source:
        raise RuntimeError("capture patch already present")
    source = replace_once(
        source,
        "import mx_driving_cloud\n",
        "import mx_driving_cloud\nimport qrv2_training_capture\n",
        "capture import",
    )
    source = replace_once(
        source,
        "            Q, _ = mx_driving_cloud.linalg.qr(power_iter)\n",
        "            Q, _ = qrv2_training_capture.qr(\n"
        "                power_iter, mx_qr=mx_driving_cloud.linalg.qr,\n"
        "                optimizer_step=int(state['step']), factor_index=ind,\n"
        "                call_site='get_orthogonal_matrix_QR',\n"
        "            )\n",
        "synchronous QR call",
    )
    source = replace_once(
        source,
        '                "original_dtype": precond_list[ind].dtype,\n',
        '                "original_dtype": precond_list[ind].dtype,\n'
        '                "optimizer_step": int(state[\'step\']),\n',
        "async plan step context",
    )
    source = replace_once(
        source,
        '            Q, _ = mx_driving_cloud.linalg.qr(entry["power_iter"])\n',
        '            Q, _ = qrv2_training_capture.qr(\n'
        '                entry["power_iter"], mx_qr=mx_driving_cloud.linalg.qr,\n'
        '                optimizer_step=entry["optimizer_step"],\n'
        '                factor_index=entry["ind"], call_site="_qr_finish",\n'
        '            )\n',
        "asynchronous QR call",
    )
    if source.count("qrv2_training_capture.qr(") != 2:
        raise RuntimeError("capture call count mismatch")
    if "mx_driving_cloud.linalg.qr(" in source:
        raise RuntimeError("unwrapped MX QR call remains")
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{64}", args.expected_sha256) is None:
        raise RuntimeError("--expected-sha256 must be lowercase 64-hex")
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.expected_sha256:
        raise RuntimeError("SOAP source SHA256 mismatch")
    patched = patch_source(raw.decode("utf-8")).encode("utf-8")
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    temporary = args.output.with_name(
        f".{args.output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(patched)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, args.output)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite output: {args.output}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
