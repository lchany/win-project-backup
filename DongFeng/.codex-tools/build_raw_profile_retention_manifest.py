from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_ARTIFACT_NAMES = {
    "kernel_details.csv",
    "operator_details.csv",
    "trace_view.json",
    "step_trace_time.csv",
    "communication.json",
    "analysis.db",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory an Ascend raw profiling tree for deliberate retention. "
            "This tool never deletes or mutates the raw tree."
        )
    )
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--hash-mode",
        choices=("all", "key", "none"),
        default="all",
        help="Hash every raw file by default so later reuse can prove integrity.",
    )
    parser.add_argument(
        "--retention-reason",
        default="current_analysis_and_candidate_reuse",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.resolve(strict=True)
    output = args.output.resolve()
    if not raw_root.is_dir():
        raise NotADirectoryError(raw_root)
    if raw_root.is_symlink():
        raise RuntimeError(f"raw root must not be a symlink: {raw_root}")
    try:
        output.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise RuntimeError("retention manifest must be outside the raw profiling tree")

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    key_counts: dict[str, int] = {}
    symlinks: list[str] = []
    for path in sorted(raw_root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(raw_root).as_posix()
        if path.is_symlink():
            symlinks.append(relative)
            continue
        if not path.is_file():
            continue
        stat = path.stat()
        total_bytes += stat.st_size
        basename = path.name
        is_key = basename in KEY_ARTIFACT_NAMES or basename.startswith("op_summary_")
        if is_key:
            key_counts[basename] = key_counts.get(basename, 0) + 1
        should_hash = args.hash_mode == "all" or (args.hash_mode == "key" and is_key)
        entries.append(
            {
                "relative_path": relative,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path) if should_hash else None,
                "key_artifact": is_key,
            }
        )

    if symlinks:
        raise RuntimeError(
            "raw profiling tree contains symlinks; refusing ambiguous inventory: "
            + ", ".join(symlinks[:20])
        )
    if not entries:
        raise RuntimeError("raw profiling tree is empty")

    resolved_text = str(raw_root).encode("utf-8")
    manifest = {
        "manifest_version": 1,
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root_basename": raw_root.name,
        "raw_root_path_sha256": hashlib.sha256(resolved_text).hexdigest(),
        "retention_state": "retained",
        "retained": True,
        "retention_reason": args.retention_reason,
        "deletion_authorized": False,
        "mutation": False,
        "mutation_performed": False,
        "hash_mode": args.hash_mode,
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "key_artifact_counts": dict(sorted(key_counts.items())),
        "symlink_count": 0,
        "files": entries,
    }
    atomic_json_write(output, manifest)
    print(
        json.dumps(
            {
                "run_name": args.run_name,
                "retention_state": "retained",
                "file_count": len(entries),
                "total_bytes": total_bytes,
                "hash_mode": args.hash_mode,
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
