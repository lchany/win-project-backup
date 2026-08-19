"""Local-only preflight for the optional Paramiko remote transport.

This script deliberately does not read ``机器IP.md`` and does not open a socket.
It only verifies that an import candidate is a real Paramiko installation rather
than an unreadable namespace package.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = (
    ROOT / ".codex-tools" / "pydeps",
    ROOT / ".codex-tools" / "python-packages",
)


def check_candidate(candidate: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "candidate": str(candidate.relative_to(ROOT)),
        "exists": candidate.exists(),
        "usable": False,
    }
    if not result["exists"]:
        result["reason"] = "missing"
        return result

    original_path = list(sys.path)
    original_module = sys.modules.pop("paramiko", None)
    try:
        sys.path[:] = [str(candidate)] + [
            entry for entry in original_path if entry != str(candidate)
        ]
        importlib.invalidate_caches()
        module = importlib.import_module("paramiko")
        required = ("SSHClient", "AutoAddPolicy", "SSHException")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            result["reason"] = "namespace_or_incomplete_package"
            result["missing_api"] = missing
        else:
            result["usable"] = True
            result["version"] = getattr(module, "__version__", "unknown")
    except (ImportError, OSError, PermissionError) as exc:
        result["reason"] = type(exc).__name__
    finally:
        sys.modules.pop("paramiko", None)
        if original_module is not None:
            sys.modules["paramiko"] = original_module
        sys.path[:] = original_path
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        type=Path,
        help="Workspace-local dependency directory; may be specified more than once.",
    )
    args = parser.parse_args()

    candidates = tuple(args.candidate or DEFAULT_CANDIDATES)
    resolved: list[Path] = []
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else ROOT / candidate
        path = path.resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            print(json.dumps({"status": "invalid_candidate_scope"}))
            return 2
        resolved.append(path)

    checks = [check_candidate(candidate) for candidate in resolved]
    usable = next((item for item in checks if item["usable"]), None)
    print(
        json.dumps(
            {
                "status": "ready" if usable else "dependency_unavailable",
                "network_attempted": False,
                "credential_file_read": False,
                "checks": checks,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
