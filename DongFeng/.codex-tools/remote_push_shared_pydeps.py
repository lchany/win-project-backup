"""Run the shared-root uploader with workspace-local Paramiko dependencies."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".codex-tools"
LOCAL_DEPS = TOOLS / "pydeps"

sys.path.insert(0, str(LOCAL_DEPS))
import paramiko

required = ("SSHClient", "AutoAddPolicy", "SSHException")
missing = [name for name in required if not hasattr(paramiko, name)]
if missing:
    raise ImportError("incomplete workspace Paramiko: " + ", ".join(missing))

sys.path.insert(0, str(TOOLS))
import remote_push_shared


if __name__ == "__main__":
    raise SystemExit(remote_push_shared.main())
