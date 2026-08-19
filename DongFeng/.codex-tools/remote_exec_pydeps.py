"""Run the existing redacting remote helper with workspace-local dependencies."""

from __future__ import annotations

import socket
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".codex-tools"
LOCAL_DEPS = TOOLS / "pydeps"

# Import the validated local package before remote_exec inserts its legacy path.
sys.path.insert(0, str(LOCAL_DEPS))
import paramiko

required = ("SSHClient", "AutoAddPolicy", "SSHException")
missing = [name for name in required if not hasattr(paramiko, name)]
if missing:
    raise ImportError("incomplete workspace Paramiko: " + ", ".join(missing))

sys.path.insert(0, str(TOOLS))
import remote_exec


if __name__ == "__main__":
    try:
        raise SystemExit(remote_exec.main())
    except (OSError, ValueError, socket.error, paramiko.SSHException) as exc:
        # Do not echo exception text: socket/auth errors may embed endpoint data.
        print(f"remote_exec failed: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(2)
