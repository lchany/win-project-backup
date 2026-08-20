#!/usr/bin/env python3
import base64
import subprocess
import sys
from pathlib import Path

cmd = sys.argv[1] if len(sys.argv) > 1 else "echo ok"
b64 = base64.b64encode(cmd.encode()).decode()
script = Path(__file__).resolve().parent / "remote_exec.py"
raise SystemExit(
    subprocess.run(
        [sys.executable, str(script), "--host", "npu", "--command-b64", b64],
        check=False,
    ).returncode
)
