#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


CMD = r"""
set -euo pipefail
echo '=== BACK8 CLEANUP BEFORE ==='

# Heuristic: only kill processes that look like our elastic launcher / spetr test runner.
# Important: keep the match tight, otherwise our own cleanup command line (which contains the pattern
# string in the python snippet) may be matched and killed.
python3 - <<'PY'
import subprocess, re
pat = re.compile(r'(pt_elastic\\b|tools/test_spetr\\.py\\b)')
out = subprocess.check_output(['ps','-eo','pid,cmd'], text=True, errors='replace')
rows = []
for line in out.splitlines()[1:]:
    if pat.search(line):
        pid = line.strip().split(None,1)[0]
        rows.append((pid, line.strip()))
print('MATCHED_PIDS:')
for pid, line in rows:
    print(pid)
print('MATCHED_CMDS:')
for pid, line in rows:
    print(line)
PY

PIDS_TO_KILL=$(python3 - <<'PY'
import subprocess, re
pat = re.compile(r'(pt_elastic\\b|tools/test_spetr\\.py\\b)')
out = subprocess.check_output(['ps','-eo','pid,cmd'], text=True, errors='replace')
pids = []
for line in out.splitlines()[1:]:
    if pat.search(line):
        pid = line.strip().split(None,1)[0]
        if pid.isdigit():
            pids.append(pid)
print(' '.join(pids))
PY
)

if [ -n "${PIDS_TO_KILL}" ]; then
  echo "KILL -9 PIDS: ${PIDS_TO_KILL}"
  kill -9 ${PIDS_TO_KILL} || true
else
  echo 'No matched processes; nothing to kill.'
fi

sleep 3
echo '=== BACK8 CLEANUP AFTER ==='
python3 - <<'PY'
import subprocess, re
pat = re.compile(r'(pt_elastic\\b|tools/test_spetr\\.py\\b)')
out = subprocess.check_output(['ps','-eo','pid,cmd'], text=True, errors='replace')
rows = []
for line in out.splitlines()[1:]:
    if pat.search(line):
        pid = line.strip().split(None,1)[0]
        rows.append((pid, line.strip()))
print('REMAIN_PIDS:')
for pid, _ in rows:
    print(pid)
PY
"""


def main() -> int:
    info = parse_machine_info()
    jump = connect(
        str(info["jump_host"]),
        int(info["jump_port"]),
        str(info["jump_user"]),
        str(info["jump_password"]),
    )
    target = None
    try:
        channel = jump.get_transport().open_channel(
            "direct-tcpip",
            (str(info["target_host"]), int(info["target_port"])),
            ("127.0.0.1", 0),
        )
        target = connect(
            str(info["target_host"]),
            int(info["target_port"]),
            str(info["target_user"]),
            str(info["target_password"]),
            sock=channel,
        )
        _, stdout, stderr = target.exec_command(CMD, timeout=180)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        text = redact(out + err, info)
        print(text, end="" if text.endswith("\n") else "\n")
        return stdout.channel.recv_exit_status()
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())

