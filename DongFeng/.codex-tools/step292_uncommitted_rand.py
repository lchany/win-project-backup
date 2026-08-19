#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
set -e
REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
cd "$REPO"
echo "pwd=$(pwd)"
echo "HEAD=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo '=== status ==='
git status --short
echo '=== name-status vs HEAD ==='
git diff --name-status HEAD
echo '=== unstaged/staged files matching rand markers ==='
git diff HEAD --name-only
echo '=== grep uncommitted diffs for 随机/seed/determ ==='
git diff HEAD | python3 -c "
import sys
text=sys.stdin.read()
keys=('随机性','seed','determ','shuffle','manual_seed','cudnn','set_random','rand_another','np.arange','np.random')
# print hunks: split by diff --git
parts=text.split('diff --git ')
print('n_file_diffs', max(0,len(parts)-1))
for p in parts[1:]:
    first=p.splitlines()[0] if p.splitlines() else ''
    body=p
    hit=any(k in body for k in keys)
    if hit:
        print('HIT_FILE', first)
        for i,l in enumerate(body.splitlines()):
            if l.startswith(('+++','---','@@')) or ((l.startswith('+') or l.startswith('-')) and not l.startswith(('+++','---'))):
                if any(k in l for k in keys) or l.startswith('@@') or l.startswith('+++') or l.startswith('---'):
                    print(l[:200])
"
echo '=== per-file stat ==='
git diff --stat HEAD
"""


def main() -> int:
    info = parse_machine_info()
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    try:
        _, stdout, stderr = jump.exec_command(CMD, timeout=60)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        return stdout.channel.recv_exit_status()
    finally:
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
