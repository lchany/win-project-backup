#!/usr/bin/env python3
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


REL = "diagnostics/step307_head27b1d6d_direct30step_20260819T171800"


def main() -> int:
    info = parse_machine_info()
    remote_dir = posixpath.join(str(info["shared"]).rstrip("/"), REL)
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    target = None
    try:
        channel = jump.get_transport().open_channel(
            "direct-tcpip", (str(info["target_host"]), int(info["target_port"])), ("127.0.0.1", 0)
        )
        target = connect(str(info["target_host"]), int(info["target_port"]), str(info["target_user"]), str(info["target_password"]), sock=channel)
        command = rf"""
DIR={remote_dir}
echo rc=$(cat "$DIR/launcher_rc.txt" 2>/dev/null || echo pending)
python3 - <<'PY'
import json
import re
from pathlib import Path

p = Path('{remote_dir}/logs/launcher.log')
text = p.read_text(encoding='utf-8', errors='replace') if p.is_file() else ''
rows = []
for line in text.splitlines():
    m = re.search(r'Iter \[(\d+)/(\d+)\].*?time: ([0-9.]+).*?loss: ([0-9.eE+-]+)', line)
    if m:
        rows.append({{"iter": int(m.group(1)), "time": float(m.group(3)), "loss": float(m.group(4))}})
print('iters', len(rows), 'latest', rows[-1] if rows else None)
print('nan_loss', sum(1 for line in text.splitlines() if re.search(r'loss:\s*(?:nan|inf)', line, re.I)))
print('fatal', sum(text.count(x) for x in ('Traceback', 'RuntimeError', '507015')))
print('rows_json', json.dumps(rows, ensure_ascii=False))
PY
docker exec mapqr-leicheng bash --noprofile --norc -lc "ps -eo pid,cmd | grep 'tools/train_spetr.py' | grep -v grep | wc -l" 2>/dev/null | awk '{{print "train_rank_processes=" $1}}'
tail -n 20 "$DIR/logs/launcher.log" 2>/dev/null || true
"""
        _, stdout, stderr = target.exec_command(command, timeout=30)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        return stdout.channel.recv_exit_status()
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
