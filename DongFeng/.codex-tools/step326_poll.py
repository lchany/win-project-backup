#!/usr/bin/env python3
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


REL = "diagnostics/step326_torch_qr_shortterm_30step_back8_20260819T233000"


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
    m = re.search(r'Iter \[(\d+)/(\d+)\].*?time: ([0-9.]+).*?loss: ([0-9.eE+-]+|nan|inf)', line, re.I)
    if m:
        rows.append({{
            'iter': int(m.group(1)),
            'total': int(m.group(2)),
            'time_s': float(m.group(3)),
            'loss': m.group(4).lower(),
        }})
print('iters', len(rows), 'latest', rows[-1] if rows else None)
print('nan_loss', sum(1 for r in rows if r['loss'] in ('nan', 'inf')))
soap = [r for r in rows if r['iter'] in (10, 20, 30)]
if soap:
    print('soap_iters', [(r['iter'], r['time_s'], r['loss']) for r in soap])
if len(rows) >= 2:
    t2_30 = sum(r['time_s'] for r in rows if 2 <= r['iter'] <= 30)
    print('time_sum_2_30', round(t2_30, 3))
Path('{remote_dir}/summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
PY
echo LOG_TAIL
tail -n 40 "$DIR/logs/launcher.log" 2>/dev/null || true
"""
        _, stdout, stderr = target.exec_command(command, timeout=45)
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
