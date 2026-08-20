#!/usr/bin/env python3
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


REL = "diagnostics/step325_qr_backend_torch_iter30_back8_20260819T230500"


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
import re
from pathlib import Path
p = Path('{remote_dir}/logs/launcher.log')
text = p.read_text(encoding='utf-8', errors='replace') if p.is_file() else ''
iters = []
for line in text.splitlines():
    m = re.search(r'Iter \[(\d+)/(\d+)\].*?loss: ([0-9.eE+-]+|nan|inf)', line, re.I)
    if m:
        iters.append((int(m.group(1)), int(m.group(2)), m.group(3)))
print('iters', len(iters), 'latest', iters[-1] if iters else None)
print('nan_loss', sum(1 for _, _, v in iters if v.lower() in ('nan', 'inf')))
PY
echo QR_FILES
docker exec mapqr-leicheng bash --noprofile --norc -lc "ls -1 '{remote_dir}/qr_tensors' 2>/dev/null | wc -l" 2>/dev/null
echo LOG_TAIL
tail -n 80 "$DIR/logs/launcher.log" 2>/dev/null || true
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
