#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


CMD = r"""
python3 - <<'PY'
from pathlib import Path
root = Path('/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang')
keys = ('loss_seg', 'gt_seg_mask', 'seg_offset', 'seg_type', 'seg_color')
for path in root.rglob('*.py'):
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        continue
    hits = [k for k in keys if k in text]
    if hits:
        print(path)
        for i, line in enumerate(text.splitlines(), 1):
            if any(k in line for k in keys):
                print(f"{i}:{line}")
        print('---FILE-END---')
PY
"""


def main() -> int:
    info = parse_machine_info()
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    target = None
    try:
        channel = jump.get_transport().open_channel(
            "direct-tcpip",
            (str(info["target_host"]), int(info["target_port"])),
            ("127.0.0.1", 0),
        )
        target = connect(str(info["target_host"]), int(info["target_port"]), str(info["target_user"]), str(info["target_password"]), sock=channel)
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
