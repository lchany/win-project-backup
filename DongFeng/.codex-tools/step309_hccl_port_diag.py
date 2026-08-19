#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


CMD = r"""
set -euo pipefail
echo HOSTNAME=$(hostname)
echo '=== HOST SS ==='
ss -ltnp 2>/dev/null || true
echo '=== HOST PS TRAIN ==='
ps -eo pid,ppid,cmd | python3 - <<'PY'
import sys
for line in sys.stdin:
    if any(k in line for k in ('train_spetr.py', 'torch.distributed.launch', 'torchrun', 'hccl')):
        print(line.rstrip())
PY
echo '=== HOST NPU-SMI ==='
npu-smi info || true
echo '=== DOCKER PS ==='
docker ps --format '{{.Names}} {{.Status}}' || true
echo '=== CONTAINER PS TRAIN ==='
docker exec mapqr-leicheng bash --noprofile --norc -lc "ps -eo pid,ppid,cmd" | python3 - <<'PY'
import sys
for line in sys.stdin:
    if any(k in line for k in ('train_spetr.py', 'torch.distributed.launch', 'torchrun', 'hccl')):
        print(line.rstrip())
PY
echo '=== CONTAINER SS ==='
docker exec mapqr-leicheng bash --noprofile --norc -lc "ss -ltnp 2>/dev/null || true"
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
        transport = jump.get_transport()
        channel = transport.open_channel(
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
        _, stdout, stderr = target.exec_command(CMD, timeout=120)
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
