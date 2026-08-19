#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
set +e
echo '=== npu-smi info (AICore/HBM) ==='
npu-smi info | sed -n '1,120p'
echo '=== proc-mem all ==='
npu-smi info -t proc-mem 2>/dev/null | sed -n '1,200p'
echo '=== per-card proc front (i=0..3) ==='
for i in 0 1 2 3; do
  echo "--- card $i chip 0 ---"
  npu-smi info proc -i "$i" -c 0 2>/dev/null | sed -n '1,40p'
  echo "--- card $i chip 1 ---"
  npu-smi info proc -i "$i" -c 1 2>/dev/null | sed -n '1,40p'
done
echo '=== docker mapqr-leicheng ==='
docker ps --filter name=^/mapqr-leicheng$ --format '{{.Names}} {{.Status}}'
echo '=== container npu processes (python/train) ==='
docker exec mapqr-leicheng bash --noprofile --norc -lc 'ps -eo pid,pcpu,pmem,etime,cmd --sort=-pcpu | grep -E "python|train|torch" | grep -v grep | head -n 30'
echo '=== host python/train (redact paths later) ==='
ps -eo pid,pcpu,pmem,etime,cmd --sort=-pcpu | grep -E "python|train|torch" | grep -v grep | head -n 20
"""


def main() -> int:
    info = parse_machine_info()
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
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
        _, stdout, stderr = target.exec_command(CMD, timeout=90)
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
