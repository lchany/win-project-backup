#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


CMD = r"""
set +e
echo hostname=$(hostname)
echo '=== npu-smi info ==='
npu-smi info | sed -n '1,40p'
echo '=== board card4 ==='
npu-smi info -t board -i 4 2>/dev/null | sed -n '1,30p'
echo '=== container torch device names ==='
docker ps --format '{{.Names}}' | head -n 1 | while read c; do
  docker exec "$c" bash --noprofile --norc -lc '
    source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1 2>/dev/null || true
    python - <<PY
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"]="0"
import torch, torch_npu
print("device0", torch.npu.get_device_name(0))
print("device1", torch.npu.get_device_name(1))
PY
  ' 2>/dev/null && break
done
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

