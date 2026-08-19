#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


def main() -> int:
    info = parse_machine_info()
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    target = None
    try:
        channel = jump.get_transport().open_channel(
            "direct-tcpip", (str(info["target_host"]), int(info["target_port"])), ("127.0.0.1", 0)
        )
        target = connect(str(info["target_host"]), int(info["target_port"]), str(info["target_user"]), str(info["target_password"]), sock=channel)
        command = r'''
echo CONTAINER
docker ps --format '{{.Names}} {{.Status}}' | grep '^mapqr-leicheng ' || true
echo NPU_PROCESSES
npu-smi info 2>/dev/null | sed -n '/Process id/,$p' | head -n 80 || true
echo TRAIN_ENVS
for p in $(docker exec mapqr bash --noprofile --norc -lc "pgrep -f 'tools/train_spetr.py'" 2>/dev/null | head -n 8); do
  docker exec mapqr bash --noprofile --norc -lc "tr '\0' '\n' </proc/$p/environ | grep -E '^(RANK|LOCAL_RANK|WORLD_SIZE|MASTER_PORT|ASCEND_RT_VISIBLE_DEVICES|HCCL_IF_BASE_PORT|HCCL_IF_IP|HCCL_SOCKET_IFNAME)=' | sort" 2>/dev/null
  break
done
echo HCCL_ENV_DEFINITIONS
docker exec mapqr-leicheng bash --noprofile --norc -lc "grep -R -n -m 8 'HCCL_IF_BASE_PORT' /usr/local/Ascend/ascend-toolkit/latest/include /usr/local/Ascend/driver/include 2>/dev/null" || true
echo HCCL_RELEVANT_ENV_NAMES
docker exec mapqr-leicheng bash --noprofile --norc -lc "grep -Rho 'HCCL_[A-Z0-9_]*PORT[A-Z0-9_]*' /usr/local/Ascend/ascend-toolkit/latest/include /usr/local/Ascend/driver/include 2>/dev/null | sort -u" || true
echo DEVICE_IPS
for i in $(seq 0 15); do
  printf 'dev=%s ' "$i"
  hccn_tool -i "$i" -ip -g 2>/dev/null | tr '\n' ' '; echo
done
echo DEVICE_LINKS
for i in 0 7 8 15; do
  printf 'dev=%s ' "$i"
  hccn_tool -i "$i" -link -g 2>/dev/null | head -n 2 | tr '\n' ' '; echo
done
'''
        _, stdout, stderr = target.exec_command(command, timeout=90)
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
