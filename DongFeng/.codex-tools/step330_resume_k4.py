#!/usr/bin/env python3
"""Unblock k0 rc marker and start k=4 run on back-8 only."""
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

ROOT = "diagnostics/step330_stale_q_ab_30step_back8_20260820T143600"
K4_PORT = 30194


def main() -> int:
    info = parse_machine_info()
    root = posixpath.join(str(info["shared"]).rstrip("/"), ROOT)
    k0 = posixpath.join(root, "k0_run")
    k4 = posixpath.join(root, "k4_run")
    launch = posixpath.join(root, "step330_launch_inside.sh")

    cmd = f"""
set -e
# mark k0 done if train finished
if [ -f {k0}/work/train.log ] && grep -q 'Iter \\[30/30\\]' {k0}/work/train.log; then
  echo 0 > {k0}/launcher_rc.txt
fi
# stop stuck host orchestrator only (not unrelated jobs)
pkill -f '{root}/step330_host_ab.sh' 2>/dev/null || true
sleep 2
if [ -f {k4}/launcher_rc.txt ]; then
  echo k4_already_done
  exit 0
fi
if ss -ltn | awk '{{print $4}}' | grep -q ':{K4_PORT}$'; then
  echo port_busy_{K4_PORT}
  exit 2
fi
mkdir -p {k4}/logs {k4}/work
setsid -f sh -c 'docker exec -e ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 -e STEP330_OUT={k4} -e SOAP_STALE_Q_K=4 -e MAX_ITERS=30 -e MASTER_PORT={K4_PORT} mapqr-leicheng bash --noprofile --norc {launch}' </dev/null
echo k4_started
"""
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    target = None
    try:
        ch = jump.get_transport().open_channel(
            "direct-tcpip", (str(info["target_host"]), int(info["target_port"])), ("127.0.0.1", 0)
        )
        target = connect(
            str(info["target_host"]), int(info["target_port"]),
            str(info["target_user"]), str(info["target_password"]), sock=ch,
        )
        _, stdout, stderr = target.exec_command(cmd, timeout=60)
        print(redact(stdout.read().decode("utf-8", errors="replace") + stderr.read().decode("utf-8", errors="replace"), info))
        return stdout.channel.recv_exit_status()
    finally:
        if target:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
