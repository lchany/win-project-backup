#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
DIR=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step285_bad8_official_qr
echo '=== pid ==='
cat "$DIR/step285.pid" 2>/dev/null || true
if [ -f "$DIR/step285.pid" ]; then
  old=$(cat "$DIR/step285.pid")
  if kill -0 "$old" 2>/dev/null; then echo running=1; else echo running=0; fi
fi
echo '=== rc ==='
cat "$DIR/launcher_rc.txt" 2>/dev/null || echo none
echo '=== cases ==='
ls "$DIR"/case_*.json 2>/dev/null | wc -l
echo '=== log tail ==='
tail -n 40 "$DIR/step285_driver.log" 2>/dev/null || echo no_log
echo '=== summary exists ==='
test -f "$DIR/step285_summary.json" && echo yes || echo no
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
        _, stdout, stderr = target.exec_command(CMD, timeout=30)
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
