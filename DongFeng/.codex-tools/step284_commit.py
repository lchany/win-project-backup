#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
set -euo pipefail
REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
cd "$REPO"
test "$(git rev-parse --abbrev-ref HEAD)" = "ascend_npu_optimize"

git add -- projects/mmdet3d_plugin/optimizers/soap.py
echo '=== staged ==='
git diff --cached --name-only
git diff --cached --stat

STAGED=$(git diff --cached --name-only)
if [ "$STAGED" != "projects/mmdet3d_plugin/optimizers/soap.py" ]; then
  echo "FAIL unexpected staged files:"
  printf '%s\n' "$STAGED"
  git reset HEAD -- projects/mmdet3d_plugin/optimizers/soap.py
  exit 1
fi

git commit -m "$(cat <<'EOF'
【npu性能优化】SOAP 使用 mx_driving_cloud.linalg.qr 替换 torch.linalg.qr

以 669a138 SOAP 为底接入客户 driving-cloud-ops 官方 QR，撤回 dump/诊断改写，不修改算子 kernel。
EOF
)"

echo '=== HEAD after ==='
git log -1 --format='%H%n%s%n%b'
echo '=== show stat ==='
git show --stat --oneline -1
echo '=== show names ==='
git show --name-only --pretty=format: -1
echo '=== status after ==='
git status --short
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
        _, stdout, stderr = target.exec_command(CMD, timeout=180)
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
