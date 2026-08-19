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

echo '=== before ==='
git rev-parse --short HEAD
git log -3 --oneline
git status --short

echo '=== soft reset HEAD~2 ==='
git reset --soft HEAD~2
git reset HEAD -- .

echo '=== stage 6 files only ==='
git add -- \
  projects/mmdet3d_plugin/datasets/internal_dataset_track_stream.py \
  projects/mmdet3d_plugin/datasets/pipelines/vectorize_local_map.py \
  projects/mmdet3d_plugin/core/hook/__init__.py \
  projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py \
  projects/mmdet3d_plugin/models/detectors/spetr3d.py \
  mmcv/runner/hooks/optimizer.py

echo '=== staged names ==='
git diff --cached --name-only

echo '=== commit ==='
git commit -m '【去除随机性固定】 去除随机性固定'

echo '=== after ==='
git rev-parse --short HEAD
git log -2 --oneline
git status --short
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
        _, stdout, stderr = target.exec_command(CMD, timeout=240)
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

