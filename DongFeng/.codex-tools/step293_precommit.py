#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
set -e
REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
cd "$REPO"
echo "HEAD=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo '=== status ==='
git status --short
echo '=== log ==='
git log -6 --oneline
echo '=== six files diffstat ==='
git diff --stat HEAD -- \
  projects/mmdet3d_plugin/datasets/internal_dataset_track_stream.py \
  projects/mmdet3d_plugin/datasets/pipelines/vectorize_local_map.py \
  projects/mmdet3d_plugin/core/hook/__init__.py \
  projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py \
  projects/mmdet3d_plugin/models/detectors/spetr3d.py \
  mmcv/runner/hooks/optimizer.py
echo '=== config one_sided in worktree ==='
grep -n 'one_sided_dim_threshold' projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py | head
"""


def main() -> int:
    info = parse_machine_info()
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    try:
        _, stdout, stderr = jump.exec_command(CMD, timeout=40)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        return stdout.channel.recv_exit_status()
    finally:
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
