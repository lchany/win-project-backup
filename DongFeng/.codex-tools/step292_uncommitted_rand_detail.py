#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
cd "$REPO"
echo '=== config diff ==='
git diff HEAD -- projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
echo '=== hook init diff ==='
git diff HEAD -- projects/mmdet3d_plugin/core/hook/__init__.py
echo '=== loading.py diff ==='
git diff HEAD -- projects/mmdet3d_plugin/datasets/pipelines/loading.py
echo '=== train_spetr dirty? ==='
git diff HEAD --stat -- tools/train_spetr.py
echo '=== remaining 随机性固定 in worktree vs HEAD ==='
echo WORKTREE
git grep -l '随机性固定' -- '*.py' | wc -l
echo HEAD_committed
git grep -l '随机性固定' HEAD -- '*.py' | wc -l
"""

def main():
    info=parse_machine_info()
    jump=connect(str(info["jump_host"]),int(info["jump_port"]),str(info["jump_user"]),str(info["jump_password"]))
    try:
        _,so,se=jump.exec_command(CMD,timeout=40)
        out=so.read().decode("utf-8","replace"); err=se.read().decode("utf-8","replace")
        print(redact(out+err,info), end="" if (out+err).endswith("\n") else "\n")
        return so.channel.recv_exit_status()
    finally:
        jump.close()
if __name__=="__main__":
    raise SystemExit(main())
