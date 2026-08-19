#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
cd "$REPO"
CFG=projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
echo '=== ignore space/cr stat vs parent ==='
git diff --stat 9565044 HEAD -- "$CFG"
git diff --ignore-cr-at-eol --stat 9565044 HEAD -- "$CFG"
git diff -w --stat 9565044 HEAD -- "$CFG"
echo '=== real content diff ignore cr ==='
git diff --ignore-cr-at-eol 9565044 HEAD -- "$CFG" | python3 -c "
import sys
t=sys.stdin.read()
print('bytes', len(t))
print(t[:2500])
"
echo '=== file crlf check ==='
python3 - <<'PY'
import subprocess
old=subprocess.check_output(['git','show','9565044:projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py'])
new=subprocess.check_output(['git','show','HEAD:projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py'])
print('old_crlf', old.count(b'\r\n'), 'old_lf', old.count(b'\n'))
print('new_crlf', new.count(b'\r\n'), 'new_lf', new.count(b'\n'))
print('old_has_cr', b'\r' in old, 'new_has_cr', b'\r' in new)
PY
"""

def main():
    info=parse_machine_info()
    jump=connect(str(info["jump_host"]),int(info["jump_port"]),str(info["jump_user"]),str(info["jump_password"]))
    try:
        _,so,se=jump.exec_command(CMD,timeout=40)
        print(redact((so.read()+se.read()).decode("utf-8","replace"), info))
        return so.channel.recv_exit_status()
    finally:
        jump.close()
if __name__=="__main__":
    raise SystemExit(main())
