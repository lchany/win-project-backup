#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
cd "$REPO"
echo '=== 63861df train_spetr seed block ==='
git show 63861df:tools/train_spetr.py | python3 -c "
import sys
lines=sys.stdin.read().splitlines()
for i,l in enumerate(lines,1):
    if any(k in l.lower() for k in ('seed','determ','cudnn','set_random')):
        if 370<=i<=650 or 'runtime_' in l:
            print(f'{i}:{l}')
"
echo '=== HEAD vs 63861df train_spetr ==='
git diff 63861df HEAD -- tools/train_spetr.py
echo '=== HEAD vs 63861df spetr3d stat ==='
git diff --stat 63861df HEAD -- projects/mmdet3d_plugin/models/detectors/spetr3d.py
echo '=== 5a37d04 ==='
git log -1 --format='%s' 5a37d04
git show --stat --format='' 5a37d04
echo '=== bf9ed6e train_spetr? ==='
git show --stat --format='%s' bf9ed6e
"""

def main():
    info=parse_machine_info()
    jump=connect(str(info["jump_host"]),int(info["jump_port"]),str(info["jump_user"]),str(info["jump_password"]))
    target=None
    try:
        tr=jump.get_transport()
        ch=tr.open_channel("direct-tcpip",(str(info["target_host"]),int(info["target_port"])),("127.0.0.1",0))
        target=connect(str(info["target_host"]),int(info["target_port"]),str(info["target_user"]),str(info["target_password"]),sock=ch)
        _,so,se=target.exec_command(CMD,timeout=40)
        out=so.read().decode(); err=se.read().decode()
        print(redact(out+err,info), end="" if (out+err).endswith("\n") else "\n")
        return so.channel.recv_exit_status()
    finally:
        if target: target.close()
        jump.close()
if __name__=="__main__":
    raise SystemExit(main())
