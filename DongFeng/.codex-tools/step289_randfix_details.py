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
echo '=== a757f29 ==='
git log -1 --format='%h %s' a757f29
git show --stat --format='' a757f29
echo '=== a757f29 names ==='
git show --name-only --pretty=format: a757f29
echo '=== seed/det markers in HEAD key files ==='
python3 - <<'PY'
import subprocess
files=[
 'tools/train_spetr.py',
 'projects/mmdet3d_plugin/core/apis/mmdet_train.py',
 'projects/mmdet3d_plugin/datasets/builder.py',
 'projects/mmdet3d_plugin/datasets/samplers/distributed_sampler.py',
 'projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py',
 'projects/mmdet3d_plugin/models/detectors/spetr3d.py',
]
keys=('seed','deterministic','cudnn','CUBLAS','PYTHONHASH','worker_init','fix_random','randomness','np.random','torch.manual')
for f in files:
    text=subprocess.check_output(['git','show',f'HEAD:{f}'], text=True, errors='replace')
    print('FILE', f)
    hits=0
    for i,l in enumerate(text.splitlines(),1):
        if any(k.lower() in l.lower() for k in keys):
            if 'seed' in l.lower() or 'determ' in l.lower() or 'cudnn' in l.lower() or 'worker_init' in l.lower() or 'HASH' in l or 'CUBLAS' in l or 'fix_random' in l.lower() or 'randomness' in l.lower():
                print(f'  {i}:{l[:160]}')
                hits+=1
    if hits==0:
        print('  (no seed/det hits)')
PY
echo '=== 63861df vs HEAD on 3 changed: seed-related diff only ==='
git diff 63861df HEAD -- tools/train_spetr.py | python3 -c "
import sys
t=sys.stdin.read()
print('train_spetr diff bytes', len(t))
"
git log --oneline 63861df..HEAD -- tools/train_spetr.py projects/mmdet3d_plugin/models/detectors/spetr3d.py projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
echo '=== show a757f29 patch head ==='
git show a757f29 --pretty=format:'%s' -- | python3 -c "import sys; t=sys.stdin.read(); print(t[:4000])"
"""

def main():
    info=parse_machine_info()
    jump=connect(str(info["jump_host"]),int(info["jump_port"]),str(info["jump_user"]),str(info["jump_password"]))
    target=None
    try:
        tr=jump.get_transport()
        ch=tr.open_channel("direct-tcpip",(str(info["target_host"]),int(info["target_port"])),("127.0.0.1",0))
        target=connect(str(info["target_host"]),int(info["target_port"]),str(info["target_user"]),str(info["target_password"]),sock=ch)
        _,so,se=target.exec_command(CMD,timeout=60)
        out=so.read().decode("utf-8","replace"); err=se.read().decode("utf-8","replace")
        print(redact(out+err,info), end="" if (out+err).endswith("\n") else "\n")
        return so.channel.recv_exit_status()
    finally:
        if target: target.close()
        jump.close()

if __name__=="__main__":
    raise SystemExit(main())
