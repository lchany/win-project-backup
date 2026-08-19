#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
cd "$REPO"
python3 - <<'PY'
import subprocess
text=subprocess.check_output(['git','show','63861df:projects/mmdet3d_plugin/optimizers/soap.py'], text=True, errors='replace')
lines=text.splitlines()
# print get_orthogonal_matrix_QR body around cpu/qr
start=None
for i,l in enumerate(lines):
    if 'def get_orthogonal_matrix_QR' in l:
        start=i
        break
if start is not None:
    for j,l in enumerate(lines[start:start+70], start+1):
        print(f'{j}:{l}')
print('--- init eigen ---')
for i,l in enumerate(lines,1):
    if any(k in l for k in ('eigh','float64','cpu()','eye(','get_orthogonal_matrix(')) and 'def ' not in l:
        if i<450 or 'eigh' in l or 'identity' in l.lower():
            if 250<=i<=450 or 'eigh' in l:
                print(f'{i}:{l}')
PY
"""

def main():
    info = parse_machine_info()
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    target=None
    try:
        transport=jump.get_transport()
        ch=transport.open_channel("direct-tcpip",(str(info["target_host"]),int(info["target_port"])),("127.0.0.1",0))
        target=connect(str(info["target_host"]),int(info["target_port"]),str(info["target_user"]),str(info["target_password"]),sock=ch)
        _,stdout,stderr=target.exec_command(CMD,timeout=40)
        out=stdout.read().decode("utf-8",errors="replace")
        err=stderr.read().decode("utf-8",errors="replace")
        print(redact(out+err,info), end="" if (out+err).endswith("\n") else "\n")
        return stdout.channel.recv_exit_status()
    finally:
        if target: target.close()
        jump.close()

if __name__=="__main__":
    raise SystemExit(main())
