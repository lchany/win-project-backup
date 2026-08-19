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
echo '=== current HEAD ==='
git log -1 --oneline
echo '=== soap log (qr related) ==='
git log --oneline -- projects/mmdet3d_plugin/optimizers/soap.py | head -n 25
echo '=== 63861df soap qr lines ==='
git show 63861df:projects/mmdet3d_plugin/optimizers/soap.py 2>/dev/null | python3 -c "
import sys
t=sys.stdin.read()
print('exists', bool(t))
print('mx', 'mx_driving_cloud' in t)
print('torch_qr', t.count('torch.linalg.qr'))
print('np_qr', t.count('np.linalg.qr')+t.count('numpy.linalg.qr'))
for i,l in enumerate(t.splitlines(),1):
    if 'linalg.qr' in l or 'mx_driving' in l or 'QR' in l and ('qr' in l.lower() or 'orthogonal' in l.lower()):
        if any(k in l for k in ('linalg','mx_driving','def get_orthogonal','power_iter','np.linalg')):
            print(f'{i}:{l}')
"
echo '=== parent of first soap optimize? ==='
git log --reverse --oneline -- projects/mmdet3d_plugin/optimizers/soap.py | head -n 15
echo '=== fb979b2 vs 63861df soap qr ==='
git show fb979b2:projects/mmdet3d_plugin/optimizers/soap.py 2>/dev/null | python3 -c "
import sys
t=sys.stdin.read()
print('mx', 'mx_driving_cloud' in t)
print('torch_qr', t.count('torch.linalg.qr'))
for i,l in enumerate(t.splitlines(),1):
    if 'linalg.qr' in l or 'import mx' in l:
        print(f'{i}:{l}')
" || echo 'no fb979b2'
echo '=== 669a138 soap qr ==='
git show 669a138:projects/mmdet3d_plugin/optimizers/soap.py | python3 -c "
import sys
t=sys.stdin.read()
print('mx', 'mx_driving_cloud' in t)
print('torch_qr', t.count('torch.linalg.qr'))
for i,l in enumerate(t.splitlines(),1):
    if 'linalg.qr' in l or 'import mx' in l:
        print(f'{i}:{l}')
"
echo '=== 9565044 soap qr ==='
git show 9565044:projects/mmdet3d_plugin/optimizers/soap.py | python3 -c "
import sys
t=sys.stdin.read()
print('mx', 'mx_driving_cloud' in t)
print('torch_qr', t.count('torch.linalg.qr'))
print('mx_qr', t.count('mx_driving_cloud.linalg.qr'))
for i,l in enumerate(t.splitlines(),1):
    if 'linalg.qr' in l or 'import mx' in l:
        print(f'{i}:{l}')
"
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
        _, stdout, stderr = target.exec_command(CMD, timeout=60)
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
