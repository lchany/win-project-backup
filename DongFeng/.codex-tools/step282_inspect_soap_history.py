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
echo '=== 669a138 soap qr ==='
git show 669a138:projects/mmdet3d_plugin/optimizers/soap.py | python3 -c "
import sys
keys=('mx_driving_cloud','torch.linalg.qr','linalg.qr','SOAP_QR_DUMP','SOAP_DIST')
for i,line in enumerate(sys.stdin,1):
    if any(k in line for k in keys):
        print(f'{i}:{line.rstrip()}')
"
echo '=== HEAD soap dump hunk stat ==='
git log --oneline -5 -- projects/mmdet3d_plugin/optimizers/soap.py
echo '=== diff 669a138 HEAD soap stat ==='
git diff --stat 669a138 HEAD -- projects/mmdet3d_plugin/optimizers/soap.py
echo '=== qr_v2 backup ==='
ls -l /home/ma-user/anaconda3/envs/PyTorch-2.7.1/lib/python3.11/site-packages/mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.cpp 2>/dev/null | head -1 || true
docker exec mapqr-leicheng bash --noprofile --norc -lc '
python - <<'"'"'PY'"'"'
from pathlib import Path
p=Path("/home/ma-user/anaconda3/envs/PyTorch-2.7.1/lib/python3.11/site-packages/mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.cpp")
print("qr_v2_exists", p.is_file(), "size", p.stat().st_size if p.is_file() else None)
if p.is_file():
    t=p.read_text(encoding="utf-8", errors="replace")
    print("has_useCoreNum_guard", "useCoreNum > 0" in t)
    print("CalcQ_core0_only", t.count("if (coreId == 0)"))
    print("CalcQ_guarded", t.count("coreId == 0 && tilingInfo.useCoreNum"))
PY
ls /mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step271_qr_patch 2>/dev/null | head
'
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
        _, stdout, stderr = target.exec_command(CMD, timeout=90)
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
