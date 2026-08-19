#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
set -euo pipefail
DUMP=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step260_qr_tensor_dump_30step_20260818T194457/qr_tensors
echo '=== dump files ==='
ls -l "$DUMP"/rank{0,1,2,3,4,5,6,7}_step10_ind0_192x192_BAD.pt
echo '=== container ==='
docker ps --filter name=^/mapqr-leicheng$ --format '{{.Names}} {{.Status}}'
echo '=== npu-smi processes 8-15 (host) ==='
npu-smi info -t proc 2>/dev/null | sed -n '1,80p' || npu-smi info | sed -n '1,50p'
echo '=== site linalg in container ==='
docker exec mapqr-leicheng bash --noprofile --norc -lc '
source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1
python - <<PY
import inspect
import mx_driving_cloud
import mx_driving_cloud.ops.linalg as L
print("pkg", getattr(mx_driving_cloud, "__version__", "?"))
print("file", L.__file__)
src = inspect.getsource(L)
print("QR_SOAP_FIXED_SHAPE", "QR_SOAP_FIXED_SHAPE" in src)
print("MX_QR_VALIDATION_BYPASS", "MX_QR_VALIDATION_BYPASS" in src)
print("has_qr", hasattr(mx_driving_cloud.linalg, "qr"))
PY
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
