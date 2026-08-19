#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
set +e
echo '=== container ==='
docker ps --filter name=^/mapqr-leicheng$ --format '{{.Names}} {{.Image}} {{.Status}}'
echo '=== npu-smi product phy8 (card4 chip0) ==='
npu-smi info -t product -i 4 -c 0 2>/dev/null | sed -n '1,40p'
echo '=== npu-smi board card4 ==='
npu-smi info -t board -i 4 2>/dev/null | sed -n '1,50p'
echo '=== inside container versions ==='
docker exec mapqr-leicheng bash --noprofile --norc -lc '
source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1
echo CANN=${ASCEND_HOME_PATH:-unset}
ls /usr/local/Ascend/ascend-toolkit/latest 2>/dev/null | head
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null | head -n 5
python - <<PY
import os, torch
print("torch", torch.__version__)
import torch_npu
print("torch_npu", getattr(torch_npu, "__version__", "?"))
print("npu_count_no_vis", torch.npu.device_count())
import mx_driving_cloud, inspect
import mx_driving_cloud.ops.linalg as L
print("mx_driving_cloud", getattr(mx_driving_cloud, "__version__", "?"))
print("mx_file", L.__file__)
print("has_fixed_shape", "QR_SOAP_FIXED_SHAPE" in inspect.getsource(L))
print("has_bypass", "MX_QR_VALIDATION_BYPASS" in inspect.getsource(L))
os.environ["ASCEND_RT_VISIBLE_DEVICES"]="8,9,10,11,12,13,14,15"
# cannot re-import torch_npu vis easily; print soc via npu-smi in container
PY
npu-smi info -t product -i 4 -c 0 2>/dev/null | sed -n "1,20p"
python - <<PY
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"]="8,9,10,11,12,13,14,15"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"]="0"
import torch, torch_npu
print("visible_count", torch.npu.device_count())
print("npu0_name", torch.npu.get_device_name(0))
print("npu2_name", torch.npu.get_device_name(2) if torch.npu.device_count()>2 else "n/a")
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
