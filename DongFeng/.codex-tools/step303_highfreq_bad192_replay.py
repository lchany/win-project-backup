#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


CMD = r"""
set -euo pipefail
OUT=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step303_highfreq_bad192_replay
mkdir -p "$OUT"
set +e
docker exec mapqr-leicheng bash --noprofile --norc -lc '
set -euo pipefail
source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
unset MX_QR_VALIDATION_BYPASS SOAP_DIST_QR SOAP_QR_SHAPE_LOG SOAP_QR_DUMP_DIR || true
python - <<\"PY\"
import json, os, time
from pathlib import Path
import torch
import torch_npu  # noqa: F401
import mx_driving_cloud

dump = Path("/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step260_qr_tensor_dump_30step_20260818T194457/qr_tensors/rank0_step10_ind0_192x192_BAD.pt")
obj = torch.load(dump, map_location="cpu", weights_only=False)
A_cpu = obj["A"].float().contiguous()

def summarize_bad_cols(Q):
    bad = (~torch.isfinite(Q).all(0)).nonzero(as_tuple=False).flatten().tolist()
    return {"count": len(bad), "start": bad[0] if bad else None, "end": bad[-1] if bad else None}

summary = {
    "visible": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    "iters_per_device": 512,
    "file": dump.name,
    "results": [],
}

for logical in (0, 1, 2):
    rec = {"logical": logical}
    torch.npu.set_device(logical)
    rec["current_device"] = int(torch.npu.current_device())
    A = A_cpu.to(f"npu:{logical}")
    t0 = time.time()
    fail = None
    max_recon = 0.0
    for step in range(1, 513):
        try:
            Q, R = mx_driving_cloud.linalg.qr(A)
            torch.npu.synchronize()
            qf = bool(torch.isfinite(Q).all())
            rf = bool(torch.isfinite(R).all())
            if not (qf and rf):
                fail = {
                    "iter": step,
                    "type": "nonfinite",
                    "Q": summarize_bad_cols(Q),
                    "R_finite": rf,
                }
                break
            recon = float((Q @ R - A).abs().max())
            if recon > max_recon:
                max_recon = recon
        except Exception as exc:  # noqa: BLE001
            fail = {
                "iter": step,
                "type": "exception",
                "error": str(exc).splitlines()[0][:500],
                "error_507015": "507015" in str(exc),
            }
            break
    rec["elapsed_s"] = round(time.time() - t0, 3)
    rec["max_recon"] = max_recon
    rec["ok"] = fail is None
    rec["fail"] = fail
    summary["results"].append(rec)
    print(json.dumps(rec, ensure_ascii=False), flush=True)

out = Path("/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step303_highfreq_bad192_replay/summary.json")
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print("WROTE", out)
print(json.dumps(summary, ensure_ascii=False))
PY
' > "$OUT/launcher.log" 2>&1
rc=$?
cat "$OUT/launcher.log"
exit $rc
"""


def main() -> int:
    info = parse_machine_info()
    jump = connect(
        str(info["jump_host"]),
        int(info["jump_port"]),
        str(info["jump_user"]),
        str(info["jump_password"]),
    )
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
        _, stdout, stderr = target.exec_command(CMD, timeout=7200)
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

