#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


ROOT = Path(__file__).resolve().parents[1]
MACHINE_INFO = ROOT / "机器IP.md"


def parse_secondary_host() -> tuple[str, int, str, str]:
    text = MACHINE_INFO.read_text(encoding="utf-8")
    m = re.search(r"npu2训练机器[\s\S]*?((?:\d{1,3}\.){3}\d{1,3})\s+(\S+)\s+密码\s*[:：]\s*(\S+)", text)
    if not m:
        raise ValueError("unable to parse secondary NPU host line")
    host, user, password = m.groups()
    port = int(re.search(r"npu2训练机器[\s\S]*?端口\s*[:：]\s*(\d+)", text).group(1))
    return host, port, user, password


def run(client, cmd: str, timeout: int = 1200) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> int:
    info = parse_machine_info()
    sec_host, sec_port, sec_user, sec_password = parse_secondary_host()
    info_for_redact = dict(info)
    info_for_redact["secondary_host"] = sec_host
    info_for_redact["secondary_user"] = sec_user
    info_for_redact["secondary_password"] = sec_password
    shared = str(info["shared"]).rstrip("/")
    stamp = "step299_mapqr_leicheng_migrate"
    tar_path = f"{shared}/diagnostics/{stamp}/mapqr-leicheng-committed.tar"
    image_tag = f"mapqr-leicheng:migrate-{stamp}"
    test_out = f"{shared}/diagnostics/{stamp}/qr171"

    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    primary = None
    secondary = None
    try:
        t = jump.get_transport()
        ch1 = t.open_channel("direct-tcpip", (str(info["target_host"]), int(info["target_port"])), ("127.0.0.1", 0))
        primary = connect(str(info["target_host"]), int(info["target_port"]), str(info["target_user"]), str(info["target_password"]), sock=ch1)
        ch2 = t.open_channel("direct-tcpip", (sec_host, sec_port), ("127.0.0.1", 0))
        secondary = connect(sec_host, sec_port, sec_user, sec_password, sock=ch2)

        primary_cmd = f"""
set -euo pipefail
mkdir -p "{shared}/diagnostics/{stamp}"
docker commit mapqr-leicheng {image_tag}
docker image inspect {image_tag} --format 'committed={{.Id}}'
docker save -o "{tar_path}" {image_tag}
ls -lh "{tar_path}"
"""
        print("=== primary export ===")
        st, out, err = run(primary, primary_cmd, timeout=7200)
        print(redact(out + err, info_for_redact), end="" if (out + err).endswith("\n") else "\n")
        if st != 0:
            return st

        secondary_cmd = f"""
set -euo pipefail
mkdir -p "{test_out}"
test -f "{tar_path}"
if docker ps -a --format '{{{{.Names}}}}' | grep -qx 'mapqr-leicheng'; then
  echo 'existing_mapqr_leicheng_found'
  docker rm -f mapqr-leicheng
fi
docker load -i "{tar_path}"
docker run -dit --name mapqr-leicheng --net host --privileged --shm-size=1099511627776 \
  -v /data:/data \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /var/log/npu:/usr/slog \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /mnt/sfs_turbo:/mnt/sfs_turbo \
  -w /home/ma-user \
  {image_tag} /bin/bash
docker ps --filter name=^/mapqr-leicheng$ --format '{{{{.Names}}}}\t{{{{.Image}}}}\t{{{{.Status}}}}'

docker exec mapqr-leicheng bash --noprofile --norc -lc '
source /home/ma-user/anaconda3/bin/activate /home/ma-user/anaconda3/envs/PyTorch-2.7.1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
unset MX_QR_VALIDATION_BYPASS SOAP_DIST_QR SOAP_QR_SHAPE_LOG SOAP_QR_DUMP_DIR || true
python - <<\"PY\"
import json, os, torch
import torch_npu  # noqa: F401
import mx_driving_cloud
from pathlib import Path

dump = Path("/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step260_qr_tensor_dump_30step_20260818T194457/qr_tensors/rank0_step10_ind0_192x192_BAD.pt")
obj = torch.load(dump, map_location="cpu", weights_only=False)
A_cpu = obj["A"].float().contiguous()
out = {{"visible": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), "file": dump.name, "results": []}}
for npu in [0, 1, 2]:
    rec = {{"npu": npu}}
    try:
        A = A_cpu.to(f"npu:{{npu}}")
        Q, R = mx_driving_cloud.linalg.qr(A)
        torch.npu.synchronize()
        rec["q_finite"] = bool(torch.isfinite(Q).all())
        rec["r_finite"] = bool(torch.isfinite(R).all())
        rec["ok"] = rec["q_finite"] and rec["r_finite"]
        if rec["ok"]:
            rec["recon_max"] = float((Q @ R - A).abs().max())
        else:
            bad_cols = (~torch.isfinite(Q).all(0)).nonzero(as_tuple=False).flatten().tolist()
            rec["bad_cols"] = bad_cols[:16]
            rec["bad_col_count"] = len(bad_cols)
    except Exception as exc:
        rec["ok"] = False
        rec["error"] = str(exc).splitlines()[0][:500]
        rec["error_507015"] = "507015" in str(exc)
    out["results"].append(rec)
Path("{test_out}/qr_bad192_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
PY
'
"""
        print("=== secondary load/run/test ===")
        st, out, err = run(secondary, secondary_cmd, timeout=7200)
        print(redact(out + err, info_for_redact), end="" if (out + err).endswith("\n") else "\n")
        return st
    finally:
        if secondary is not None:
            secondary.close()
        if primary is not None:
            primary.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())

