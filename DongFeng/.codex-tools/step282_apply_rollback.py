#!/usr/bin/env python3
"""Rollback 192 linalg overlay and SOAP dump rewrite; SOAP only swaps QR to official mx_driving_cloud."""
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

LOCAL = Path(__file__).resolve().parent
REPO = "/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang"
SOAP = "projects/mmdet3d_plugin/optimizers/soap.py"
SITE = (
    "/home/ma-user/anaconda3/envs/PyTorch-2.7.1/lib/python3.11/"
    "site-packages/mx_driving_cloud"
)
LINALG = SITE + "/ops/linalg.py"
QRV2 = SITE + "/packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.cpp"
REL = "diagnostics/step282_rollback_mx_qr_overlay"


def run(client, cmd: str, timeout: int = 60):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> int:
    info = parse_machine_info()
    shared = str(info["shared"]).rstrip("/")
    remote_dir = posixpath.join(shared, REL)
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
        st, out, err = run(target, f"mkdir -p {remote_dir}")
        print("mkdir", st, redact((out + err).strip(), info))
        sftp = target.open_sftp()
        try:
            sftp.put(str(LOCAL / "linalg_official_26.0.7.py"), posixpath.join(remote_dir, "linalg.py"))
            sftp.put(str(LOCAL / "qr_v2.cpp"), posixpath.join(remote_dir, "qr_v2.cpp"))
        finally:
            sftp.close()
        cmd = f"""
set -e
REPO={REPO}
SOAP={SOAP}
LINALG={LINALG}
QRV2={QRV2}
SRC={remote_dir}
STAMP=$(date +%Y%m%dT%H%M%S)
docker exec mapqr-leicheng bash --noprofile --norc -lc "
set -e
cp -a '{LINALG}' '{remote_dir}/linalg.py.site.bak.$STAMP'
cp -a '{QRV2}' '{remote_dir}/qr_v2.cpp.site.bak.$STAMP'
cp -a '{remote_dir}/linalg.py' '{LINALG}'
cp -a '{remote_dir}/qr_v2.cpp' '{QRV2}'
python -m py_compile '{LINALG}'
python - <<'PY'
import inspect
import mx_driving_cloud.ops.linalg as L
src = inspect.getsource(L)
assert 'QR_SOAP_FIXED_SHAPE' not in src
assert 'MX_QR_VALIDATION_BYPASS' not in src
print('site_linalg_official_ok', L.__file__)
from pathlib import Path
t = Path('{QRV2}').read_text(encoding='utf-8', errors='replace')
print('qr_v2_useCoreNum_guard', 'coreId == 0 && tilingInfo.useCoreNum' in t)
PY
"
cd "$REPO"
git checkout 669a138 -- "$SOAP"
python3 - <<'PY'
from pathlib import Path
p = Path("{REPO}") / "{SOAP}"
text = p.read_text(encoding="utf-8")
if "import mx_driving_cloud" not in text:
    needle = "import torch\\n"
    if needle not in text:
        raise SystemExit("cannot find import torch anchor")
    text = text.replace(needle, "import torch\\nimport mx_driving_cloud\\n", 1)
count = text.count("torch.linalg.qr(")
if count < 1:
    raise SystemExit("no torch.linalg.qr calls")
text = text.replace("torch.linalg.qr(", "mx_driving_cloud.linalg.qr(")
p.write_text(text, encoding="utf-8")
print("replaced_qr_calls", count)
print("remaining_torch_qr", text.count("torch.linalg.qr("))
print("mx_qr_calls", text.count("mx_driving_cloud.linalg.qr("))
PY
python3 -m py_compile "$REPO/$SOAP"
echo '=== soap diffstat vs HEAD ==='
git diff --stat -- "$SOAP"
echo '=== soap diff vs 669a138 ==='
git diff 669a138 -- "$SOAP"
"""
        st, out, err = run(target, cmd, timeout=90)
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        return st
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
