#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

CMD = r"""
set -euo pipefail
REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
cd "$REPO"
test -d .git
test -f projects/mmdet3d_plugin/optimizers/soap.py
echo "repo=$(pwd)"
echo "head=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"

# Shared-disk diagnostics lives next to the repo, not inside it. Never touch it.
SHARED_DIAG=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics
test -d "$SHARED_DIAG"
test "$REPO/diagnostics" != "$SHARED_DIAG"

git reset HEAD -- projects/mmdet3d_plugin/optimizers/soap.py

if [ -e "$REPO/diagnostics" ]; then
  echo "removing $REPO/diagnostics"
  rm -rf -- "$REPO/diagnostics"
fi
if [ -e "$REPO/kernel_meta" ]; then
  echo "removing $REPO/kernel_meta"
  rm -rf -- "$REPO/kernel_meta"
fi
find "$REPO" -maxdepth 1 -type f -name 'no_track_huawei-maptr2-worker-0_*.pt.trace.7z' -print -delete

echo '=== leftover checks ==='
if [ -e "$REPO/diagnostics" ]; then echo "FAIL repo diagnostics still exists"; exit 1; else echo "repo diagnostics gone"; fi
if [ -e "$REPO/kernel_meta" ]; then echo "FAIL kernel_meta still exists"; exit 1; else echo "kernel_meta gone"; fi
ls "$REPO"/no_track_huawei-maptr2-worker-0_*.pt.trace.7z 2>/dev/null && { echo FAIL trace leftover; exit 1; } || echo "trace 7z gone"
test -d "$SHARED_DIAG" && echo "shared diagnostics still present (untouched)"
echo "shared diagnostics top:"; ls "$SHARED_DIAG" | wc -l

echo '=== soap ==='
python3 - <<'PY'
from pathlib import Path
text=Path("projects/mmdet3d_plugin/optimizers/soap.py").read_text(encoding="utf-8", errors="replace")
keys=("SOAP_QR_DUMP","SOAP_QR_SHAPE","SOAP_DIST_QR","pytest","unittest")
hits=[(i,line) for i,line in enumerate(text.splitlines(),1) if any(k in line for k in keys)]
print("dump_hits", len(hits))
for i,line in hits[:20]:
    print(f"{i}:{line}")
print("mx_qr", text.count("mx_driving_cloud.linalg.qr("))
print("torch_qr", text.count("torch.linalg.qr("))
print("import_mx", "import mx_driving_cloud" in text)
PY
echo '=== diff vs 669a138 soap ==='
git diff --stat 669a138 -- projects/mmdet3d_plugin/optimizers/soap.py
echo '=== git status ==='
git status --short
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
        _, stdout, stderr = target.exec_command(CMD, timeout=120)
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
