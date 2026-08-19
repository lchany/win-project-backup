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

python3 - <<'PY'
import re, subprocess

CFG='projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py'
KEYS=[
  'one_sided_dim_threshold',
  'gradient_fingerprint',
  'fingerprint',
  'custom_imports',
]
pat=re.compile('|'.join(re.escape(k) for k in KEYS))

def show_blob(rev):
  text=subprocess.check_output(['git','show',f'{rev}:{CFG}'], text=True, errors='replace')
  lines=text.splitlines()
  hits=[(i+1,l) for i,l in enumerate(lines) if pat.search(l)]
  print(f'=== {rev} matched {len(hits)} lines ===')
  for i,l in hits[:80]:
    print(f'{i}:{l}')

def show_diff(rev):
  diff=subprocess.check_output(['git','show',rev,'--',CFG], text=True, errors='replace')
  hits=[(i,l) for i,l in enumerate(diff.splitlines(),1) if pat.search(l)]
  print(f'=== {rev} diff matched {len(hits)} lines (in unified diff) ===')
  for i,l in hits[:120]:
    print(f'{i}:{l}')

show_blob('ed8678c')
show_diff('ed8678c')
PY
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

