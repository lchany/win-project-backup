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
echo "HEAD=$(git rev-parse --short HEAD)"
echo '=== grep 随机性固定 ==='
git grep -n '随机性固定' HEAD -- '*.py' '*.sh'
echo '=== dump contexts ==='
python3 - <<'PY'
import subprocess
files=[
 'tools/train_spetr.py',
 'projects/mmdet3d_plugin/models/detectors/spetr3d.py',
 'projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py',
 'mmcv/runner/hooks/optimizer.py',
 'projects/mmdet3d_plugin/core/hook/__init__.py',
 'projects/mmdet3d_plugin/core/hook/gradient_fingerprint_optimizer_hook.py',
 'projects/mmdet3d_plugin/datasets/internal_dataset_track_stream.py',
 'projects/mmdet3d_plugin/datasets/pipelines/vectorize_local_map.py',
]
for path in files:
    blob=subprocess.check_output(['git','show','HEAD:'+path])
    lines=blob.decode('utf-8','replace').splitlines()
    idxs=[i for i,l in enumerate(lines) if '随机性固定' in l]
    print('====', path, 'hits', len(idxs))
    for ln0 in idxs:
        s=max(0, ln0-3); e=min(len(lines), ln0+5)
        print('--', ln0+1, '--')
        for i in range(s, e):
            mark='>>' if i==ln0 else '  '
            print('%s%d:%s' % (mark, i+1, lines[i][:150]))
PY
echo '=== train_spetr live seed lines ==='
git show HEAD:tools/train_spetr.py | python3 -c "
import sys
for i,l in enumerate(sys.stdin.read().splitlines(),1):
    s=l.strip()
    if s.startswith('#'):
        continue
    low=l.lower()
    if any(k in low for k in ('seed','determ','set_random','cudnn','cublas','hashseed')):
        print('%d:%s'%(i,l[:160]))
"
"""


def main() -> int:
    info = parse_machine_info()
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    try:
        _, stdout, stderr = jump.exec_command(CMD, timeout=60)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        return stdout.channel.recv_exit_status()
    finally:
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
