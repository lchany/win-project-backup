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
test "$(git rev-parse --abbrev-ref HEAD)" = "ascend_npu_optimize"

CFG=projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
python3 - <<'PY'
from pathlib import Path
p=Path("projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py")
text=p.read_text(encoding="utf-8")
old="    one_sided_dim_threshold=None,"
new="    one_sided_dim_threshold=1024,"
if old not in text:
    raise SystemExit("one_sided None not found; abort")
p.write_text(text.replace(old, new, 1), encoding="utf-8")
print("config_one_sided_restored_to_1024_for_commit")
PY

git add -- \
  projects/mmdet3d_plugin/datasets/internal_dataset_track_stream.py \
  projects/mmdet3d_plugin/datasets/pipelines/vectorize_local_map.py \
  projects/mmdet3d_plugin/core/hook/__init__.py \
  "$CFG" \
  projects/mmdet3d_plugin/models/detectors/spetr3d.py \
  mmcv/runner/hooks/optimizer.py

echo '=== staged names ==='
git diff --cached --name-only
echo '=== staged must not contain one_sided None or eval files ==='
if git diff --cached | grep -q 'one_sided_dim_threshold=None'; then
  echo FAIL one_sided None staged
  git reset HEAD -- .
  exit 1
fi
if git diff --cached --name-only | grep -E 'eval|loading|run_|fusion_result|ddp_eval'; then
  echo FAIL extra files
  git reset HEAD -- .
  exit 1
fi
echo '=== staged stat ==='
git diff --cached --stat

git commit -m "$(cat <<'EOF'
[去除随机性固定] 去除随机性固定

EOF
)"

python3 - <<'PY'
from pathlib import Path
p=Path("projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py")
text=p.read_text(encoding="utf-8")
old="    one_sided_dim_threshold=1024,"
new="    one_sided_dim_threshold=None,"
# after commit, HEAD has 1024; worktree should keep colleague SOAP None if it was dirty
if old not in text:
    print("warn: 1024 not in worktree after commit")
else:
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("restored_worktree_one_sided_None")
PY

echo '=== HEAD after ==='
git log -1 --format='%H%n%s'
echo '=== show names ==='
git show --name-only --pretty=format: -1
echo '=== status after ==='
git status --short
"""


def main() -> int:
    info = parse_machine_info()
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    try:
        _, stdout, stderr = jump.exec_command(CMD, timeout=180)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        print(redact(out + err, info), end="" if (out + err).endswith("\n") else "\n")
        return stdout.channel.recv_exit_status()
    finally:
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
