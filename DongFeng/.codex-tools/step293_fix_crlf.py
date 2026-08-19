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
CFG=projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
python3 - <<'PY'
from pathlib import Path
import subprocess
p=Path("projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py")
# keep current worktree SOAP one_sided aside
wt=p.read_bytes()
head=subprocess.check_output(["git","show","HEAD:"+str(p).replace("\\","/")])
if b"\r\n" in head:
    print("already crlf")
else:
    text=head.decode("utf-8")
    crlf=text.replace("\r\n","\n").replace("\n","\r\n").encode("utf-8")
    p.write_bytes(crlf)
    print("wrote_head_content_as_crlf")
PY
git add -- "$CFG"
echo '=== staged stat ==='
git diff --cached --stat
if git diff --cached --name-only | grep -v "$CFG"; then
  echo FAIL extra
  exit 1
fi
# content vs previous commit ignoring eol should be empty
git diff --cached --ignore-cr-at-eol --stat || true
git commit -m "$(cat <<'EOF'
[去除随机性固定] 恢复正式config为CRLF换行

EOF
)"
# restore colleague one_sided=None on CRLF file
python3 - <<'PY'
from pathlib import Path
p=Path("projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py")
raw=p.read_bytes()
text=raw.decode("utf-8")
if "one_sided_dim_threshold=None" in text:
    print("already None")
elif "one_sided_dim_threshold=1024" in text:
    # preserve newline style of that line
    if "\r\n" in text:
        text=text.replace("    one_sided_dim_threshold=1024,\r\n","    one_sided_dim_threshold=None,\r\n",1)
    else:
        text=text.replace("    one_sided_dim_threshold=1024,","    one_sided_dim_threshold=None,",1)
    p.write_bytes(text.encode("utf-8"))
    print("worktree_one_sided_None")
else:
    print("one_sided line missing")
PY
echo '=== log ==='
git log -2 --oneline
echo '=== status ==='
git status --short
echo '=== crlf head cfg ==='
python3 - <<'PY'
import subprocess
new=subprocess.check_output(['git','show','HEAD:projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py'])
print('head_crlf', new.count(b'\r\n'), 'head_lf', new.count(b'\n'))
PY
"""

def main():
    info=parse_machine_info()
    jump=connect(str(info["jump_host"]),int(info["jump_port"]),str(info["jump_user"]),str(info["jump_password"]))
    try:
        _,so,se=jump.exec_command(CMD,timeout=120)
        print(redact((so.read()+se.read()).decode("utf-8","replace"), info))
        return so.channel.recv_exit_status()
    finally:
        jump.close()
if __name__=="__main__":
    raise SystemExit(main())
