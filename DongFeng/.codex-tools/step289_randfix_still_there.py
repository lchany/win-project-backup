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
echo '=== HEAD ==='
git log -1 --oneline
echo '=== 63861df ancestor? ==='
git merge-base --is-ancestor 63861df HEAD && echo YES || echo NO
echo '=== 63861df commit ==='
git log -1 --oneline 63861df
echo '=== files in 63861df ==='
git show --name-only --pretty=format: 63861df
echo '=== later commits touching those files? ==='
git log --oneline 63861df..HEAD -- \
  $(git show --name-only --pretty=format: 63861df | tr -d '\r')
echo '=== key markers still in HEAD? ==='
python3 - <<'PY'
import subprocess
files=subprocess.check_output(['git','show','--name-only','--pretty=format:','63861df'], text=True).splitlines()
files=[f.strip() for f in files if f.strip()]
print('n_files', len(files))
# search common randomness markers in HEAD tree for those files
keys=('cudnn','CUBLAS','PYTHONHASHSEED','deterministic','torch.use_deterministic','cudnn.benchmark','seed','random.seed','np.random','Generator(')
for f in files:
    try:
        text=subprocess.check_output(['git','show',f'HEAD:{f}'], text=True, errors='replace')
    except subprocess.CalledProcessError:
        print('MISSING', f)
        continue
    old=subprocess.check_output(['git','show',f'63861df:{f}'], text=True, errors='replace')
    same = text==old
    print(('UNCHANGED' if same else 'CHANGED_SINCE'), f)
PY
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
        _, stdout, stderr = target.exec_command(CMD, timeout=60)
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
