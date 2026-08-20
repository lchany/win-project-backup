#!/usr/bin/env python3
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


REL = "diagnostics/step315_qr_dump_monkeypatch_iter6_back8_20260819T222000"


def main() -> int:
    info = parse_machine_info()
    remote_dir = posixpath.join(str(info["shared"]).rstrip("/"), REL)

    jump = connect(
        str(info["jump_host"]),
        int(info["jump_port"]),
        str(info["jump_user"]),
        str(info["jump_password"]),
    )
    target = None
    try:
        channel = jump.get_transport().open_channel(
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

        command = f"""python3 - <<'PY'
from pathlib import Path
import re
from collections import Counter

dump_dir = Path(r'{remote_dir}/qr_tensors')
files = sorted([x.name for x in dump_dir.glob('*.pt')])
print('total', len(files))

nonfinite = [x for x in files if 'nonfinite' in x]
print('nonfinite_count', len(nonfinite))

rx_note = re.compile(r'^rank\\d+_.*?_qr\\d+_\\d+x\\d+_(?P<note>.+)\\.pt$')
ctr = Counter()
for fn in files:
    if 'nonfinite' in fn:
        continue
    m = rx_note.match(fn)
    if not m:
        continue
    ctr[m.group('note')] += 1
print('top_notes_excluding_nonfinite', ctr.most_common(8))

print('last_20_files')
for fn in files[-20:]:
    print(fn)
PY"""

        _, stdout, stderr = target.exec_command(command, timeout=180)
        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')
        text = redact(out + err, info)
        print(text, end='' if text.endswith('\\n') else '\\n')
        return stdout.channel.recv_exit_status()
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())

