#!/usr/bin/env python3
"""Fetch post-669a138 commit diffs from remote for adapt doc update."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact

COMMITS = ["10f897d", "5899e94", "9565044", "27b1d6d", "3a1d763"]

CMD = r"""
set -euo pipefail
REPO=/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang
OUT=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/adapt_doc_diffs_tmp
mkdir -p "$OUT"
cd "$REPO"
for c in 10f897d 5899e94 9565044 27b1d6d 3a1d763; do
  git show "$c" --format= > "$OUT/${c}.patch"
  echo "WROTE $c $(wc -c < "$OUT/${c}.patch")"
done
tar -czf "$OUT/bundle.tgz" -C "$OUT" 10f897d.patch 5899e94.patch 9565044.patch 27b1d6d.patch 3a1d763.patch
base64 -w0 "$OUT/bundle.tgz"
echo
"""


def main() -> int:
    materials = Path(__file__).resolve().parents[1] / "_adapt_doc_materials"
    materials.mkdir(parents=True, exist_ok=True)

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
        _, stdout, stderr = target.exec_command(CMD, timeout=300)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        print(redact(err, info), end="" if err.endswith("\n") else "\n")
        lines = out.strip().splitlines()
        import base64
        import tarfile
        import io

        b64 = lines[-1]
        for ln in lines[:-1]:
            print(redact(ln, info))
        data = base64.b64decode(b64)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for member in tf.getmembers():
                raw = tf.extractfile(member).read()
                name = member.name.replace(".patch", "")
                dest = materials / f"diff_{name}.patch"
                dest.write_bytes(raw)
                print(f"local {dest.name} bytes={len(raw)}")
        return stdout.channel.recv_exit_status()
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
