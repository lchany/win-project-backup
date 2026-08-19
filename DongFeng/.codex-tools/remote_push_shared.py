from __future__ import annotations

import argparse
import posixpath
import sys
from pathlib import Path, PurePosixPath

from remote_exec import connect, parse_machine_info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-rel", required=True)
    parser.add_argument("--local", required=True)
    args = parser.parse_args()

    rel = PurePosixPath(args.remote_rel)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("remote path must stay inside the configured shared root")
    source = Path(args.local).resolve(strict=True)
    info = parse_machine_info()
    remote = posixpath.join(str(info["shared"]), *rel.parts)

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
        sftp = target.open_sftp()
        try:
            sftp.put(str(source), remote)
        finally:
            sftp.close()
        print(f"push complete: {args.remote_rel}")
        return 0
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
