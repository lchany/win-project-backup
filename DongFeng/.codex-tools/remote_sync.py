from __future__ import annotations

import argparse
import posixpath
import sys
from pathlib import Path, PurePosixPath

from remote_exec import connect, parse_machine_info


REPO_DIR = "l2.9-df-for-yuexiang"


def safe_remote_path(shared: str, relative: str) -> str:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("remote path must stay inside the target repository")
    return posixpath.join(shared, REPO_DIR, *rel.parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("pull", "push"))
    parser.add_argument("--remote-rel", required=True)
    parser.add_argument("--local", required=True)
    args = parser.parse_args()

    info = parse_machine_info()
    remote_path = safe_remote_path(str(info["shared"]), args.remote_rel)
    local_path = Path(args.local).resolve()

    jump = connect(
        str(info["jump_host"]),
        int(info["jump_port"]),
        str(info["jump_user"]),
        str(info["jump_password"]),
    )
    try:
        sftp = jump.open_sftp()
        try:
            if args.action == "pull":
                local_path.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote_path, str(local_path))
            else:
                if not local_path.is_file():
                    raise FileNotFoundError(local_path)
                with local_path.open("rb") as source, sftp.open(remote_path, "wb") as dest:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        dest.write(block)
            print(f"{args.action} complete: {args.remote_rel}")
            return 0
        finally:
            sftp.close()
    finally:
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
