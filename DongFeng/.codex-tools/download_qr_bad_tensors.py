from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info

SHARED_REL = (
    "diagnostics/step260_qr_tensor_dump_30step_20260818T194457/qr_tensors"
)
NAMES = [f"rank{i}_step10_ind0_192x192_BAD.pt" for i in range(8)]
LOCAL_DIR = Path(__file__).resolve().parents[1] / "step260_qr_bad_tensors"


def main() -> int:
    info = parse_machine_info()
    shared = str(info["shared"]).rstrip("/")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    jump = connect(
        str(info["jump_host"]),
        int(info["jump_port"]),
        str(info["jump_user"]),
        str(info["jump_password"]),
    )
    try:
        sftp = jump.open_sftp()
        try:
            total = 0
            for name in NAMES:
                remote = posixpath.join(shared, SHARED_REL, name)
                local = LOCAL_DIR / name
                sftp.get(remote, str(local))
                size = local.stat().st_size
                total += size
                print(f"ok {name} {size}")
            print(f"done {len(NAMES)} total_bytes={total}")
        finally:
            sftp.close()
    finally:
        jump.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
