"""Legacy arbitrary-host uploader disabled by the project target-42 policy."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "disabled: arbitrary-host SSH is forbidden; use remote_push_shared.py "
        "through the guarded 42 target",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
