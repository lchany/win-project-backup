from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

SUFFIX = r'''

# STEP-331: capture training iters 10-16 on rank0 (wait8/warmup1/active7).
custom_imports["imports"] = list(custom_imports["imports"]) + [
    "step331_cycle_profiler_hook"
]
custom_hooks = [
    dict(
        type="Step331CycleProfilerHook",
        output_dir=os.environ["STEP331_PROFILE_OUTPUT"],
        wait=8,
        warmup=1,
        active=7,
        priority="LOWEST",
    )
]
checkpoint_config = None
work_dir = os.environ["STEP331_PROFILE_WORK_DIR"]
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repo", default=os.environ.get("STEP331_REPO", ""))
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    if not args.repo:
        raise RuntimeError("STEP331_REPO is required")
    if source.count("import os") < 1:
        raise RuntimeError("source must import os")
    for marker in ("custom_imports = dict(", "checkpoint_config = dict("):
        if source.count(marker) != 1:
            raise RuntimeError(f"expected one {marker!r}, found {source.count(marker)}")
    if "Step331CycleProfilerHook" in source:
        raise RuntimeError("source already contains STEP-331 profiler overlay")

    base_runtime = Path(args.repo) / "mmdetection3d-0.17.1/configs/_base_/default_runtime.py"
    base_runtime = base_runtime.as_posix()
    source, n = re.subn(
        r"_base_ = \[.*?\]",
        f'_base_ = ["{base_runtime}"]',
        source,
        count=1,
    )
    if n != 1:
        raise RuntimeError("failed to rewrite _base_ to absolute runtime path")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(source.rstrip() + "\n" + SUFFIX, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
