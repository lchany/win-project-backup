from __future__ import annotations

import argparse
from pathlib import Path


SUFFIX = r'''

# One-time, rank-0-only profiling overlay. The source config above is an exact
# repository-external copy of the GPU-aligned NPU runtime contract.
custom_imports["imports"] = list(custom_imports["imports"]) + [
    "gpu_contract_fullstage_profiler_hook"
]
custom_hooks = [
    dict(
        type="GPUContractFullStageProfilerHook",
        output_dir=os.environ["GPU_CONTRACT_PROFILE_OUTPUT"],
        wait=22,
        warmup=1,
        active=4,
        priority="LOWEST",
    )
]
checkpoint_config = None
work_dir = os.environ["GPU_CONTRACT_PROFILE_WORK_DIR"]
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    if source.count("import os") < 1:
        raise RuntimeError("source must import os")
    required_unique = (
        "custom_imports = dict(",
        "gradient_fingerprint_optimizer_hook",
        "checkpoint_config = dict(",
    )
    for marker in required_unique:
        count = source.count(marker)
        if count != 1:
            raise RuntimeError(f"expected one {marker!r}, found {count}")
    if "GPUContractFullStageProfilerHook" in source:
        raise RuntimeError("source already contains profiler overlay")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(source.rstrip() + "\n" + SUFFIX, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
