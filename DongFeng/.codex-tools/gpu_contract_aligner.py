from __future__ import annotations

import argparse
import re
from pathlib import Path


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-base", required=True)
    args = parser.parse_args()

    text = Path(args.source).read_text(encoding="utf-8")
    text = re.sub(
        r'^_base_\s*=\s*\["\.\./\.\./\.\./mmdetection3d-0\.17\.1/configs/_base_/default_runtime\.py"\]$',
        f'_base_ = ["{args.runtime_base}"]',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if args.runtime_base not in text:
        raise RuntimeError("runtime base replacement failed")
    for old, new, label in (
        ("use_grid_mask=False", "use_grid_mask=True", "grid mask"),
        ("lidar_dropout_prob=0,", "lidar_dropout_prob=0.1,", "lidar dropout"),
        ("lidar_spatial_rate=0,", "lidar_spatial_rate=0.2,", "lidar spatial"),
        ("lidar_mask_ratio=0,", "lidar_mask_ratio=0.2,", "lidar mask"),
        ("dropout_sd_prob=0,", "dropout_sd_prob=0.2,", "active train dropout"),
    ):
        text = replace_exact(text, old, new, label)

    hook_pattern = re.compile(
        r"optimizer_config\s*=\s*dict\(\s*"
        r"type='GradientFingerprintOptimizerHook',\s*"
        r"grad_clip=dict\(max_norm=35, norm_type=2\),\s*"
        r"fingerprint_iters=\(\),\s*"
        r"fingerprint_phases=\(\),\s*"
        r"synchronize_after_backward=False,\s*"
        r"\)",
        re.MULTILINE,
    )
    text, count = hook_pattern.subn(
        "optimizer_config = dict(\n"
        "    type='Fp16OptimizerHookProtectGradNan', loss_scale='dynamic',\n"
        "    grad_clip=dict(max_norm=35, norm_type=2))",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"optimizer hook: expected one block, found {count}")

    Path(args.output).write_text(text, encoding="utf-8")
    print("ALIGNMENT_PATCH=PASS")


if __name__ == "__main__":
    main()
