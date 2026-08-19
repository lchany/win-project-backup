from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    text = args.source.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from mmdet import __version__ as mmdet_version\n"
        "from mmdet3d import __version__ as mmdet3d_version\n",
        "from mmdet import __version__ as mmdet_version\n"
        "from mmdet.apis import set_random_seed\n"
        "from mmdet3d import __version__ as mmdet3d_version\n",
        "set_random_seed import",
    )
    text = replace_once(
        text,
        'project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))\n',
        'project_root = os.environ.get(\n'
        '    "REPO_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))\n',
        "diagnostic runtime project root",
    )
    text = replace_once(
        text,
        "    # Do not pin the process RNG. Samplers that require a shared seed across\n"
        "    # distributed ranks generate and broadcast a fresh seed at runtime.\n"
        "    cfg.seed = None\n"
        "    meta['exp_name'] = osp.basename(args.config)\n",
        "    # Match the customer GPU runtime contract exactly.\n"
        "    runtime_seed = 0\n"
        "    runtime_deterministic = False\n"
        "    logger.info(f'Set random seed to {runtime_seed}, '\n"
        "                f'deterministic: {runtime_deterministic}')\n"
        "    set_random_seed(runtime_seed, deterministic=runtime_deterministic)\n"
        "    cfg.seed = runtime_seed\n"
        "    meta['seed'] = runtime_seed\n"
        "    meta['exp_name'] = osp.basename(args.config)\n",
        "runtime seed block",
    )
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"OUTPUT_SHA256={digest}")


if __name__ == "__main__":
    main()
