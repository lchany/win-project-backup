#!/usr/bin/env python3
"""Eight-rank NPU liveness probe; does not execute the mechanism benchmark."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-dir", required=True)
    parser.add_argument("--hold-seconds", type=int, default=45)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    marker = torch.ones(1, device=f"npu:{local_rank}") * (rank + 1)
    torch.npu.synchronize(local_rank)
    ready = Path(args.ready_dir)
    ready.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "visible": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "npu_available": bool(torch.npu.is_available()),
        "device_count": int(torch.npu.device_count()),
        "marker": int(marker.cpu().item()),
    }
    (ready / f"rank{rank}.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8")
    dist.barrier()
    time.sleep(args.hold_seconds)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
