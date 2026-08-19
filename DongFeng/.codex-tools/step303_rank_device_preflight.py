#!/usr/bin/env python3
"""Eight-rank explicit NPU device binding and HCCL preflight."""
from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def main() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    count = int(torch.npu.device_count())
    if world != 8 or count != 8:
        raise RuntimeError(f"expected world=8 and npu_count=8, got world={world} count={count}")

    torch.npu.set_device(local_rank)
    current = int(torch.npu.current_device())
    if current != local_rank:
        raise RuntimeError(f"current device mismatch: local_rank={local_rank} current={current}")

    dist.init_process_group(backend="hccl")
    value = torch.tensor(float(rank + 1), device=f"npu:{local_rank}")
    dist.all_reduce(value)
    torch.npu.synchronize()
    reduced = float(value.cpu())
    if reduced != 36.0:
        raise RuntimeError(f"all_reduce mismatch: {reduced}")

    print(
        "STEP303_DEVICE_AUDIT "
        + json.dumps(
            {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world,
                "npu_count": count,
                "current_device": current,
                "tensor_device": str(value.device),
                "all_reduce": reduced,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
