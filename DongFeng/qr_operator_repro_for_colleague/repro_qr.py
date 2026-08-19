#!/usr/bin/env python3
"""Replay one BAD 192x192 A on mx_driving_cloud.linalg.qr. Run as a standalone process."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pt",
        type=Path,
        default=Path(__file__).resolve().parent / "inputs" / "rank0_step10_ind0_192x192_BAD.pt",
    )
    p.add_argument("--npu", type=int, default=0, help="visible npu index")
    p.add_argument("--eye", action="store_true", help="use eye(192) instead of dumped A")
    args = p.parse_args()

    import torch_npu  # noqa: F401
    import mx_driving_cloud

    torch.npu.set_device(args.npu)
    current = int(torch.npu.current_device())
    if current != args.npu:
        raise RuntimeError(f"current device mismatch: expected {args.npu}, got {current}")
    print("visible device_count", torch.npu.device_count(), "current_device", current)

    nvis = torch.npu.device_count()
    vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    print("visible", vis, "device_count", nvis, "npu", args.npu)
    if args.npu >= nvis:
        raise SystemExit(
            f"npu:{args.npu} 超出 device_count={nvis}；请检查 ASCEND_RT_VISIBLE_DEVICES。"
        )

    if args.eye:
        A = torch.eye(192, dtype=torch.float32, device=f"npu:{args.npu}")
    else:
        obj = torch.load(args.pt, map_location="cpu", weights_only=False)
        A_cpu = obj["A"].float().contiguous()
        print("meta", obj.get("meta"))
        print("A finite", bool(torch.isfinite(A_cpu).all()), "shape", tuple(A_cpu.shape), "absmax", float(A_cpu.abs().max()))
        A = A_cpu.to(f"npu:{args.npu}")

    Q, R = mx_driving_cloud.linalg.qr(A)
    torch.npu.synchronize()
    qf = bool(torch.isfinite(Q).all())
    rf = bool(torch.isfinite(R).all())
    print("Q finite", qf, "R finite", rf)
    if not qf:
        bad_cols = (~torch.isfinite(Q).all(0)).nonzero(as_tuple=False).flatten().tolist()
        print("Q nonfinite cols", bad_cols[:16], "... count", len(bad_cols))
    if qf and rf:
        print("recon_max", float((Q @ R - A).abs().max()))
    return 0 if qf and rf else 1


if __name__ == "__main__":
    raise SystemExit(main())
