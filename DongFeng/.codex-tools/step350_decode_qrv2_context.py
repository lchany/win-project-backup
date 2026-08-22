#!/usr/bin/env python3
"""Decode and validate STEP-350 QrV2 context data from raw 192x192 R."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


MAGIC = 350350.0


def decode(raw_r: torch.Tensor) -> dict[str, object]:
    if raw_r.device.type != "cpu" or raw_r.dtype != torch.float32 or tuple(raw_r.shape) != (192, 192):
        raise ValueError("raw_r must be a CPU float32 tensor with shape 192x192")
    stable_parts = [raw_r[row, : (min(row, 128) // 8) * 8] for row in range(1, 192)]
    stable = torch.cat(stable_parts)
    if stable.numel() != 15_872:
        raise RuntimeError("stable diagnostic region length mismatch")
    final_parts = [raw_r[row, 128 : 128 + ((row - 128) // 8) * 8] for row in range(129, 192)]
    final = torch.cat(final_parts)
    if final.numel() != 1_792:
        raise RuntimeError("final diagnostic region length mismatch")
    payload = torch.cat((stable, final[:512]))
    header = final[512:544]
    expected = torch.tensor(
        [MAGIC, 1, 2, 0, 0, 3, 64, 4096, 4096, 4096, 4096, 1, 2, 3, 15872, 512],
        dtype=torch.float32,
    )
    if not torch.equal(header[:16], expected) or torch.count_nonzero(header[16:]).item() != 0:
        raise RuntimeError("diagnostic header/magic/completion gate failed")
    if payload.numel() != 16_384:
        raise RuntimeError("diagnostic payload length mismatch")
    names = ("t_before_free", "v_before_free", "t_after_free", "v_after_free")
    tensors = {
        name: payload[index * 4096 : (index + 1) * 4096].reshape(64, 64).clone()
        for index, name in enumerate(names)
    }
    return {"schema": "step350-qrv2-context-v1", "header": header.clone(), **tensors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_r", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = torch.load(args.raw_r, map_location="cpu", weights_only=True)
    if not isinstance(raw, torch.Tensor):
        raise TypeError("input file must contain one tensor")
    decoded = decode(raw)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(decoded, args.output / "context.pt")
    summary = {
        "schema": decoded["schema"],
        "tensors": {
            name: {
                "finite": bool(torch.isfinite(decoded[name]).all()),
                "nan_count": int(torch.isnan(decoded[name]).sum()),
                "inf_count": int(torch.isinf(decoded[name]).sum()),
            }
            for name in ("t_before_free", "v_before_free", "t_after_free", "v_after_free")
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
