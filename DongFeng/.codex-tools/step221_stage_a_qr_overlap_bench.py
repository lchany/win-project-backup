#!/usr/bin/env python3
"""STEP-221 Stage A: can background-stream QR hide behind foreground compute?

Read-only probe: no training, no business code, results as JSON in the run dir.
Value line (pre-declared): hidden_ratio >= 0.70 and fg_slowdown < 0.05.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch_npu  # noqa: F401

# Authoritative 23-class Q-factor inventory (543 calls/cycle), from the
# STEP-216 checkpoint contract.
COUNTS = {
    1: 106, 3: 30, 4: 6, 7: 37, 8: 1, 11: 1, 22: 1, 32: 4,
    40: 9, 64: 28, 96: 3, 120: 1, 128: 18, 160: 1, 192: 32,
    220: 4, 256: 181, 352: 1, 440: 4, 512: 43, 768: 22,
    1024: 6, 2560: 4,
}
assert sum(COUNTS.values()) == 543


def make_inputs(device: torch.device) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    mats = []
    for n in sorted(COUNTS):
        for _ in range(COUNTS[n]):
            mats.append(torch.randn(n, n, generator=gen, dtype=torch.float32).to(device))
    torch.npu.synchronize()
    return mats


def dispatch_qr(mats: list[torch.Tensor]) -> tuple[list[torch.Tensor], float]:
    start = time.perf_counter()
    outs = [torch.linalg.qr(a)[0] for a in mats]
    return outs, time.perf_counter() - start


def fg_step(a, b, aicpu_probe):
    c = a @ b
    if aicpu_probe is not None:
        # Approximate the normal step's AICPU traffic (contention probe only).
        torch.nonzero(aicpu_probe)
    return c


def timed_fg(a, b, iters, aicpu_probe):
    e0 = torch.npu.Event(enable_timing=True)
    e1 = torch.npu.Event(enable_timing=True)
    wall0 = time.perf_counter()
    e0.record()
    keep = None
    for _ in range(iters):
        keep = fg_step(a, b, aicpu_probe)
    e1.record()
    torch.npu.synchronize()
    wall = time.perf_counter() - wall0
    del keep
    return wall, e0.elapsed_time(e1) / 1000.0


def run(device_str: str, out_path: str) -> None:
    device = torch.device(device_str)
    torch.npu.set_device(device)
    mats = make_inputs(device)
    size = 4096
    a = torch.randn(size, size, dtype=torch.float32, device=device)
    b = torch.randn(size, size, dtype=torch.float32, device=device)
    bool_probe = torch.zeros(4096, dtype=torch.bool, device=device)
    bool_probe[::7] = True

    # Warmup both paths once.
    outs, _ = dispatch_qr(mats[:50])
    fg_step(a, b, bool_probe)
    torch.npu.synchronize()
    del outs

    # 1) QR alone on the default stream.
    t0 = time.perf_counter()
    outs, qr_dispatch_alone = dispatch_qr(mats)
    torch.npu.synchronize()
    t_qr_alone = time.perf_counter() - t0
    del outs

    result = {
        "device": device_str,
        "t_qr_alone_s": t_qr_alone,
        "qr_dispatch_alone_s": qr_dispatch_alone,
        "scenarios": {},
    }

    # Calibrate fg iterations so fg alone is roughly comparable to the QR wall.
    wall1, _ = timed_fg(a, b, 8, None)
    per_iter = wall1 / 8
    iters = max(8, int(t_qr_alone * 1.1 / per_iter))

    side = torch.npu.Stream(device=device)
    for name, probe in (("matmul_only", None), ("matmul_plus_aicpu", bool_probe)):
        fg_alone_wall, fg_alone_evt = timed_fg(a, b, iters, probe)

        torch.npu.synchronize()
        t0 = time.perf_counter()
        side.wait_stream(torch.npu.current_stream(device))
        with torch.npu.stream(side):
            outs, qr_dispatch = dispatch_qr(mats)
        fg_over_wall, fg_over_evt = timed_fg(a, b, iters, probe)
        side.synchronize()
        torch.npu.synchronize()
        t_overlap = time.perf_counter() - t0
        del outs

        t_seq = t_qr_alone + fg_alone_wall
        hidden = (t_seq - t_overlap) / t_qr_alone
        slowdown = fg_over_evt / fg_alone_evt - 1.0
        result["scenarios"][name] = {
            "fg_iters": iters,
            "fg_alone_wall_s": fg_alone_wall,
            "fg_alone_evt_s": fg_alone_evt,
            "qr_dispatch_overlap_s": qr_dispatch,
            "t_overlap_wall_s": t_overlap,
            "t_seq_s": t_seq,
            "hidden_ratio": hidden,
            "fg_slowdown": slowdown,
        }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print("BENCH_DONE", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.device, args.out)
