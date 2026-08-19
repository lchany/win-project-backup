#!/usr/bin/env python3
"""STEP-274: compare logged total loss and iter time vs GPU baseline (local GPU log only)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

GPU = Path(r"C:\project\win-project-backup\DongFeng\gpu去除随机性固定后loss.log")

# NPU rows parsed remotely from step274 launcher (Iter loss/time); do not pull full log.
NPU = {
    1: (51.638, 435.7073),
    2: (4.597, 426.184),
    3: (4.675, 420.0385),
    4: (208.162, 421.9305),
    5: (20.081, 423.992),
    6: (4.029, 415.9301),
    7: (3.967, 411.0485),
    8: (3.947, 398.8566),
    9: (4.167, 390.1657),
    10: (4.728, 386.6013),
    11: (4.402, 355.0448),
    12: (4.615, 328.0534),
    13: (4.439, 315.9492),
    14: (10.614, 302.2998),
    15: (4.334, 295.398),
    16: (4.262, 296.0328),
    17: (4.342, 276.1033),
    18: (4.703, 293.5391),
    19: (4.074, 268.777),
    20: (4.302, 273.494),
    21: (11.347, 265.6343),
    22: (7.268, 275.0368),
    23: (4.173, 258.6079),
    24: (10.300, 240.6649),
    25: (4.061, 243.9207),
    26: (4.223, 232.8462),
    27: (4.334, 231.8578),
    28: (4.304, 237.9659),
    29: (4.346, 230.8185),
    30: (10.148, 225.5574),
}


def extract_gpu(path: Path, max_it: int = 30) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"Iter \[(\d+)/", line)
        if not m:
            continue
        it = int(m.group(1))
        if it > max_it:
            continue
        tm = re.search(r"time: ([0-9.]+)", line)
        lm = re.search(r"loss: ([0-9.eE+-]+)", line)
        if tm and lm:
            out[it] = (float(tm.group(1)), float(lm.group(1)))
    return out


def main() -> int:
    gpu = extract_gpu(GPU)
    fail1: list[tuple[int, float]] = []
    fail2: list[tuple[int, float]] = []
    print("iter | NPU loss | GPU loss | diff% | <=1% | <=2%")
    for it in range(1, 31):
        if it not in gpu or it not in NPU:
            continue
        _, n_loss = NPU[it]
        _, g_loss = gpu[it]
        d = (n_loss - g_loss) / g_loss * 100
        ok1 = abs(d) <= 1.0
        ok2 = abs(d) <= 2.0
        if not ok1:
            fail1.append((it, d))
        if not ok2:
            fail2.append((it, d))
        if it in (1, 10, 11, 12, 20, 21, 22, 30) or not ok2:
            mark1 = "OK" if ok1 else "FAIL"
            mark2 = "OK" if ok2 else "FAIL"
            print(f"{it:4d} | {n_loss:8.4f} | {g_loss:8.4f} | {d:+6.2f}% | {mark1} | {mark2}")

    t_npu = sum(NPU[i][0] for i in range(2, 31))
    t_gpu = sum(gpu[i][0] for i in range(2, 31))
    print(f"\nIter2-30 sum time: NPU={t_npu:.1f}s GPU={t_gpu:.1f}s ratio={t_npu/t_gpu:.2f}x")
    print(f"<=1% pass: {30 - len(fail1)}/30  worst: {max(fail1, key=lambda x: abs(x[1])) if fail1 else 'none'}")
    print(f"<=2% pass: {30 - len(fail2)}/30  worst: {max(fail2, key=lambda x: abs(x[1])) if fail2 else 'none'}")
    soap = [10, 20, 30]
    print("SOAP step times (NPU):", {k: NPU[k][0] for k in soap})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
