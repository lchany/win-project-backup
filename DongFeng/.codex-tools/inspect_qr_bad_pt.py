from __future__ import annotations

import math
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "step260_qr_bad_tensors"


def load_storages(path: Path) -> list[np.ndarray]:
    """Read raw float32 storages from a torch 2 zip .pt without importing torch."""
    arrays: list[np.ndarray] = []
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if "/data/" in n.replace("\\", "/") and not n.endswith("/")]
        names = sorted(names)
        if not names:
            names = [n for n in zf.namelist() if n.endswith(".storage") or "/data/" in n]
        for name in names:
            raw = zf.read(name)
            arr = np.frombuffer(raw, dtype=np.float32).copy()
            arrays.append(arr)
    return arrays


def stats(name: str, x: np.ndarray) -> str:
    finite = np.isfinite(x)
    n_nf = int((~finite).sum())
    xf = x[finite]
    if xf.size == 0:
        return f"{name}: all_nonfinite n={x.size}"
    return (
        f"{name}: shape={x.shape} finite={finite.all()} nonfinite={n_nf} "
        f"min={float(xf.min()):.6g} max={float(xf.max()):.6g} "
        f"mean={float(xf.mean()):.6g} absmax={float(np.abs(xf).max()):.6g}"
    )


def cond_est(A: np.ndarray) -> float:
    s = np.linalg.svd(A.astype(np.float64), compute_uv=False)
    smax = float(s.max())
    smin = float(s.min())
    if smin == 0:
        return float("inf")
    return smax / smin


def main() -> int:
    files = sorted(ROOT.glob("rank*_step10_ind0_192x192_BAD.pt"))
    print(f"files={len(files)}")
    for path in files:
        print(f"\n=== {path.name} size={path.stat().st_size} ===")
        with zipfile.ZipFile(path) as zf:
            print("zip:", zf.namelist())
        arrs = load_storages(path)
        print("n_storages", len(arrs), "lens", [a.size for a in arrs])
        mats = []
        for a in arrs:
            if a.size == 192 * 192:
                mats.append(a.reshape(192, 192))
        if len(mats) < 3:
            print("WARN expected 3 192x192 storages")
            continue
        # dump order A, Q, R as written in soap logger
        A, Q, R = mats[0], mats[1], mats[2]
        print(stats("A", A))
        print(stats("Q", Q))
        print(stats("R", R))
        print("A nan/inf", int(~np.isfinite(A).sum()), "Q", int(~np.isfinite(Q).sum()), "R", int(~np.isfinite(R).sum()))
        print("A cond2(fp64)", cond_est(A))
        print("A fro", float(np.linalg.norm(A)))
        Q64, R64 = np.linalg.qr(A.astype(np.float64), mode="reduced")
        recon = Q64 @ R64 - A.astype(np.float64)
        print(
            "cpu_fp64_qr finite",
            bool(np.isfinite(Q64).all() and np.isfinite(R64).all()),
            "recon_max",
            float(np.abs(recon).max()),
            "Q_orth_max",
            float(np.abs(Q64.T @ Q64 - np.eye(192)).max()),
        )
        q32, r32 = np.linalg.qr(A.astype(np.float32), mode="reduced")
        print(
            "cpu_fp32_qr finite",
            bool(np.isfinite(q32).all() and np.isfinite(r32).all()),
            "recon_max",
            float(np.abs(q32.astype(np.float64) @ r32.astype(np.float64) - A.astype(np.float64)).max()),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
