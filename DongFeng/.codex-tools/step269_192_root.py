#!/usr/bin/env python3
"""STEP-269: isolate mx QR failure condition for the dumped 192x192 A.

Each case is a subprocess. Tests layout, alignment, and workspace reuse.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

DUMP = Path(
    "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/"
    "step260_qr_tensor_dump_30step_20260818T194457/qr_tensors/"
    "rank0_step10_ind0_192x192_BAD.pt"
)
OUT_DIR = Path(os.environ.get("STEP269_OUT", "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step269_192_root"))


def load_A():
    import torch

    return torch.load(DUMP, map_location="cpu", weights_only=False)["A"].float()


def make_A(kind: str):
    import torch

    A = load_A()
    if kind == "loaded_as_is":
        return A
    if kind == "clone":
        return A.clone()
    if kind == "contiguous":
        return A.contiguous()
    if kind == "clone_contiguous":
        return A.clone().contiguous()
    if kind == "t_contiguous":
        return A.t().contiguous()
    if kind == "t_only":
        return A.t()  # likely non-contiguous
    if kind == "narrow_from_256":
        big = torch.zeros(256, 256)
        big[:192, :192] = A
        return big.narrow(0, 0, 192).narrow(1, 0, 192)
    if kind == "offset_1_from_193":
        big = torch.zeros(193, 193)
        big[1:, 1:] = A
        return big.narrow(0, 1, 192).narrow(1, 1, 192)
    if kind == "channels_last_like":
        x = A.unsqueeze(0).unsqueeze(0).contiguous(memory_format=torch.channels_last).squeeze()
        return x
    if kind == "add_zero":
        return (A + 0).contiguous()
    raise KeyError(kind)


def summarize(x):
    import torch

    xf = x.detach().float().cpu()
    finite = torch.isfinite(xf)
    nan_cols = torch.where(~torch.isfinite(xf).all(0))[0].tolist() if xf.ndim == 2 else []
    return {
        "finite": bool(finite.all()),
        "nonfinite": int((~finite).sum().item()),
        "nan_col_start": nan_cols[0] if nan_cols else None,
        "nan_col_count": len(nan_cols),
        "absmax": float(xf[finite].abs().max()) if bool(finite.any()) else None,
    }


def child() -> int:
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    import torch_npu  # noqa: F401
    import mx_driving_cloud

    kind = os.environ["STEP269_KIND"]
    warmup = int(os.environ.get("STEP269_WARMUP", "0"))
    npu = int(os.environ.get("STEP269_NPU", "0"))
    device = torch.device(f"npu:{npu}")
    A_cpu = make_A(kind)
    rec = {
        "kind": kind,
        "warmup": warmup,
        "npu": npu,
        "shape": list(A_cpu.shape),
        "cpu_contiguous": bool(A_cpu.is_contiguous()),
        "cpu_stride": list(A_cpu.stride()),
        "storage_offset": int(A_cpu.storage_offset()),
        "storage_size": int(A_cpu.untyped_storage().size() // 4),
    }
    A = A_cpu.to(device)
    rec["npu_contiguous"] = bool(A.is_contiguous())
    rec["npu_stride"] = list(A.stride())
    rec["npu_offset"] = int(A.storage_offset())
    try:
        for i in range(warmup):
            w = torch.randn(192, 192, device=device)
            q, r = mx_driving_cloud.linalg.qr(w)
            del q, r, w
        torch.npu.synchronize()
        Q, R = mx_driving_cloud.linalg.qr(A)
        torch.npu.synchronize()
        rec["crash"] = False
        rec["Q"] = summarize(Q)
        rec["R"] = summarize(R)
        if rec["Q"]["finite"] and rec["R"]["finite"]:
            rec["recon_max"] = float((Q @ R - A).abs().max())
            rec["ok"] = rec["recon_max"] < 1e-3
        else:
            rec["ok"] = False
            rec["nan_last64"] = rec["Q"]["nan_col_start"] == 128
    except Exception as exc:  # noqa: BLE001
        rec["crash"] = True
        rec["ok"] = False
        rec["error"] = type(exc).__name__
        rec["error_head"] = str(exc).splitlines()[0][:300]
    q, r = torch.linalg.qr(A_cpu.contiguous())
    rec["cpu_ok"] = bool(torch.isfinite(q).all() and torch.isfinite(r).all())
    Path(os.environ["STEP269_CASE_OUT"]).write_text(json.dumps(rec), encoding="utf-8")
    return 0


def driver() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for kind in (
        "loaded_as_is",
        "clone",
        "contiguous",
        "clone_contiguous",
        "t_contiguous",
        "t_only",
        "narrow_from_256",
        "offset_1_from_193",
        "add_zero",
    ):
        jobs.append((kind, 0, 0))
    for warmup in (0, 1, 8, 32, 64, 128):
        jobs.append(("loaded_as_is", warmup, 0))
        jobs.append(("clone_contiguous", warmup, 0))
    # 8-card exact loaded tensor, no warmup
    for npu in range(int(os.environ.get("STEP269_NPU_COUNT", "8"))):
        jobs.append(("loaded_as_is", 0, npu))

    results = []
    py = sys.executable
    script = str(Path(__file__).resolve())
    for i, (kind, warmup, npu) in enumerate(jobs):
        outp = OUT_DIR / f"case_{i:03d}.json"
        env = os.environ.copy()
        env.update(
            {
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "STEP269_CHILD": "1",
                "STEP269_KIND": kind,
                "STEP269_WARMUP": str(warmup),
                "STEP269_NPU": str(npu),
                "STEP269_CASE_OUT": str(outp),
            }
        )
        t0 = time.perf_counter()
        proc = subprocess.run([py, script], env=env, capture_output=True, text=True, timeout=180)
        rec = {"kind": kind, "warmup": warmup, "npu": npu, "rc": proc.returncode, "s": round(time.perf_counter() - t0, 3)}
        if outp.is_file():
            rec.update(json.loads(outp.read_text(encoding="utf-8")))
        else:
            rec["crash"] = True
            rec["ok"] = False
            rec["stderr_tail"] = (proc.stderr or "")[-400:]
        results.append(rec)
        print(
            f"{i:03d} {kind:20s} wu={warmup:3d} npu{npu} ok={rec.get('ok')} crash={rec.get('crash')} "
            f"contig={rec.get('npu_contiguous')} stride={rec.get('npu_stride')} "
            f"nan={rec.get('Q', {}).get('nan_col_count') if isinstance(rec.get('Q'), dict) else None}",
            flush=True,
        )

    fails = [r for r in results if not r.get("ok")]
    summary = {
        "n": len(results),
        "n_fail": len(fails),
        "fail_kinds": [(r.get("kind"), r.get("warmup"), r.get("npu"), r.get("crash"), r.get("error_head")) for r in fails],
        "results": results,
    }
    (OUT_DIR / "step269_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE fail", len(fails), "/", len(results), flush=True)
    return 0


if __name__ == "__main__":
    if os.environ.get("STEP269_CHILD") == "1":
        raise SystemExit(child())
    raise SystemExit(driver())
