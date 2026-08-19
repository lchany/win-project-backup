#!/usr/bin/env python3
"""STEP-273: re-validate STEP-271 bypass on front 8 NPUs (devices 0-7)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(
    os.environ.get(
        "STEP273_OUT",
        "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step273_front8_bypass_verify",
    )
)


def load_bad():
    import torch

    p = Path(
        "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/"
        "step260_qr_tensor_dump_30step_20260818T194457/qr_tensors/"
        "rank0_step10_ind0_192x192_BAD.pt"
    )
    return torch.load(p, map_location="cpu", weights_only=False)["A"].float().contiguous()


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
    }


def child() -> int:
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    os.environ["MX_QR_VALIDATION_BYPASS"] = "1"
    import torch
    import torch_npu  # noqa: F401
    import mx_driving_cloud

    mode = os.environ["STEP273_MODE"]
    npu = int(os.environ.get("STEP273_NPU", "0"))
    device = torch.device(f"npu:{npu}")
    A_cpu = torch.eye(192) if mode == "eye" else load_bad()
    rec = {"mode": mode, "npu": npu, "shape": list(A_cpu.shape), "bypass": "1"}
    A = A_cpu.to(device)
    try:
        Q, R = mx_driving_cloud.linalg.qr(A)
        torch.npu.synchronize()
        rec["crash"] = False
        rec["Q"] = summarize(Q)
        rec["R"] = summarize(R)
        rec["ok"] = rec["Q"]["finite"] and rec["R"]["finite"]
        if rec["ok"]:
            rec["recon_max"] = float((Q @ R - A).abs().max())
    except Exception as exc:  # noqa: BLE001
        rec["crash"] = True
        rec["ok"] = False
        rec["error_head"] = str(exc).splitlines()[0][:400]
    Path(os.environ["STEP273_CASE_OUT"]).write_text(json.dumps(rec), encoding="utf-8")
    return 0


def driver() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("eye", n) for n in range(8)] + [("bad", n) for n in range(8)]
    py = sys.executable
    script = str(Path(__file__).resolve())
    results = []
    for i, (mode, npu) in enumerate(jobs):
        outp = OUT / f"case_{i:03d}.json"
        env = os.environ.copy()
        env.update(
            {
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "MX_QR_VALIDATION_BYPASS": "1",
                "STEP273_CHILD": "1",
                "STEP273_MODE": mode,
                "STEP273_NPU": str(npu),
                "STEP273_CASE_OUT": str(outp),
            }
        )
        t0 = time.perf_counter()
        subprocess.run([py, script], env=env, capture_output=True, text=True, timeout=180, check=False)
        rec = {"mode": mode, "npu": npu, "s": round(time.perf_counter() - t0, 3)}
        if outp.is_file():
            rec.update(json.loads(outp.read_text(encoding="utf-8")))
        else:
            rec["crash"] = True
            rec["ok"] = False
        results.append(rec)
        print(
            f"{i:03d} {mode:4s} npu{npu} ok={rec.get('ok')} crash={rec.get('crash')} recon={rec.get('recon_max')}",
            flush=True,
        )
    fails = [r for r in results if not r.get("ok")]
    summary = {
        "patch": "STEP-271 MX_QR_VALIDATION_BYPASS=1",
        "visible": "0-7",
        "n": len(results),
        "n_fail": len(fails),
        "fails": [(r.get("mode"), r.get("npu"), r.get("crash"), r.get("error_head")) for r in fails],
        "results": results,
    }
    (OUT / "step273_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE fail", len(fails), "/", len(results), flush=True)
    return 0


if __name__ == "__main__":
    if os.environ.get("STEP273_CHILD") == "1":
        raise SystemExit(child())
    raise SystemExit(driver())
