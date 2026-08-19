#!/usr/bin/env python3
"""STEP-272: eye(192) mx QR probe on physical devices 0-7.

Do NOT set MX_QR_VALIDATION_BYPASS: this must hit QrV2, not the STEP-271 Python fallback.
Each case is a subprocess so one 507015 cannot poison later devices.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(
    os.environ.get(
        "STEP272_OUT",
        "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step272_front8_eye192",
    )
)


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
    os.environ.pop("MX_QR_VALIDATION_BYPASS", None)
    import torch
    import torch_npu  # noqa: F401
    import mx_driving_cloud

    npu = int(os.environ["STEP272_NPU"])
    rec = {
        "npu": npu,
        "visible": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "device_name": torch.npu.get_device_name(npu),
        "bypass_env": os.environ.get("MX_QR_VALIDATION_BYPASS"),
    }
    A = torch.eye(192, dtype=torch.float32, device=f"npu:{npu}")
    rec["npu_contiguous"] = bool(A.is_contiguous())
    try:
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
        rec["error_head"] = str(exc).splitlines()[0][:400]
    Path(os.environ["STEP272_CASE_OUT"]).write_text(json.dumps(rec), encoding="utf-8")
    return 0


def driver() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    script = str(Path(__file__).resolve())
    results = []
    for npu in range(8):
        outp = OUT / f"case_{npu:03d}.json"
        env = os.environ.copy()
        env.pop("MX_QR_VALIDATION_BYPASS", None)
        env.update(
            {
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "STEP272_CHILD": "1",
                "STEP272_NPU": str(npu),
                "STEP272_CASE_OUT": str(outp),
            }
        )
        t0 = time.perf_counter()
        proc = subprocess.run([py, script], env=env, capture_output=True, text=True, timeout=180)
        rec = {"npu": npu, "rc": proc.returncode, "s": round(time.perf_counter() - t0, 3)}
        if outp.is_file():
            rec.update(json.loads(outp.read_text(encoding="utf-8")))
        else:
            rec["crash"] = True
            rec["ok"] = False
            rec["stderr_tail"] = (proc.stderr or "")[-500:]
        results.append(rec)
        print(
            f"npu{npu} ok={rec.get('ok')} crash={rec.get('crash')} "
            f"name={rec.get('device_name')} recon={rec.get('recon_max')} "
            f"err={str(rec.get('error_head') or '')[:80]}",
            flush=True,
        )
    fails = [r for r in results if not r.get("ok")]
    summary = {
        "visible": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "n": len(results),
        "n_fail": len(fails),
        "fails": [(r.get("npu"), r.get("crash"), r.get("error_head")) for r in fails],
        "results": results,
    }
    (OUT / "step272_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE fail", len(fails), "/", len(results), flush=True)
    return 0


if __name__ == "__main__":
    if os.environ.get("STEP272_CHILD") == "1":
        raise SystemExit(child())
    raise SystemExit(driver())
