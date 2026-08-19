#!/usr/bin/env python3
"""STEP-285: isolated official mx_driving_cloud.linalg.qr on 8 BAD 192x192 dumps."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

DUMP_DIR = Path(
    "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/"
    "step260_qr_tensor_dump_30step_20260818T194457/qr_tensors"
)
OUT = Path(
    os.environ.get(
        "STEP285_OUT",
        "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step285_bad8_official_qr",
    )
)
FILES = [f"rank{i}_step10_ind0_192x192_BAD.pt" for i in range(8)]


def summarize(x):
    import torch

    xf = x.detach().float().cpu()
    finite = torch.isfinite(xf)
    nan_cols = torch.where(~torch.isfinite(xf).all(0))[0].tolist() if xf.ndim == 2 else []
    return {
        "finite": bool(finite.all()),
        "nonfinite": int((~finite).sum().item()),
        "nan_col_start": nan_cols[0] if nan_cols else None,
        "nan_col_end": nan_cols[-1] if nan_cols else None,
        "nan_col_count": len(nan_cols),
    }


def dump_snapshot(path: Path) -> dict:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    rec = {"file": path.name, "keys": sorted(obj.keys()) if isinstance(obj, dict) else type(obj).__name__}
    A = obj["A"].float().contiguous()
    rec["A"] = summarize(A)
    rec["A_shape"] = list(A.shape)
    rec["A_absmax"] = float(A.abs().max())
    rec["A_sum"] = float(A.sum())
    for name in ("Q", "R"):
        if name in obj and obj[name] is not None:
            rec[f"dump_{name}"] = summarize(obj[name].float())
    q, r = torch.linalg.qr(A)
    rec["cpu_torch_ok"] = bool(torch.isfinite(q).all() and torch.isfinite(r).all())
    if rec["cpu_torch_ok"]:
        rec["cpu_recon_max"] = float((q @ r - A).abs().max())
    return rec


def child() -> int:
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    os.environ.pop("MX_QR_VALIDATION_BYPASS", None)
    import torch
    import torch_npu  # noqa: F401
    import mx_driving_cloud

    npu = int(os.environ["STEP285_NPU"])
    path = Path(os.environ["STEP285_PT"])
    rec = {
        "file": path.name,
        "npu": npu,
        "visible": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "device_count": int(torch.npu.device_count()),
        "qr_mod": mx_driving_cloud.linalg.qr.__module__,
    }
    obj = torch.load(path, map_location="cpu", weights_only=False)
    A_cpu = obj["A"].float().contiguous()
    rec["A"] = summarize(A_cpu)
    device = torch.device(f"npu:{npu}")
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
        elif rec["Q"]["nan_col_start"] is not None:
            rec["q_nan_span"] = [rec["Q"]["nan_col_start"], rec["Q"]["nan_col_end"]]
    except Exception as exc:  # noqa: BLE001
        rec["crash"] = True
        rec["ok"] = False
        rec["error_head"] = str(exc).splitlines()[0][:400]
        rec["error_has_507015"] = "507015" in str(exc)
    Path(os.environ["STEP285_CASE_OUT"]).write_text(json.dumps(rec), encoding="utf-8")
    return 0


def driver() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for name in FILES:
        path = DUMP_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        snap = dump_snapshot(path)
        snapshots.append(snap)
        print(
            f"dump {name} A_finite={snap['A']['finite']} "
            f"dumpQ_finite={snap.get('dump_Q', {}).get('finite')} "
            f"cpu_ok={snap['cpu_torch_ok']}",
            flush=True,
        )
    (OUT / "dump_snapshots.json").write_text(json.dumps(snapshots, indent=2), encoding="utf-8")

    jobs = [(name, npu) for name in FILES for npu in range(8)]
    py = sys.executable
    script = str(Path(__file__).resolve())
    results = []
    for i, (name, npu) in enumerate(jobs):
        outp = OUT / f"case_{i:03d}_rank{name[4]}_npu{npu}.json"
        env = os.environ.copy()
        env.pop("MX_QR_VALIDATION_BYPASS", None)
        env.update(
            {
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "STEP285_CHILD": "1",
                "STEP285_NPU": str(npu),
                "STEP285_PT": str(DUMP_DIR / name),
                "STEP285_CASE_OUT": str(outp),
            }
        )
        t0 = time.perf_counter()
        proc = subprocess.run([py, script], env=env, capture_output=True, text=True, timeout=180)
        rec = {
            "file": name,
            "npu": npu,
            "rc": proc.returncode,
            "s": round(time.perf_counter() - t0, 3),
        }
        if outp.is_file():
            rec.update(json.loads(outp.read_text(encoding="utf-8")))
        else:
            rec["crash"] = True
            rec["ok"] = False
            rec["stderr_tail"] = (proc.stderr or "")[-500:]
            rec["error_has_507015"] = "507015" in (proc.stderr or "")
        results.append(rec)
        print(
            f"{i:02d} {name} npu{npu} ok={rec.get('ok')} crash={rec.get('crash')} "
            f"507015={rec.get('error_has_507015')} recon={rec.get('recon_max')}",
            flush=True,
        )

    fails = [r for r in results if not r.get("ok")]
    by_npu = {}
    for r in results:
        key = f"npu{r['npu']}"
        by_npu.setdefault(key, {"n": 0, "ok": 0, "crash": 0, "nonfinite": 0})
        by_npu[key]["n"] += 1
        if r.get("ok"):
            by_npu[key]["ok"] += 1
        if r.get("crash"):
            by_npu[key]["crash"] += 1
        if (not r.get("ok")) and (not r.get("crash")):
            by_npu[key]["nonfinite"] += 1
    summary = {
        "operator": "mx_driving_cloud.linalg.qr official 26.0.7, no bypass",
        "visible": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "n": len(results),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_fail": len(fails),
        "by_npu": by_npu,
        "fail_heads": [
            {
                "file": r.get("file"),
                "npu": r.get("npu"),
                "crash": r.get("crash"),
                "error_has_507015": r.get("error_has_507015"),
                "error_head": r.get("error_head"),
                "q_nan_span": r.get("q_nan_span"),
            }
            for r in fails
        ],
        "results": results,
    }
    (OUT / "step285_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE", OUT / "step285_summary.json", "fail", len(fails), "/", len(results), flush=True)
    return 0


if __name__ == "__main__":
    if os.environ.get("STEP285_CHILD") == "1":
        raise SystemExit(child())
    raise SystemExit(driver())
