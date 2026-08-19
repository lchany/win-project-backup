#!/usr/bin/env python3
"""STEP-280: one-by-one mx QrV2 vs CPU FP64 QR over real SOAP square shapes.

Runs inside mapqr-leicheng. Does not use the site-packages 192 Python bypass:
>80 always calls mx_driving_cloud._C.qr (QrV2). <=80 follows the official
wrapper and uses torch.linalg.qr (AICPU).

CPU golden is torch.linalg.qr(A.double().cpu()), the previous SOAP CPU scheme.
Each (shape, case, npu) is an isolated process so 507015 cannot poison later cases.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

SOAP_SHAPES = [
    1, 3, 4, 7, 8, 11, 22, 32, 40, 64,
    96, 120, 128, 160, 192, 220, 256, 352, 440, 512, 768, 1024, 2560, 5120,
]
SOAP_COUNTS = {
    1: 106, 3: 30, 4: 6, 7: 37, 8: 1, 11: 1, 22: 1, 32: 4,
    40: 9, 64: 28, 96: 3, 120: 1, 128: 18, 160: 1, 192: 32,
    220: 4, 256: 181, 352: 1, 440: 4, 512: 43, 768: 22,
    1024: 6, 2560: 8, 5120: 4,
}
BAD192 = Path(
    "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/"
    "step260_qr_tensor_dump_30step_20260818T194457/qr_tensors/"
    "rank0_step10_ind0_192x192_BAD.pt"
)
SAMPLE192 = Path(str(BAD192).replace("_BAD.pt", "_SAMPLE.pt"))
OUT = Path(
    os.environ.get(
        "STEP280_OUT",
        "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step280_qr_cpu_vs_mx",
    )
)
RECON_FAIL = 1.0e-3
QR_AICPU_THRESHOLD = 80
BLOCK = 64


def path_kind(n: int) -> str:
    return "aicpu" if n <= QR_AICPU_THRESHOLD else "qrv2"


def load_case(n: int, case: str):
    import torch

    if case == "eye":
        return torch.eye(n, dtype=torch.float32)
    if case == "randn":
        g = torch.Generator().manual_seed(20260819 + n)
        return torch.randn(n, n, generator=g, dtype=torch.float32)
    if case == "small":
        g = torch.Generator().manual_seed(20260819 + 10000 + n)
        return torch.randn(n, n, generator=g, dtype=torch.float32) * 1.0e-8
    if case == "bad192":
        return torch.load(BAD192, map_location="cpu", weights_only=False)["A"].float().contiguous()
    if case == "sample192":
        return torch.load(SAMPLE192, map_location="cpu", weights_only=False)["A"].float().contiguous()
    raise KeyError(case)


def mx_qr_force_kernel(A):
    """Official linalg.py pad + _C.qr, skipping the 192 Python bypass."""
    import torch
    import torch.nn.functional as F
    import mx_driving_cloud

    dim = A.shape
    if dim[0] <= QR_AICPU_THRESHOLD or dim[1] <= QR_AICPU_THRESHOLD:
        return torch.linalg.qr(A)
    lda = max(int(dim[0]), int(dim[1]))
    pad = (BLOCK - (lda % BLOCK)) % BLOCK
    lda_pad = lda + pad
    A_pad = F.pad(A, (0, lda_pad - int(dim[1]), 0, lda_pad - int(dim[0]))).contiguous()
    Q, R = mx_driving_cloud._C.qr(A_pad)
    Q = Q[: dim[0], : dim[0]]
    R = torch.triu(R[: dim[0], : dim[1]])
    return Q, R


def finite_stats(x):
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


def align_max_abs(a, b):
    import torch

    signs = torch.sign((a * b).sum(0, keepdim=True))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return float((a * signs - b).abs().max().item())


def child() -> int:
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    import torch_npu  # noqa: F401
    import mx_driving_cloud  # noqa: F401

    n = int(os.environ["STEP280_N"])
    case = os.environ["STEP280_CASE"]
    npu = int(os.environ["STEP280_NPU"])
    outp = Path(os.environ["STEP280_OUT_JSON"])
    rec = {
        "n": n,
        "case": case,
        "npu": npu,
        "path": path_kind(n),
        "soap_count": SOAP_COUNTS.get(n),
        "visible": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "ok": False,
        "crash": False,
        "fail_reason": None,
    }
    try:
        A = load_case(n, case).contiguous()
        rec["A"] = {
            "shape": list(A.shape),
            "absmax": float(A.abs().max().item()),
            "finite": bool(torch.isfinite(A).all().item()),
        }
        A64 = A.double()
        t0 = time.perf_counter()
        Qc, Rc = torch.linalg.qr(A64)
        rec["cpu64_s"] = round(time.perf_counter() - t0, 4)
        rec["cpu64_finite"] = bool(torch.isfinite(Qc).all() and torch.isfinite(Rc).all())
        rec["cpu64_recon"] = float((Qc @ Rc - A64).abs().max().item()) if rec["cpu64_finite"] else None
        A32 = A.float()
        Q32, R32 = torch.linalg.qr(A32)
        rec["cpu32_finite"] = bool(torch.isfinite(Q32).all() and torch.isfinite(R32).all())
        rec["cpu32_recon"] = float((Q32 @ R32 - A32).abs().max().item()) if rec["cpu32_finite"] else None

        An = A32.to(f"npu:{npu}")
        t1 = time.perf_counter()
        Qn, Rn = mx_qr_force_kernel(An)
        torch.npu.synchronize()
        rec["npu_s"] = round(time.perf_counter() - t1, 4)
        rec["Q"] = finite_stats(Qn)
        rec["R"] = finite_stats(Rn)
        if rec["Q"]["finite"] and rec["R"]["finite"]:
            recon = (Qn.float() @ Rn.float() - An).abs().max()
            rec["npu_recon"] = float(recon.item())
            q_cpu = Q32.to(Qn.device)
            rec["q_vs_cpu32_aligned"] = align_max_abs(Qn.float(), q_cpu)
            if rec["npu_recon"] > RECON_FAIL:
                rec["fail_reason"] = "recon"
            else:
                rec["ok"] = True
        else:
            rec["npu_recon"] = None
            rec["fail_reason"] = "nonfinite"
    except Exception as exc:
        rec["crash"] = True
        rec["ok"] = False
        rec["fail_reason"] = "exception"
        rec["exc_type"] = type(exc).__name__
        rec["exc"] = str(exc)[:300]
        rec["tb_tail"] = traceback.format_exc()[-500:]
        blob = (rec.get("exc") or "") + rec.get("tb_tail", "")
        rec["hit_507015"] = "507015" in blob or "DDR address out of range" in blob
    outp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return 0 if rec.get("ok") else 1


def jobs():
    out = []
    for n in SOAP_SHAPES:
        for case in ("eye", "randn", "small"):
            out.append((n, case, 0))
        if n > QR_AICPU_THRESHOLD:
            out.append((n, "eye", 2))
    if BAD192.is_file():
        out.append((192, "bad192", 0))
    if SAMPLE192.is_file():
        out.append((192, "sample192", 0))
    return out


def driver() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = jobs()
    py = sys.executable
    script = str(Path(__file__).resolve())
    jsonl = OUT / "results.jsonl"
    if jsonl.exists():
        jsonl.unlink()
    results = []
    print(f"STEP280 jobs={len(cases)} out={OUT}", flush=True)
    for i, (n, case, npu) in enumerate(cases):
        case_json = OUT / f"case_{i:03d}_{n}_{case}_npu{npu}.json"
        env = os.environ.copy()
        env.update(
            {
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "STEP280_CHILD": "1",
                "STEP280_N": str(n),
                "STEP280_CASE": case,
                "STEP280_NPU": str(npu),
                "STEP280_OUT_JSON": str(case_json),
                "MX_QR_VALIDATION_BYPASS": "0",
            }
        )
        timeout = 240 if n >= 2560 else 90
        t0 = time.perf_counter()
        proc = subprocess.run(
            [py, "-u", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        rec = {
            "i": i,
            "n": n,
            "case": case,
            "npu": npu,
            "path": path_kind(n),
            "rc": proc.returncode,
            "wall_s": round(time.perf_counter() - t0, 3),
        }
        if case_json.is_file():
            rec.update(json.loads(case_json.read_text(encoding="utf-8")))
        else:
            rec["ok"] = False
            rec["crash"] = True
            rec["fail_reason"] = "no_json"
            rec["stderr_tail"] = (proc.stderr or "")[-400:]
            rec["hit_507015"] = "507015" in (proc.stderr or "")
        if not rec.get("ok"):
            rec["stderr_tail"] = rec.get("stderr_tail") or (proc.stderr or "")[-400:]
        results.append(rec)
        with jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        mark = "PASS" if rec.get("ok") else "FAIL"
        print(
            f"{i:03d}/{len(cases)} {mark} n={n:4d} {case:9s} npu{npu} "
            f"path={rec.get('path')} recon={rec.get('npu_recon')} "
            f"reason={rec.get('fail_reason')} 507015={rec.get('hit_507015')} "
            f"{rec['wall_s']}s",
            flush=True,
        )

    fails = [r for r in results if not r.get("ok")]
    by_shape = {}
    for r in results:
        bucket = by_shape.setdefault(r["n"], {"n": r["n"], "path": r["path"], "pass": 0, "fail": 0, "reasons": []})
        if r.get("ok"):
            bucket["pass"] += 1
        else:
            bucket["fail"] += 1
            bucket["reasons"].append(
                {
                    "case": r.get("case"),
                    "npu": r.get("npu"),
                    "fail_reason": r.get("fail_reason"),
                    "hit_507015": r.get("hit_507015"),
                    "npu_recon": r.get("npu_recon"),
                    "Q": r.get("Q"),
                    "R": r.get("R"),
                }
            )
    summary = {
        "step": 280,
        "cpu_golden": "torch.linalg.qr(A.double().cpu())",
        "npu_impl": "mx_driving_cloud._C.qr for >80 (QrV2, no 192 python bypass); torch.linalg.qr for <=80",
        "recon_fail": RECON_FAIL,
        "n_jobs": len(results),
        "n_pass": sum(1 for r in results if r.get("ok")),
        "n_fail": len(fails),
        "fail_shapes": sorted({r["n"] for r in fails}),
        "by_shape": [by_shape[k] for k in sorted(by_shape)],
        "fails": [
            {
                "n": r["n"],
                "case": r.get("case"),
                "npu": r.get("npu"),
                "path": r.get("path"),
                "fail_reason": r.get("fail_reason"),
                "hit_507015": r.get("hit_507015"),
                "npu_recon": r.get("npu_recon"),
                "cpu64_recon": r.get("cpu64_recon"),
                "cpu32_recon": r.get("cpu32_recon"),
                "Q": r.get("Q"),
                "exc_type": r.get("exc_type"),
            }
            for r in fails
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SUMMARY pass={n_pass} fail={n_fail} fail_shapes={fail_shapes}".format(**summary), flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    if os.environ.get("STEP280_CHILD") == "1":
        raise SystemExit(child())
    raise SystemExit(driver())
