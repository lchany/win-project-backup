#!/usr/bin/env python3
"""Replay one local BAD tensor on seven rear NPUs, one visible device per process."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


RUN_DIR = Path(os.environ["STEP299_RUN_DIR"])
INPUT = RUN_DIR / "rank0_step10_ind0_192x192_BAD.pt"
PHYSICAL_DEVICES = list(range(9, 16))


def tensor_stats(x):
    import torch

    x = x.detach().float().cpu()
    finite = torch.isfinite(x)
    bad_cols = (
        torch.where(~torch.isfinite(x).all(dim=0))[0].tolist()
        if x.ndim == 2
        else []
    )
    return {
        "finite": bool(finite.all()),
        "nan_count": int(torch.isnan(x).sum()),
        "posinf_count": int(torch.isposinf(x).sum()),
        "neginf_count": int(torch.isneginf(x).sum()),
        "bad_col_start": bad_cols[0] if bad_cols else None,
        "bad_col_end": bad_cols[-1] if bad_cols else None,
        "bad_col_count": len(bad_cols),
    }


def child() -> int:
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    os.environ.pop("MX_QR_VALIDATION_BYPASS", None)

    import torch
    import torch_npu  # noqa: F401
    import mx_driving_cloud

    physical = int(os.environ["STEP299_PHYSICAL"])
    out_path = Path(os.environ["STEP299_CASE_OUT"])
    rec = {
        "physical_device": physical,
        "visible": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "logical_device": 0,
        "device_count": int(torch.npu.device_count()),
        "ok": False,
        "crash": False,
        "error_has_507015": False,
    }
    try:
        if rec["device_count"] != 1:
            raise RuntimeError(f"expected exactly one visible NPU, got {rec['device_count']}")
        torch.npu.set_device(0)
        rec["current_device"] = int(torch.npu.current_device())

        obj = torch.load(INPUT, map_location="cpu", weights_only=False)
        a_cpu = obj["A"].float().contiguous()
        rec["A"] = tensor_stats(a_cpu)
        rec["A_shape"] = list(a_cpu.shape)
        rec["A_absmax"] = float(a_cpu.abs().max())
        rec["A_norm"] = float(torch.linalg.vector_norm(a_cpu))

        a = a_cpu.to("npu:0")
        started = time.perf_counter()
        q, r = mx_driving_cloud.linalg.qr(a)
        torch.npu.synchronize()
        rec["elapsed_s"] = round(time.perf_counter() - started, 6)
        rec["Q"] = tensor_stats(q)
        rec["R"] = tensor_stats(r)
        rec["finite_ok"] = rec["Q"]["finite"] and rec["R"]["finite"]
        if rec["finite_ok"]:
            residual = torch.linalg.vector_norm((q.float() @ r.float()) - a.float())
            denom = torch.linalg.vector_norm(a.float()).clamp_min(torch.finfo(torch.float32).tiny)
            eye = torch.eye(q.shape[1], dtype=q.dtype, device=q.device)
            rec["recon_absmax"] = float(((q @ r) - a).abs().max())
            rec["recon_rel_fro"] = float(residual / denom)
            rec["orth_absmax"] = float(((q.transpose(0, 1) @ q) - eye).abs().max())
            rec["r_lower_absmax"] = float(torch.tril(r, diagonal=-1).abs().max())
            rec["ok"] = True
        else:
            rec["ok"] = False
    except Exception as exc:  # noqa: BLE001
        rec["crash"] = True
        rec["ok"] = False
        rec["error_type"] = type(exc).__name__
        rec["error_head"] = str(exc).splitlines()[0][:500]
        rec["error_has_507015"] = "507015" in str(exc)
    out_path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return 0


def driver() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    import torch

    obj = torch.load(INPUT, map_location="cpu", weights_only=False)
    a = obj["A"].float().contiguous()
    input_summary = {
        "shape": list(a.shape),
        "finite": bool(torch.isfinite(a).all()),
        "absmax": float(a.abs().max()),
        "norm": float(torch.linalg.vector_norm(a)),
    }
    q_cpu, r_cpu = torch.linalg.qr(a)
    input_summary["cpu_finite"] = bool(torch.isfinite(q_cpu).all() and torch.isfinite(r_cpu).all())
    input_summary["cpu_recon_absmax"] = float(((q_cpu @ r_cpu) - a).abs().max())

    results = []
    for physical in PHYSICAL_DEVICES:
        case_out = RUN_DIR / f"physical_{physical}.json"
        env = os.environ.copy()
        env.pop("MX_QR_VALIDATION_BYPASS", None)
        env.update(
            {
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "ASCEND_RT_VISIBLE_DEVICES": str(physical),
                "STEP299_CHILD": "1",
                "STEP299_PHYSICAL": str(physical),
                "STEP299_CASE_OUT": str(case_out),
            }
        )
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve())],
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            rec = {
                "physical_device": physical,
                "process_rc": proc.returncode,
                "wall_s": round(time.perf_counter() - started, 3),
            }
            if case_out.is_file():
                rec.update(json.loads(case_out.read_text(encoding="utf-8")))
            else:
                rec.update(
                    {
                        "ok": False,
                        "crash": True,
                        "error_head": (proc.stderr or "no case json")[-500:],
                        "error_has_507015": "507015" in (proc.stderr or ""),
                    }
                )
        except subprocess.TimeoutExpired:
            rec = {
                "physical_device": physical,
                "ok": False,
                "crash": True,
                "timeout": True,
                "wall_s": round(time.perf_counter() - started, 3),
            }
        results.append(rec)
        print(
            f"physical={physical} single_visible logical=0 ok={rec.get('ok')} "
            f"crash={rec.get('crash')} finite={rec.get('finite_ok')} "
            f"nanQ={rec.get('Q', {}).get('nan_count')} nanR={rec.get('R', {}).get('nan_count')} "
            f"recon={rec.get('recon_absmax')} rel={rec.get('recon_rel_fro')} "
            f"507015={rec.get('error_has_507015')}",
            flush=True,
        )

    summary = {
        "operator": "mx_driving_cloud.linalg.qr",
        "mode": "one physical NPU visible per independent process",
        "input": input_summary,
        "n": len(results),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_nonfinite": sum(
            1 for r in results if not r.get("crash") and not r.get("finite_ok", False)
        ),
        "n_crash": sum(1 for r in results if r.get("crash")),
        "n_507015": sum(1 for r in results if r.get("error_has_507015")),
        "results": results,
    }
    (RUN_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"SUMMARY n={summary['n']} ok={summary['n_ok']} "
        f"nonfinite={summary['n_nonfinite']} crash={summary['n_crash']} "
        f"507015={summary['n_507015']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(child() if os.environ.get("STEP299_CHILD") == "1" else driver())
