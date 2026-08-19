#!/usr/bin/env python3
"""STEP-268: extra mx QR tests on 192x192, one subprocess per case.

A previous in-process run hit QrV2 AICore MTE out-of-range on the exact BAD A.
Isolate each case so one kernel crash cannot poison the rest of the sweep.
"""
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
OUT_DIR = Path(os.environ.get("STEP268_OUT", "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step268_192_qr_extra"))


def last_tile64(n: int) -> int:
    start = (n // 64) * 64
    if start == n:
        start = max(0, n - 64)
    return start


def load_dump_A(path: Path):
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["A"].float().contiguous()


def summarize(name: str, x) -> dict:
    import torch

    xf = x.detach().float().cpu()
    finite = torch.isfinite(xf)
    nan_cols = []
    if xf.ndim == 2:
        nan_cols = torch.where(~torch.isfinite(xf).all(dim=0))[0].tolist()
    return {
        f"{name}_finite": bool(finite.all()),
        f"{name}_nonfinite": int((~finite).sum().item()),
        f"{name}_absmax": float(xf[finite].abs().max()) if bool(finite.any()) else float("nan"),
        f"{name}_nan_col_count": len(nan_cols),
        f"{name}_nan_col_start": nan_cols[0] if nan_cols else None,
        f"{name}_nan_col_end": nan_cols[-1] if nan_cols else None,
    }


def build_named_A(name: str):
    import torch

    bad = load_dump_A(DUMP_DIR / "rank0_step10_ind0_192x192_BAD.pt")
    sample_path = DUMP_DIR / "rank0_step10_ind0_192x192_SAMPLE.pt"
    sample = load_dump_A(sample_path) if sample_path.is_file() else None
    n = 192
    g = torch.Generator().manual_seed(192)
    tiny = float(bad.abs().max())
    if name == "bad_exact":
        return bad
    if name == "sample_192_ok":
        if sample is None:
            raise FileNotFoundError("sample missing")
        return sample
    if name.startswith("bad_scale_"):
        token = name[len("bad_scale_"):]
        if token.startswith("1em"):
            k = -int(token[3:])
        elif token.startswith("1e"):
            k = int(token[2:])
        else:
            raise KeyError(name)
        return (bad * (10.0 ** k)).contiguous()
    if name == "bad_unit_fro":
        return (bad / bad.norm()).contiguous()
    if name == "rand_1":
        return (torch.randn(n, n, generator=g) * 1.0).contiguous()
    if name == "rand_1e-3":
        return (torch.randn(n, n, generator=g) * 1e-3).contiguous()
    if name == "rand_1e-7":
        return (torch.randn(n, n, generator=g) * 1e-7).contiguous()
    if name == "rand_badabsmax":
        return (torch.randn(n, n, generator=g) * tiny).contiguous()
    if name == "identity":
        return torch.eye(n)
    if name == "well_conditioned_qr":
        q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
        r = torch.triu(torch.randn(n, n, generator=g))
        return (q @ r).contiguous()
    if name.startswith("rand1_n"):
        nn = int(name[len("rand1_n"):])
        g2 = torch.Generator().manual_seed(1000 + nn)
        return torch.randn(nn, nn, generator=g2).contiguous()
    if name.startswith("tiny_n"):
        nn = int(name[len("tiny_n"):])
        g2 = torch.Generator().manual_seed(2000 + nn)
        return (torch.randn(nn, nn, generator=g2) * tiny).contiguous()
    if name.startswith("bad_crop_n"):
        nn = int(name[len("bad_crop_n"):])
        return bad[:nn, :nn].contiguous()
    if name.startswith("bad_pad_n"):
        nn = int(name[len("bad_pad_n"):])
        pad = torch.zeros(nn, nn)
        pad[:192, :192] = bad
        return pad.contiguous()
    raise KeyError(name)


def case_list() -> list[str]:
    names = ["sample_192_ok", "identity", "well_conditioned_qr", "rand_1", "rand_1e-3"]
    # scale sweep around the known-bad magnitude ~1e-8
    for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8):
        if k < 0:
            names.append(f"bad_scale_1em{-k}")
        else:
            names.append(f"bad_scale_1e{k}")
    names += ["bad_exact", "bad_unit_fro", "rand_1e-7", "rand_badabsmax"]
    for nn in (64, 128, 160, 191, 192, 193, 224, 256):
        names.append(f"rand1_n{nn}")
        names.append(f"tiny_n{nn}")
        if nn <= 192:
            names.append(f"bad_crop_n{nn}")
        else:
            names.append(f"bad_pad_n{nn}")
    return names


def run_one(name: str, npu: int) -> dict:
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    import torch_npu  # noqa: F401
    import mx_driving_cloud

    A = build_named_A(name)
    device = torch.device(f"npu:{npu}")
    rec = {
        "case": name,
        "npu": npu,
        "shape": list(A.shape),
        "A_cpu_absmax": float(A.abs().max()),
        "mx_version": getattr(mx_driving_cloud, "__version__", "?"),
    }
    A_n = A.detach().to(device=device, dtype=torch.float32).contiguous()
    rec.update(summarize("A", A_n))
    try:
        torch.npu.synchronize()
        t0 = time.perf_counter()
        Q, R = mx_driving_cloud.linalg.qr(A_n)
        torch.npu.synchronize()
        rec["mx_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        rec.update(summarize("Q", Q))
        rec.update(summarize("R", R))
        rec["mx_crash"] = False
        if rec["Q_finite"] and rec["R_finite"]:
            rec["mx_recon_max"] = float((Q @ R - A_n).abs().max())
            rec["mx_ok"] = rec["mx_recon_max"] < 1e-3
        else:
            rec["mx_ok"] = False
            rec["nan_is_last_64_tile"] = rec.get("Q_nan_col_start") == last_tile64(A.shape[0])
    except Exception as exc:  # noqa: BLE001
        rec["mx_crash"] = True
        rec["mx_ok"] = False
        rec["mx_error"] = type(exc).__name__
        rec["mx_error_head"] = str(exc).splitlines()[0][:240]
    q32, r32 = torch.linalg.qr(A.cpu().float())
    rec["cpu32_finite"] = bool(torch.isfinite(q32).all() and torch.isfinite(r32).all())
    rec["cpu32_recon_max"] = float((q32 @ r32 - A.cpu().float()).abs().max())
    return rec


def driver() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = case_list()
    results = []
    py = sys.executable
    script = str(Path(__file__).resolve())
    npu_count = int(os.environ.get("STEP268_NPU_COUNT", "8"))
    # replay exact BAD on all 8 after safer cases, as separate names
    replay_names = [f"__replay_npu{i}__" for i in range(npu_count)]
    all_names = names + replay_names
    for idx, name in enumerate(all_names):
        npu = 0
        real = name
        if name.startswith("__replay_npu"):
            npu = int(name.replace("__replay_npu", "").replace("__", ""))
            real = "bad_exact"
        env = os.environ.copy()
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        env["STEP268_CASE"] = real
        env["STEP268_NPU"] = str(npu)
        env["STEP268_CHILD"] = "1"
        out_path = OUT_DIR / f"case_{idx:03d}.json"
        env["STEP268_CASE_OUT"] = str(out_path)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [py, script],
            env=env,
            cwd=str(OUT_DIR),
            capture_output=True,
            text=True,
            timeout=180,
        )
        rec = {
            "case": real,
            "npu": npu,
            "driver_s": round(time.perf_counter() - t0, 3),
            "child_rc": proc.returncode,
        }
        if out_path.is_file():
            rec.update(json.loads(out_path.read_text(encoding="utf-8")))
        else:
            rec["mx_crash"] = True
            rec["mx_ok"] = False
            rec["mx_error"] = "no_case_json"
            rec["stderr_tail"] = (proc.stderr or "")[-500:]
        results.append(rec)
        print(
            f"{idx:03d} {real:22s} npu{npu} rc={proc.returncode} "
            f"ok={rec.get('mx_ok')} crash={rec.get('mx_crash')} "
            f"Qfin={rec.get('Q_finite')} nan_cols={rec.get('Q_nan_col_count')} "
            f"last64={rec.get('nan_is_last_64_tile')}",
            flush=True,
        )

    fails = [r["case"] for r in results if not r.get("mx_ok")]
    summary = {
        "n_cases": len(results),
        "n_fail": len(fails),
        "fails": fails,
        "crash_cases": [r["case"] for r in results if r.get("mx_crash")],
        "nan_last64_cases": [r["case"] for r in results if r.get("nan_is_last_64_tile")],
        "results": results,
    }
    out = OUT_DIR / "step268_192_qr_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE", out, "fail", len(fails), "/", len(results), flush=True)
    return 0


def child() -> int:
    rec = run_one(os.environ["STEP268_CASE"], int(os.environ.get("STEP268_NPU", "0")))
    Path(os.environ["STEP268_CASE_OUT"]).write_text(json.dumps(rec), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if os.environ.get("STEP268_CHILD") == "1":
        raise SystemExit(child())
    raise SystemExit(driver())
