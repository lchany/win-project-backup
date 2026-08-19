#!/usr/bin/env python3
"""STEP-270: device probe + npu2 vs npu0 value/shape isolation for BAD 192 QR."""
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
OUT_DIR = Path(
    os.environ.get("STEP270_OUT", "/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step270_device_pin")
)


def load_named(name: str):
    import torch

    bad = torch.load(DUMP_DIR / "rank0_step10_ind0_192x192_BAD.pt", map_location="cpu", weights_only=False)["A"].float().contiguous()
    sample_p = DUMP_DIR / "rank0_step10_ind0_192x192_SAMPLE.pt"
    if name == "bad":
        return bad
    if name == "sample":
        return torch.load(sample_p, map_location="cpu", weights_only=False)["A"].float().contiguous()
    if name == "identity":
        return torch.eye(192)
    if name == "randn":
        g = torch.Generator().manual_seed(192)
        return torch.randn(192, 192, generator=g)
    if name == "bad_scale_1e4":
        return (bad * 1e4).contiguous()
    if name == "bad_pad256":
        out = torch.zeros(256, 256)
        out[:192, :192] = bad
        return out
    if name == "bad_128":
        return bad[:128, :128].contiguous()
    if name == "bad_last64":
        return bad[128:, 128:].contiguous()
    if name == "bad_191":
        return bad[:191, :191].contiguous()
    if name == "bad_193":
        out = torch.zeros(193, 193)
        out[:192, :192] = bad
        out[-1, -1] = 1.0
        return out
    raise KeyError(name)


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


def probe_child() -> int:
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    import torch_npu  # noqa: F401

    recs = []
    n = torch.npu.device_count()
    recs.append({"device_count": n, "visible": os.environ.get("ASCEND_RT_VISIBLE_DEVICES")})
    for i in range(n):
        props = torch.npu.get_device_properties(i)
        item = {"npu": i, "name": torch.npu.get_device_name(i)}
        for attr in (
            "name",
            "total_memory",
            "multi_processor_count",
            "major",
            "minor",
            "is_multi_process_capable",
        ):
            if hasattr(props, attr):
                val = getattr(props, attr)
                item[attr] = int(val) if isinstance(val, (int, bool)) else str(val)
        recs.append(item)
        try:
            torch.npu.set_device(i)
            x = torch.ones(8, device=f"npu:{i}")
            torch.npu.synchronize()
            item["sync_ok"] = True
            item["ones_sum"] = float(x.sum().cpu())
        except Exception as exc:  # noqa: BLE001
            item["sync_ok"] = False
            item["error"] = type(exc).__name__ + ":" + str(exc).splitlines()[0][:200]
    try:
        from tbe.common.platform import get_soc_spec

        recs.append(
            {
                "soc_version": str(get_soc_spec("SOC_VERSION")),
                "short_soc": str(get_soc_spec("SHORT_SOC_VERSION")),
                "core_num": str(get_soc_spec("CORE_NUM")),
                "cube_core": str(get_soc_spec("CUBE_CORE_CNT") if True else ""),
            }
        )
    except Exception as exc:  # noqa: BLE001
        recs.append({"soc_error": type(exc).__name__, "soc_msg": str(exc)[:300]})
        # try torch_npu utils
        try:
            recs.append({"npu_utils": str(torch.npu.get_device_properties(0))})
        except Exception:
            pass
    Path(os.environ["STEP270_CASE_OUT"]).write_text(json.dumps(recs, indent=2, default=str), encoding="utf-8")
    return 0


def qr_child() -> int:
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    import torch_npu  # noqa: F401
    import mx_driving_cloud

    name = os.environ["STEP270_NAME"]
    npu = int(os.environ.get("STEP270_NPU", "0"))
    A_cpu = load_named(name)
    rec = {
        "name": name,
        "npu": npu,
        "shape": list(A_cpu.shape),
        "cpu_absmax": float(A_cpu.abs().max()),
    }
    q, r = torch.linalg.qr(A_cpu)
    rec["cpu_ok"] = bool(torch.isfinite(q).all() and torch.isfinite(r).all())
    A = A_cpu.to(f"npu:{npu}")
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
    Path(os.environ["STEP270_CASE_OUT"]).write_text(json.dumps(rec), encoding="utf-8")
    return 0


def driver() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [("probe", 0, "probe")]
    names = [
        "sample",
        "identity",
        "randn",
        "bad",
        "bad_scale_1e4",
        "bad_pad256",
        "bad_128",
        "bad_last64",
        "bad_191",
        "bad_193",
    ]
    for npu in (0, 2):
        for name in names:
            jobs.append((name, npu, "qr"))
    # also bad on every npu for device map confirmation
    for npu in range(8):
        jobs.append(("bad", npu, "qr"))

    py = sys.executable
    script = str(Path(__file__).resolve())
    results = []
    for i, (name, npu, kind) in enumerate(jobs):
        outp = OUT_DIR / f"case_{i:03d}.json"
        env = os.environ.copy()
        env.update(
            {
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "STEP270_CHILD": kind,
                "STEP270_NAME": name,
                "STEP270_NPU": str(npu),
                "STEP270_CASE_OUT": str(outp),
            }
        )
        t0 = time.perf_counter()
        proc = subprocess.run([py, script], env=env, capture_output=True, text=True, timeout=180)
        rec = {"name": name, "npu": npu, "kind": kind, "rc": proc.returncode, "s": round(time.perf_counter() - t0, 3)}
        if outp.is_file():
            payload = json.loads(outp.read_text(encoding="utf-8"))
            if kind == "probe":
                rec["probe"] = payload
            else:
                rec.update(payload)
        else:
            rec["crash"] = True
            rec["ok"] = False
            rec["stderr_tail"] = (proc.stderr or "")[-500:]
        results.append(rec)
        print(
            f"{i:03d} {kind:5s} {name:14s} npu{npu} ok={rec.get('ok')} crash={rec.get('crash')} "
            f"err={str(rec.get('error_head') or '')[:80]}",
            flush=True,
        )

    fails = [r for r in results if r.get("kind") != "probe" and not r.get("ok")]
    summary = {"n": len(results), "n_fail": len(fails), "fails": [(r.get("name"), r.get("npu"), r.get("crash"), r.get("error_head")) for r in fails], "results": results}
    (OUT_DIR / "step270_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("WROTE fail", len(fails), "/", len(results), flush=True)
    return 0


if __name__ == "__main__":
    kind = os.environ.get("STEP270_CHILD")
    if kind == "probe":
        raise SystemExit(probe_child())
    if kind == "qr":
        raise SystemExit(qr_child())
    raise SystemExit(driver())
