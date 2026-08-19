#!/usr/bin/env python3
"""B1 channel32 structured atomic: direct versus register-aggregated."""
from __future__ import annotations
import argparse, json, math, os, statistics, time, traceback
from pathlib import Path

_lr = os.environ.get("LOCAL_RANK", "unlaunched")
if _lr.isdigit():
    os.environ["TRITON_CACHE_DIR"] = str(Path(os.environ["OUTPUT_DIR"]) / "triton_cache" / f"rank{_lr}")
import torch, torch_npu, triton
import triton.language as tl

VISIBLE="8,9,10,11,12,13,14,15"

@triton.jit
def direct_kernel(values, out, n, BLOCK:tl.constexpr, COLLISION:tl.constexpr, REPEAT:tl.constexpr):
    pid=tl.program_id(0); lane=pid % COLLISION; group=pid // COLLISION
    offs=group*BLOCK+tl.arange(0,BLOCK); mask=offs<n
    for r in tl.static_range(REPEAT):
        v=tl.load(values+(lane*REPEAT+r)*n+offs, mask=mask, other=0.0)
        tl.atomic_add(out+offs, v, mask=mask)

@triton.jit
def aggregate_kernel(values, out, n, BLOCK:tl.constexpr, COLLISION:tl.constexpr, REPEAT:tl.constexpr):
    pid=tl.program_id(0); lane=pid % COLLISION; group=pid // COLLISION
    offs=group*BLOCK+tl.arange(0,BLOCK); mask=offs<n
    acc=tl.zeros((BLOCK,), tl.float32)
    for r in tl.static_range(REPEAT):
        acc += tl.load(values+(lane*REPEAT+r)*n+offs, mask=mask, other=0.0)
    tl.atomic_add(out+offs, acc, mask=mask)

def launch(kernel, values, out, collision=32, repeat=27, block=1024):
    kernel[(triton.cdiv(out.numel(),block)*collision,)](values,out,out.numel(),BLOCK=block,COLLISION=collision,REPEAT=repeat)

def estats(v):
    s=sorted(v); return {"min_ms":min(v),"median_ms":statistics.median(v),"p95_ms":s[max(0,math.ceil(.95*len(s))-1)],"max_ms":max(v)}

def measure(kernel, values, out):
    out.zero_(); torch.npu.synchronize(); start=torch.npu.Event(enable_timing=True); end=torch.npu.Event(enable_timing=True)
    wall=time.perf_counter(); start.record(); launch(kernel,values,out); end.record(); end.synchronize()
    return float(start.elapsed_time(end)),(time.perf_counter()-wall)*1000.0

def errors(actual, oracle):
    diff=actual-oracle; denom=torch.sqrt(torch.mean(oracle*oracle)); num=torch.sqrt(torch.mean(diff*diff))
    return {"max_abs":float(diff.abs().max().cpu()),"nrmse":float((num/denom).cpu()),"finite":bool(torch.isfinite(actual).all().cpu())}

def run_variant(kernel, values, oracle, samples):
    out=torch.empty_like(oracle); events=[]; walls=[]; errs=[]; first=None; repeat_max=[]
    for _ in range(2): measure(kernel,values,out)
    for _ in range(samples):
        e,w=measure(kernel,values,out); events.append(e); walls.append(w); errs.append(errors(out,oracle))
        if first is None: first=out.clone()
        repeat_max.append(float((out-first).abs().max().cpu()))
    return {"event":estats(events),"wall":estats(walls),"max_abs_max":max(x["max_abs"] for x in errs),
            "nrmse_max":max(x["nrmse"] for x in errs),"finite_all":all(x["finite"] for x in errs),
            "repeat_max_abs":max(repeat_max),"repeat_exact":max(repeat_max)==0.0}

def atomic_json(path,payload):
    t=path.with_suffix(path.suffix+f".tmp.{os.getpid()}"); t.write_text(json.dumps(payload,sort_keys=True),encoding="utf-8"); t.replace(path)

def main_run(args):
    rank=int(os.environ["RANK"]); local=int(os.environ["LOCAL_RANK"]); world=int(os.environ["WORLD_SIZE"])
    root=Path(args.output_dir).resolve(strict=True)
    if world!=8 or rank!=local or rank not in range(8) or os.environ.get("ASCEND_RT_VISIBLE_DEVICES")!=VISIBLE: raise RuntimeError("world/rank/visible mismatch")
    cache=root/"triton_cache"/f"rank{local}"; cache.mkdir(parents=True,exist_ok=False)
    for d in (root/"ready",root/"done",root/"failure"): d.mkdir(parents=True,exist_ok=True)
    torch.npu.set_device(local); device=torch.device(f"npu:{local}"); n=576*8*32
    torch.manual_seed(20260815+rank); torch.npu.manual_seed_all(20260815+rank)
    values=torch.randn((32,27,n),dtype=torch.float32,device=device)
    oracle=values.sum(dim=1,dtype=torch.float32).sum(dim=0,dtype=torch.float32)
    direct=run_variant(direct_kernel,values,oracle,args.samples)
    aggregate=run_variant(aggregate_kernel,values,oracle,args.samples)
    out_d=torch.zeros_like(oracle); out_a=torch.zeros_like(oracle); launch(direct_kernel,values,out_d); launch(aggregate_kernel,values,out_a); torch.npu.synchronize()
    cross=errors(out_a,out_d)
    estimate=aggregate["event"]["median_ms"]*112.0; perf_pass=estimate<123.88
    gate_pass=direct["finite_all"] and aggregate["finite_all"] and cross["finite"]
    payload={"pid":os.getpid(),"rank":rank,"local_rank":local,"world_size":world,"visible":VISIBLE,"gate_pass":gate_pass,
      "logical_shape":[4608,32],"values_shape":[32,27,n],"input_bytes":values.numel()*4,"contributions_per_output":864,
      "direct_gm_atomics_per_output":864,"aggregate_gm_atomics_per_output":32,"direct":direct,"aggregate":aggregate,
      "aggregate_vs_direct":cross,"estimated_b112_ms":estimate,"performance_threshold_ms":123.88,"performance_projection_pass":perf_pass,
      "torch":torch.__version__,"torch_npu":torch_npu.__version__,"triton_module":str(Path(triton.__file__).resolve())}
    if not gate_pass: raise AssertionError("finite gate failed")
    atomic_json(root/"ready"/f"rank{rank}.json",payload)
    deadline=time.monotonic()+args.hold_timeout_seconds
    while not (root/"release_after_npu_smi").exists():
        if time.monotonic()>deadline: raise TimeoutError("release timeout")
        time.sleep(.2)
    atomic_json(root/"done"/f"rank{rank}.json",payload)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",required=True); p.add_argument("--samples",type=int,default=5); p.add_argument("--hold-timeout-seconds",type=int,default=120); a=p.parse_args()
    try: main_run(a); return 0
    except BaseException as e:
        root=os.environ.get("OUTPUT_DIR"); rank=os.environ.get("RANK","unknown")
        if root:
            d=Path(root)/"failure"; d.mkdir(parents=True,exist_ok=True); (d/f"rank{rank}.txt").write_text("".join(traceback.format_exception(e)),encoding="utf-8")
        raise
if __name__=="__main__": raise SystemExit(main())
