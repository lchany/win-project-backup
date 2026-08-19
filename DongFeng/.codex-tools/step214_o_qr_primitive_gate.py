#!/usr/bin/env python3
"""STEP-214-O QR primitive gate: linalg.qr versus geqrf+orgqr."""
from __future__ import annotations
import argparse,json,math,os,statistics,time,traceback
from pathlib import Path
_lr=os.environ.get('LOCAL_RANK','x')
if _lr.isdigit(): os.environ['TRITON_CACHE_DIR']=str(Path(os.environ['OUTPUT_DIR'])/'triton_cache'/f'rank{_lr}')
import torch,torch_npu
VISIBLE='8,9,10,11,12,13,14,15'
def stat(v):
 s=sorted(v);return {'min_ms':min(v),'median_ms':statistics.median(v),'p95_ms':s[max(0,math.ceil(.95*len(s))-1)],'max_ms':max(v)}
def event(fn):
 torch.npu.synchronize();a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);t=time.perf_counter();a.record();x=fn();b.record();b.synchronize();return x,float(a.elapsed_time(b)),(time.perf_counter()-t)*1000
def err(x,y):
 d=x-y;den=torch.sqrt(torch.mean(y*y));return {'bitwise':bool(torch.equal(x,y)),'max_abs':float(d.abs().max().cpu()),'nrmse':float((torch.sqrt(torch.mean(d*d))/den).cpu())}
def run(a):
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);root=Path(a.output_dir).resolve(strict=True)
 if world!=8 or rank!=local or os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!=VISIBLE:raise RuntimeError('contract')
 for d in (root/'ready',root/'done',root/'failure'):d.mkdir(parents=True,exist_ok=True)
 torch.npu.set_device(local);dev=torch.device(f'npu:{local}');torch.manual_seed(20260815);torch.npu.manual_seed_all(20260815)
 x=torch.randn((2560,2560),dtype=torch.float32,device=dev)
 direct=lambda:torch.linalg.qr(x,mode='reduced')
 def candidate():
  packed,tau=torch.geqrf(x);q=torch.orgqr(packed,tau);return q,packed,tau
 event(direct);event(candidate)
 dt=[];dw=[];ct=[];cw=[]
 torch.npu.reset_peak_memory_stats(local);base_alloc=int(torch.npu.memory_allocated(local));base_reserved=int(torch.npu.memory_reserved(local))
 for i in range(a.samples):
  order=(('d',direct),('c',candidate)) if i%2==0 else (('c',candidate),('d',direct))
  for name,fn in order:
   z,e,w=event(fn)
   if name=='d':dt.append(e);dw.append(w)
   else:ct.append(e);cw.append(w)
   del z
 torch.npu.synchronize();peak_alloc=int(torch.npu.max_memory_allocated(local));peak_reserved=int(torch.npu.max_memory_reserved(local))
 (qd,rd),_,_=event(direct);(qc,packed,tau),_,_=event(candidate);rc=torch.triu(packed);torch.npu.synchronize()
 qe=err(qc,qd);re=err(rc,rd);finite=all(bool(torch.isfinite(z).all().cpu()) for z in (qd,rd,qc,packed,tau,rc))
 eye=torch.eye(2560,dtype=torch.float32,device=dev);orth=float((qc.T@qc-eye).abs().max().cpu());recon=float(torch.linalg.vector_norm(qc@rc-x).cpu()/torch.linalg.vector_norm(x).cpu())
 payload={'pid':os.getpid(),'rank':rank,'local_rank':local,'world_size':world,'visible':VISIBLE,'gate_pass':finite,
 'input_shape':[2560,2560],'input_dtype':str(x.dtype),'q_shape':list(qc.shape),'r_shape':list(rc.shape),'packed_shape':list(packed.shape),'tau_shape':list(tau.shape),
 'q_dtype':str(qc.dtype),'r_dtype':str(rc.dtype),'packed_dtype':str(packed.dtype),'tau_dtype':str(tau.dtype),'finite_all':finite,
 'direct_event':stat(dt),'direct_wall':stat(dw),'candidate_event':stat(ct),'candidate_wall':stat(cw),'q_error':qe,'r_error':re,
 'orthogonality_max_abs':orth,'reconstruction_rel_l2':recon,'base_allocated':base_alloc,'peak_allocated':peak_alloc,'extra_peak_allocated':peak_alloc-base_alloc,
 'base_reserved':base_reserved,'peak_reserved':peak_reserved,'extra_peak_reserved':peak_reserved-base_reserved}
 def wr(p,z):t=p.with_suffix('.tmp');t.write_text(json.dumps(z,sort_keys=True),encoding='utf-8');t.replace(p)
 wr(root/'ready'/f'rank{rank}.json',payload);deadline=time.monotonic()+120
 while not (root/'release_after_npu_smi').exists():
  if time.monotonic()>deadline:raise TimeoutError('release')
  time.sleep(.2)
 wr(root/'done'/f'rank{rank}.json',payload)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--samples',type=int,default=7);a=p.parse_args()
 try:run(a);return 0
 except BaseException as e:
  if os.environ.get('OUTPUT_DIR'):
   d=Path(os.environ['OUTPUT_DIR'])/'failure';d.mkdir(parents=True,exist_ok=True);(d/f"rank{os.environ.get('RANK','x')}.txt").write_text(''.join(traceback.format_exception(e)),encoding='utf-8')
  raise
if __name__=='__main__':raise SystemExit(main())
