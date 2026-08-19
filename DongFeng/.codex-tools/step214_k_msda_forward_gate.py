#!/usr/bin/env python3
"""B1 true-shape FP32 MSDA forward: DrivingSDK versus static Triton prototype."""
from __future__ import annotations
import argparse,json,math,os,statistics,time,traceback
from pathlib import Path
_lr=os.environ.get('LOCAL_RANK','x')
if _lr.isdigit(): os.environ['TRITON_CACHE_DIR']=str(Path(os.environ['OUTPUT_DIR'])/'triton_cache'/f'rank{_lr}')
import torch,torch_npu,triton
import triton.language as tl
import mx_driving
VISIBLE='8,9,10,11,12,13,14,15'

@triton.jit
def msda_fwd(value,loc,weight,out,Q:tl.constexpr,H:tl.constexpr,C:tl.constexpr,IH:tl.constexpr,IW:tl.constexpr,P:tl.constexpr):
 pid=tl.program_id(0); q=pid//H; head=pid-q*H; ch=tl.arange(0,C); acc=tl.zeros((C,),tl.float32)
 for p in tl.static_range(P):
  lp=((((q*H+head)*P+p)*2)); x=tl.load(loc+lp)*IW-0.5; y=tl.load(loc+lp+1)*IH-0.5
  x0=tl.floor(x).to(tl.int32); y0=tl.floor(y).to(tl.int32); x1=x0+1; y1=y0+1
  lx=x-x0.to(tl.float32); ly=y-y0.to(tl.float32); hx=1.0-lx; hy=1.0-ly
  w=tl.load(weight+(q*H+head)*P+p)
  m00=(x0>=0)&(x0<IW)&(y0>=0)&(y0<IH); i00=tl.where(m00,((y0*IW+x0)*H+head)*C+ch,0)
  m01=(x1>=0)&(x1<IW)&(y0>=0)&(y0<IH); i01=tl.where(m01,((y0*IW+x1)*H+head)*C+ch,0)
  m10=(x0>=0)&(x0<IW)&(y1>=0)&(y1<IH); i10=tl.where(m10,((y1*IW+x0)*H+head)*C+ch,0)
  m11=(x1>=0)&(x1<IW)&(y1>=0)&(y1<IH); i11=tl.where(m11,((y1*IW+x1)*H+head)*C+ch,0)
  v00=tl.load(value+i00,mask=m00,other=0.0);v01=tl.load(value+i01,mask=m01,other=0.0)
  v10=tl.load(value+i10,mask=m10,other=0.0);v11=tl.load(value+i11,mask=m11,other=0.0)
  acc += w*(v00*hx*hy+v01*lx*hy+v10*hx*ly+v11*lx*ly)
 tl.store(out+(q*H+head)*C+ch,acc)

def launch(v,l,w,o): msda_fwd[(15360*8,)](v,l,w,o,Q=15360,H=8,C=32,IH=18,IW=32,P=8)
def stat(v):
 s=sorted(v);return {'min_ms':min(v),'median_ms':statistics.median(v),'p95_ms':s[max(0,math.ceil(.95*len(s))-1)],'max_ms':max(v)}
def measure(fn):
 torch.npu.synchronize();a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);t=time.perf_counter();a.record();x=fn();b.record();b.synchronize();return x,float(a.elapsed_time(b)),(time.perf_counter()-t)*1000
def err(a,b):
 d=a-b;return {'max_abs':float(d.abs().max().cpu()),'nrmse':float((torch.sqrt(torch.mean(d*d))/torch.sqrt(torch.mean(b*b))).cpu()),'finite':bool(torch.isfinite(a).all().cpu())}
def main_run(a):
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);root=Path(a.output_dir).resolve(strict=True)
 if world!=8 or rank!=local or os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!=VISIBLE:raise RuntimeError('contract')
 (root/'triton_cache'/f'rank{rank}').mkdir(parents=True,exist_ok=False)
 for d in (root/'ready',root/'done',root/'failure'):d.mkdir(parents=True,exist_ok=True)
 torch.npu.set_device(local);dev=torch.device(f'npu:{local}');torch.manual_seed(20260815+rank);torch.npu.manual_seed_all(20260815+rank)
 v=torch.randn((1,576,8,32),device=dev,dtype=torch.float32);l=torch.rand((1,15360,8,1,8,2),device=dev,dtype=torch.float32)*1.5-0.25
 w=torch.rand((1,15360,8,1,8),device=dev,dtype=torch.float32);w=w/w.sum(dim=-1,keepdim=True)
 shapes=torch.tensor([[18,32]],device=dev,dtype=torch.int32);offset=torch.tensor([0],device=dev,dtype=torch.int32);out=torch.empty((1,15360,8,32),device=dev)
 sdk=lambda:mx_driving.multi_scale_deformable_attn(v,shapes,offset,l,w)
 tri=lambda:(launch(v,l,w,out),out)[1]
 for _ in range(3):measure(sdk);measure(tri)
 se=[];sw=[];te=[];tw=[];errs=[];tr=None;sr=None;trep=[];srep=[]
 for _ in range(a.samples):
  so,e,wa=measure(sdk);se.append(e);sw.append(wa);so=so.view_as(out);sr=so.clone() if sr is None else sr;srep.append(float((so-sr).abs().max().cpu()))
  to,e,wa=measure(tri);te.append(e);tw.append(wa);to=to.clone();tr=to.clone() if tr is None else tr;trep.append(float((to-tr).abs().max().cpu()));errs.append(err(to,so))
 med=statistics.median(te);estimate=med*112;payload={'pid':os.getpid(),'rank':rank,'local_rank':local,'world_size':world,'visible':VISIBLE,'gate_pass':all(x['finite'] for x in errs),
 'shape_value':[1,576,8,32],'shape_sampling':[1,15360,8,1,8,2],'shape_weight':[1,15360,8,1,8],'sampling_min':float(l.min().cpu()),'sampling_max':float(l.max().cpu()),
 'sdk_event':stat(se),'sdk_wall':stat(sw),'triton_event':stat(te),'triton_wall':stat(tw),'max_abs_max':max(x['max_abs'] for x in errs),'nrmse_max':max(x['nrmse'] for x in errs),
 'sdk_repeat_max':max(srep),'triton_repeat_max':max(trep),'triton_repeat_exact':max(trep)==0.0,'estimated_b112_ms':estimate,'threshold_ms':53.938,'performance_projection_pass':estimate<53.938}
 if not payload['gate_pass']:raise AssertionError('finite')
 def wr(p,z):t=p.with_suffix('.tmp');t.write_text(json.dumps(z,sort_keys=True),encoding='utf-8');t.replace(p)
 wr(root/'ready'/f'rank{rank}.json',payload);deadline=time.monotonic()+120
 while not (root/'release_after_npu_smi').exists():
  if time.monotonic()>deadline:raise TimeoutError('release')
  time.sleep(.2)
 wr(root/'done'/f'rank{rank}.json',payload)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--samples',type=int,default=7);a=p.parse_args()
 try:main_run(a);return 0
 except BaseException as e:
  if os.environ.get('OUTPUT_DIR'):
   d=Path(os.environ['OUTPUT_DIR'])/'failure';d.mkdir(parents=True,exist_ok=True);(d/f"rank{os.environ.get('RANK','x')}.txt").write_text(''.join(traceback.format_exception(e)),encoding='utf-8')
  raise
if __name__=='__main__':raise SystemExit(main())
