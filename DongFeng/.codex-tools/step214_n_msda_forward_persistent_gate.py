#!/usr/bin/env python3
"""Final STEP-214-N: persistent grid64 MSDA forward."""
import step214_k_msda_forward_gate as base
import triton
import triton.language as tl

@triton.jit
def msda_fwd_persistent(value,loc,weight,out,TOTAL:tl.constexpr,H:tl.constexpr,C:tl.constexpr,IH:tl.constexpr,IW:tl.constexpr,P:tl.constexpr,BLOCK:tl.constexpr,GRID:tl.constexpr):
 pid=tl.program_id(0)
 for start in tl.range(pid*BLOCK,TOTAL,GRID*BLOCK):
  idx=start+tl.arange(0,BLOCK);mask=idx<TOTAL;ch=idx%C;t=idx//C;head=t%H;q=t//H;acc=tl.zeros((BLOCK,),tl.float32)
  for p in tl.static_range(P):
   lp=(q*H+head)*P*2+p*2;x=tl.load(loc+lp,mask=mask,other=0.0)*IW-0.5;y=tl.load(loc+lp+1,mask=mask,other=0.0)*IH-0.5
   x0=tl.floor(x).to(tl.int32);y0=tl.floor(y).to(tl.int32);x1=x0+1;y1=y0+1
   lx=x-x0.to(tl.float32);ly=y-y0.to(tl.float32);hx=1.0-lx;hy=1.0-ly;w=tl.load(weight+(q*H+head)*P+p,mask=mask,other=0.0)
   m00=mask&(x0>=0)&(x0<IW)&(y0>=0)&(y0<IH);i00=((y0*IW+x0)*H+head)*C+ch
   m01=mask&(x1>=0)&(x1<IW)&(y0>=0)&(y0<IH);i01=((y0*IW+x1)*H+head)*C+ch
   m10=mask&(x0>=0)&(x0<IW)&(y1>=0)&(y1<IH);i10=((y1*IW+x0)*H+head)*C+ch
   m11=mask&(x1>=0)&(x1<IW)&(y1>=0)&(y1<IH);i11=((y1*IW+x1)*H+head)*C+ch
   acc+=w*(tl.load(value+i00,mask=m00,other=0.0)*hx*hy+tl.load(value+i01,mask=m01,other=0.0)*lx*hy+tl.load(value+i10,mask=m10,other=0.0)*hx*ly+tl.load(value+i11,mask=m11,other=0.0)*lx*ly)
  tl.store(out+idx,acc,mask=mask)

def launch(v,l,w,o):msda_fwd_persistent[(64,)](v,l,w,o,TOTAL=15360*8*32,H=8,C=32,IH=18,IW=32,P=8,BLOCK=256,GRID=64)
base.launch=launch
if __name__=='__main__':raise SystemExit(base.main())
