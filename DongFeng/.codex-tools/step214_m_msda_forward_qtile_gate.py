#!/usr/bin/env python3
"""Independent STEP-214-M: B1 MSDA forward, 32-query tile per program."""
import step214_k_msda_forward_gate as base
import triton
import triton.language as tl

@triton.jit
def msda_fwd_qtile(value,loc,weight,out,Q:tl.constexpr,H:tl.constexpr,C:tl.constexpr,IH:tl.constexpr,IW:tl.constexpr,P:tl.constexpr,BQ:tl.constexpr):
 pid=tl.program_id(0);qb=pid//H;head=pid-qb*H;q=qb*BQ+tl.arange(0,BQ)[:,None];ch=tl.arange(0,C)[None,:];qm=q<Q
 acc=tl.zeros((BQ,C),tl.float32)
 for p in tl.static_range(P):
  lp=(q*H+head)*P*2+p*2;x=tl.load(loc+lp,mask=qm,other=0.0)*IW-0.5;y=tl.load(loc+lp+1,mask=qm,other=0.0)*IH-0.5
  x0=tl.floor(x).to(tl.int32);y0=tl.floor(y).to(tl.int32);x1=x0+1;y1=y0+1
  lx=x-x0.to(tl.float32);ly=y-y0.to(tl.float32);hx=1.0-lx;hy=1.0-ly;w=tl.load(weight+(q*H+head)*P+p,mask=qm,other=0.0)
  m00=qm&(x0>=0)&(x0<IW)&(y0>=0)&(y0<IH);i00=((y0*IW+x0)*H+head)*C+ch
  m01=qm&(x1>=0)&(x1<IW)&(y0>=0)&(y0<IH);i01=((y0*IW+x1)*H+head)*C+ch
  m10=qm&(x0>=0)&(x0<IW)&(y1>=0)&(y1<IH);i10=((y1*IW+x0)*H+head)*C+ch
  m11=qm&(x1>=0)&(x1<IW)&(y1>=0)&(y1<IH);i11=((y1*IW+x1)*H+head)*C+ch
  v00=tl.load(value+i00,mask=m00,other=0.0);v01=tl.load(value+i01,mask=m01,other=0.0)
  v10=tl.load(value+i10,mask=m10,other=0.0);v11=tl.load(value+i11,mask=m11,other=0.0)
  acc+=w*(v00*hx*hy+v01*lx*hy+v10*hx*ly+v11*lx*ly)
 tl.store(out+(q*H+head)*C+ch,acc,mask=qm)

def launch(v,l,w,o):msda_fwd_qtile[(480*8,)](v,l,w,o,Q=15360,H=8,C=32,IH=18,IW=32,P=8,BQ=32)
base.launch=launch
if __name__=='__main__':raise SystemExit(base.main())
