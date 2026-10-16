from typing import List
import numpy as np
import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self):
        super().__init__();self.projection=nn.Linear(20,64);self.core=nn.GRUCell(64,545)
        self.residual_gate=nn.Parameter(torch.zeros(()));self.head=nn.Linear(545,10)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        h=torch.zeros(x.size(0),545,dtype=x.dtype,device=x.device)
        out=torch.jit.annotate(List[torch.Tensor],[])
        for i in range(x.size(1)):
            h=self.core(self.projection(x[:,i]),h)+torch.tanh(self.residual_gate)*h
            out.append(self.head(h))
        return torch.stack(out,1)

def bits(v,n):return [(v>>i)&1 for i in range(n)]
def integer(dd):
    if any(v is None for v in dd):return None
    result=0
    for d in reversed(dd):result=(result<<1)|d
    return result
def transformed(r,route='P'):
    w=r['width'];b=r['b'];cin=0;n=w+1;flag=0
    if r['op']=='SUB':
        if route=='P':b=(-b)%(1<<w)
        elif route=='B1_native':b=((1<<n)-1)^b;cin=1
        elif route=='B0_native':flag=1;n=w
        else:raise ValueError('no B0 Cin path')
    return r['a'],b,n,cin,flag
def encode(rows,route='P',labels=False):
    lengths=[transformed(r,route)[2] for r in rows];maxlen=max(lengths)
    x=np.zeros((len(rows),maxlen,3),np.float32);y=np.full((len(rows),maxlen),-100,np.int64)
    for i,r in enumerate(rows):
        a,b,n,cin,flag=transformed(r,route);x[i,:n,0]=bits(a,n);x[i,:n,1]=bits(b,n);x[i,:n,2]=flag
        if cin:x[i,0,2]=1
        if labels:
            target=(a-b if route=='B0_native' and r['op']=='SUB' else a+b+cin)%(1<<n)
            y[i,:n]=bits(target,n)
    return torch.from_numpy(x),torch.from_numpy(y),lengths
def bucket(n):return '0' if not n else '1' if n==1 else '2-3' if n<=3 else '4-7' if n<=7 else '8-15' if n<=15 else '16+'
def diagnostics(r,route='P'):
    a,b,n,cin,flag=transformed(r,route);aa=bits(a,n);bb=bits(b,n);state=cin;events=[];run=longest=0;truth=[]
    for x,y in zip(aa,bb):
        old=state
        if flag:s=x-y-state;state=int(s<0)
        else:s=x+y+state;state=s//2
        truth.append(s%2);events.append((old,state));run=run+1 if state else 0;longest=max(longest,run)
    out=[]
    for i,(old,new) in enumerate(events):
        out.append('computation_start' if i==0 else 'start' if not old and new else 'continue' if old and new else 'release' if old and not new else 'inactive')
    onesrun=maxones=0
    for d in bb[:r['width']]:onesrun=onesrun+1 if d else 0;maxones=max(maxones,onesrun)
    edge=(r['width']+3)//4;physical=events[:r['width']];run=corelong=0
    for a,b in physical:run=run+1 if b else 0;corelong=max(corelong,run)
    return dict(raw_truth_digits=truth,phases=out,longest_carry=corelong,longest_carry_full_sequence=longest,carry_bucket=bucket(corelong),B_ones=sum(bb[:r['width']]),B_longest_ones=maxones,B_LSB_edge_ones=sum(bb[:edge]),B_MSB_edge_ones=sum(bb[r['width']-edge:r['width']]),edge_length=edge,correct_incoming_carry_ones=sum(v[0] for v in physical),correct_outgoing_carry_ones=sum(v[1] for v in physical),carry_start=sum(not a and b for a,b in physical),carry_continue=sum(a and b for a,b in physical),carry_release=sum(a and not b for a,b in physical),binary_propagation=r['longest_propagation'],binary_bucket=bucket(r['longest_propagation']))
