from typing import List
import numpy as np
import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self,kind='B0'):
        super().__init__();self.projection=nn.Linear(3,64);self.core=nn.GRUCell(64,545)
        self.residual_gate=nn.Parameter(torch.zeros(()));self.head=nn.Linear(545,3 if kind.startswith('T') else 2)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        h=torch.zeros(x.size(0),545,dtype=x.dtype,device=x.device)
        out=torch.jit.annotate(List[torch.Tensor],[])
        for i in range(x.size(1)):
            h=self.core(self.projection(x[:,i]),h)+torch.tanh(self.residual_gate)*h
            out.append(self.head(h))
        return torch.stack(out,1)

def ternary_length(w):
    n=1;capacity=1
    while capacity<2*((1<<w)-1):n+=1;capacity=3*capacity+1
    return n

def encode(v,n,radix):
    if radix==2:
        assert 0<=v<(1<<n);return [(v>>i)&1 for i in range(n)]
    out=[]
    for _ in range(n):
        digit=(v+1)%3-1;out.append(digit);v=(v-digit)//3
    assert v==0,'insufficient ternary width'
    return out

def decode(d,radix):return sum(v*radix**i for i,v in enumerate(d))
def length(r,k):return ternary_length(r['width']) if k.startswith('T') else r['width']+int(k=='B1' or r['op']=='ADD')

def inputs(r,k):
    n=length(r,k);radix=3 if k.startswith('T') else 2
    a=encode(r['a'],n,radix);b=encode(r['b'],n,radix);third=[0]*n
    if k in ('B0','T0'):third=[int(r['op']=='SUB')]*n
    elif r['op']=='SUB':
        if k=='B1':b=[1-v for v in b];third[0]=1
        else:b=[-v for v in b]
    return a,b,third

def batch(rows,k,labels=True):
    ll=[length(r,k) for r in rows];x=np.zeros((len(rows),max(ll),3),np.float32)
    y=np.full((len(rows),max(ll)),-100,np.int64)
    for i,(r,n) in enumerate(zip(rows,ll)):
        a,b,c=inputs(r,k);x[i,:n]=np.array([a,b,c],np.float32).T
        if labels:
            value=r['a']+r['b'] if r['op']=='ADD' else r['a']-r['b']
            y[i,:n]=np.array(encode(value,n,3 if k.startswith('T') else 2))+(1 if k.startswith('T') else 0)
    return torch.from_numpy(x),torch.from_numpy(y),ll

def bucket(n):return '0' if n==0 else '1' if n==1 else '2-3' if n<=3 else '4-7' if n<=7 else '8-15' if n<=15 else '16+'

def propagation(r,k):
    aa,bb,extra=inputs(r,k);state=int(k=='B1' and r['op']=='SUB');out=[];events=[];run=longest=0
    for a,b in zip(aa,bb):
        incoming=state
        if k.startswith('T'):
            s=a+(-b if k=='T0' and r['op']=='SUB' else b)+state
            z=(s+1)%3-1;state=(s-z)//3
        elif k=='B0' and r['op']=='SUB':
            s=a-b-state;z=s%2;state=int(s<0)
        else:s=a+b+state;z=s%2;state=s//2
        out.append(z);events.append((incoming,state));run=run+1 if state!=0 else 0;longest=max(longest,run)
    assert decode(out,3 if k.startswith('T') else 2)==(r['a']+r['b'] if r['op']=='ADD' else r['a']-r['b'])
    return out,events,longest

def phases(events):
    result=[]
    for i,(a,b) in enumerate(events):
        phase='propagation_start' if not a and b else 'continue' if a and b else 'release' if a and not b else 'inactive'
        if i==0:phase='computation_start'
        elif phase=='inactive' and events[i-1][0] and not events[i-1][1]:phase='post_release'
        result.append(phase)
    return result
