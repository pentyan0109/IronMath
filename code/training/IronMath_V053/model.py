from typing import List
import numpy as np
import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self):
        super().__init__();self.projection=nn.Linear(3,64);self.core=nn.GRUCell(64,545)
        self.residual_gate=nn.Parameter(torch.zeros(()));self.head=nn.Linear(545,2)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        h=torch.zeros(x.size(0),545,dtype=x.dtype,device=x.device)
        out=torch.jit.annotate(List[torch.Tensor],[])
        for i in range(x.size(1)):
            h=self.core(self.projection(x[:,i]),h)+torch.tanh(self.residual_gate)*h
            out.append(self.head(h))
        return torch.stack(out,1)

class NativeReference(Residual):
    def forward(self,x):
        z=self.projection(x);h=torch.zeros(x.shape[0],545);states=[]
        for i in range(x.shape[1]):
            h=self.core(z[:,i],h)+torch.tanh(self.residual_gate)*h;states.append(h)
        return self.head(torch.stack(states,1))

def bits(v,n):return [(v>>i)&1 for i in range(n)]
def batch(rows,direction):
    lengths=[r['width']+int(r['op']=='ADD') for r in rows];maxlen=max(lengths)
    x=np.zeros((len(rows),maxlen,3),dtype=np.float32);y=np.full((len(rows),maxlen),-100,dtype=np.int64)
    for i,(r,n) in enumerate(zip(rows,lengths)):
        a=bits(r['a'],n);b=bits(r['b'],n);answer=bits(r['a']+r['b'] if r['op']=='ADD' else r['a']-r['b'],n)
        if direction=='L':a=a[::-1];b=b[::-1];answer=answer[::-1]
        x[i,:n,0]=a;x[i,:n,1]=b;x[i,:n,2]=int(r['op']=='SUB');y[i,:n]=answer
    return torch.from_numpy(x),torch.from_numpy(y),lengths

def bit_oracle(a,b,op,n):
    aa=bits(a,n);bb=bits(b,n);out=[];state=0;events=[];longest=run=0
    for x,y in zip(aa,bb):
        incoming=state
        if op=='ADD':z=x^y^state;state=(x&y)|((x^y)&state)
        else:z=x^y^state;state=((1^x)&y)|((1^(x^y))&state)
        out.append(z);events.append(dict(incoming=incoming,outgoing=state))
        run=run+1 if state else 0;longest=max(longest,run)
    assert state==0
    return out,events,longest

def annotate(r):
    n=r['width']+int(r['op']=='ADD');bb,ev,longest=bit_oracle(r['a'],r['b'],r['op'],n)
    value=sum(x<<i for i,x in enumerate(bb));assert value==(r['a']+r['b'] if r['op']=='ADD' else r['a']-r['b'])
    attrs=['no_propagation' if longest==0 else 'short_propagation' if longest<max(4,r['width']//2) else 'long_propagation']
    if r['a']==0 or r['b']==0:attrs.append('zero_operand')
    if (1<<r['width'])-1-max(r['a'],r['b'])<=8:attrs.append('near_maximum')
    if any(abs(r['a']-(1<<k))<=1 or abs(r['b']-(1<<k))<=1 for k in range(r['width'])):attrs.append('bit_boundary')
    bucket='none' if not longest else '1' if longest==1 else '2-4' if longest<=4 else '5-16' if longest<=16 else '17-64' if longest<=64 else '65-256' if longest<=256 else '257+'
    return dict(r,answer=value,length=n,longest_propagation=longest,propagation_bucket=bucket,attributes=attrs)
