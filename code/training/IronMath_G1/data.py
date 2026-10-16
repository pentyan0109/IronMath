import numpy as np
import torch
from common import R,read,sha,dump,now
P=[.1,.1,.5,.3];SYMBOLS=['W0','W1','N','R'];LENGTHS=[32,64,128,512,2048,8192,65000]
def draw(seed,n,t):return np.random.default_rng(seed).choice(4,size=(n,t),p=P).astype(np.uint8)
def labels(a):
    # Vectorized independent oracle: most recent write index, default value0.
    writes=a<2;idx=np.maximum.accumulate(np.where(writes,np.arange(a.shape[1])[None,:],-1),axis=1)
    picked=np.take_along_axis(a,np.maximum(idx,0),axis=1)
    return np.where(idx<0,0,picked).astype(np.uint8)
def sequential_labels(a):
    out=np.zeros_like(a)
    for j,row in enumerate(a):
        state=0
        for i,s in enumerate(row):
            if s==0:state=0
            elif s==1:state=1
            out[j,i]=state
    return out
def tensor(a):return torch.nn.functional.one_hot(torch.from_numpy(a.astype(np.int64)),4).float()
def longset(t):
    rows=[];meta=[]
    def add(a,cat,value=None,j=None,multiplicity=1):
        i=len(rows);rows.append(a);meta.append(dict(id=f'T{t}_{cat}_{i:02}',category=cat,value=value,j=j,multiplicity=multiplicity))
    for j,a in enumerate(draw(50000+t,32,t)):add(a,'random',j=j)
    for cat in ('long_N','long_R'):
        for v in (0,1):
            for j in range(8):
                a=np.full(t,2 if cat=='long_N' else 3,dtype=np.uint8);a[:8]=draw(70000+j,1,8)[0];a[8]=v;a[-1]=3;add(a,cat,v,j)
    for cat in ('rewrite','mixed'):
        for v in (0,1):
            a=np.full(t,2,dtype=np.uint8)
            if cat=='rewrite':
                pos=[0,t//4,t//2,3*t//4]
                for k,i in enumerate(pos):
                    a[i]=v if k%2==0 else 1-v;a[i+1]=3
                    if k<3:a[pos[k+1]-1]=3
            else:
                a[0]=v;b=max(4,(t-1)//16);i=1;symbol=2
                while i+b<=t:
                    a[i:i+b]=symbol;i+=b;symbol=3 if symbol==2 else 2
            a[-1]=3;add(a,cat,v,0,8)
    return np.stack(rows),meta