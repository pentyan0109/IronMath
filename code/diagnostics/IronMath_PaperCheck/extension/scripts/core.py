from common import *
import torch,random
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
class Residual(nn.Module):
    """Identical parameter creation order and recurrence to V054 B0, no new weights."""
    def __init__(self):
        super().__init__();self.projection=nn.Linear(3,64);self.core=nn.GRUCell(64,545);self.residual_gate=nn.Parameter(torch.zeros(()));self.head=nn.Linear(545,2)
    def forward(self,x:torch.Tensor):
        h=torch.zeros(x.size(0),545,dtype=x.dtype,device=x.device);out=torch.jit.annotate(list[torch.Tensor],[])
        for i in range(x.size(1)):
            h=self.core(self.projection(x[:,i]),h)+torch.tanh(self.residual_gate)*h;out.append(self.head(h))
        return torch.stack(out,1)
    @torch.jit.export
    def advance(self,x:torch.Tensor,h:torch.Tensor):return self.core(self.projection(x),h)+torch.tanh(self.residual_gate)*h
    @torch.jit.export
    def evolution(self,x:torch.Tensor,h:torch.Tensor,target:torch.Tensor,count:int):
        margins=torch.empty(count,x.size(0));moves=torch.empty_like(margins);saved=torch.empty(min(count,512)+1,x.size(0),545);saved[0]=h
        for t in range(count):
            old=h;h=self.advance(x,h);z=self.head(h);margins[t]=z.gather(1,target[:,None])[:,0]-z.gather(1,(1-target)[:,None])[:,0];moves[t]=torch.linalg.vector_norm(h-old,dim=1)
            if t<512:saved[t+1]=h
        return h,saved,margins,moves
def modelhash(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def setup(seed):torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
def load_model(path):
    cp=torch.load(path,map_location='cpu',weights_only=False);m=Residual();m.load_state_dict(cp['model']);m.eval();return m,cp
def batch(rr):
    ll=[r['width']+int(r['op']=='ADD') for r in rr];x=np.zeros((len(rr),max(ll),3),np.float32);y=np.full((len(rr),max(ll)),-100,np.int64)
    for j,(r,n) in enumerate(zip(rr,ll)):
        a=int(r['a']);b=int(r['b']);v=a+b if r['op']=='ADD' else a-b
        for t in range(n):x[j,t]=[(a>>t)&1,(b>>t)&1,int(r['op']=='SUB')];y[j,t]=(v>>t)&1
    return torch.from_numpy(x),torch.from_numpy(y),ll
def data():return {f'{op}_{sp}':read(W/f'IronMath_V053/data/{op}_{sp}.json') for op in ['ADD','SUB'] for sp in ['Train','Validation','Test8','Test64','Test256','Test2048']}
def selected_probes():
    with (W/'v054t_transition/data/probes.jsonl').open(encoding='utf-8') as f:rr=[json.loads(s) for s in f]
    return [r for r in rr if r['k']==4096 and r['prefix']=='min' and r['suffix']=='s0']
def starts(m,rr):
    hs=[];xx=[];yy=[]
    with torch.no_grad():
        for r in rr:
            a=int(r['a'],16);b=int(r['b'],16);h=torch.zeros(1,545)
            for t in range(r['hold_start']):h=m.advance(torch.tensor([[(a>>t)&1,(b>>t)&1,int(r['op']=='SUB')]],dtype=torch.float32),h)
            hs.append(h[0]);xx.append([int(r['pair'][0]),int(r['pair'][1]),int(r['op']=='SUB')]);s=r['state'];pa,pb=map(int,r['pair']);yy.append((pa+pb+s)%2 if r['op']=='ADD' else (pa-pb-s)%2)
    return torch.stack(hs),torch.tensor(xx,dtype=torch.float32),torch.tensor(yy)
@torch.no_grad()
def evaluate(m,rr):
    result=[];loss=nc=0.;count=0
    for j in range(0,len(rr),128):
        r=rr[j:j+128];x,y,ll=batch(r);z=m(x);p=z.argmax(-1);valid=torch.isfinite(z).all(-1);mask=y!=-100
        loss+=float(F.cross_entropy(z.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum'));count+=int(mask.sum())
        for a,pred,fin,n in zip(r,p.numpy(),valid.numpy(),ll):
            pred=pred[:n];fin=fin[:n];value=sum(int(v)<<i for i,v in enumerate(pred)) if fin.all() else None;truth=a['a']+a['b'] if a['op']=='ADD' else a['a']-a['b']
            result.append(dict(id=a['id'],op=a['op'],width=a['width'],a=hex(a['a']),b=hex(a['b']),truth=hex(truth),prediction=hex(value) if value is not None else None,correct=value==truth,finite=bool(fin.all())))
    return loss/count,result
