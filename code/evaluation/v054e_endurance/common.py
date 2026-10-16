"""V054E shared code: paths, V054 model loading, inference engines, data generation, per-problem metrics.
Model classes are imported from IronMath_V054/model.py (not reimplemented). No training, no checkpoint writes."""
import csv,hashlib,json,random,sys,time
from pathlib import Path
from typing import List,Tuple
import numpy as np
import torch
from torch import nn

E=Path(__file__).resolve().parent;WORK=E.parent;V054=WORK/'IronMath_V054'
sys.path.insert(0,str(V054))
from model import Residual,batch as v054_batch  # noqa: E402  V054 code, imported unchanged

SEEDS=[0,1,2]
def ckpt_path(seed):return V054/f'checkpoints/C1_B0_s{seed}/best_validation.pt'

def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
def modelhash(m):  # same definition as V054 run.py
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def dump(p,x):
    p=E/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def load(p):return json.loads((E/p).read_text(encoding='utf-8'))
def csv_write(p,rows,mode='w'):
    p=E/p;keys=list(dict.fromkeys(k for r in rows for k in r)) if rows else ['no_rows']
    exists=p.exists() and mode=='a'
    with p.open(mode,encoding='utf-8-sig' if not exists else 'utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys)
        if not exists:w.writeheader()
        w.writerows({k:('NA' if v is None else v) for k,v in r.items()} for r in rows)
def csv_read(p):
    with (E/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def setup():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)

def load_base(seed,dtype=torch.float32):
    cp=torch.load(ckpt_path(seed),weights_only=False,map_location='cpu')
    assert cp['condition']=='C1' and cp['kind']=='B0' and cp['seed']==seed
    base=Residual('B0');base.load_state_dict(cp['model']);base=base.to(dtype).eval()
    for p in base.parameters():p.requires_grad_(False)
    return base,cp

# ---------------- engines ----------------
class Chunked(nn.Module):
    """Same per-bit recurrence as V054 Residual.forward (h=core(projection(x_i),h)+tanh(gate)*h, out=head(h)),
    using the V054 submodules themselves; hidden is carried across time chunks. Optional hidden summary every
    `every` steps on the first `tn` rows. `pre`=1 precomputes projection for the whole chunk (fast variant)."""
    def __init__(self,base:Residual,pre:int):
        super().__init__();self.projection=base.projection;self.core=base.core;self.head=base.head;self.residual_gate=base.residual_gate;self.pre=pre
    def forward(self,x:torch.Tensor,h:torch.Tensor,offset:int,tn:int,every:int)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        T=x.size(1);pred=torch.empty(x.size(0),T,dtype=torch.int64)
        g=torch.tanh(self.residual_gate)
        stats=torch.jit.annotate(List[torch.Tensor],[])
        if self.pre==1:
            p=self.projection(x)
            for i in range(T):
                h=self.core(p[:,i],h)+g*h;pred[:,i]=self.head(h).argmax(-1)
                if tn>0 and (offset+i+1)%every==0:
                    hh=h[:tn];n=torch.linalg.vector_norm(hh,2.0,[1])
                    stats.append(torch.stack([torch.full_like(n.mean(),float(offset+i+1)),n.mean(),n.max(),(hh.abs()>0.99).to(h.dtype).mean(),(g*n).abs().mean()]))
        else:
            for i in range(T):
                h=self.core(self.projection(x[:,i]),h)+g*h;pred[:,i]=self.head(h).argmax(-1)
                if tn>0 and (offset+i+1)%every==0:
                    hh=h[:tn];n=torch.linalg.vector_norm(hh,2.0,[1])
                    stats.append(torch.stack([torch.full_like(n.mean(),float(offset+i+1)),n.mean(),n.max(),(hh.abs()>0.99).to(h.dtype).mean(),(g*n).abs().mean()]))
        st=torch.stack(stats) if len(stats)>0 else torch.zeros(0,5)
        return pred,h,st

def rows_input(rows,dtype=torch.float32):
    x,_,ll=v054_batch(rows,'B0',labels=False);return x.to(dtype),ll

def run_base(scripted,rows):
    """V054 path exactly: model.batch -> scripted Residual forward -> argmax."""
    x,ll=rows_input(rows);return scripted(x).argmax(-1).numpy(),ll

def run_chunked(engine,rows,chunk=4096,tn=0,every=256,dtype=torch.float32):
    x,ll=rows_input(rows,dtype);B,T,_=x.shape;h=torch.zeros(B,545,dtype=dtype);out=[];st=[]
    for s in range(0,T,chunk):
        p,h,t=engine(x[:,s:s+chunk].contiguous(),h,s,tn,every);out.append(p);st.append(t)
    return torch.cat(out,1).numpy(),ll,torch.cat(st,0).numpy()

def make_engine(base,pre):return torch.jit.script(Chunked(base,pre)).eval()

# ---------------- data generation ----------------
EDGES=lambda L:[(0,0),(0,L),(L,L),(L,1),(L,L-1),(1,1),(0,1),(L-1,L-1),(L-2,L-2),(L-3,L-3),(L,L-2),(L,L-3),(0,L-1),(0,L-2),(0,2),(0,3)]
V050_CATS=[('boundary',16),('random',128),('propagation',64),('alternating',16),('sparse',16),('dense',16)]
V053_KINDS=('random','no_propagation','long_propagation','zero_operand','extrema','boundary')
def bits_int(v):return [int(b) for b in v]
def integer(v):return sum(int(b)<<j for j,b in enumerate(v))

def counts(scale_half=False):
    """Category counts: V050-part (256) + V053-fill (256 cycling 6 kinds -> 43,43,43,43,42,42)."""
    v050=[(c,n) for c,n in V050_CATS];v053=[(k,43 if i<4 else 42) for i,k in enumerate(V053_KINDS)]
    if scale_half:
        v050=[(c,max(8,(n+1)//2)) for c,n in v050];v053=[(k,max(8,(n+1)//2)) for k,n in v053]
    return v050,v053

def generate(width,seed,set_name,scale_half=False,probe=False):
    """V050 prepare.generate (256 part) + V053 dataset.generate fill (256 part), ADD only, one rng seeded `seed`.
    Changes for long widths are listed in generator_notes.md."""
    rng=random.Random(seed);limit=(1<<width)-1;used=set();rows=[]
    def add(a,b,cat,origin,L=None):
        key=tuple(sorted((a,b)))
        if key in used:return False
        used.add(key);rows.append(dict(id=f'{set_name}_ADD_{width}_{len(rows)}',op='ADD',a=a,b=b,width=width,category=cat,origin=origin,_L=L));return True
    v050,v053=counts(scale_half)
    if probe:v050=[('boundary',8),('random',16),('propagation',8)];v053=[]
    for cat,n in v050:
        count=0;attempt=0
        candidates=[l for l in [1,2,3,4,5,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536] if l<=width]
        while count<n:
            attempt+=1;assert attempt<100000;L=None
            if cat=='boundary':
                a,b=EDGES(limit)[count];assert add(a,b,'boundary','V050gen_boundary');count+=1;continue
            if cat=='random':a=rng.getrandbits(width);b=rng.getrandbits(width)
            elif cat=='propagation':
                L=candidates[count%len(candidates)];start=rng.randrange(width-L+1);non=[(0,0),(1,0),(0,1)]
                pairs=[rng.choice(non) for _ in range(width)];pairs[start]=(1,1)
                for j in range(start+1,start+L):pairs[j]=rng.choice([(0,1),(1,0),(1,1)])
                if start+L<width:pairs[start+L]=(0,0)
                a=integer([v[0] for v in pairs]);b=integer([v[1] for v in pairs])
            elif cat=='alternating':
                base=int('01'*(width//2+1),2)&limit;a=base;b=limit^base
                for _ in range(rng.randrange(1,4)):a^=1<<rng.randrange(width);b^=1<<rng.randrange(width)
            else:
                a=sum(1<<j for j in rng.sample(range(width),rng.randrange(1,min(8,width)+1)));b=sum(1<<j for j in rng.sample(range(width),rng.randrange(1,min(8,width)+1)))
                if cat=='dense':a=limit^a;b=limit^b
            if add(a,b,cat,'V050gen_'+cat,L):count+=1
    for kind,n in v053:
        count=0;attempt=0
        while count<n:
            attempt+=1;assert attempt<100000
            if kind=='random':a,b=rng.getrandbits(width),rng.getrandbits(width)
            elif kind=='no_propagation':a=rng.getrandbits(width);b=rng.getrandbits(width)&~a
            elif kind=='long_propagation':
                k=rng.randrange(max(1,width//2),width);upper=rng.getrandbits(width-k)<<k
                a=(upper|((1<<k)-1))&limit;b=rng.randrange(1,4)
            elif kind=='zero_operand':a,b=rng.getrandbits(width),0
            elif kind=='extrema':a,b=limit-rng.randrange(9),rng.getrandbits(width)
            else:a,b=(1<<rng.randrange(width)),rng.getrandbits(width)
            a,b=sorted((a,b))
            if add(a,b,kind,'V053gen_'+kind):count+=1
    for r in rows:annotate(r)
    return rows

def carry_vectors(a,b,width):
    """carry-in / carry-out per output position 0..width (Python integers)."""
    cin=(a+b)^a^b;cout=cin>>1;return cin,cout
def longest_ones(v):
    s=bin(v)[2:];return max((len(t) for t in s.split('0')),default=0) if v else 0
def annotate(r):
    a,b,w=r['a'],r['b'],r['width'];cin,cout=carry_vectors(a,b,w);L=longest_ones(cout)
    planted=r.pop('_L',None)
    if planted is not None:assert L==planted,(planted,L)
    r['answer']=a+b;r['length']=w+1;r['longest_propagation']=L
    return r

def boolean_ripple(rows):
    """Independent vectorised Boolean ripple-carry adder over the whole set (numpy, per bit)."""
    w=rows[0]['width'];n=len(rows)
    def mat(vals):
        by=np.frombuffer(b''.join(v.to_bytes((w+8)//8+1,'little') for v in vals),np.uint8).reshape(n,-1)
        return np.unpackbits(by,axis=1,bitorder='little')[:,:w+1]
    A=mat([r['a'] for r in rows]);Bm=mat([r['b'] for r in rows]);c=np.zeros(n,np.uint8);S=np.zeros_like(A)
    for i in range(w+1):
        x=A[:,i]^Bm[:,i];S[:,i]=x^c;c=(A[:,i]&Bm[:,i])|(c&x)
    assert not c.any()
    return S

def to_bits(vals,w):
    n=len(vals);by=np.frombuffer(b''.join(v.to_bytes((w+8)//8+1,'little') for v in vals),np.uint8).reshape(n,-1)
    return np.unpackbits(by,axis=1,bitorder='little')[:,:w+1]
def bits_to_int(bits):
    return int.from_bytes(np.packbits(bits.astype(np.uint8),bitorder='little').tobytes(),'little')

def save_set(name,rows):
    p=E/'data'/f'{name}.json';p.parent.mkdir(parents=True,exist_ok=True)
    out=[dict(id=r['id'],op=r['op'],width=r['width'],category=r['category'],origin=r['origin'],a=hex(r['a']),b=hex(r['b']),answer=hex(r['answer']),length=r['length'],longest_propagation=r['longest_propagation']) for r in rows]
    p.write_text(json.dumps(out),encoding='utf-8');return sha_file(p)
def load_set(name):
    rows=json.loads((E/'data'/f'{name}.json').read_text(encoding='utf-8'))
    for r in rows:r['a']=int(r['a'],16);r['b']=int(r['b'],16);r['answer']=int(r['answer'],16)
    return rows
def verify_rows(rows):
    """Python integer answers vs independent Boolean ripple, uniqueness, planted propagation lengths."""
    S=boolean_ripple(rows);w=rows[0]['width']
    for r,s in zip(rows,S):
        assert r['answer']==r['a']+r['b']==bits_to_int(s) and 0<=r['a']<(1<<w) and 0<=r['b']<(1<<w)
    assert len({tuple(sorted((r['a'],r['b']))) for r in rows})==len(rows)
    return True

# ---------------- per-problem metrics ----------------
def problem_metrics(r,pred_bits):
    w=r['width'];n=w+1;truth=to_bits([r['answer']],w)[0];p=np.asarray(pred_bits[:n],np.uint8)
    wrong=np.nonzero(p!=truth)[0];pred_int=bits_to_int(p)
    cin,cout=carry_vectors(r['a'],r['b'],w)
    if len(wrong):
        f=int(wrong[0]);ci=(cin>>f)&1;co=(cout>>f)&1
        phase='start' if not ci and co else 'middle' if ci and co else 'end' if ci and not co else 'none'
        d=np.diff(wrong);breaks=np.nonzero(d!=1)[0];runs=np.diff(np.concatenate([[-1],breaks,[len(wrong)-1]]));longest=int(runs.max())
    else:f=None;phase='correct';longest=0
    return dict(problem_id=r['id'],width=w,category=r['category'],origin=r['origin'],length=n,prediction_hex=hex(pred_int),
                exact=int(pred_int==r['answer'] and len(wrong)==0),bit_accuracy=1-len(wrong)/n,first_error=f,last_error=int(wrong[-1]) if len(wrong) else None,
                wrong_bits=int(len(wrong)),longest_error_run=longest,first_error_phase=phase,longest_propagation=r['longest_propagation'])
