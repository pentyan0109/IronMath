"""Independent arithmetic/statistical helpers; never imports experiment modules."""
import csv,json,math,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; W=R.parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(name,x): (R/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(name,rs):
    if not rs:return
    keys=list(dict.fromkeys(k for r in rs for k in r))
    with (R/name).open('w',encoding='utf-8',newline='') as f:
        q=csv.DictWriter(f,keys);q.writeheader();q.writerows(rs)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def score(p,y):
    p=np.asarray(p,bool);y=np.asarray(y,bool)
    tp=int((p&y).sum());fp=int((p&~y).sum());tn=int((~p&~y).sum());fn=int((~p&y).sum())
    return dict(N=len(y),hits=tp+tn,TP=tp,FP=fp,TN=tn,FN=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)
def paired(a,b,y,seed):
    a=np.asarray(a,bool)==y;b=np.asarray(b,bool)==y
    onlya=int((a&~b).sum());onlyb=int((b&~a).sum());n=onlya+onlyb
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(onlya,onlyb)+1))/2**n) if n else 1.
    d=a.astype(int)-b.astype(int);inds=np.random.default_rng(seed).integers(0,len(d),size=(10000,len(d)))
    ci=np.percentile(d[inds].mean(axis=1),[2.5,97.5]).tolist()
    return dict(A_only_correct=onlya,other_only_correct=onlyb,McNemar_exact_two_sided_p=p,accuracy_difference=float(d.mean()),CI_low=ci[0],CI_high=ci[1],bootstrap_seed=seed,bootstrap_samples=10000)
def equal(a,b):
    if isinstance(a,(float,int)) and isinstance(b,(float,int)):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
class Comparisons:
    def __init__(self):self.data=[]
    def add(self,e,m,old,new,source,raw,limitation=''):
        self.data.append(dict(experiment=e,metric=m,reported_value=json.dumps(old,ensure_ascii=False),recomputed_value=json.dumps(new,ensure_ascii=False),match='not_reported' if old=='NOT_REPORTED' else equal(old,new),reported_source=source,raw_source=raw,limitation=limitation))
