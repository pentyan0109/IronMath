from common import *
import numpy as np
from models import ARCHS,make
def identity(arch,length,seed):return f'{arch}_L{length}_s{seed:02}'
def conditions(seeds):return [(identity(a,l,s),a,l,s) for a in ARCHS for l in [8,16] for s in seeds]
def pilot_conditions():return conditions([20])
def write_utf8(name,text):
    p=R/name
    if p.exists():raise FileExistsError(p)
    p.write_text(text,encoding='utf-8')
    actual=p.read_text(encoding='utf-8',errors='strict')
    assert actual==text and '\ufffd' not in actual and '???' not in actual
    return sha(p)
def load_checkpoint(asset):
    import torch
    p=R/asset['path'];assert sha(p)==asset['SHA256']
    cp=torch.load(p,map_location='cpu',weights_only=False);m=make(cp['architecture']).eval();m.load_state_dict(cp['model'])
    for q in m.parameters():q.requires_grad_(False)
    return m,cp
def scientific_metrics(models,seed=81000):
    import math
    n=len(models);methods=['A','B','C_all_success','C_all_failure','D'];stats=[]
    for method in methods:
        tp=sum(m[method] and m['actual_success'] for m in models);fp=sum(m[method] and not m['actual_success'] for m in models);tn=sum(not m[method] and not m['actual_success'] for m in models);fn=sum(not m[method] and m['actual_success'] for m in models)
        stats.append(dict(method=method,N=n,hits=tp+tn,TP=tp,FP=fp,TN=tn,FN=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None))
    b=sum((m['A']==m['actual_success']) and (m['D']!=m['actual_success']) for m in models);c=sum((m['D']==m['actual_success']) and (m['A']!=m['actual_success']) for m in models);discord=b+c
    p=min(1.,2*sum(math.comb(discord,k) for k in range(min(b,c)+1))/2**discord) if discord else 1.
    d=np.array([int(m['A']==m['actual_success'])-int(m['D']==m['actual_success']) for m in models])
    ci=np.percentile(np.random.default_rng(seed).choice(d,size=(10000,n),replace=True).mean(1),[2.5,97.5]).tolist()
    return dict(methods=stats,A_only_correct=b,D_only_correct=c,McNemar_exact_two_sided_p=p,accuracy_difference=float(d.mean()),bootstrap95_CI=ci,bootstrap_seed=seed,bootstrap_samples=10000)
