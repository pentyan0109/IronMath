"""R2 (Holm) and R3 (seed-cluster bootstrap) from stage-1 independently rescored per-model records. No training, no inference."""
import csv,json,math,datetime
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;X=S.parent
ARCH=['residual_gru','gru','lstm','rnn'];N_BOOT=10000
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(name,rs):
    keys=list(dict.fromkeys(k for r in rs for k in r))
    with (S/name).open('w',encoding='utf-8',newline='') as f:
        q=csv.DictWriter(f,keys);q.writeheader();q.writerows(rs)
def mcnemar(ca,cb):
    oa=int((ca&~cb).sum());ob=int((cb&~ca).sum());n=oa+ob
    return oa,ob,(min(1.,2*sum(math.comb(n,k) for k in range(min(oa,ob)+1))/2**n) if n else 1.)
def holm(ps):
    order=sorted(range(len(ps)),key=lambda i:ps[i]);m=len(ps);adj=[0.]*m;run=0.
    for r,i in enumerate(order):run=max(run,min(1.,(m-r)*ps[i]));adj[i]=run
    return adj
def model_ci(d,seed):
    inds=np.random.default_rng(seed).integers(0,len(d),size=(N_BOOT,len(d)))
    return np.percentile(d[inds].mean(axis=1),[2.5,97.5]).tolist()
def seed_ci(d,seeds,rng_seed):
    us=np.unique(seeds);idx=[np.flatnonzero(seeds==s) for s in us]
    draws=np.random.default_rng(rng_seed).integers(0,len(us),size=(N_BOOT,len(us)));out=np.empty(N_BOOT)
    for b in range(N_BOOT):
        sel=np.concatenate([idx[k] for k in draws[b]]);out[b]=d[sel].mean()
    return np.percentile(out,[2.5,97.5]).tolist()
def run():
    ms=[m for m in load(X/'memory_models.json') if m['study']=='G1' and m['architecture']!='linear']
    assert len(ms)==160 and len({m['model'] for m in ms})==160
    seeds=np.array([m['seed'] for m in ms]);arch=np.array([m['architecture'] for m in ms])
    assert sorted(set(seeds.tolist()))==list(range(20,40)) and all((seeds==s).sum()==8 for s in range(20,40))
    y=np.array([m['actual3'] for m in ms]);A=np.array([m['A'] for m in ms]);D=np.array([m['D'] for m in ms])
    bcum=lambda ell:np.array([all(m['lengths'][str(t)] for t in [32,64,128,512] if t<=ell) for m in ms])
    others={'D':(D,81000),'B_cum128':(bcum(128),82000),'B_cum512':(bcum(512),82000)}
    r1=rows(X/'r1_comparison.csv');checks=[]
    # R2
    r2=[]
    for name,(B,_) in others.items():
        res=[mcnemar((A==y)[arch==a],(B==y)[arch==a]) for a in ARCH];adj=holm([r[2] for r in res])
        for a,(oa,ob,p),q in zip(ARCH,res,adj):
            r2.append(dict(group={'D':'1: A vs D','B_cum128':'2: A vs B_cum(128)','B_cum512':'3: A vs B_cum(512)'}[name],structure=a,N=int((arch==a).sum()),A_hits=int((A==y)[arch==a].sum()),other_hits=int((B==y)[arch==a].sum()),A_only_correct=oa,other_only_correct=ob,p_unadjusted=p,p_holm=q,significant_unadjusted_005=p<.05,significant_holm_005=q<.05))
            if name!='D':
                ref=[r for r in r1 if r['experiment']=='G1' and r['structure']==a and r['ell']==name[-3:] and r['baseline']=='B_cum'][0]
                checks.append(dict(check=f'R2 p matches r1_comparison.csv {a} {name}',passed=math.isclose(float(ref['McNemar_exact_two_sided_p']),p,rel_tol=1e-12)))
    table('r2_holm.csv',r2)
    # R3
    r3=[]
    for name,(B,mseed) in others.items():
        d=(A==y).astype(int)-(B==y).astype(int)
        mci=model_ci(d,mseed);sci=seed_ci(d,seeds,83000)
        r3.append(dict(comparison='A - '+name.replace('B_cum','B_cum(')+(')' if 'B_cum' in name else ''),N_models=160,N_seeds=20,accuracy_difference=float(d.mean()),seed_cluster_CI_low=sci[0],seed_cluster_CI_high=sci[1],seed_cluster_rng=83000,model_CI_low=mci[0],model_CI_high=mci[1],model_rng=mseed,bootstrap_samples=N_BOOT,seed_cluster_width=sci[1]-sci[0],model_width=mci[1]-mci[0]))
        if name!='D':
            ref=[r for r in r1 if r['experiment']=='G1' and r['structure']=='nonlinear' and r['ell']==name[-3:] and r['baseline']=='B_cum'][0]
            checks.append(dict(check=f'R3 model CI reproduces r1_comparison.csv {name}',passed=all(math.isclose(float(ref[k]),v,abs_tol=1e-12) for k,v in [('CI_low',mci[0]),('CI_high',mci[1]),('accuracy_difference',float(d.mean()))])))
    ref=load(X/'memory_group_statistics.json')
    table('r3_seed_cluster.csv',r3)
    # per-seed accuracy table (descriptive)
    per=[dict(seed=int(s),A_hits=int((A==y)[seeds==s].sum()),D_hits=int((D==y)[seeds==s].sum()),B_cum128_hits=int((others['B_cum128'][0]==y)[seeds==s].sum()),B_cum512_hits=int((others['B_cum512'][0]==y)[seeds==s].sum()),actual_success=int(y[seeds==s].sum()),N=8) for s in range(20,40)]
    table('r3_per_seed.csv',per)
    checks.append(dict(check='Holm adjusted p monotone and >= unadjusted',passed=all(r['p_holm']>=r['p_unadjusted'] for r in r2)))
    checks.append(dict(check='Holm reference example [0.01,0.04,0.03,0.005]',passed=np.allclose(holm([.01,.04,.03,.005]),[.03,.06,.06,.02])))
    (S/'r2r3_record.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source='IronMath_PaperCheck/memory_models.json (stage-1 independent raw-bit rescoring)',rules_SHA256=load(S/'S1S2_RULES_SHA256.json')['SHA256'],checks=checks,all_passed=all(c['passed'] for c in checks),memory_group_statistics_keys=list(ref)[:20] if isinstance(ref,dict) else None,new_training=False,new_inference=False),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False,indent=1))
if __name__=='__main__':run()
