"""Saved-results analysis only (no model execution): summary, boundary/confirmation tables, hazard, KM survival,
failure cases, reference prediction, preservation check."""
import csv,json,math,statistics,time
from collections import defaultdict
from common import E,WORK,SEEDS,dump,load,csv_read,csv_write,sha_file
from prep import manifest,AUDIT_DIRS

def num(v):return None if v in ('NA','',None) else int(v)
def sd(x):return statistics.stdev(x) if len(x)>1 else None

def main():
    cfg=load('config.json');prog=json.loads((E/'progress.json').read_text());st=prog['state'];R=csv_read('results_per_problem.csv')
    for r in R:r['seed']=int(r['seed']);r['width']=int(r['width']);r['length']=int(r['length']);r['exact']=int(r['exact']);r['first_error']=num(r['first_error'])
    # summary: per set x seed, set mean, categories
    summ=[];groups=defaultdict(list)
    for r in R:groups[(r['set_type'],r['set_name'],r['width'])].append(r)
    for (t,name,w),rr in sorted(groups.items(),key=lambda k:(k[0][2],k[0][0])):
        per={s:[r for r in rr if r['seed']==s] for s in SEEDS}
        rates={s:sum(r['exact'] for r in x)/len(x) for s,x in per.items() if x}
        for s,x in per.items():
            if x:summ.append(dict(set_type=t,set_name=name,width=w,seed=s,category='ALL',n=len(x),exact=sum(r['exact'] for r in x),exact_rate=rates[s],bit_accuracy=statistics.mean(float(r['bit_accuracy']) for r in x),wall_seconds=sum(float(r['wall_seconds']) for r in x)))
        summ.append(dict(set_type=t,set_name=name,width=w,seed='mean',category='ALL',n=len(per[0]),exact_rate=statistics.mean(rates.values()),exact_rate_sample_SD=sd(list(rates.values())),all_seeds_ge_95=all(v>=0.95 for v in rates.values()),mean_ge_95=statistics.mean(rates.values())>=0.95))
        cats=sorted({r['category'] for r in rr})
        for c in cats:
            cr={s:[r for r in per[s] if r['category']==c] for s in SEEDS}
            cv=[sum(r['exact'] for r in x)/len(x) for x in cr.values() if x]
            for s,x in cr.items():
                if x:summ.append(dict(set_type=t,set_name=name,width=w,seed=s,category=c,n=len(x),exact=sum(r['exact'] for r in x),exact_rate=sum(r['exact'] for r in x)/len(x)))
            summ.append(dict(set_type=t,set_name=name,width=w,seed='mean',category=c,n=len(cr[0]),exact_rate=statistics.mean(cv),exact_rate_sample_SD=sd(cv)))
        if t=='confirmation':
            ov=set(prog['sets'][name].get('overlap_ids',[]))
            ex={s:[r for r in per[s] if r['problem_id'] not in ov] for s in SEEDS};ev=[sum(r['exact'] for r in x)/len(x) for x in ex.values() if x]
            summ.append(dict(set_type=t,set_name=name,width=w,seed='mean',category='ALL_excluding_shared_edges',n=len(ex[0]),exact_rate=statistics.mean(ev),exact_rate_sample_SD=sd(ev),per_seed=json.dumps(ev)))
    csv_write('summary.csv',summ)
    # boundary / confirmation tables
    bs=st.get('boundary',{}).get('midpoints',[]);csv_write('boundary_search.csv',[dict(k=x['k'],width=x['width'],lo=x.get('lo'),hi=x.get('hi'),mean_exact=x.get('mean'),seed0=(x.get('per_seed') or [None]*3)[0],seed1=(x.get('per_seed') or [None]*3)[1],seed2=(x.get('per_seed') or [None]*3)[2],passed=x.get('passed'),note='exploratory') for x in bs])
    c=st.get('confirmation');csv_write('confirmation.csv',[dict(width=c['width'],n=c['n'],mean_exact=c['mean'],seed0=c['per_seed'][0],seed1=c['per_seed'][1],seed2=c['per_seed'][2],mean_ge_95=c['passed'],all_seeds_ge_95=all(v>=0.95 for v in c['per_seed']))] if c and c.get('per_seed') else [])
    # hazard on main-evaluation problems
    M=[r for r in R if r['set_type']=='main'];H=[]
    def hazard(label,key,rows):
        for s,e in cfg['hazard_intervals']:
            risk=[r for r in rows if r['length']>s and (r['first_error'] is None or r['first_error']>=s)]
            ev=sum(1 for r in risk if r['first_error'] is not None and s<=r['first_error']<e)
            exp=sum(min(e,r['length'],(r['first_error']+1) if r['first_error'] is not None else 10**9)-s for r in risk)
            H.append(dict(grouping=label,group=key,interval_start=s,interval_end=e,at_risk_at_entry=len(risk),events=ev,exposed_bits=exp,hazard_per_bit=ev/exp if exp else None,hazard_per_2048_bits=ev/exp*2048 if exp else None))
    hazard('all','main_all',M)
    for k in sorted({r['width'] for r in M}):hazard('sequence_length',str(k+1),[r for r in M if r['width']==k])
    for k in sorted({r['category'] for r in M}):hazard('category',k,[r for r in M if r['category']==k])
    for k in SEEDS:hazard('seed',str(k),[r for r in M if r['seed']==k])
    for k in sorted({r['width'] for r in M}):
        for s in SEEDS:hazard('width_seed',f'{k}|s{s}',[r for r in M if r['width']==k and r['seed']==s])
    csv_write('hazard.csv',H)
    # Kaplan-Meier survival of 'no error yet' by output position
    K=[]
    def km(label,key,rows):
        events=defaultdict(int)
        for r in rows:
            if r['first_error'] is not None:events[r['first_error']]+=1
        S=1.0;K.append(dict(grouping=label,group=key,position=0,at_risk=len(rows),events=0,censored_before=0,survival=1.0))
        for t in sorted(events):
            n=sum(1 for r in rows if r['length']>t and (r['first_error'] is None or r['first_error']>=t))
            S*=1-events[t]/n;K.append(dict(grouping=label,group=key,position=t,at_risk=n,events=events[t],censored_before=sum(1 for r in rows if r['first_error'] is None and r['length']<=t),survival=S))
        mx=max(r['length'] for r in rows);K.append(dict(grouping=label,group=key,position=mx,at_risk=sum(1 for r in rows if r['first_error'] is None and r['length']>=mx),events=0,censored_before=sum(1 for r in rows if r['first_error'] is None and r['length']<mx),survival=S))
    km('all','main_all',M)
    for s in SEEDS:km('seed',str(s),[r for r in M if r['seed']==s])
    for k in sorted({r['width'] for r in M}):km('sequence_length',str(k+1),[r for r in M if r['width']==k])
    for k in sorted({r['category'] for r in M}):km('category',k,[r for r in M if r['category']==k])
    csv_write('survival_curve.csv',K)
    # failure cases (all sets)
    csv_write('failure_cases.csv',[{k:v for k,v in r.items()} for r in R if r['exact']==0])
    # reference prediction
    p2048={s:next(x['exact_rate'] for x in summ if x['set_name']=='main_2048' and x['seed']==s and x['category']=='ALL') for s in SEEDS}
    ref=[]
    for w in cfg['widths']:
        pr={s:p2048[s]**(w/2048) for s in SEEDS};obs=next((x for x in summ if x['set_name']==f'main_{w}' and x['seed']=='mean' and x['category']=='ALL'),None)
        ref.append(dict(width=w,predicted_seed0=pr[0],predicted_seed1=pr[1],predicted_seed2=pr[2],predicted_mean=statistics.mean(pr.values()),predicted_from_mean_p=statistics.mean(p2048.values())**(w/2048),observed_mean=obs['exact_rate'] if obs else None))
    csv_write('reference_prediction.csv',ref)
    pm=statistics.mean(p2048.values());dump('analysis_stats.json',dict(p2048=p2048,p2048_mean=pm,predicted_95_crossing_width_from_mean=2048*math.log(0.95)/math.log(pm) if pm<1 else None,state=st))
    # preservation
    before=json.loads((E/'preservation_before.json').read_text())['sha256'];t=time.perf_counter();after=manifest()
    changed=[k for k in before if k in after and before[k]!=after[k]];missing=[k for k in before if k not in after];added=[k for k in after if k not in before]
    dump('preservation_check.json',dict(dirs=AUDIT_DIRS,excluded=['v054e_endurance','IronMath_V055'],files_before=len(before),files_after=len(after),changed=changed,missing=missing,added=added,identical=not(changed or missing or added),seconds=time.perf_counter()-t,checked_at=time.strftime('%Y-%m-%dT%H:%M:%S')))
    print('ANALYSIS DONE; preservation identical =',not(changed or missing or added),flush=True)

if __name__=='__main__':main()
