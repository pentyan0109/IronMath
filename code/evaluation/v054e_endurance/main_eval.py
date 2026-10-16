"""Stages C-E: main evaluation (ascending widths, all 3 seeds per width), boundary search, independent confirmation,
float64 replay (exploratory), optional 65000-bit probe. Resumable: completed (set, seed) units are not recomputed."""
import datetime,json,math,statistics,sys,time
import numpy as np
import torch
from common import (E,SEEDS,setup,load_base,modelhash,dump,load,csv_write,csv_read,load_set,save_set,generate,verify_rows,
                    make_engine,run_base,rows_input,to_bits,problem_metrics,sha_file)

CFG=load('config.json');EST=load('estimate.json');ENV=load('environment.json')
T0=datetime.datetime.fromisoformat(ENV['started']).timestamp();BUDGET=CFG['time_budget_hours']*3600
PROG=E/'progress.json'
def progress():return json.loads(PROG.read_text()) if PROG.exists() else dict(done={},sets={},log=[])
def save_progress(p):PROG.write_text(json.dumps(p,indent=1),encoding='utf-8')
def elapsed():return time.time()-T0
def note(p,msg):p['log'].append(dict(time=datetime.datetime.now().isoformat(timespec='seconds'),msg=msg));save_progress(p);print(msg,flush=True)

def fast_input(rows,dtype=torch.float32):
    w=rows[0]['width'];A=to_bits([r['a'] for r in rows],w);B=to_bits([r['b'] for r in rows],w)
    x=np.zeros((len(rows),w+1,3),np.float32);x[:,:,0]=A;x[:,:,1]=B;return torch.from_numpy(x).to(dtype)

def check_fast_input(rows):
    sub=rows[:16];x,_=rows_input(sub);return bool(torch.equal(x,fast_input(sub)))

ENGINES={}
def engine(seed,dtype=torch.float32):
    k=(seed,dtype)
    if k not in ENGINES:
        base,_=load_base(seed,dtype);ENGINES[k]=(base,modelhash(base),make_engine(base,1),torch.jit.script(base).eval())
    return ENGINES[k]

@torch.no_grad()
def infer(seed,rows,tn,dtype=torch.float32):
    base,h0,fast,scripted=engine(seed,dtype)
    if EST['engine']=='fast':
        x=fast_input(rows,dtype);B,T,_=x.shape;h=torch.zeros(B,545,dtype=dtype);out=[];st=[]
        for s in range(0,T,4096):
            p,h,t=fast(x[:,s:s+4096].contiguous(),h,s,tn,256);out.append(p);st.append(t)
        return torch.cat(out,1).numpy(),torch.cat(st,0).numpy()
    x,_=rows_input(rows,dtype);return scripted(x).argmax(-1).numpy(),np.zeros((0,5))

def subset_half(rows):
    groups={};out=[]
    for r in rows:groups.setdefault((r['origin'],r['category']),[]).append(r)
    for g in groups.values():out+=g[:max(8,(len(g)+1)//2)]
    keep={id(r) for r in out};return [r for r in rows if id(r) in keep]

def eval_unit(p,set_type,set_name,rows,seed,batch=128,trace=True):
    key=f'{set_name}|s{seed}'
    if key in p['done']:return p['done'][key]
    t_unit=time.perf_counter();res=[];trace_rows=[];base,h0,_,_=engine(seed);tanh_g=float(torch.tanh(base.residual_gate))
    for j in range(0,len(rows),batch):
        rr=rows[j:j+batch];tn=min(16,len(rr)) if (trace and j==0) else 0
        t=time.perf_counter();pp,st=infer(seed,rr,tn);dt=time.perf_counter()-t
        for i,(r,pr) in enumerate(zip(rr,pp)):
            m=problem_metrics(r,pr);m.update(set_type=set_type,set_name=set_name,seed=seed,batch_size=len(rr),wall_seconds=dt/len(rr),traced=int(i<tn));res.append(m)
        for s in st:trace_rows.append(dict(set_type=set_type,set_name=set_name,width=rows[0]['width'],seed=seed,rows=tn,step=int(s[0]),norm_mean=float(s[1]),norm_max=float(s[2]),frac_abs_gt_099=float(s[3]),residual_term_mean=float(s[4]),tanh_gate=tanh_g))
        print(f'  {set_name} s{seed} batch {j//batch} {len(rr)} rows {dt:.1f}s exact {sum(m["exact"] for m in res)}/{len(res)} elapsed {elapsed()/3600:.2f}h',flush=True)
    assert modelhash(base)==h0
    csv_write('results_per_problem.csv',res,'a')
    if trace_rows:csv_write('hidden_trace.csv',trace_rows,'a')
    s=dict(set_type=set_type,set_name=set_name,width=rows[0]['width'],seed=seed,n=len(rows),exact=sum(m['exact'] for m in res),exact_rate=sum(m['exact'] for m in res)/len(rows),
           bit_accuracy=statistics.mean(m['bit_accuracy'] for m in res),wall_seconds=time.perf_counter()-t_unit,weights_unchanged=True,finished=datetime.datetime.now().isoformat(timespec='seconds'))
    p['done'][key]=s;note(p,f'DONE {key} exact {s["exact"]}/{s["n"]} ({s["exact_rate"]:.4f}) wall {s["wall_seconds"]:.0f}s');return s

def register_set(p,name,rows,seed,extra=None):
    if name not in p['sets']:
        f=E/'data'/f'{name}.json'
        if f.exists():
            sha=sha_file(f);man=json.loads((E/'data/manifest.json').read_text())
            assert name in man and man[name]['sha256']==sha,'pre-generated set SHA mismatch'
        else:sha=save_set(name,rows)
        p['sets'][name]=dict(sha256=sha,n=len(rows),width=rows[0]['width'],seed=seed,fast_input_equal=check_fast_input(rows),registered=datetime.datetime.now().isoformat(timespec='seconds'),**(extra or {}))
        note(p,f'SET {name} n={len(rows)} sha={sha[:12]}')
    assert sha_file(E/'data'/f'{name}.json')==p['sets'][name]['sha256'] and p['sets'][name]['fast_input_equal']
    return load_set(name)

def width_cost(w,n):return EST['per_bit_per_problem_wall_seconds']*(w+1)*n

def run_set3(p,set_type,name,rows):
    """Evaluate all 3 seeds; return mean exact or None if budget would be exceeded before finishing."""
    out=[]
    for seed in SEEDS:
        key=f'{name}|s{seed}'
        if key not in p['done'] and elapsed()+width_cost(rows[0]['width'],len(rows))>BUDGET:
            note(p,f'BUDGET: {key} not started (elapsed {elapsed()/3600:.2f}h)');return None
        out.append(eval_unit(p,set_type,name,rows,seed)['exact_rate'])
    return statistics.mean(out),out

def main():
    setup();p=progress();p.setdefault('state',{});st=p['state']
    if 'main' not in st:
        note(p,f'START main evaluation engine={EST["engine"]} reduction={EST["reduction_needed"]} elapsed {elapsed()/3600:.2f}h')
        st['main']=dict(results={},unevaluated=[],W_p=None,W_f=None,complete=False);save_progress(p)
    m=st['main']
    if not m['complete']:
        for i,w in enumerate(CFG['widths']):
            if str(w) in m['results']:continue
            rows=register_set(p,f'main_{w}',load_set(f'main_{w}'),'V054 existing Test' if w==2048 else 80000+w)
            if EST['reduction_needed'] and w>=16384:rows=subset_half(rows)
            r=run_set3(p,'main',f'main_{w}',rows)
            if r is None:m['unevaluated']+=[x for x in CFG['widths'][i:]];m['budget_stop']=True;break
            mean,per=r;m['results'][str(w)]=dict(mean=mean,per_seed=per,n=len(rows),passed=mean>=0.95);save_progress(p)
            if mean>=0.95:m['W_p']=w
            else:m['W_f']=w;m['unevaluated']+=[x for x in CFG['widths'][i+1:]];note(p,f'FIRST FAIL at {w}: mean {mean:.4f}');break
        m['complete']=True;save_progress(p)
    # boundary search
    if 'boundary' not in st:st['boundary']=dict(midpoints=[],complete=False);save_progress(p)
    b=st['boundary']
    if m['W_f'] is not None and m['W_p'] is not None and not b['complete']:
        lo,hi=m['W_p'],m['W_f']
        for mp in b['midpoints']:
            if mp.get('passed') is True:lo=max(lo,mp['width'])
            elif mp.get('passed') is False:hi=min(hi,mp['width'])
        for k in range(len(b['midpoints'])+1,4):
            mid=int(round((lo+hi)/2/64))*64
            if mid<=lo or mid>=hi:note(p,f'BOUNDARY stop: midpoint {mid} not strictly between {lo},{hi}');break
            name=f'boundary_{k}_{mid}';rows=generate(mid,81000+k,name);verify_rows(rows);rows=register_set(p,name,rows,81000+k)
            if EST['reduction_needed'] and mid>=16384:rows=subset_half(rows)
            r=run_set3(p,'boundary',name,rows)
            if r is None:b['midpoints'].append(dict(k=k,width=mid,passed=None,note='budget'));break
            mean,per=r;b['midpoints'].append(dict(k=k,width=mid,lo=lo,hi=hi,mean=mean,per_seed=per,passed=mean>=0.95));save_progress(p)
            if mean>=0.95:lo=mid
            else:hi=mid
        b['complete']=True;save_progress(p)
    passing=[int(w) for w,v in m['results'].items() if v['passed']]+[x['width'] for x in b['midpoints'] if x.get('passed')]
    if m['W_f'] is None and not m.get('budget_stop'):W_b=CFG['widths'][-1]
    else:W_b=max(passing) if passing else None
    st['W_b']=W_b;save_progress(p)
    # confirmation
    if 'confirmation' not in st and W_b is not None:
        name=f'confirmation_{W_b}';rows=generate(W_b,82000+W_b,name);verify_rows(rows)
        prior=[n for n in p['sets'] if p['sets'][n]['width']==W_b and not n.startswith('confirmation')];prior_keys=set()
        for n in prior:prior_keys|={tuple(sorted((r['a'],r['b']))) for r in load_set(n)}
        overlap=[r['id'] for r in rows if tuple(sorted((r['a'],r['b']))) in prior_keys]
        rows=register_set(p,name,rows,82000+W_b,dict(overlap_with=prior,overlap_ids=overlap,overlap_all_edges=all(r['origin']=='V050gen_boundary' for r in rows if r['id'] in overlap)))
        if EST['reduction_needed'] and W_b>=16384:rows=subset_half(rows)
        r=run_set3(p,'confirmation',name,rows)
        st['confirmation']=dict(width=W_b,mean=r[0] if r else None,per_seed=r[1] if r else None,passed=(r[0]>=0.95) if r else None,n=len(rows),budget_stop=r is None);save_progress(p)
    # float64 replay (exploratory)
    if 'float64' not in st:
        res=[r for r in csv_read('results_per_problem.csv') if r['set_type']=='main' and r['exact']=='0'];out=[]
        order={f'main_{w}':i for i,w in enumerate(CFG['widths'])}
        for seed in SEEDS:
            fails=[r for r in res if int(r['seed'])==seed];pos={}
            for n in {r['set_name'] for r in fails}:pos.update({(n,x['id']):i for i,x in enumerate(load_set(n))})
            fails=sorted(fails,key=lambda r:(order[r['set_name']],pos[(r['set_name'],r['problem_id'])]))[:10]
            by={}
            for r in fails:by.setdefault(r['set_name'],[]).append(r)
            for name,rr in by.items():
                rows={x['id']:x for x in load_set(name)};sub=[rows[r['problem_id']] for r in rr]
                if elapsed()+width_cost(sub[0]['width'],len(sub))*4>BUDGET:note(p,f'BUDGET: float64 skipped {name} s{seed}');continue
                t=time.perf_counter();pp,_=infer(seed,sub,0,torch.float64);dt=time.perf_counter()-t
                for r32,row,pr in zip(rr,sub,pp):
                    m64=problem_metrics(row,pr);out.append(dict(seed=seed,set_name=name,width=row['width'],problem_id=row['id'],category=row['category'],first_error_float32=r32['first_error'],first_error_float64=m64['first_error'],
                                   exact_float32=0,exact_float64=m64['exact'],wrong_bits_float32=r32['wrong_bits'],wrong_bits_float64=m64['wrong_bits'],first_error_changed=str(m64['first_error'])!=r32['first_error'],prediction_float64_hex=m64['prediction_hex'],wall_seconds_batch=dt))
                print('F64',seed,name,len(sub),round(dt,1),flush=True)
        csv_write('float64_replay.csv',out);st['float64']=dict(n=len(out));save_progress(p)
    # probe (exploratory, only if time remains)
    if 'probe' not in st:
        cost=EST['per_bit_per_problem_wall_seconds']*65001*32*3*2.0
        if elapsed()+cost<BUDGET:
            name='probe_65000';rows=generate(65000,83000+65000,name,probe=True);verify_rows(rows);rows=register_set(p,name,rows,83000+65000)
            out=[eval_unit(p,'probe',name,rows,seed,batch=32,trace=False) for seed in SEEDS]
            pr=[r for r in csv_read('results_per_problem.csv') if r['set_type']=='probe'];csv_write('probe_65000.csv',pr)
            st['probe']=dict(done=True,per_seed=[o['exact_rate'] for o in out])
        else:st['probe']=dict(done=False,reason=f'insufficient time: elapsed {elapsed()/3600:.2f}h, estimated cost {cost/3600:.2f}h')
        save_progress(p)
    note(p,f'ALL STAGES FINISHED elapsed {elapsed()/3600:.2f}h')

if __name__=='__main__':main()
