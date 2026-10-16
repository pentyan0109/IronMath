"""Stage A: engine equivalence vs V054 saved predictions (all 9 widths <= 2048, C1 B0 ADD, 3 seeds), 4096-bit
whole/chunked/fast agreement, per-problem wall timing, hidden-trace overhead, estimate."""
import gzip,json,time
import numpy as np
import torch
from common import E,WORK,SEEDS,setup,load_base,modelhash,dump,load,load_set,make_engine,run_base,run_chunked

WIDTHS_OLD=[8,16,32,64,128,256,512,1024,2048]

def saved_predictions():
    out={}
    with gzip.open(WORK/'IronMath_V054/predictions.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['condition']=='C1' and r['model']=='B0' and r['op']=='ADD':out[(r['seed'],r['width'],r['id'])]=r['prediction_digits']
    assert len(out)==13824;return out

@torch.no_grad()
def main():
    setup();saved=saved_predictions();data={w:json.loads((WORK/f'IronMath_V053/data/ADD_Test{w}.json').read_text()) for w in WIDTHS_OLD}
    result=dict(reproduction=[],chunk_4096=[],timing=[],trace_overhead=[],weights=[],torch_compile=None)
    for seed in SEEDS:
        base,_=load_base(seed);h0=modelhash(base);scripted=torch.jit.script(base).eval();fast=make_engine(base,1);chunk=make_engine(base,0)
        for name,fn in [('base',lambda rr:run_base(scripted,rr)[0]),('fast',lambda rr:run_chunked(fast,rr)[0])]:
            for w in WIDTHS_OLD:
                rows=data[w];mism=0;t=time.perf_counter()
                for j in range(0,len(rows),128):
                    rr=rows[j:j+128];pp=fn(rr)
                    for r,p in zip(rr,pp):
                        if [int(v) for v in p[:w+1]]!=saved[(seed,w,r['id'])]:mism+=1
                dt=time.perf_counter()-t
                result['reproduction'].append(dict(seed=seed,engine=name,width=w,n=len(rows),mismatched_problems=mism,all_match=mism==0,wall_seconds=dt,wall_per_problem=dt/len(rows)))
                print('REPRO',seed,name,w,mism,round(dt,2),flush=True)
        # 4096: whole-sequence base vs chunked (chunk 1000) vs fast (chunk 4096 -> 2 chunks)
        rows=load_set('main_4096')[:8];t=time.perf_counter();pb,_=run_base(scripted,rows);tb=time.perf_counter()-t
        t=time.perf_counter();pc,_,_=run_chunked(chunk,rows,chunk=1000);tc=time.perf_counter()-t
        t=time.perf_counter();pf,_,_=run_chunked(fast,rows);tf=time.perf_counter()-t
        result['chunk_4096'].append(dict(seed=seed,n=8,base_vs_chunked_equal=bool((pb==pc).all()),base_vs_fast_equal=bool((pb==pf).all()),base_wall=tb,chunked_wall=tc,fast_wall=tf))
        print('C4096',seed,(pb==pc).all(),(pb==pf).all(),flush=True)
        if seed==0:
            for w in (4096,8192):
                rows=load_set(f'main_{w}')[:8]
                for name,fn in [('base',lambda rr:run_base(scripted,rr)),('fast',lambda rr:run_chunked(fast,rr))]:
                    t=time.perf_counter();fn(rows);dt=time.perf_counter()-t
                    result['timing'].append(dict(seed=0,width=w,engine=name,batch=8,n=8,wall_seconds=dt,wall_per_problem=dt/8));print('TIME',w,name,round(dt/8,3),flush=True)
            rows=load_set('main_8192')[:8]
            for tn in (0,8,0,8):
                t=time.perf_counter();p,_,st=run_chunked(fast,rows,tn=tn);dt=time.perf_counter()-t
                result['trace_overhead'].append(dict(width=8192,n=8,trace_rows=tn,every=256,wall_seconds=dt,trace_points=len(st)));print('TRACE',tn,round(dt,2),flush=True)
        result['weights'].append(dict(seed=seed,sha_before=h0,sha_after=modelhash(base),unchanged=h0==modelhash(base)))
    result['torch_compile']='not attempted: TorchScript (torch.jit.script, as in V054) used for the loop; torch.compile on Windows requires a C++ compiler toolchain and was not pursued per instruction 3.2'
    rep=result['reproduction']
    result['adoption']={e:all(r['all_match'] for r in rep if r['engine']==e) and all(c['base_vs_fast_equal'] for c in result['chunk_4096']) if e=='fast' else all(r['all_match'] for r in rep if r['engine']==e) for e in ('base','fast')}
    result['chunked_adoption']=all(c['base_vs_chunked_equal'] for c in result['chunk_4096'])
    sp={e:sum(r['wall_seconds'] for r in rep if r['engine']==e) for e in ('base','fast')}
    result['measured_speed']=dict(total_reproduction_wall_seconds=sp,ratio_base_over_fast=sp['base']/sp['fast'],note='measured wall times only, concurrent V055 load present')
    dump('engine_equivalence.json',result)
    if not result['adoption']['base']:print('BASE ENGINE DOES NOT REPRODUCE V054 -> STOP');return
    # estimate
    eng='fast' if result['adoption']['fast'] and sp['fast']<sp['base'] else 'base'
    cfg=load('config.json');r2048=[r for r in rep if r['engine']==eng and r['width']==2048]
    per_bit=max(r['wall_per_problem'] for r in r2048)/2049
    main_s={w:per_bit*(w+1)*512*3 for w in cfg['widths']}
    total=sum(main_s.values())
    half={w:(per_bit*(w+1)*(sum(max(8,(n+1)//2) for n in [16,128,64,16,16,16,43,43,43,43,42,42]))*3 if w>=16384 else main_s[w]) for w in cfg['widths']}
    est=dict(engine=eng,per_bit_per_problem_wall_seconds=per_bit,basis='max over seeds of 2048-bit reproduction wall/problem (batch 128) / 2049 bits, adopted engine',
             main_seconds_by_width=main_s,main_total_hours=total/3600,limit_hours=cfg['main_estimate_limit_hours'],reduction_needed=total/3600>cfg['main_estimate_limit_hours'],
             halved_total_hours=sum(half.values())/3600,
             boundary_search_note='bisection between W_p and W_f: each midpoint <= W_f, 3 iterations x 3 seeds x 512 problems; upper bound = 3 x cost(W_f)',
             confirmation_note='one width W_b x 3 seeds x 512 problems; <= cost(W_f)',
             eight_problem_timings=result['timing'],
             trace_overhead=result['trace_overhead'])
    dump('estimate.json',est);print('ESTIMATE',json.dumps({k:v for k,v in est.items() if k in('engine','main_total_hours','reduction_needed','halved_total_hours')}),flush=True)

if __name__=='__main__':main()
