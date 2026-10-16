import csv,gzip,hashlib,json,platform,random,statistics,sys,time,unittest
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import Residual,batch,encode,decode,length,inputs,propagation,bucket,phases

R=Path(__file__).resolve().parent;WORK=R.parent;C=json.loads((R/'config.json').read_text())
def dump(name,x):
    (R/name).parent.mkdir(parents=True,exist_ok=True);(R/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def csvout(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (R/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:('NA' if v is None else v) for k,v in r.items()} for r in rows)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
def protected():
    return {str(p.relative_to(WORK)):sha(p) for folder in ('IronMath_V4/v050','IronMath_V4/v051','IronMath_V052','IronMath_V053','IronMath_Abacus') for p in (WORK/folder).rglob('*') if p.is_file()}
def setup(seed):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
def modelhash(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()
def memory():
    import ctypes
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('fault',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in ('peak','working','qp','q','qn','n','page','peakpage')]
    p=PMC();p.cb=ctypes.sizeof(p);kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=wintypes.HANDLE
    fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD];assert fn(kernel.GetCurrentProcess(),ctypes.byref(p),p.cb)
    if p.peak>C['max_peak_RAM_MiB']*2**20:raise MemoryError('registered RAM ceiling exceeded')
    return p.peak
def loss(z,y):
    return (F.cross_entropy(z[:8].flatten(0,1),y[:8].flatten(),ignore_index=-100)+F.cross_entropy(z[8:].flatten(0,1),y[8:].flatten(),ignore_index=-100))/2
@torch.no_grad()
def fit(m,data,k,split):
    stats={}
    for op in ('ADD','SUB'):
        rr=data[f'{op}_{split}'];nc=dc=tokens=0;sl=0.
        for j in range(0,len(rr),128):
            rows=rr[j:j+128];x,y,ll=batch(rows,k);z=m(x);p=z.argmax(-1);mask=y!=-100
            nc+=int(((p==y)|~mask).all(-1).sum());dc+=int(((p==y)&mask).sum());tokens+=int(mask.sum())
            sl+=float(F.cross_entropy(z.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum'))
        stats[op]=dict(loss=sl/tokens,correct=nc,n=len(rr),exact_rate=nc/len(rr),digit_accuracy=dc/tokens)
    return sum(v['loss'] for v in stats.values())/2,stats
def sample(data,rng,hashobj=None):
    rows=[]
    for op in ('ADD','SUB'):
        pool=data[f'{op}_Train']
        for _ in range(8):
            r=dict(pool[rng.randrange(len(pool))])
            if op=='ADD' and rng.randrange(2):r['a'],r['b']=r['b'],r['a']
            rows.append(r)
            if hashobj is not None:hashobj.update(f"{r['id']}:{r['a']}:{r['b']}\n".encode())
    return rows
def train(data,condition,k,seed):
    setup(seed);base=Residual(k);initial=modelhash(base);m=torch.jit.script(base);params=sum(p.numel() for p in m.parameters())
    opt=torch.optim.AdamW(m.parameters(),lr=C['learning_rate'],weight_decay=C['weight_decay'])
    prefix=f'{condition}_{k}_s{seed}';folder=R/'checkpoints'/prefix;folder.mkdir(parents=True)
    c=time.process_time();w=time.perf_counter()
    for _ in range(3):m.zero_grad();m(torch.zeros(16,9,3)).square().mean().backward()
    warmcpu=time.process_time()-c;warmwall=time.perf_counter()-w;assert modelhash(m)==initial
    rng=random.Random(50000+seed);oh=hashlib.sha256();cpu=wall=valcpu=valwall=savecpu=savewall=batchcpu=batchwall=0.;step=0;best=float('inf');best_step=0;logs=[];start=time.perf_counter();digits=0
    def save(kind,score=None):
        nonlocal savecpu,savewall
        c=time.process_time();w=time.perf_counter()
        torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),condition=condition,kind=k,seed=seed,update=step,validation_loss=score,initial_sha256=initial,problem_order_sha256=oh.hexdigest(),train_cpu_seconds=cpu,train_wall_seconds=wall,config=C,rng_state=rng.getstate(),torch_rng=torch.get_rng_state()),folder/f'{kind}.pt')
        savecpu+=time.process_time()-c;savewall+=time.perf_counter()-w
    while (step<1200 if condition=='C1' else cpu<90):
        c=time.process_time();w=time.perf_counter();rows=sample(data,rng,oh);x,y,ll=batch(rows,k);digits+=sum(ll);batchcpu+=time.process_time()-c;batchwall+=time.perf_counter()-w
        c=time.process_time();w=time.perf_counter();m.train();opt.zero_grad(set_to_none=True);z=m(x);l=loss(z,y);assert torch.isfinite(l)
        l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();cpu+=time.process_time()-c;wall+=time.perf_counter()-w;step+=1
        ending=step>=1200 if condition=='C1' else cpu>=90
        if step%200==0 or ending:
            c=time.process_time();w=time.perf_counter();m.eval();score,st=fit(m,data,k,'Validation');valcpu+=time.process_time()-c;valwall+=time.perf_counter()-w
            if score<best:best=score;best_step=step;save('best_validation',score)
            logs.append(dict(condition=condition,model=k,seed=seed,update=step,ADD_presentations=step*8,SUB_presentations=step*8,training_digits=digits,minibatch_loss=float(l.detach()),validation_loss=score,Validation_ADD_exact=st['ADD']['exact_rate'],Validation_SUB_exact=st['SUB']['exact_rate'],train_cpu_seconds=cpu,train_wall_seconds=wall,selected_update=best_step))
            print('TRAIN',prefix,step,'CPU',round(cpu,2),'VAL',round(score,5),flush=True);memory()
    save('final');cp=torch.load(folder/'best_validation.pt',weights_only=False,map_location='cpu');m.load_state_dict(cp['model']);m.eval()
    c=time.process_time();w=time.perf_counter();_,ts=fit(m,data,k,'Train');_,vs=fit(m,data,k,'Validation');fitcpu=time.process_time()-c;fitwall=time.perf_counter()-w
    resource=dict(stage='training',condition=condition,model=k,seed=seed,parameters=params,updates=step,ADD_presentations=step*8,SUB_presentations=step*8,training_digits=digits,selected_update=best_step,initial_sha256=initial,problem_order_sha256=oh.hexdigest(),train_cpu_seconds=cpu,train_wall_seconds=wall,validation_cpu_seconds=valcpu,validation_wall_seconds=valwall,batch_cpu_seconds=batchcpu,batch_wall_seconds=batchwall,save_cpu_seconds=savecpu,save_wall_seconds=savewall,warmup_cpu_seconds=warmcpu,warmup_wall_seconds=warmwall,fit_cpu_seconds=fitcpu,fit_wall_seconds=fitwall,total_wall_seconds=time.perf_counter()-start,CPU_budget_overshoot=max(0,cpu-90) if condition=='C2' else 0,peak_RAM_bytes=memory())
    fitting=[dict(condition=condition,model=k,seed=seed,split=sp,op=op,**st[op]) for sp,st in [('Train',ts),('Validation',vs)] for op in ('ADD','SUB')]
    return logs,resource,fitting
@torch.no_grad()
def evaluate(m,rows,k):
    outputs=[];cpu=wall=encpu=enwall=0.
    for j in range(0,len(rows),128):
        rr=rows[j:j+128];c=time.process_time();w=time.perf_counter();x,_,ll=batch(rr,k,labels=False);encpu+=time.process_time()-c;enwall+=time.perf_counter()-w
        c=time.process_time();w=time.perf_counter();z=m(x);finite=torch.isfinite(z).all(-1).numpy();pp=z.argmax(-1).numpy();cpu+=time.process_time()-c;wall+=time.perf_counter()-w
        for r,p,ok,n in zip(rr,pp,finite,ll):
            dd=[int(v)-(1 if k.startswith('T') else 0) if f else None for v,f in zip(p[:n],ok[:n])];answer=r['a']+r['b'] if r['op']=='ADD' else r['a']-r['b']
            truth=encode(answer,n,3 if k.startswith('T') else 2);wrong=[i for i,(a,b) in enumerate(zip(dd,truth)) if a!=b]
            _,events,longest=propagation(r,k);ph=phases(events);numeric=decode(dd,3 if k.startswith('T') else 2) if all(v is not None for v in dd) else None
            aa,bb,ex=inputs(r,k)
            outputs.append(dict(id=r['id'],a=r['a'],b=r['b'],op=r['op'],width=r['width'],length=n,answer=answer,prediction=numeric,correct=int(numeric==answer),prediction_digits=dd,wrong_positions=wrong,digit_correct=n-len(wrong),digit_accuracy=1-len(wrong)/n,first_error=wrong[0] if wrong else None,first_error_phase=ph[wrong[0]] if wrong else 'correct',phase_opportunities=dict(Counter(ph)),phase_errors=dict(Counter(ph[i] for i in wrong)),binary_propagation_length=r['longest_propagation'],binary_bucket=bucket(r['longest_propagation']),internal_propagation_length=longest,internal_bucket=bucket(longest),internal_incoming_counts=dict(Counter(str(a) for a,b in events)),input_pair_counts=dict(Counter(f'{a},{b}' for a,b in zip(aa,bb))),attributes=r['attributes'],category=r['category'],origin=r['origin'],numeric_error=numeric-answer if numeric is not None else None))
    return outputs,dict(stage='inference',model=k,width=rows[0]['width'],op=rows[0]['op'],n=len(rows),model_cpu_seconds=cpu,model_wall_seconds=wall,input_encoding_cpu_seconds=encpu,input_encoding_wall_seconds=enwall,peak_RAM_bytes=memory())
def load_data():
    return {p.stem:json.loads(p.read_text()) for p in sorted((WORK/C['data_source']).glob('*.json'))}
def key(r):
    a,b=sorted((r['a'],r['b'])) if r['op']=='ADD' else (r['a'],r['b'])
    return r['width'],r['op'],a,b
def preflight(data):
    suite=unittest.defaultTestLoader.discover(str(R/'tests'));result=unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful()
    seen=set();count=0;cats=[]
    for name,rr in data.items():
        if '_Test' in name:assert len(rr)==512
        for r in rr:
            assert key(r) not in seen;seen.add(key(r));assert r['a']>=0 and r['b']>=0 and (r['op']=='ADD' or r['a']>=r['b'])
            for k in C['models']:
                dd,ev,n=propagation(r,k);assert dd==encode(r['a']+r['b'] if r['op']=='ADD' else r['a']-r['b'],length(r,k),3 if k.startswith('T') else 2)
            count+=1
        for cat,n in Counter(a for r in rr for a in r['attributes']).items():cats.append(dict(dataset=name,category=cat,n=n,overlapping=True))
        for label,n in Counter(bucket(r['longest_propagation']) for r in rr).items():cats.append(dict(dataset=name,category='binary_'+label,n=n,overlapping=True))
    csvout('category_counts.csv',cats)
    table=[]
    for a in (-1,0,1):
        for b in (-1,0,1):
            for c in (-1,0,1):
                s=a+b+c;d=(s+1)%3-1;carry=(s-d)//3;assert s==d+3*carry and d in (-1,0,1) and carry in (-1,0,1)
                table.append(dict(a=a,b=b,carry_in=c,sum=s,output=d,carry_out=carry))
    csvout('ternary_carry_table.csv',table)
    return dict(passed=True,unit_tests=result.testsRun,all_dataset_rows_checked=count,carry_transitions=27,split_leakage=False)
def main():
    if (R/'STARTED.json').exists():raise RuntimeError('already started; no restart')
    setup(0);dump('STARTED.json',dict(client_date='2026-10-15',registered_SHA={str(p.relative_to(R)):sha(p) for p in list(R.glob('*.py'))+list((R/'tests').glob('*.py'))+[R/'config.json',R/'SPEC_REPRESENTATION.md',R/'PREREGISTRATION.md']}))
    before=protected();dump('protected_sha_before.json',before);data=load_data();dump('dataset_manifest.json',{name:dict(n=len(rr),source=str((WORK/C['data_source']/f'{name}.json').relative_to(WORK)),sha256=sha(WORK/C['data_source']/f'{name}.json')) for name,rr in data.items()})
    print('PRETRAIN QA',flush=True);dump('preflight.json',preflight(data));dump('runtime.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),threads=2,device='CPU',date='2026-10-15',memory_scope='same process lifetime peak, not isolated run peak',training_CPU_scope='forward/backward/clip/AdamW only; validation/save/batch/warmup excluded',inference_CPU_scope='forward finite checks and argmax; encoding/serialization excluded'))
    logs=[];resources=[];fits=[]
    for condition in C['conditions']:
        for seed in C['seeds']:
            for k in C['models']:
                print('START',condition,k,seed,flush=True);ll,rr,ff=train(data,condition,k,seed);logs+=ll;resources.append(rr);fits+=ff
                csvout('training_log.csv',logs);csvout('training_results.csv',fits);csvout('cpu_memory.csv',resources)
    for cond in C['conditions']:
        for seed in C['seeds']:
            rs=[r for r in resources if r['condition']==cond and r['seed']==seed]
            assert rs[0]['initial_sha256']==rs[1]['initial_sha256'] and rs[2]['initial_sha256']==rs[3]['initial_sha256']
            if cond=='C1':assert len({r['problem_order_sha256'] for r in rs})==1
    dump('training_complete.json',dict(runs=24,Test_used=False,additional_training=0));print('ALL24 TRAINING COMPLETE; FIXED TEST BEGIN',flush=True)
    scores=[];carry=[]
    with gzip.open(R/'predictions.jsonl.gz','wt',encoding='utf-8') as out,gzip.open(R/'failure_cases.jsonl.gz','wt',encoding='utf-8') as fail:
        for cond in C['conditions']:
            for seed in C['seeds']:
                for k in C['models']:
                    cp=torch.load(R/f'checkpoints/{cond}_{k}_s{seed}/best_validation.pt',weights_only=False,map_location='cpu');base=Residual(k);base.load_state_dict(cp['model']);m=torch.jit.script(base).eval()
                    for w in C['widths']:
                        print('EVAL',cond,k,seed,w,flush=True)
                        for op in ('ADD','SUB'):
                            preds,res=evaluate(m,data[f'{op}_Test{w}'],k);res.update(condition=cond,seed=seed);resources.append(res)
                            for p in preds:
                                p.update(condition=cond,model=k,seed=seed);out.write(json.dumps(p)+'\n')
                                if not p['correct']:fail.write(json.dumps(p)+'\n')
                            scores.append(dict(condition=cond,model=k,seed=seed,width=w,op=op,n=len(preds),correct=sum(p['correct'] for p in preds),exact_rate=sum(p['correct'] for p in preds)/len(preds),digit_accuracy=sum(p['digit_correct'] for p in preds)/sum(p['length'] for p in preds),sequence_length=preds[0]['length'],model_cpu_seconds=res['model_cpu_seconds'],model_wall_seconds=res['model_wall_seconds']))
                            for definition,field in [('binary','binary_bucket'),('internal','internal_bucket')]:
                                for cat in ('0','1','2-3','4-7','8-15','16+'):
                                    ps=[p for p in preds if p[field]==cat];carry.append(dict(condition=cond,model=k,seed=seed,width=w,op=op,definition=definition,category=cat,n=len(ps),correct=sum(p['correct'] for p in ps),exact_rate=sum(p['correct'] for p in ps)/len(ps) if ps else None))
                        csvout('evaluation_results.csv',scores);csvout('carry_borrow_results.csv',carry);csvout('cpu_memory.csv',resources)
    aggregate=[];groups=defaultdict(list)
    for r in scores:groups[(r['condition'],r['model'],r['width'],r['op'])].append(r)
    for k,rs in groups.items():
        aggregate.append(dict(condition=k[0],model=k[1],width=k[2],op=k[3],seeds=len(rs),exact_mean=statistics.mean(r['exact_rate'] for r in rs),exact_sample_SD=statistics.stdev(r['exact_rate'] for r in rs),digit_accuracy_mean=statistics.mean(r['digit_accuracy'] for r in rs),sequence_length=rs[0]['sequence_length']))
    csvout('length_generalization.csv',aggregate);assert protected()==before
    for name,h in json.loads((R/'STARTED.json').read_text())['registered_SHA'].items():assert sha(R/name)==h
    dump('STOPPED.json',dict(status='all_training_and_fixed_evaluation_completed_stopped',runs=24,additional_training=0,next='independent saved-data audit and report only'))
    print('ALL24 RUNS AND9WIDTHS COMPLETE; STOPPED',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:dump('STOPPED.json',dict(status='failed_or_invalid_stopped',error=repr(e),additional_training=0));raise
