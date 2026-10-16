import gzip,hashlib,importlib.util,itertools,json,platform,random,sys,time,unittest
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from core import Residual,encode,integer,bits,diagnostics,bucket
from data import prepare,candidates,phase,identity
from utils import *

@torch.no_grad()
def fit(m,data,split):
    stats={}
    for op in ('ADD','SUB'):
        rows=data[op+'_'+split];fullcorrect=corecorrect=bc=tokens=0;sl=0.
        for j in range(0,len(rows),128):
            rr=rows[j:j+128];x,y,ll=encode(rr,labels=True);z=m(x);p=z.argmax(-1)
            fullcorrect+=int((p==y).all(-1).sum());corecorrect+=int((p[:,:8]==y[:,:8]).all(-1).sum());bc+=int((p==y).sum());tokens+=y.numel()
            sl+=float(F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='sum'))
        stats[op]=dict(loss=sl/tokens,n=len(rows),correct=fullcorrect,full_exact=fullcorrect/len(rows),result_exact=(fullcorrect if op=='ADD' else corecorrect)/len(rows),bit_accuracy=bc/tokens)
    return sum(v['loss'] for v in stats.values())/2,stats

def record(r,dd,route,condition,budget,seed,cp_sha,source='new inference'):
    w=r['width'];diag=diagnostics(r,route);truth=diag.pop('raw_truth_digits');ph=diag.pop('phases');n=len(dd);resultn=w if r['op']=='SUB' else n
    core=dd[:resultn];result=integer(core);answer=r['a']+r['b'] if r['op']=='ADD' else r['a']-r['b']
    wrong=[i for i,(a,b) in enumerate(zip(core,truth[:resultn])) if a!=b];guard=dd[w] if n>w else None
    return dict(condition=condition,budget=budget,seed=seed,route=route,id=r['id'],op=r['op'],width=w,a=r['a'],b=r['b'],answer=answer,prediction=result,correct=int(result==answer),prediction_digits=dd,result_length=resultn,sequence_length=n,raw_prediction=integer(dd),raw_truth=integer(truth),guard_prediction=guard,guard_truth=truth[w] if n>w else None,guard_correct=int(guard==truth[w]) if n>w else None,wrong_positions=wrong,bit_correct=resultn-len(wrong),first_error=wrong[0] if wrong else None,first_error_phase=ph[wrong[0]] if wrong else 'correct',old_ADD_train_input_familiar=r.get('old_ADD_train_input_familiar',0) if route=='P' else None,phase_opportunities=dict(Counter(ph[:resultn])),phase_errors=dict(Counter(ph[i] for i in wrong)),attributes=r['attributes'],checkpoint_SHA256=cp_sha,source=source,**diag)

def references(data):
    manifest={name:hashval for hashval,name in (line.split('  ',1) for line in (WORK/'IronMath_V054/sha256_manifest.txt').read_text().splitlines())}
    registry={};stats=readcsv(WORK/'IronMath_V054/cpu_memory.csv')
    for budget in C['budgets']:
        for seed in C['seeds']:
            for kind in ('B0','B1'):
                rel=f'checkpoints/{budget}_{kind}_s{seed}/best_validation.pt';path=WORK/'IronMath_V054'/rel
                expected=manifest.get(str(Path(rel)),manifest.get(rel));assert sha(path)==expected
                cp=torch.load(path,map_location='cpu',weights_only=False);s=next(r for r in stats if r['stage']=='training' and r['condition']==budget and r['model']==kind and int(r['seed'])==seed)
                assert cp['update']==int(s['selected_update']) and cp['seed']==seed and cp['condition']==budget and cp['kind']==kind
                registry[f'{budget}_{kind}_s{seed}']=dict(path=str(path.relative_to(WORK)),sha256=expected,selected_update=cp['update'],model_state_SHA256=modelhash_state(cp['model']),old_inference=[r for r in stats if r['stage']=='inference' and r['condition']==budget and r['model']==kind and int(r['seed'])==seed])
    archive=WORK/'IronMath_V054/predictions.jsonl.gz';assert sha(archive)==manifest['predictions.jsonl.gz']
    dump('reference_manifest.json',dict(checkpoints=registry,archive_SHA256=sha(archive),archive_source='IronMath_V054/predictions.jsonl.gz',B0_third_feature='is_SUB, no Carry-in',Z2='not applicable',zero_shot_optimizer_updates=0))
    return registry

def modelhash_state(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.numpy().tobytes())
    return h.hexdigest()

def preflight(source,data,registry):
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(R/'tests')));assert result.wasSuccessful()
    train={identity(r) for op in ('ADD','SUB') for r in data[op+'_Train']};val={identity(r) for op in ('ADD','SUB') for r in data[op+'_Validation']};test={identity(r) for name,rr in data.items() if '_Test' in name for r in rr};assert not(train&val or train&test or val&test)
    spec=importlib.util.spec_from_file_location('V054_readonly_reference',WORK/'IronMath_V054/model.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy);comparisons=[]
    rows=source['ADD_Validation'][:16]+source['SUB_Validation'][:16]
    for key,meta in registry.items():
        kind='B1' if '_B1_' in key else 'B0';route='B1_native' if kind=='B1' else 'B0_native'
        cp=torch.load(WORK/meta['path'],map_location='cpu',weights_only=False);a=Residual();a.load_state_dict(cp['model']);b=legacy.Residual(kind);b.load_state_dict(cp['model'])
        x,y,ll=encode(rows,route,True);xx,yy,lll=legacy.batch(rows,kind,True);assert torch.equal(x,xx) and torch.equal(y,yy) and ll==lll
        with torch.no_grad():za=torch.jit.script(a)(x);zb=torch.jit.script(b)(xx)
        diff=float((za-zb).abs().max());assert diff<=1e-5 and torch.equal(za.argmax(-1),zb.argmax(-1));assert modelhash(a)==meta['model_state_SHA256']
        comparisons.append(dict(checkpoint=key,max_logit_difference=diff,exact_inputs_labels_lengths=True,weight_unchanged=True))
    checked=0
    for name,rr in data.items():
        for r in rr:
            assert 0<=r['a']<(1<<r['width']) and 0<=r['b']<(1<<r['width']) and (r['op']=='ADD' or r['a']>=r['b'])
            d=diagnostics(r);raw=d['raw_truth_digits'];assert integer(raw)&((1<<r['width'])-1)==(r['a']-r['b']) if r['op']=='SUB' else integer(raw)==r['a']+r['b'];checked+=1
    return dict(passed=True,unit_tests=result.testsRun,eligible_rows_checked=checked,derived_split_disjoint=True,native_compatibility=comparisons,parameters=1000334,Z2_not_applicable=True)

def main():
    if (R/'STARTED.json').exists():raise RuntimeError('already started; no automatic rerun')
    setup(0);dump('STARTED.json',dict(client_date='2026-10-15',registered_SHA={str(p.relative_to(R)):sha(p) for p in list(R.glob('*.py'))+list((R/'tests').glob('*.py'))+[R/'config.json',R/'SPEC_COMPLEMENT.md',R/'PREREGISTRATION.md']}))
    before=protected();dump('protected_sha_before.json',before);source,data,excluded=prepare();(R/'data').mkdir();manifest={}
    for name,rows in data.items():dump(f'data/{name}.json',rows);manifest[name]=dict(n=len(rows),sha256=sha(R/f'data/{name}.json'),original_n=len(source[name]),original_SHA256=sha(WORK/C['data_source']/f'{name}.json'))
    dump('dataset_manifest.json',manifest);csvout('split_exclusions.csv',excluded);registry=references(data);print('PRETRAIN QA',flush=True);dump('preflight.json',preflight(source,data,registry))
    dump('runtime.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),CPU_threads=2,device='CPU',memory_scope='process lifetime peak, not isolated per run',learning_CPU_scope='forward/backward/clip/AdamW only',date='2026-10-15'))
    resources=[];scores=[];logs=[];fits=[]
    with gzip.open(R/'predictions.jsonl.gz','wt',encoding='utf-8') as out,gzip.open(R/'failure_cases.jsonl.gz','wt',encoding='utf-8') as fail:
        def save_preds(pp):
            scores.append(score(pp))
            for p in pp:
                line=json.dumps(p)+'\n';out.write(line)
                if not p['correct']:fail.write(line)
        # Cached native baselines are independently re-scored under fixed guard rules.
        with gzip.open(WORK/'IronMath_V054/predictions.jsonl.gz','rt',encoding='utf-8') as f:
            selected=(p for p in (json.loads(line) for line in f) if p['model'] in ('B0','B1'))
            for (budget,seed,kind,w,op),group in itertools.groupby(selected,key=lambda p:(p['condition'],p['seed'],p['model'],p['width'],p['op'])):
                oldrows=list(group);lookup={r['id']:r for r in data[f'{op}_Test{w}']};assert len(oldrows)==512 and {p['id'] for p in oldrows}==set(lookup);key=f'{budget}_{kind}_s{seed}'
                condition=('Z0' if op=='ADD' else 'R_B0') if kind=='B0' else 'Z3';route='B0_native' if kind=='B0' else 'B1_native';pp=[]
                for p in oldrows:
                    r=lookup[p['id']];assert (p['a'],p['b'],p['answer'])==(r['a'],r['b'],r['answer']);assert p['correct']==int(p['prediction']==p['answer'])
                    q=record(r,p['prediction_digits'],route,condition,budget,seed,registry[key]['sha256'],'audited V054 cached native inference');q.update(native_V054_correct=p['correct'],native_V054_prediction=p['prediction']);pp.append(q)
                save_preds(pp);original=next(r for r in registry[key]['old_inference'] if int(r['width'])==w and r['op']==op)
                resources.append(dict(stage='cached_native_inference_reference',condition=condition,budget=budget,seed=seed,width=w,op=op,n=512,model_CPU_seconds=float(original['model_cpu_seconds']),model_wall_seconds=float(original['model_wall_seconds']),new_inference_CPU_seconds=0,peak_RAM_bytes=memory(),time_scope='archived V054 inference; not new V055 execution'))
        print('FIXED Z1 ZERO SHOT BEGIN',flush=True)
        for budget in C['budgets']:
            for seed in C['seeds']:
                meta=registry[f'{budget}_B0_s{seed}'];cp=torch.load(WORK/meta['path'],map_location='cpu',weights_only=False);base=Residual();base.load_state_dict(cp['model']);m=torch.jit.script(base).eval();initial=modelhash(m)
                for w in C['widths']:
                    print('Z1',budget,seed,w,flush=True);pp,res=evaluate(m,data[f'SUB_Test{w}'],'P','Z1',budget,seed,meta['sha256']);save_preds(pp);resources.append(res);csvout('zero_shot_results.csv',scores);csvout('cpu_memory.csv',resources)
                assert modelhash(m)==initial==meta['model_state_SHA256']
        dump('zero_shot_complete.json',dict(conditions=['Z0','Z1','Z3','R_B0'],Z2='not applicable: B0 feature is operation flag',optimizer_updates=0,weights_unchanged=True,adaptive_changes_to_F_conditions=False))
        print('ZERO SHOT DONE;18 PREREGISTERED F RUNS',flush=True)
        for budget in C['budgets']:
            for seed in C['seeds']:
                for condition in C['training_conditions']:
                    ll,res,ff=train(data,condition,budget,seed);logs+=ll;resources.append(res);fits+=ff
                    csvout('training_log.csv',logs);csvout('training_fits.csv',fits);csvout('cpu_memory.csv',resources)
        trains=[r for r in resources if r['stage']=='training'];assert len(trains)==18
        for budget in C['budgets']:
            for seed in C['seeds']:
                rr=[r for r in trains if r['budget']==budget and r['seed']==seed];assert len({r['initial_SHA256'] for r in rr})==1
                if budget=='C1':assert len({r['candidate_SHA256'] for r in rr})==1 and all(r['total_presentations']==19200 for r in rr)
        dump('training_complete.json',dict(runs=18,F_Test_used=False,zero_shot_did_not_change_training=True));print('ALL18 TRAINING DONE; F FIXED TEST',flush=True)
        for budget in C['budgets']:
            for seed in C['seeds']:
                for condition in C['training_conditions']:
                    path=R/f'checkpoints/{budget}_{condition}_s{seed}/best_validation.pt';cp=torch.load(path,map_location='cpu',weights_only=False);base=Residual();base.load_state_dict(cp['model']);m=torch.jit.script(base).eval();h=sha(path)
                    for w in C['widths']:
                        print('F EVAL',budget,condition,seed,w,flush=True)
                        for op in ('ADD','SUB'):
                            pp,res=evaluate(m,data[f'{op}_Test{w}'],'P',condition,budget,seed,h);save_preds(pp);resources.append(res)
                        csvout('finetuning_results.csv',[r for r in scores if r['condition'].startswith('F')]);csvout('cpu_memory.csv',resources)
    assert protected()==before
    for p,h in json.loads((R/'STARTED.json').read_text())['registered_SHA'].items():assert sha(R/p)==h
    dump('STOPPED.json',dict(status='fixed_learning_and_evaluation_completed_stopped',new_training_runs=18,extra_training=0,next='saved-data independent audit and report only'))
    print('ALL FIXED EVALUATION COMPLETE; STOPPED',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:dump('STOPPED.json',dict(status='failed_or_invalid_stopped',error=repr(e),extra_training=0));raise

@torch.no_grad()
def evaluate(m,rows,route,condition,budget,seed,cp_sha):
    pp=[];cpu=wall=encpu=enwall=0.
    for j in range(0,len(rows),128):
        rr=rows[j:j+128];c=time.process_time();w=time.perf_counter();x,_,ll=encode(rr,route,False);encpu+=time.process_time()-c;enwall+=time.perf_counter()-w
        c=time.process_time();w=time.perf_counter();z=m(x);ok=torch.isfinite(z).all(-1).numpy();p=z.argmax(-1).numpy();cpu+=time.process_time()-c;wall+=time.perf_counter()-w
        for r,q,v,n in zip(rr,p,ok,ll):pp.append(record(r,[int(d) if f else None for d,f in zip(q[:n],v[:n])],route,condition,budget,seed,cp_sha))
    res=dict(stage='inference',condition=condition,budget=budget,seed=seed,op=rows[0]['op'],width=rows[0]['width'],n=len(rows),model_CPU_seconds=cpu,model_wall_seconds=wall,encoding_CPU_seconds=encpu,encoding_wall_seconds=enwall,peak_RAM_bytes=memory())
    return pp,res

def score(pp):
    p=pp[0];n=len(pp);tokens=sum(q['result_length'] for q in pp)
    return dict(condition=p['condition'],budget=p['budget'],seed=p['seed'],width=p['width'],op=p['op'],n=n,correct=sum(q['correct'] for q in pp),exact_rate=sum(q['correct'] for q in pp)/n,bit_accuracy=sum(q['bit_correct'] for q in pp)/tokens,guard_accuracy=sum(q['guard_correct'] or 0 for q in pp)/n if p['sequence_length']>p['width'] else None,sequence_length=p['sequence_length'],checkpoint_SHA256=p['checkpoint_SHA256'],source=p['source'])

def train(data,condition,budget,seed):
    setup(seed);base=Residual();initial=modelhash(base);m=torch.jit.script(base);opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01)
    folder=R/f'checkpoints/{budget}_{condition}_s{seed}';folder.mkdir(parents=True)
    c=time.process_time();w=time.perf_counter()
    for _ in range(3):m.zero_grad();m(torch.zeros(16,9,3)).square().mean().backward()
    warmcpu=time.process_time()-c;warmwall=time.perf_counter()-w;assert modelhash(m)==initial
    rng=random.Random(50000+seed);order=hashlib.sha256();candidatehash=hashlib.sha256();step=0;best=float('inf');beststep=0;cpu=wall=valcpu=valwall=batchcpu=batchwall=savecpu=savewall=0.;add=sub=0;logs=[];boundary=None;boundaryCPU=boundaryPrevious=None;lastUpdateStart=0.
    def save(version,loss=None):
        nonlocal savecpu,savewall
        c=time.process_time();w=time.perf_counter()
        torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),condition=condition,budget=budget,seed=seed,update=step,validation_loss=loss,initial_SHA256=initial,order_SHA256=order.hexdigest(),candidate_SHA256=candidatehash.hexdigest(),train_CPU_seconds=cpu,ADD_presentations=add,SUB_presentations=sub,phase_boundary_update=boundary,phase_boundary_CPU_seconds=boundaryCPU,config=C,rng_state=rng.getstate(),torch_rng=torch.get_rng_state()),folder/f'{version}.pt')
        savecpu+=time.process_time()-c;savewall+=time.perf_counter()-w
    while (step<1200 if budget=='C1' else cpu<90):
        thisphase=phase(condition,budget,step,cpu)
        if condition=='F2' and thisphase=='mixed' and boundary is None:boundary=step;boundaryCPU=cpu;boundaryPrevious=lastUpdateStart;print('BOUNDARY',budget,condition,seed,step,cpu,flush=True)
        c=time.process_time();w=time.perf_counter();streams=candidates(data,rng)
        for ss in streams:
            for r in ss:candidatehash.update(f"{r['id']}:{r['a']}:{r['b']}\n".encode())
        rows=streams[0]+streams[2 if thisphase=='mixed' else 1]
        for r in rows:order.update(f"{r['id']}:{r['a']}:{r['b']}\n".encode());add+=r['op']=='ADD';sub+=r['op']=='SUB'
        x,y,_=encode(rows,labels=True);batchcpu+=time.process_time()-c;batchwall+=time.perf_counter()-w
        lastUpdateStart=cpu;c=time.process_time();w=time.perf_counter();m.train();opt.zero_grad(set_to_none=True);z=m(x);loss=F.cross_entropy(z.flatten(0,1),y.flatten());assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();cpu+=time.process_time()-c;wall+=time.perf_counter()-w;step+=1
        ending=step>=1200 if budget=='C1' else cpu>=90
        if step%200==0 or ending:
            c=time.process_time();w=time.perf_counter();m.eval();vl,stats=fit(m,data,'Validation');valcpu+=time.process_time()-c;valwall+=time.perf_counter()-w
            if vl<best:best=vl;beststep=step;save('best_validation',vl)
            logs.append(dict(condition=condition,budget=budget,seed=seed,update=step,phase=thisphase,ADD_presentations=add,SUB_presentations=sub,train_CPU_seconds=cpu,train_wall_seconds=wall,minibatch_loss=float(loss.detach()),validation_loss=vl,validation_ADD_exact=stats['ADD']['result_exact'],validation_SUB_exact=stats['SUB']['result_exact'],selected_update=beststep))
            print('TRAIN',budget,condition,seed,step,'CPU',round(cpu,3),'VAL',round(vl,6),flush=True);memory()
    save('final');cp=torch.load(folder/'best_validation.pt',map_location='cpu',weights_only=False);m.load_state_dict(cp['model']);m.eval();c=time.process_time();w=time.perf_counter();_,ts=fit(m,data,'Train');_,vs=fit(m,data,'Validation');fitcpu=time.process_time()-c;fitwall=time.perf_counter()-w
    resource=dict(stage='training',condition=condition,budget=budget,seed=seed,parameters=1000334,updates=step,ADD_presentations=add,SUB_presentations=sub,total_presentations=add+sub,training_bit_tokens=(add+sub)*9,phase_boundary_update=boundary,phase_boundary_CPU_seconds=boundaryCPU,phase_boundary_previous_update_start_CPU=boundaryPrevious,initial_SHA256=initial,order_SHA256=order.hexdigest(),candidate_SHA256=candidatehash.hexdigest(),selected_update=beststep,train_CPU_seconds=cpu,train_wall_seconds=wall,validation_CPU_seconds=valcpu,validation_wall_seconds=valwall,batch_CPU_seconds=batchcpu,batch_wall_seconds=batchwall,save_CPU_seconds=savecpu,save_wall_seconds=savewall,fit_CPU_seconds=fitcpu,fit_wall_seconds=fitwall,warmup_CPU_seconds=warmcpu,warmup_wall_seconds=warmwall,CPU_budget_overshoot=max(0,cpu-90) if budget=='C2' else 0,peak_RAM_bytes=memory())
    fits=[dict(condition=condition,budget=budget,seed=seed,split=sp,op=op,**stats[op]) for sp,stats in [('Train',ts),('Validation',vs)] for op in ('ADD','SUB')]
    return logs,resource,fits
