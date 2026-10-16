import csv,gzip,hashlib,json,math,platform,random,statistics,sys,time,unittest
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import Residual,NativeReference,batch,bits,annotate
from dataset import generate,key

R=Path(__file__).resolve().parent;WORK=R.parent;C=json.loads((R/'config.json').read_text())
def dump(name,x):
    (R/name).parent.mkdir(parents=True,exist_ok=True);(R/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def csvout(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (R/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:('NA' if v is None else v) for k,v in r.items()} for r in rows)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
def protected():
    return {str(p.relative_to(WORK)):sha(p) for folder in ('IronMath_V4/v050','IronMath_V4/v051','IronMath_V052','IronMath_Abacus') for p in (WORK/folder).rglob('*') if p.is_file()}
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
    if p.peak>2048*2**20:raise MemoryError('registered RAM ceiling exceeded; stop')
    return p.peak
def loss(z,y,rows):
    return sum(F.cross_entropy(z[[i for i,r in enumerate(rows) if r['op']==op]].flatten(0,1),y[[i for i,r in enumerate(rows) if r['op']==op]].flatten(),ignore_index=-100) for op in ('ADD','SUB'))/2
@torch.no_grad()
def fit(m,data,direction,split):
    total=0.;opstats={}
    for op in ('ADD','SUB'):
        rows=data[f'{op}_{split}'];ncorrect=bc=tokens=0;total_loss=0.
        for j in range(0,len(rows),128):
            rr=rows[j:j+128];x,y,ll=batch(rr,direction);z=m(x);pred=z.argmax(-1);mask=y!=-100
            ncorrect+=int(((pred==y)|~mask).all(-1).sum());bc+=int(((pred==y)&mask).sum());tokens+=int(mask.sum())
            total_loss+=float(F.cross_entropy(z.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum'))
        opstats[op]=dict(loss=total_loss/tokens,exact=ncorrect/len(rows),correct=ncorrect,n=len(rows),bit_accuracy=bc/tokens)
    return sum(v['loss'] for v in opstats.values())/2,opstats

def train(data,condition,direction,seed):
    setup(seed);base=Residual();initial=modelhash(base);m=torch.jit.script(base);opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01)
    prefix=f'{condition}_{direction}_s{seed}';folder=R/'checkpoints'/prefix;folder.mkdir(parents=True)
    # No optimizer update: specialize the common execution graph before timing.
    w=time.perf_counter();c=time.process_time()
    for _ in range(3):m.zero_grad();m(torch.zeros(16,9,3)).square().mean().backward()
    warm_cpu=time.process_time()-c;warm_wall=time.perf_counter()-w;assert modelhash(m)==initial
    rng=random.Random(50000+seed);orderhash=hashlib.sha256();cpu=wall=valcpu=valwall=savecpu=savewall=batchcpu=batchwall=0.;step=0;best=float('inf');best_step=0;logs=[];started=time.perf_counter()
    def save(kind,score=None):
        nonlocal savecpu,savewall
        c=time.process_time();w=time.perf_counter()
        torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),condition=condition,direction=direction,seed=seed,update=step,validation_loss=score,initial_sha256=initial,problem_order_sha256=orderhash.hexdigest(),train_cpu_seconds=cpu,train_wall_seconds=wall,config=C),folder/f'{kind}.pt')
        savecpu+=time.process_time()-c;savewall+=time.perf_counter()-w
    while (step<1200 if condition=='C1' else cpu<90):
        c=time.process_time();w=time.perf_counter();rows=[]
        for op in ('ADD','SUB'):
            pool=data[f'{op}_Train']
            for _ in range(8):
                r=dict(pool[rng.randrange(len(pool))])
                if op=='ADD' and rng.randrange(2):r['a'],r['b']=r['b'],r['a']
                rows.append(r);orderhash.update(f"{r['id']}:{r['a']}:{r['b']}\n".encode())
        x,y,ll=batch(rows,direction);batchcpu+=time.process_time()-c;batchwall+=time.perf_counter()-w
        c=time.process_time();w=time.perf_counter();m.train();opt.zero_grad(set_to_none=True);z=m(x);l=loss(z,y,rows);assert torch.isfinite(l)
        l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();cpu+=time.process_time()-c;wall+=time.perf_counter()-w;step+=1
        ending=step>=1200 if condition=='C1' else cpu>=90
        if step%200==0 or ending:
            c=time.process_time();w=time.perf_counter();m.eval();score,stats=fit(m,data,direction,'Validation');valcpu+=time.process_time()-c;valwall+=time.perf_counter()-w
            if score<best:best=score;best_step=step;save('best_validation',score)
            logs.append(dict(condition=condition,direction=direction,seed=seed,update=step,problem_presentations=step*16,training_bit_tokens=step*136,train_minibatch_loss=float(l.detach()),validation_loss=score,Validation_ADD_exact=stats['ADD']['exact'],Validation_SUB_exact=stats['SUB']['exact'],train_cpu_seconds=cpu,train_wall_seconds=wall,selected_update=best_step))
            print('TRAIN',prefix,step,'CPU',round(cpu,2),'Val',round(score,5),flush=True);memory()
    save('final');cp=torch.load(folder/'best_validation.pt',weights_only=False,map_location='cpu');m.load_state_dict(cp['model']);m.eval()
    c=time.process_time();w=time.perf_counter();_,trainstats=fit(m,data,direction,'Train');_,valstats=fit(m,data,direction,'Validation');fitcpu=time.process_time()-c;fitwall=time.perf_counter()-w
    resource=dict(stage='training',condition=condition,direction=direction,seed=seed,parameters=1000334,updates=step,problem_presentations=step*16,training_bit_tokens=step*136,selected_update=best_step,initial_sha256=initial,problem_order_sha256=orderhash.hexdigest(),train_cpu_seconds=cpu,train_wall_seconds=wall,batch_cpu_seconds=batchcpu,batch_wall_seconds=batchwall,validation_cpu_seconds=valcpu,validation_wall_seconds=valwall,save_cpu_seconds=savecpu,save_wall_seconds=savewall,warmup_cpu_seconds=warm_cpu,warmup_wall_seconds=warm_wall,fit_cpu_seconds=fitcpu,fit_wall_seconds=fitwall,total_wall_seconds=time.perf_counter()-started,CPU_budget_overshoot=max(0,cpu-90) if condition=='C2' else 0,peak_RAM_bytes=memory())
    fitting=[dict(condition=condition,direction=direction,seed=seed,split=split,op=op,**stats[op]) for split,stats in [('Train',trainstats),('Validation',valstats)] for op in ('ADD','SUB')]
    return logs,resource,fitting

@torch.no_grad()
def evaluate(m,rows,direction):
    outputs=[];cpu=wall=0.;encoding_cpu=encoding_wall=0.
    for j in range(0,len(rows),128):
        rr=rows[j:j+128];c=time.process_time();w=time.perf_counter();x,y,ll=batch(rr,direction);encoding_cpu+=time.process_time()-c;encoding_wall+=time.perf_counter()-w
        c=time.process_time();w=time.perf_counter();z=m(x);finite=torch.isfinite(z).all(-1).numpy();pp=z.argmax(-1).numpy();cpu+=time.process_time()-c;wall+=time.perf_counter()-w
        for r,p,valid,n in zip(rr,pp,finite,ll):
            bb=[int(v) if f else None for v,f in zip(p[:n],valid[:n])];physical=bb if direction=='R' else bb[::-1];truth=bits(r['answer'],n)
            wrong=[i for i,(a,b) in enumerate(zip(physical,truth)) if a!=b];processing=wrong if direction=='R' else [n-1-i for i in wrong]
            incoming=[];outgoing=[];state=0
            for i in range(n):
                incoming.append(state);a=(r['a']>>i)&1;b=(r['b']>>i)&1
                state=((a&b)|((a^b)&state)) if r['op']=='ADD' else (((1^a)&b)|((1^(a^b))&state))
                outgoing.append(state)
            phases=['start' if not a and b else 'continue' if a and b else 'release' if a and not b else 'none' for a,b in zip(incoming,outgoing)]
            firstproc=min(processing) if processing else None;firstphysical=(firstproc if direction=='R' else n-1-firstproc) if firstproc is not None else None
            head=sum(i*3<n for i in processing);middle=sum(n<=i*3<2*n for i in processing);tail=len(processing)-head-middle
            outputs.append(dict(id=r['id'],a=r['a'],b=r['b'],op=r['op'],width=r['width'],length=n,answer=r['answer'],prediction=sum(int(b)<<i for i,b in enumerate(physical)) if all(b is not None for b in physical) else None,numeric_valid=int(all(b is not None for b in physical)),nonfinite_positions=[i for i,b in enumerate(physical) if b is None],prediction_bits=physical,correct=int(not wrong),wrong_positions=wrong,wrong_bits=len(wrong),bit_accuracy=1-len(wrong)/n,first_error_lsb=min(wrong) if wrong else None,first_error_processing=firstproc,first_processing_error_physical=firstphysical,first_error_phase=phases[firstphysical] if firstphysical is not None else 'correct',phase_opportunities=dict(Counter(phases)),phase_errors=dict(Counter(phases[i] for i in wrong)),head_errors=head,middle_errors=middle,tail_errors=tail,errors_at_incoming_propagation=sum(incoming[i] for i in wrong),incoming_propagation_positions=sum(incoming),longest_propagation=r['longest_propagation'],propagation_bucket=r['propagation_bucket'],attributes=r['attributes'],origin=r['origin']))
    return outputs,dict(stage='inference',direction=direction,width=rows[0]['width'],op=rows[0]['op'],n=len(rows),model_cpu_seconds=cpu,model_wall_seconds=wall,input_encoding_cpu_seconds=encoding_cpu,input_encoding_wall_seconds=encoding_wall,peak_RAM_bytes=memory())

def statistics_pair(rs,ls):
    assert [r['id'] for r in rs]==[r['id'] for r in ls]
    n=len(rs);classes=Counter();same_wrong=missing_wrong=0;overlaps=[];wrong_overlaps=[];agreements=agree_correct=0;pair_rows=[]
    for a,b in zip(rs,ls):
        label='Both Correct' if a['correct'] and b['correct'] else 'R Only' if a['correct'] else 'L Only' if b['correct'] else 'Both Wrong';classes[label]+=1
        missing=a['prediction'] is None or b['prediction'] is None;same=not missing and a['prediction']==b['prediction'];agreements+=same;agree_correct+=same and a['correct']
        same_wrong+=label=='Both Wrong' and same;missing_wrong+=label=='Both Wrong' and missing;ar=set(a['wrong_positions']);br=set(b['wrong_positions']);union=ar|br;overlap=len(ar&br)/len(union) if union and not missing else None
        if overlap is not None:overlaps.append(overlap)
        if label=='Both Wrong' and overlap is not None:wrong_overlaps.append(overlap)
        pair_rows.append(dict(id=a['id'],class_label=label,R_correct=a['correct'],L_correct=b['correct'],R_prediction=a['prediction'],L_prediction=b['prediction'],answers_agree=int(same),error_bit_overlap=overlap))
    a,b,c,d=[classes[k] for k in ('Both Correct','R Only','L Only','Both Wrong')];den=math.sqrt((a+b)*(c+d)*(a+c)*(b+d));phi=(a*d-b*c)/den if den else None
    return dict(n=n,both_correct=a,R_only=b,L_only=c,both_wrong=d,both_correct_rate=a/n,R_only_rate=b/n,L_only_rate=c/n,both_wrong_rate=d/n,same_wrong=same_wrong,different_wrong=d-same_wrong-missing_wrong,both_wrong_missing=missing_wrong,same_wrong_rate=same_wrong/n,different_wrong_rate=(d-same_wrong-missing_wrong)/n,same_wrong_given_both_wrong=same_wrong/d if d else None,different_wrong_given_both_wrong=(d-same_wrong-missing_wrong)/d if d else None,correctness_phi=phi,error_overlap_mean=statistics.mean(overlaps) if overlaps else None,error_overlap_both_wrong_mean=statistics.mean(wrong_overlaps) if wrong_overlaps else None,oracle_selection_upper_bound=1-d/n),dict(n=n,agreement_count=agreements,agreement_rate=agreements/n,agreement_correct=agree_correct,agreement_accuracy=agree_correct/agreements if agreements else None,agreed_wrong=agreements-agree_correct,disagreement_count=n-agreements,disagreement_rate=1-agreements/n,abstention_rate=1-agreements/n),pair_rows

def baseline_reference(data):
    wanted={r['id']:r for name,rr in data.items() if '_Test' in name for r in rr};out=[];seen=set()
    with (WORK/'IronMath_V4/v051/predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['model']!='A1_residual' or r['version']!='1200' or r['problem_id'] not in wanted:continue
            p=wanted[r['problem_id']];seed=int(r['seed']);identity=(seed,p['id']);assert identity not in seen;seen.add(identity)
            assert (int(r['a']),int(r['b']),int(r['width']),r['op'])==(p['a'],p['b'],p['width'],p['op'])
            prediction=int(r['prediction']);bb=json.loads(r['output_bits']);assert sum(int(x)<<i for i,x in enumerate(bb))==prediction and len(bb)==p['length']
            assert int(r['expected'])==p['answer']
            out.append(dict(model='V051_archived_native',seed=seed,width=p['width'],op=p['op'],id=p['id'],a=p['a'],b=p['b'],answer=p['answer'],prediction=prediction,correct=int(prediction==p['answer']),source='exact archived native prediction; no new inference'))
    return out

def preflight(data):
    suite=unittest.defaultTestLoader.discover(str(R/'tests'));test=unittest.TextTestRunner(verbosity=2).run(suite);assert test.wasSuccessful()
    seen=set()
    for name,rows in data.items():
        if '_Test' in name:assert len(rows)==512
        for r in rows:
            assert key(r) not in seen;seen.add(key(r));assert r['a']>=0 and r['b']>=0 and (r['op']=='ADD' or r['a']>=r['b'])
            annotate(r)
    errors=[]
    for seed in C['seeds']:
        old=torch.load(WORK/f'IronMath_V4/v051/checkpoints/A1_residual_s{seed}_1200.pt',weights_only=True,map_location='cpu');m=Residual();n=NativeReference();m.load_state_dict(old['model']);n.load_state_dict(old['model'])
        sm=torch.jit.script(m);rows=data['ADD_Validation'][:16]+data['SUB_Validation'][:16];x,y,ll=batch(rows,'R')
        with torch.no_grad():a=sm(x);b=n(x)
        difference=float((a-b).abs().max());assert difference<=1e-4 and torch.equal(a.argmax(-1),b.argmax(-1));errors.append(dict(seed=seed,native_max_abs_difference=difference,labels_identical=True))
    # all-correct vectors make phi and empty-union overlap explicitly NA
    toy=dict(id='x',correct=1,prediction=0,wrong_positions=[]);stats,agreement,_=statistics_pair([toy],[toy]);assert stats['correctness_phi'] is None and stats['error_overlap_mean'] is None
    return dict(passed=True,unit_tests=test.testsRun,all_dataset_rows_checked=len(seen),parameters=1000334,native_checkpoint_equivalence=errors,MSB_future_carry_constraint_demonstrated=True,zero_denominator_NA=True)

def main():
    if (R/'STARTED.json').exists():raise RuntimeError('already started; no automatic restart')
    dump('STARTED.json',dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),registered_SHA={name:sha(R/name) for name in ('config.json','SPEC_DIRECTION.md','PREREGISTRATION.md','model.py','dataset.py','run.py','tests/test_direction.py')}))
    setup(0);before=protected();dump('protected_sha_before.json',before);c=time.process_time();w=time.perf_counter();data,meta=generate(WORK,C);(R/'data').mkdir();manifest={};cats=[]
    for name,rows in data.items():
        dump('data/'+name+'.json',rows);manifest[name]=dict(n=len(rows),sha256=sha(R/'data'/f'{name}.json'))
        for cat,count in Counter(a for r in rows for a in r['attributes']).items():cats.append(dict(dataset=name,category=cat,n=count,definition='overlapping attributes'))
    prepare=dict(stage='data_generation_and_save',cpu_seconds=time.process_time()-c,wall_seconds=time.perf_counter()-w,peak_RAM_bytes=memory())
    dump('dataset_manifest.json',dict(data=manifest,inherited_counts=meta));csvout('category_counts.csv',cats);print('DATA FIXED; QA',flush=True)
    qa=preflight(data);dump('preflight.json',qa);dump('runtime.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),threads=2,device='CPU',memory_scope='same process lifetime peak, not isolated run peak',training_CPU_scope='forward/backward/gradient clip/AdamW only; graph warming,batch,data,val,save separate',inference_scope='model argmax incl; bit packing input separately timed',streaming_implementation_difference='per-step projection/head, common scripted kernel; mathematical equivalence with FP32 tolerance verified'))
    print('QA PASS;12 MAIN RUNS START',flush=True);logs=[];resources=[prepare];fits=[];training=[]
    for condition in C['conditions']:
        for seed in C['seeds']:
            for direction in C['directions']:
                print('START',condition,direction,seed,flush=True);ll,rr,ff=train(data,condition,direction,seed);logs+=ll;resources.append(rr);fits+=ff;training.append(rr)
                csvout('training_log.csv',logs);csvout('training_results.csv',fits);csvout('cpu_memory.csv',resources)
    for condition in C['conditions']:
        for seed in C['seeds']:
            rr=[r for r in training if r['condition']==condition and r['seed']==seed];assert rr[0]['initial_sha256']==rr[1]['initial_sha256']
            if condition=='C1':assert rr[0]['problem_order_sha256']==rr[1]['problem_order_sha256'] and rr[0]['updates']==rr[1]['updates']==1200
    dump('training_complete.json',dict(runs=12,Test_used=False,additional_training=0));print('ALL TRAINING DONE; ALL9 WIDTHS BEGIN',flush=True)
    scores=[];carry=[];paired=[];correlations=[];agreements=[]
    with gzip.open(R/'predictions.jsonl.gz','wt',encoding='utf-8') as out,gzip.open(R/'failure_cases.jsonl.gz','wt',encoding='utf-8') as fail,gzip.open(R/'paired_predictions.jsonl.gz','wt',encoding='utf-8') as pairs:
        for condition in C['conditions']:
            for seed in C['seeds']:
                models={}
                for direction in C['directions']:
                    cp=torch.load(R/f'checkpoints/{condition}_{direction}_s{seed}/best_validation.pt',weights_only=False,map_location='cpu');m=Residual();m.load_state_dict(cp['model']);models[direction]=torch.jit.script(m).eval()
                for width in C['widths']:
                    print('EVALUATE',condition,seed,width,flush=True)
                    for op in ('ADD','SUB'):
                        rows=data[f'{op}_Test{width}'];direction_out={}
                        for direction,m in models.items():
                            preds,res=evaluate(m,rows,direction);direction_out[direction]=preds;res.update(condition=condition,seed=seed);resources.append(res)
                            for p in preds:
                                p.update(condition=condition,direction=direction,seed=seed);out.write(json.dumps(p)+'\n')
                                if not p['correct']:fail.write(json.dumps(p)+'\n')
                            scores.append(dict(condition=condition,direction=direction,seed=seed,width=width,op=op,n=512,correct=sum(p['correct'] for p in preds),exact_rate=sum(p['correct'] for p in preds)/512,bit_accuracy=statistics.mean(p['bit_accuracy'] for p in preds),head_errors=sum(p['head_errors'] for p in preds),middle_errors=sum(p['middle_errors'] for p in preds),tail_errors=sum(p['tail_errors'] for p in preds),errors_at_incoming_propagation=sum(p['errors_at_incoming_propagation'] for p in preds),incoming_propagation_positions=sum(p['incoming_propagation_positions'] for p in preds),**{k:res[k] for k in ('model_cpu_seconds','model_wall_seconds')}))
                            for category in sorted({a for p in preds for a in p['attributes']}|{p['propagation_bucket'] for p in preds}):
                                ps=[p for p in preds if category in p['attributes'] or category==p['propagation_bucket']];carry.append(dict(condition=condition,direction=direction,seed=seed,width=width,op=op,category=category,n=len(ps),correct=sum(p['correct'] for p in ps),exact_rate=sum(p['correct'] for p in ps)/len(ps)))
                        stat,agreement,pp=statistics_pair(direction_out['R'],direction_out['L']);identity=dict(condition=condition,seed=seed,width=width,op=op)
                        correlations.append(dict(**identity,**stat));agreements.append(dict(**identity,**agreement))
                        paired.append(dict(**identity,R_exact=sum(p['correct'] for p in direction_out['R'])/512,L_exact=sum(p['correct'] for p in direction_out['L'])/512,R_minus_L=(sum(p['correct'] for p in direction_out['R'])-sum(p['correct'] for p in direction_out['L']))/512))
                        for p in pp:pairs.write(json.dumps(dict(**identity,**p))+'\n')
                    csvout('evaluation_results.csv',scores);csvout('carry_borrow_results.csv',carry);csvout('paired_seed_comparison.csv',paired);csvout('error_correlation.csv',correlations);csvout('agreement_analysis.csv',agreements);csvout('cpu_memory.csv',resources)
    refs=baseline_reference(data);csvout('V051_common_reference_predictions.csv',refs);groups=defaultdict(list)
    for r in refs:groups[(r['seed'],r['width'],r['op'])].append(r)
    refrows=[dict(seed=k[0],width=k[1],op=k[2],common_subset_n=len(rs),V053_total_n=512,correct=sum(r['correct'] for r in rs),exact_rate=sum(r['correct'] for r in rs)/len(rs),source='archived exact common native predictions') for k,rs in groups.items()];csvout('V051_common_reference.csv',refrows)
    aggregate=[];groups=defaultdict(list)
    for r in scores:groups[(r['condition'],r['direction'],r['width'],r['op'])].append(r)
    for k,rs in groups.items():
        v=[r['exact_rate'] for r in rs];aggregate.append(dict(condition=k[0],direction=k[1],width=k[2],op=k[3],seeds=len(rs),exact_mean=statistics.mean(v),exact_sample_SD=statistics.stdev(v),bit_accuracy_mean=statistics.mean(r['bit_accuracy'] for r in rs)))
    csvout('length_generalization.csv',aggregate);assert protected()==before
    for name,h in json.loads((R/'STARTED.json').read_text())['registered_SHA'].items():assert sha(R/name)==h
    dump('STOPPED.json',dict(status='training_and_all_evaluation_completed_stopped',runs=12,additional_training=0,next='independent saved-data audit and report only'))
    print('ALL12 RUNS / ALL9 WIDTHS COMPLETE; TRAINING AND INFERENCE STOPPED',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:dump('STOPPED.json',dict(status='failed_or_invalid_stopped',error=repr(e),additional_training=0));raise
