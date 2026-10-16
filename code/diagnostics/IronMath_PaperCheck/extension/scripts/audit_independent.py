"""Independent scoring and E2 audit replay: imports no E1/E2/E3 evaluator."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from common import *
import torch,gzip,math
import random
from torch import nn
torch.set_num_threads(2)
class AuditModel(nn.Module):
    def __init__(self,sd):
        super().__init__();self.projection=nn.Linear(3,64);self.core=nn.GRUCell(64,545);self.head=nn.Linear(545,2);self.residual_gate=nn.Parameter(torch.zeros(()))
        if sd is not None:self.load_state_dict(sd)
    def forward(self,initial:torch.Tensor,x:torch.Tensor,targets:torch.Tensor,count:int):
        h=initial;out=torch.empty(count+1,len(targets));z=self.head(h);out[0]=z.gather(1,targets[:,None])[:,0]-z.gather(1,(1-targets)[:,None])[:,0]
        for t in range(count):
            h=self.core(self.projection(x),h)+torch.tanh(self.residual_gate)*h;z=self.head(h);out[t+1]=z.gather(1,targets[:,None])[:,0]-z.gather(1,(1-targets)[:,None])[:,0]
        return out
def load_weights(path):return torch.load(path,weights_only=False,map_location='cpu')['model']
def audit_e2():
    md('audit/E2_AUDIT_REPLAY_RULES.md','''# E2独立監査の追加読み取り専用再生
初回E2保存curvesは代表条件だけであるため、全1350測定行の初回/持続時刻を独立再採点するために、同じcheckpoint・入力・介入条件・seed・時刻を再生する。学習・新条件はなし。評価モジュールはimportせず、独立定義の標準GRUCellとheadで再生。保存された受容/ドナー状態を用い、全曲線をauditに追加保存する。各条件の初回時刻・持続開始は元CSVと厳密一致を要求。無介入margin差<=1e-3、ノルム相対差1e-5・margin変化差1e-3を品質許容差とする。条件でなく監査精度の指定であり、異なる結果は隠さず停止する。
''')
    dump('audit/E2_AUDIT_REPLAY_FROZEN.json',dict(utc=utc(),rules_SHA256=sha(R/'audit/E2_AUDIT_REPLAY_RULES.md'),script_SHA256=sha(Path(__file__))))
    reference={(int(r['seed']),r['op'],int(r['state']),r['pair'],int(r['at']),r['condition']):r for r in rows(R/'E2_causal_intervention/E2_RESULTS.csv') if r['status']=='measured'};checks=[];geometry=[];saved=[]
    with (W/'v054t_transition/data/probes.jsonl').open() as f:rr=[json.loads(l) for l in f]
    rr=[r for r in rr if r['k']==4096 and r['prefix']=='min' and r['suffix']=='s0']
    with gzip.open(R/'audit/E2_all_curves.jsonl.gz','wt',encoding='utf-8') as output:
        for seed in range(3):
            sd=load_weights(W/f'IronMath_V054/checkpoints/C1_B0_s{seed}/best_validation.pt');model=torch.jit.script(AuditModel(sd).eval());cache=np.load(R/f'E1_finite_time/cache/V054_s{seed}_trajectory.npz');st=cache['states'];x=cache['inputs'];y=cache['targets'];baseline=cache['margins'][:1024]
            weights=sd['head.weight'].numpy()
            for i,r in enumerate(rr):
                for at in [1,4,8]:
                    rng=np.random.default_rng(92000+seed*1000+i*20+at);base=st[at,i];cases=[('baseline',base,0.,None)]
                    for donor,point in [('correct',1),('failed',512)]:
                        dm=cache['margins'][point-1,i]
                        if not (dm>0 if donor=='correct' else dm<0):continue
                        ds=st[point,i];n=float(np.linalg.norm(ds-base));w=weights[y[i]]-weights[1-y[i]];u=w/np.linalg.norm(w);cases.append(('transplant_'+donor,ds,n,donor))
                        for sign in [-1,1]:cases.append((f'normal_{sign}_{donor}',base+sign*n*u,n,donor))
                        for j in range(3):
                            q=rng.normal(size=545).astype(np.float32);q=q/np.linalg.norm(q);orth=q-(q@u)*u;orth=orth/np.linalg.norm(orth);cases.extend([(f'random{j}_{donor}',base+n*q,n,donor),(f'orthogonal{j}_{donor}',base+n*orth,n,donor)])
                    states=torch.from_numpy(np.stack([a[1] for a in cases]));xx=torch.from_numpy(np.tile(x[i],(len(cases),1)));targets=torch.full((len(cases),),int(y[i]),dtype=torch.long)
                    with torch.no_grad():marg=model(states,xx,targets,1024-at).numpy()
                    for j,(name,state,n,donor) in enumerate(cases):
                        key=(seed,r['op'],r['state'],r['pair'],at,name);old=reference[key];full=baseline[:,i].copy();full[at-1:]=marg[:,j];badmask=(full<0)|((full==0)&(y[i]==1))|~np.isfinite(full);bad=np.flatnonzero(badmask);good=np.flatnonzero(~badmask);first=int(bad[0])+1 if len(bad) else None;pers=int(good[-1])+2 if badmask[-1] and len(good) else 1 if not len(good) else None
                        eq=(str(first) if first is not None else '')==old['first_error'] and (str(pers) if pers is not None else '')==old['persistent_error_start_observed']
                        checks.append(dict(seed=seed,op=r['op'],state=r['state'],pair=r['pair'],at=at,condition=name,first_error=first,persistent_error_start_observed=pers,passed=eq));assert eq,checks[-1]
                        delta=state.astype(float)-base.astype(float);actualnorm=float(np.linalg.norm(delta));wnormal=(weights[y[i]]-weights[1-y[i]]).astype(float);exact=float(wnormal@delta);measured=float(old['immediate_margin_change']);match=np.isclose(actualnorm,n,rtol=1e-5,atol=1e-6) and abs(measured-exact)<1e-3
                        geometry.append(dict(seed=seed,index=i,at=at,condition=name,intended_norm=n,actual_norm=actualnorm,linear_margin_prediction=exact,measured_margin_change=measured,passed=bool(match)));assert match,geometry[-1]
                        output.write(json.dumps(dict(seed=seed,index=i,at=at,condition=name,margins=full.tolist()))+'\n')
            print('Independent E2 replay audited seed',seed,flush=True)
    table('audit/E2_METRIC_CHECKS.csv',checks);table('audit/E2_GEOMETRY_CHECKS.csv',geometry)
    arithmetic=[]
    for r in rows(R/'E2_causal_intervention/E2_ARITHMETIC.csv'):
        a=int(r['a'],16);b=int(r['b'],16);truth=a+b if r['op']=='ADD' else a-b;baseline=int(r['baseline_prediction'],16);pred=int(r['prediction'],16)
        assert truth==int(r['truth'],16) and (baseline==truth)==(r['baseline_correct']=='True') and (pred==truth)==(r['intervention_correct']=='True')
        if r['intervened']=='False':assert baseline==pred
        arithmetic.append(dict(seed=r['seed'],id=r['id'],mode=r['mode'],passed=True))
    table('audit/E2_ARITHMETIC_RESCORE.csv',arithmetic)
def synthetic_errors():
    from decimal import Decimal,localcontext
    config=read(R/'E1_finite_time/FROZEN.json')['cases'];rr=[]
    for name,AA,HH,WW,BB in config:
        for dtype,typ in [('FP32',np.float32),('FP64',np.float64)]:
            with localcontext() as ctx:
                ctx.prec=100;a=[[Decimal(v) for v in row] for row in AA];ref=list(map(Decimal,HH));v=np.array(HH,typ);af=np.array(AA,typ);maximum=Decimal(0)
                for t in range(4097):
                    error=sum((Decimal.from_float(float(x))-y)**2 for x,y in zip(v,ref)).sqrt();maximum=max(maximum,error);v=af@v;ref=[sum(x*y for x,y in zip(row,ref)) for row in a]
                rr.append(dict(case=name,dtype=dtype,max_state_error_decimal=str(maximum),note='Decimal norm avoids binary64 squared-norm underflow; original float norm retained separately'))
    table('audit/E1_HIGH_PRECISION_ERROR_AUDIT.csv',rr)
    dump('audit/NUMERICAL_CLARIFICATION.json',dict(utc=utc(),issue='Original FP64 boundary state-error norm is displayed as zero due to binary64 conversion/squared-norm underflow',cause_identified=True,correction='Reference errors recomputed in Decimal norm and stored separately; timing outcomes unchanged',experimental_conditions_changed=False))
def e3_rescore():
    test={};hold={};probe={}
    with (W/'v054t_transition/data/probes.jsonl').open() as f:
        for line in f:
            r=json.loads(line);probe[r['pid']]=r
    raw=R/'E3_independent_replication/predictions.jsonl.gz'
    with gzip.open(raw,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);a=int(r['a'],16);b=int(r['b'],16);truth=a+b if r['op']=='ADD' else a-b;assert truth==int(r['truth'],16);prediction=int(r['prediction'],16) if r['prediction'] is not None else None
            if r['kind']=='test':
                correct=prediction==truth;assert correct==r['correct'];key=(r['model'],r['op'],r['width']);count=test.setdefault(key,[0,0]);count[0]+=correct;count[1]+=1
            else:
                assert prediction is not None,'Missing finite output: cannot infer first error from an absent bit vector'
                p=probe[int(r['id'])];difference=prediction^truth;within=(difference>>p['hold_start'])&((1<<p['k'])-1);first=(within&-within).bit_length() if within else None
                assert first==r['first_hold_error'];hold[r['model'],p['pid']]=dict(first=first,exact=difference==0)
    checks=[]
    for r in rows(R/'E3_independent_replication/E3_TEST_RESULTS.csv'):
        hit,n=test[r['model'],r['op'],int(r['width'])];assert hit==int(r['correct']) and n==int(r['N']);checks.append(dict(model=r['model'],op=r['op'],width=r['width'],correct=hit,N=n,passed=True))
    for r in rows(R/'E3_independent_replication/E3_HOLD_RESULTS.csv'):
        x=hold[r['model'],int(r['pid'])];assert x['exact']==(r['exact']=='True')
    table('audit/E3_INDEPENDENT_RESCORE.csv',checks)
    models=rows(R/'E3_independent_replication/E3_MODEL_RESULTS.csv');assert len(models)==20 and len({r['initial_SHA256'] for r in models})==10 and len({r['data_seed'] for r in models})==20
    for seed in range(100,110):assert len({r['initial_SHA256'] for r in models if int(r['initialization_seed'])==seed})==1
    pool={op:read(W/f'IronMath_V053/data/{op}_Train.json') for op in ['ADD','SUB']};trainchecks=[]
    for m in models:
        path=R/f"E3_independent_replication/checkpoints/{m['model']}/best_validation.pt";assert sha(path)==m['selected_checkpoint_SHA256']
        torch.manual_seed(int(m['initialization_seed']));fresh=AuditModel(None);hh=hashlib.sha256()
        for k,v in fresh.state_dict().items():hh.update(k.encode());hh.update(v.detach().numpy().tobytes())
        assert hh.hexdigest()==m['initial_SHA256']
        rng=random.Random(int(m['data_seed']));draws=[]
        for step in range(1200):
            for op in ['ADD','SUB']:
                for _ in range(8):
                    p=pool[op][rng.randrange(len(pool[op]))];a,b=p['a'],p['b']
                    if op=='ADD' and rng.randrange(2):a,b=b,a
                    draws.append(f"{p['id']}:{a}:{b}")
        assert hashlib.sha256('\n'.join(draws).encode()).hexdigest()==m['data_presentation_SHA256']
        logs=rows(R/f"E3_independent_replication/checkpoints/{m['model']}/training_log.csv");selected=min(logs,key=lambda r:float(r['validation_loss']));assert selected['update']==m['selected_update'] and int(m['updates'])==1200
        trainchecks.append(dict(model=m['model'],initialization_independently_reconstructed=True,presentation_sequence_independently_reconstructed=True,Validation_selection_reproduced=True))
    table('audit/E3_TRAINING_CHECKS.csv',trainchecks)
    # Independent geometric classification from saved end states/movements.
    pp=rows(R/'E3_independent_replication/E3_PAIR_RESULTS.csv');base=[r for r in probe.values() if r['k']==4096 and r['prefix']=='min' and r['suffix']=='s0'];geochecks=[];ds=rows(R/'E3_independent_replication/E3_DIAGNOSTICS.csv');hh=rows(R/'E3_independent_replication/E3_HOLD_RESULTS.csv')
    for m in models:
        z=np.load(R/f"E3_independent_replication/diagnostics/{m['model']}_trajectory.npz")
        for r in [r for r in pp if r['model']==m['model']]:
            ix=[i for i,b in enumerate(base) if b['op']==r['op'] and b['pair']==r['pair']];assert len(ix)==2;conv=all(np.isfinite(z['end'][i]).all() and z['movements'][-100:,i].max()<1e-6 for i in ix);distance=float(np.linalg.norm(z['end'][ix[0]].astype(float)-z['end'][ix[1]].astype(float)));shared=conv and distance<1e-3
            assert shared==(r['shared']=='True') and conv==(r['both_converged']=='True');geochecks.append(dict(model=m['model'],op=r['op'],pair=r['pair'],shared=shared,distance=distance,passed=True))
            dd=[d for d in ds if d['model']==m['model'] and d['op']==r['op'] and d['pair']==r['pair']];stable=all(d['rho'] and float(d['rho'])<1 for d in dd)
            same_side_observed=any(d['endpoint_correct']=='False' and any(h['model']==m['model'] and h['op']==r['op'] and h['pair']==r['pair'] and h['state']==d['state'] and h['first_hold_error'] for h in hh) for d in dd)
            intended=shared and stable and any(d['endpoint_correct']=='False' for d in dd) and same_side_observed
            assert intended==(r['strict_mechanism_reproduced']=='True'),('same-side strict mechanism mismatch',m['model'],r)
    table('audit/E3_GEOMETRY_CHECKS.csv',geochecks)
    return len(test),len(hold)
def statistics():
    init=rows(R/'E3_independent_replication/E3_INITIALIZATION_RESULTS.csv');recorded=rows(R/'E3_independent_replication/E3_CLUSTER_STATISTICS.csv');out=[]
    for r in recorded:
        vals=np.array([float(x[r['metric']]) for x in init if x[r['metric']]!= '']);ix=np.random.default_rng(99000).integers(0,len(vals),(10000,len(vals)));ci=np.quantile(vals[ix].mean(1),[.025,.975])
        assert math.isclose(float(r['mean']),float(vals.mean()),abs_tol=1e-12) and np.allclose(ci,[float(r['CI95_low']),float(r['CI95_high'])],atol=1e-12,rtol=0);out.append(dict(metric=r['metric'],passed=True))
    table('audit/STATISTICS_CHECKS.csv',out)
def spectral_audit():
    # Independently differentiate the GRU gate algebra in NumPy float64.
    # No evaluator or torch.autograd code is imported.
    md('audit/SPECTRAL_AUDIT_RULES.md','''# Jacobian独立監査
checkpointからGRUのr/z/nゲートの偏微分式をNumPy FP64で独立計算し、全保存収束終点のrhoを固有値から再計算する。元FP32自動微分JをFP64固有値化した値とのrtol1e-3/atol1e-4およびrho<1判定一致を要求。差があれば採用を停止する。新しい実験閾値ではなく数値検証許容差。未収束終点のrhoは補完しない。
''')
    results=[];models=[(f'V054_s{s}',W/f'IronMath_V054/checkpoints/C1_B0_s{s}/best_validation.pt',R/'E1_finite_time/cache') for s in range(3)]
    models.extend((r['model'],R/f"E3_independent_replication/checkpoints/{r['model']}/best_validation.pt",R/'E3_independent_replication/diagnostics') for r in rows(R/'E3_independent_replication/E3_MODEL_RESULTS.csv'))
    for mid,cp,folder in models:
        sd={k:v.detach().numpy().astype(float) for k,v in load_weights(cp).items()};z=np.load(folder/f'{mid}_trajectory.npz');records=read(folder/f'{mid}_diagnostic.json')['records'];wh=sd['core.weight_hh'];H=545
        for r in records:
            if not r['converged']:continue
            idx=r['index'];h=z['end'][idx].astype(float);x=z['inputs'][idx].astype(float);projection=sd['projection.weight']@x+sd['projection.bias'];gi=sd['core.weight_ih']@projection+sd['core.bias_ih'];gh=wh@h+sd['core.bias_hh'];reset=1/(1+np.exp(-(gi[:H]+gh[:H])));update=1/(1+np.exp(-(gi[H:2*H]+gh[H:2*H])));new=np.tanh(gi[2*H:]+reset*gh[2*H:]);J=((1-update)*(1-new**2))[:,None]*(reset[:,None]*wh[2*H:]+(gh[2*H:]*reset*(1-reset))[:,None]*wh[:H])+((h-new)*update*(1-update))[:,None]*wh[H:2*H];J[np.diag_indices(H)]+=update+np.tanh(sd['residual_gate']);rho=float(np.abs(np.linalg.eigvals(J)).max());old=r['rho'];passed=np.isclose(rho,old,rtol=1e-3,atol=1e-4) and (rho<1)==(old<1)
            results.append(dict(model=mid,index=idx,rho_independent_FP64=rho,rho_original_FP32_J=old,passed=bool(passed)));assert passed,results[-1]
        print('Jacobian audit',mid,flush=True)
    table('audit/INDEPENDENT_SPECTRAL_CHECKS.csv',results)
def main():
    assert read(R/'E3_independent_replication/COMPLETE.json')['models']==20
    synthetic_errors();audit_e2();n,h=e3_rescore();statistics();spectral_audit();changed=[x['path'] for x in read(R/'audit/INPUT_MANIFEST.json')['files'] if sha(Path(x['path']))!=x['SHA256']];assert not changed
    dump('audit/INDEPENDENT_AUDIT.json',dict(utc=utc(),passed=True,E3_test_groups=n,E3_hold_predictions=h,all_input_SHA_unchanged=True,changed_inputs=changed,E2_additional_readonly_audit_replay=True,E1_boundary_error_representation_resolved=True,original_models_modified=False))
if __name__=='__main__':
    try:main()
    except Exception as e:stage_failure('independent audit',e);raise
