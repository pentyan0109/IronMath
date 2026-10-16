"""ABL1 independent analysis of saved files only. No torch or model loading."""
from prepare import R,W,read,write,sha,utc,processes
import csv,json,math,time,traceback,itertools
import numpy as np
METHODS=['A0','A1','A2','A3','A4','A5']
PAIRS=[('A0','A1'),('A0','A2'),('A0','A3'),('A0','A4'),('A0','A5'),('A1','A3'),('A3','A0')]
LENGTHS=[32,64,128,512,2048,8192,65000];MAIN=[2048,8192,65000]
def csvrows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
from pathlib import Path
def table(name,records):
    if not records:return
    p=R/name;assert not p.exists(),f'No overwrite: {p}'
    keys=list(dict.fromkeys(k for r in records for k in r))
    with p.open('w',encoding='utf-8',newline='') as f:q=csv.DictWriter(f,keys);q.writeheader();q.writerows(records)
def finite(v):return v is not None and math.isfinite(v)
def metrics(pred,truth):
    p=np.array(pred,bool);y=np.array(truth,bool);tp=int((p&y).sum());fp=int((p&~y).sum());tn=int((~p&~y).sum());fn=int((~p&y).sum())
    tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None;den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return dict(N=len(y),hits=tp+tn,accuracy=(tp+tn)/len(y),TP=tp,TN=tn,FP=fp,FN=fn,TPR=tpr,TNR=tnr,balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,MCC=(tp*tn-fp*fn)/math.sqrt(den) if den else None,actual_success=int(y.sum()),actual_failure=int((~y).sum()),predicted_success=int(p.sum()),predicted_failure=int((~p).sum()))
def cluster_pair(ms,left,right):
    seeds=sorted({m['seed'] for m in ms});sums=[];counts=[]
    for s in seeds:
        ss=[m for m in ms if m['seed']==s];ds=[int(m[left]==m['actual_success'])-int(m[right]==m['actual_success']) for m in ss]
        sums.append(sum(ds));counts.append(len(ds))
    sums=np.array(sums);counts=np.array(counts);ix=np.random.default_rng(87000).integers(0,len(seeds),(10000,len(seeds)))
    samples=sums[ix].sum(1)/counts[ix].sum(1);ci=np.percentile(samples,[2.5,97.5]);b=sum(m[left]==m['actual_success'] and m[right]!=m['actual_success'] for m in ms);c=sum(m[right]==m['actual_success'] and m[left]!=m['actual_success'] for m in ms);n=b+c
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.
    return dict(left=left,right=right,N=len(ms),seed_count=len(seeds),left_hits=sum(m[left]==m['actual_success'] for m in ms),right_hits=sum(m[right]==m['actual_success'] for m in ms),predictions_differ=sum(m[left]!=m[right] for m in ms),left_only_correct=int(b),right_only_correct=int(c),accuracy_difference=float(sums.sum()/counts.sum()),CI95_low=float(ci[0]),CI95_high=float(ci[1]),cluster='seed',bootstrap_samples=10000,bootstrap_seed=87000,McNemar_exact_p=p,McNemar_role='supplementary; ignores inter-model dependence',clear_difference='left higher' if ci[0]>0 else 'right higher' if ci[1]<0 else 'not detected',interpretation='exploratory unadjusted interval; no equivalence claim')
def labels(x):
    pos=np.broadcast_to(np.arange(x.shape[1]),x.shape);last=np.maximum.accumulate(np.where(x<2,pos,-1),axis=1)
    return np.where(last<0,0,np.take_along_axis(x,np.maximum(last,0),axis=1))
def run():
    start=time.perf_counter();cpu_start=time.process_time();freeze=read(R/'FREEZE.json')
    for name,v in freeze['files'].items():assert sha(R/name)==v['SHA256']
    write('ANALYSIS_CODE_FROZEN.json',dict(utc=utc(),before_comparison=True,files={p.name:sha(p) for p in (R/'scripts').glob('*.py')}))
    g=W/'IronMath_G1';frozen=read(g/'main_PREDICTIONS_FROZEN.json')['models'];manifest=read(R/'INPUT_MANIFEST.json');inputmap={r['path']:r for r in manifest['files']}
    reported={r['model']:r for r in csvrows(g/'g1_results.csv')};inputs={}
    for t in LENGTHS:
        with np.load(g/'main_long_data'/f'T{t}.npz',allow_pickle=False) as z:x=z['symbols']
        meta=read(g/'main_long_data'/f'T{t}_metadata.json');assert x.shape==(68,t) and len(meta)==68
        inputs[t]=(labels(x),x==3,meta)
    models=[];trajectories=[];scores=[];original_checks=[];categories=[];costs=[];margin_anomalies=[]
    timing={m:0. for m in METHODS};shared_read_wall=0.
    for r in frozen:
        mid=r['model'];d=read(g/'diagnostic_records'/f'{mid}.json');asset=read(g/'assets'/f'{mid}.json');assert len(d['details'])==16
        cp=inputmap[(g/asset['path']).relative_to(W).as_posix()];assert cp['SHA256']==asset['SHA256']==r['checkpoint_SHA256']==d['checkpoint_SHA256']
        original_checks.append(dict(model=mid,check='selected checkpoint hash',passed=True))
        m=dict(model=mid,architecture=r['architecture'],L=r['L'],seed=r['seed'],A0=bool(r['A']),A5=bool(r['D']),checkpoint_SHA256=cp['SHA256'])
        dd=d['details'];keyed={(x['neutral'],x['prefix'],x['value']):x for x in dd};assert len(keyed)==16
        m['nonconvergent_trajectories']=sum(not x['converged'] for x in dd)
        m['endpoint_readout_errors']=sum(not x['readout_finite'] or x['readout']!=x['value'] for x in dd)
        m['rho_missing_nonconvergent']=sum(x['spectral_radius'] is None and not x['converged'] for x in dd)
        m['rho_not_below_one_converged']=sum(x['converged'] and (not finite(x['spectral_radius']) or x['spectral_radius']>=1) for x in dd)
        for method,predicate in [('A1',lambda x:x['readout_finite'] and x['readout']==x['value']),('A3',lambda x:x['converged'] and x['readout_finite'] and x['readout']==x['value']),('A4',lambda x:x['converged'] and finite(x['spectral_radius']) and x['spectral_radius']<1)]:
            t0=time.perf_counter();m[method]=all(predicate(x) for x in dd);timing[method]+=time.perf_counter()-t0
        originalA=all(x['converged'] and finite(x['spectral_radius']) and x['spectral_radius']<1 and x['readout_finite'] and x['readout']==x['value'] for x in dd)
        assert originalA==m['A0']==d['A'];original_checks.append(dict(model=mid,check='original A logical reproduction',passed=True))
        path_ok=True;Dfinite=True;Dconv=True;wrong_steps=0;first_bad=None;nonfinite_diag=False;nonfinite_state=0
        for neutral in ['N','R']:
            t0=time.perf_counter()
            with np.load(g/'diagnostic_traces'/f'{mid}_{neutral}.npz',allow_pickle=False) as z:
                en=z['endpoints'];mv=z['movements'];mg=z['margins'];sf=z['state_finite'];lf=z['logit_finite']
            shared_read_wall+=time.perf_counter()-t0
            assert mv.shape==mg.shape==sf.shape==lf.shape==(20000,8)
            Dfinite &= bool(all(np.isfinite(z).all() for z in [en,mv,mg]) and sf.all() and lf.all());nonfinite_state+=int((~sf).sum())
            t0=time.perf_counter();values=np.arange(8)%2
            correct=np.where(values[None,:]==0,mg>=0,mg>0)&~np.isnan(mg)&lf
            bad=~correct;path_ok &= bool(correct.all());wrong_steps+=int(bad.sum());nonfinite_diag |= bool((~lf).any())
            ambiguous=(~lf)&np.isfinite(mg)
            if ambiguous.any():margin_anomalies.append(dict(model=mid,neutral=neutral,finite_margin_with_invalid_logit_guard=int(ambiguous.sum()),treatment='same G1 combined validity guard; no invalid step accepted'))
            for j in range(8):
                prefix,value=divmod(j,2);x=keyed[neutral,prefix,value];conv=bool(np.isfinite(en[j]).all() and np.isfinite(mv[-100:,j]).all() and mv[-100:,j].max()<1e-6)
                assert conv==x['converged'];Dconv &= conv
                idx=np.flatnonzero(bad[:,j]);last_good=bool(correct[-1,j]);endpoint_good=bool(x['readout_finite'] and x['readout']==value)
                if last_good!=endpoint_good:margin_anomalies.append(dict(model=mid,neutral=neutral,prefix=prefix,value=value,terminal_margin_disagrees_recorded_endpoint=True,margin=float(mg[-1,j]) if np.isfinite(mg[-1,j]) else None,treatment='A1 uses saved direct endpoint label; A2 uses saved margins and validity as frozen'))
                trajectories.append(dict(model=mid,architecture=m['architecture'],L=m['L'],seed=m['seed'],neutral=neutral,prefix=prefix,value=value,converged=conv,readout=x['readout'],readout_finite=x['readout_finite'],readout_correct=endpoint_good,spectral_radius=x['spectral_radius'],rho_eligible=conv and finite(x['spectral_radius']) and x['spectral_radius']<1,trajectory_A2_good=not len(idx),first_bad_read_step=int(idx[0]+1) if len(idx) else None,wrong_or_invalid_steps=len(idx),nonfinite_state_steps=int((~sf[:,j]).sum()),nonfinite_logit_steps=int((~lf[:,j]).sum())))
            timing['A2']+=time.perf_counter()-t0
        m['A2']=path_ok;m['intermediate_wrong_or_invalid_steps']=wrong_steps;m['diagnostic_nonfinite_logits']=nonfinite_diag;m['diagnostic_nonfinite_states']=nonfinite_state
        assert bool(Dconv and Dfinite)==m['A5']==d['D'];original_checks.append(dict(model=mid,check='original D logical reproduction',passed=True))
        m['all16_converged']=bool(Dconv);m['diagnostic_all_finite']=bool(Dfinite)
        allvalid=True;any_nonfinite=False;main_correct=0;main_count=0;failedcats=set();finitewrong=0;immediatefail=0
        for t in LENGTHS:
            yy,mask,meta=inputs[t]
            with np.load(g/'saved_predictions'/f'{mid}_T{t}.npz',allow_pickle=False) as z:
                bits=np.unpackbits(z['bits'],axis=1,count=t,bitorder='little');inv=np.unpackbits(z['invalid'],axis=1,count=t,bitorder='little').astype(bool)
            assert bits.shape==inv.shape==yy.shape
            wrong=((bits!=yy)|inv)&mask;ok=not wrong.any();any_nonfinite |= bool(inv.any());finitewrong+=int(((bits!=yy)&~inv&mask).sum())
            scores.append(dict(model=mid,T=t,success=bool(ok),wrong_R=int(wrong.sum()),nonfinite_positions=int(inv.sum())))
            if t in MAIN:
                allvalid &= bool(ok);main_count+=int(mask.sum());main_correct+=int((mask&~wrong).sum())
                for i,item in enumerate(meta):
                    if wrong[i].any():failedcats.add(item['category'])
                    categories.append(dict(model=mid,T=t,category=item['category'],id=item['id'],failed=bool(wrong[i].any()),wrong_R=int(wrong[i].sum())))
            er=read(g/'evaluation_records'/f'{mid}.json') if t==LENGTHS[0] else er
            # Immediate rewrite score uses recorded item-level counts; saved bit scoring above is independent for category outcomes.
        for item in er['items']:
            if item['T'] in MAIN:immediatefail+=item['immediate_rewrite_R_count']-item['immediate_rewrite_R_correct']
        m['actual_success']=bool(allvalid);m['long_nonfinite']=bool(any_nonfinite);m['finite_failure']=not allvalid and not any_nonfinite;m['failed_categories_main']=';'.join(sorted(failedcats));m['rewrite_or_mixed_failure']=bool(failedcats&{'rewrite','mixed'});m['immediate_rewrite_R_errors']=immediatefail
        m['classification']='発散' if any_nonfinite else '成功' if allvalid else '有限誤答';m['main_R_correct']=main_correct;m['main_R_count']=main_count
        old=reported[mid];assert allvalid==(old['actual_success']=='True') and main_correct==int(old['main_R_correct']) and main_count==int(old['main_R_count']) and any_nonfinite==(int(old['nonfinite_logit_positions'])>0)
        original_checks.append(dict(model=mid,check='saved-output independent scoring vs G1',passed=True))
        assert not m['A0'] or (m['A1'] and m['A3'] and m['A4']);assert not m['A3'] or m['A1']
        models.append(m)
        if len(models)%20==0:print('Analyzed',len(models),'/200',flush=True)
    # Existing FP64 is explicitly separate from primary FP32 results.
    fp_rows=csvrows(W/'IronMath_PaperCheck/addon/fp64/fp64_trajectories.csv');fp_by={}
    for row in fp_rows:fp_by.setdefault(row['model'],[]).append(row)
    fp_sensitivity=[]
    for m in models:
        rr=fp_by[m['model']];assert len(rr)==16
        cv=lambda x,k:x[k]=='True'
        fpa0=all(cv(x,'converged_FP64') and finite(float(x['spectral_radius_FP64'])) and float(x['spectral_radius_FP64'])<1 and cv(x,'readout_correct_FP64') for x in rr if x['spectral_radius_FP64'])
        # A nonconvergent row can have empty rho, and must not be silently omitted.
        fpa0=fpa0 and all(cv(x,'converged_FP64') and x['spectral_radius_FP64'] for x in rr)
        fpa1=all(cv(x,'readout_correct_FP64') for x in rr)
        fp_sensitivity.append(dict(model=m['model'],architecture=m['architecture'],L=m['L'],seed=m['seed'],actual_success=m['actual_success'],A0_FP32=m['A0'],A0_FP64=bool(fpa0),A1_FP32=m['A1'],A1_FP64=fpa1,changed_convergence=sum(x['converged_FP32']!=x['converged_FP64'] for x in rr),changed_readout=sum(x['readout_correct_FP32']!=x['readout_correct_FP64'] for x in rr),shared_converged_rho_threshold_changes=sum((float(x['spectral_radius_FP32'])<1)!=(float(x['spectral_radius_FP64'])<1) for x in rr if cv(x,'converged_FP32') and cv(x,'converged_FP64') and x['spectral_radius_FP32'] and x['spectral_radius_FP64'])))
    groups={'all200':models,'nonlinear160':[m for m in models if m['architecture']!='linear']}
    groups.update({arch:[m for m in models if m['architecture']==arch] for arch in ['residual_gru','gru','lstm','rnn','linear']})
    summaries=[];comparisons=[];disagreements=[];seedrows=[]
    for group,ms in groups.items():
        for method in METHODS:summaries.append(dict(group=group,diagnostic=method,**metrics([m[method] for m in ms],[m['actual_success'] for m in ms])))
        for s in sorted({m['seed'] for m in ms}):
            ss=[m for m in ms if m['seed']==s]
            for method in METHODS:seedrows.append(dict(group=group,seed=s,diagnostic=method,**metrics([m[method] for m in ss],[m['actual_success'] for m in ss])))
        for left,right in PAIRS:comparisons.append(dict(group=group,**cluster_pair(ms,left,right)))
    for left,right in PAIRS:
        for m in models:
            if m[left]!=m[right]:disagreements.append(dict(left=left,right=right,model=m['model'],architecture=m['architecture'],L=m['L'],seed=m['seed'],actual_success=m['actual_success'],left_prediction=m[left],right_prediction=m[right],left_correct=m[left]==m['actual_success'],right_correct=m[right]==m['actual_success'],nonconvergent_trajectories=m['nonconvergent_trajectories'],endpoint_readout_errors=m['endpoint_readout_errors'],rho_not_below_one_converged=m['rho_not_below_one_converged'],classification=m['classification'],failed_categories_main=m['failed_categories_main']))
    # Costs are observations from old logs, not extrapolated A1/A2 online costs.
    for method in METHODS:
        costs.append(dict(diagnostic=method,component='total from scratch',scope='dedicated '+method,status='not separately measured',wall_seconds=None,CPU_seconds=None,source='existing logs cannot isolate dedicated ablation pipeline'))
        costs.append(dict(diagnostic=method,component='predicate analysis on saved arrays',scope='200 models',status='current offline analysis only; I/O shared separately',wall_seconds=timing[method],CPU_seconds=None,source='ABL1 perf_counter around predicates; excludes raw file read'))
    for row in csvrows(W/'IronMath_PaperCheck/stage2/s2_time_breakdown.csv'):
        if row['group']!='nonlinear':continue
        for component,col in [('prep','prep_mean'),('state iteration plus intermediate readout','iterate_mean'),('endpoint readout','readout_mean'),('Jacobian plus eigendecomposition','jacobian_eig_mean'),('total excluding scripting','total_excluding_script_mean')]:
            if row.get(col):costs.append(dict(diagnostic='A0 implementation' if row['method']=='A' else 'direct '+row['method'],component=component,scope=f"nonlinear160 mean/model, iterations={row['iterations']}, ell={row.get('ell','')}",status='historical measured wall time, implementation dependent',wall_seconds=float(row[col]),CPU_seconds=None,source='stage2/s2_time_breakdown.csv'))
    for component,count in [('trajectory state updates',320000),('extra N trial-R state updates during intermediate readout',160000),('Bcum128 direct sequence steps',15232),('Bcum512 direct sequence steps',50048)]:costs.append(dict(diagnostic='structural count',component=component,scope='per model',status='operation count only; not measured runtime',state_updates=count))
    costs.extend([dict(diagnostic='shared',component='read/decompress diagnostic arrays',scope='200 models',status='current offline file I/O only',wall_seconds=shared_read_wall),dict(diagnostic='A0-A5',component='Jacobian alone / eigensolver alone / convergence alone',scope='historical online run',status='not separable in saved timing logs',wall_seconds=None)])
    # Hash audit is exact and covers every selected input, not just result tables.
    changed=[]
    for i,entry in enumerate(manifest['files'],1):
        p=W/entry['path']
        if not p.exists() or sha(p)!=entry['SHA256']:changed.append(entry['path'])
    assert not changed,changed
    table('abl1_model_results.csv',models);table('abl1_architecture_summary.csv',summaries);table('abl1_pairwise.csv',comparisons);table('abl1_seed_bootstrap.csv',comparisons);table('abl1_seed_results.csv',seedrows)
    if disagreements:table('abl1_disagreements.csv',disagreements)
    else:table('abl1_disagreements.csv',[dict(left='',right='',model='',note='No disagreements')])
    table('abl1_trajectory_details.csv',trajectories);table('abl1_long_scores.csv',scores);table('abl1_category_failures.csv',categories);table('abl1_fp64_sensitivity.csv',fp_sensitivity);table('abl1_original_checks.csv',original_checks);table('abl1_costs.csv',costs)
    write('MARGIN_ANOMALIES.json',margin_anomalies)
    write('ANALYSIS_AUDIT.json',dict(completed_utc=utc(),models=len(models),trajectories=len(trajectories),input_files_checked=len(manifest['files']),changed_inputs=changed,all_original_A_D_and_truth_checks_passed=True,rule_hashes_unchanged=all(sha(R/n)==v['SHA256'] for n,v in freeze['files'].items()),new_training=False,new_inference=False,model_checkpoint_loaded=False,analysis_wall_seconds=time.perf_counter()-start,analysis_CPU_seconds=time.process_time()-cpu_start,notes=['Paired bootstrap cluster=seed, not independent models.','Reported compute stages not separable are NA.']))
    print('ANALYSIS COMPLETE',flush=True)
if __name__=='__main__':
    try:run()
    except Exception:
        p=R/'FAILURE_LOG.jsonl'
        with p.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=utc(),traceback=traceback.format_exc(),no_values_imputed=True),ensure_ascii=False)+'\n')
        raise
