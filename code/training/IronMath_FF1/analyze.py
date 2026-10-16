"""Regrade bit-packed predictions without model inference; preregistered comparisons."""
import collections,statistics,time
from common import *
def records(p):
    with (R/p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def analyze():
    import numpy as np
    from data import sequential_labels,LENGTHS
    registrations_ok();assert read('LONG_EVALUATION_COMPLETE.json')['models']==40;t0=time.perf_counter()
    source=records('long_item_results.csv');lookup={(r['model'],int(r['T']),r['id']):r for r in source};regraded=[];failures=[]
    for f in read('LONG_DATA_FROZEN.json')['files']:
        t=f['T'];assert sha(R/f['path'])==f['SHA256'];x=np.load(R/f['path'])['symbols'];y=sequential_labels(x);meta=read(f'long_data/T{t}_metadata.json')
        for mid,h,l,seed in model_ids():
            packed=np.load(R/'saved_predictions'/f'{mid}_T{t}.npz');bits=np.unpackbits(packed['bits'],axis=1,count=t,bitorder='little');bad=np.unpackbits(packed['invalid'],axis=1,count=t,bitorder='little').astype(bool)
            for j,a in enumerate(meta):
                old=lookup[(mid,t,a['id'])];state=0;mask=x[j]==3;wrong=mask&((bits[j]!=y[j])|bad[j]);positions=np.flatnonzero(wrong);ret=np.arange(t)>=9;retpos=np.flatnonzero(wrong&ret);imm=np.zeros(t,bool);imm[1:]=(x[j,:-1]<2)&mask[1:]
                expected=dict(R_count=int(mask.sum()),R_correct=int((mask&~wrong).sum()),sequence_exact=not bool(wrong.any()),first_error_position=int(positions[0]) if len(positions) else None,first_retention_R_error_position=int(retpos[0]) if len(retpos) else None,retention_R_count=int((mask&ret).sum()),retention_R_correct=int((mask&ret&~wrong).sum()),immediate_rewrite_R_count=int(imm.sum()),immediate_rewrite_R_correct=int((imm&~wrong).sum()),nonfinite_positions=int(bad[j].sum()))
                for k,v in expected.items():assert old[k]==('' if v is None else str(v)),(mid,t,a['id'],k,old[k],v)
                r=dict(model=mid,H=h,L=l,seed=seed,T=t,**a,**expected);regraded.append(r)
                if len(positions):
                    p=int(positions[0]);last=np.flatnonzero(x[j,:p+1]<2);failures.append(dict(model=mid,T=t,id=a['id'],category=a['category'],position=p,true_value=int(y[j,p]),prediction=int(bits[j,p]),invalid=bool(bad[j,p]),last_write=int(last[-1]) if len(last) else None,local_symbols=x[j,max(0,p-8):min(t,p+9)].tolist(),dataset=f['path'],predictions=f'saved_predictions/{mid}_T{t}.npz'))
    assert len(regraded)==40*7*68;csvout('failure_cases.csv',failures)
    A=read('PREDICTIONS_A_FROZEN.json');B=read('PREDICTIONS_B_FROZEN.json');aa={r['model']:r for r in A['models']};bb={r['model']:r for r in B['models']};comparison=[];category=[];units=[];timing=[];attractors=[]
    for mid,h,l,seed in model_ids():
        rr=[r for r in regraded if r['model']==mid];long=[r for r in rr if r['T'] in (2048,8192,65000)];actual=all(r['R_correct']==r['R_count'] for r in long);pa=aa[mid]['predict_success'];pb=bb[mid]['predict_success']
        comparison.append(dict(model=mid,H=h,L=l,seed=seed,A_success=pa,B_success=pb,actual_success=actual,A_correct=pa==actual,B_correct=pb==actual,B_short_R_correct=bb[mid]['correct'],B_short_R_count=bb[mid]['R_count'],long_R_correct=sum(r['R_correct'] for r in long),long_R_count=sum(r['R_count'] for r in long),selected_update=read('CHECKPOINTS_FROZEN.json')['models'][mid]['selected_update']))
        for t in LENGTHS:
            for cat in ('random','long_N','long_R','rewrite','mixed'):
                zz=[r for r in rr if r['T']==t and r['category']==cat];correct=sum(r['R_correct'] for r in zz);n=sum(r['R_count'] for r in zz)
                category.append(dict(model=mid,H=h,L=l,seed=seed,T=t,category=cat,unique_or_item_count=len(zz),nominal_count=sum(r['multiplicity'] for r in zz),R_correct=correct,R_count=n,R_accuracy=correct/n,sequence_correct=sum(r['sequence_exact'] for r in zz),rewrite_R_correct=sum(r['immediate_rewrite_R_correct'] for r in zz),rewrite_R_count=sum(r['immediate_rewrite_R_count'] for r in zz)))
        for neutral in ('N','R'):
            attractors.append(dict(model=mid,H=h,L=l,seed=seed,neutral=neutral,two_attractors=aa[mid]['two_attractors'][neutral]))
            for v in (0,1):
                unit=next(u for u in aa[mid]['units'] if u['neutral']==neutral and u['value']==v);zz=[r for r in rr if r['category']=='long_'+neutral and r['value']==v];truth=all(r['retention_R_correct']==r['retention_R_count'] for r in zz if r['T'] in (2048,8192,65000));all7=all(r['retention_R_correct']==r['retention_R_count'] for r in zz)
                units.append(dict(model=mid,H=h,L=l,seed=seed,neutral=neutral,value=v,A_holds=unit['predict_holds'],actual_holds_main_lengths=truth,correct_main_lengths=unit['predict_holds']==truth,actual_holds_all_lengths=all7,correct_all_lengths=unit['predict_holds']==all7,predicted_break_step=unit['predicted_break_step']))
                for j in range(8):
                    items=sorted([r for r in zz if r['j']==j],key=lambda r:r['T']);ok=[r['retention_R_correct']==r['retention_R_count'] for r in items];prediction=unit['predicted_break_step'];holds=unit['predict_holds'];maxrow=items[-1];badmax=maxrow['first_retention_R_error_position'];nonmono=any(not ok[i] and any(ok[i+1:]) for i in range(len(ok)-1));right=all(ok);lower=upper=observed=error=None;inside=None
                    if neutral=='R':
                        observed=None if badmax is None else badmax-8
                        for r in items:
                            seen=r['first_retention_R_error_position'];seen=None if seen is None else seen-8;assert seen==(observed if observed is not None and observed<=r['T']-9 else None),'R prefix consistency failed'
                        if not holds and prediction is not None and observed is not None:error=prediction-observed
                    elif not right and not nonmono:
                        firstbad=ok.index(False);upper=items[firstbad]['T']-10;lower=items[firstbad-1]['T']-10 if firstbad else None
                        if not holds and prediction is not None:inside=(lower is None or prediction>lower) and prediction<=upper
                    timing.append(dict(model=mid,H=h,L=l,seed=seed,neutral=neutral,value=v,j=j,A_holds=holds,predicted_break_step=prediction,observed_R_first_failure_step=observed,N_lower_open=lower,N_upper_closed=upper,N_left_censored=neutral=='N' and upper is not None and lower is None,N_interval_prediction_inside=inside,nonmonotonic=nonmono,unobserved_at_65000=right,unpredicted_failure=holds and not right,predicted_failure_without_finite_step=not holds and prediction is None,predicted_minus_observed=error,absolute_error=None if error is None else abs(error)))
    a_hits=sum(r['A_correct'] for r in comparison);b_hits=sum(r['B_correct'] for r in comparison);success=sum(r['actual_success'] for r in comparison);c_success=success;c_failure=40-success;c=max(c_success,c_failure);a_positive=sum(r['A_success'] for r in comparison);reasons=[]
    if success in (0,40):reasons.append('識別不能：実際の結果が全モデル同一')
    if min(a_positive,40-a_positive)<5:reasons.append('件数不足：Aの成功/失敗予測の片方が5件未満')
    margin=a_hits-max(b_hits,c);verdict='判定不能' if reasons else '支持' if margin>=2 else '不支持' if margin<=0 else '判定不能'
    if not reasons and margin==1:reasons.append('優位差が1件で規定の2件に未達')
    csvout('model_comparison.csv',comparison);csvout('category_results.csv',category);csvout('retention_units.csv',units);csvout('break_step_analysis.csv',timing);csvout('attractor_conditions.csv',attractors)
    groups=[]
    for h in (32,64):
        for l in (8,16):
            cc=[r for r in comparison if r['H']==h and r['L']==l];groups.append(dict(H=h,L=l,models=10,actual_success=sum(r['actual_success'] for r in cc),A_success=sum(r['A_success'] for r in cc),A_correct=sum(r['A_correct'] for r in cc),B_correct=sum(r['B_correct'] for r in cc),N_two=sum(r['two_attractors'] for r in attractors if r['H']==h and r['L']==l and r['neutral']=='N'),R_two=sum(r['two_attractors'] for r in attractors if r['H']==h and r['L']==l and r['neutral']=='R')))
    csvout('condition_summary.csv',groups);errs=[r['predicted_minus_observed'] for r in timing if r['neutral']=='R' and r['predicted_minus_observed'] is not None];abses=[abs(x) for x in errs];eligible=[r for r in timing if r['neutral']=='N' and r['N_interval_prediction_inside'] is not None]
    metrics=dict(models=40,A_hits=a_hits,B_hits=b_hits,C_always_success_hits=c_success,C_always_failure_hits=c_failure,C_best_hits=c,actual_success=success,A_predict_success=a_positive,margin_over_best_comparator=margin,verdict=verdict,indeterminate_reasons=reasons,unit_hits_main_lengths=sum(r['correct_main_lengths'] for r in units),unit_hits_all_lengths=sum(r['correct_all_lengths'] for r in units),unit_denominator=160,R_break_pairs=len(errs),R_median_signed_error=None if not errs else statistics.median(errs),R_median_absolute_error=None if not abses else statistics.median(abses),N_intervals_compared=len(eligible),N_intervals_hit=sum(r['N_interval_prediction_inside'] for r in eligible),nonmonotonic_items=sum(r['nonmonotonic'] for r in timing if r['neutral']=='N'),unobserved_items=sum(r['unobserved_at_65000'] for r in timing),unpredicted_failure_items=sum(r['unpredicted_failure'] for r in timing),saved_items_regraded=len(regraded),failed_items=len(failures),audit_wall_seconds=time.perf_counter()-t0)
    dump('metrics.json',metrics)
    for k,n in (('A','PREDICTIONS_A'),('B','PREDICTIONS_B')):assert sha(R/f'{n}_FROZEN.json')==read(f'{n}_SHA256.json')['SHA256']
    for a in read('CHECKPOINTS_FROZEN.json')['models'].values():assert sha(R/a['path'])==a['SHA256']
    dump('OUTPUT_AUDIT.json',dict(passed=True,independent_saved_prediction_regrade_items=len(regraded),frozen_predictions_unchanged=True,selected_checkpoints_unchanged=True,R_prefix_consistency=True,utc=now()))
    print('ANALYSIS',metrics,flush=True)
