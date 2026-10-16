"""Independent saved-output scorer; no model calls and no training."""
from g1_utils import *
from data import sequential_labels

def audit_outputs(assets):
    started=time.perf_counter();ds=read('main_LONG_DATA_FROZEN.json');frozen=read('main_PREDICTIONS_FROZEN.json');nfmodels={};regraded=0;failures=[];models=[]
    assert sha(R/'main_PREDICTIONS_FROZEN.json')==read('main_PREDICTIONS_SHA256.json')['SHA256']
    assert ds['prediction_SHA256']==sha(R/'main_PREDICTIONS_FROZEN.json')
    freeze_ns=(R/'main_PREDICTIONS_FROZEN.json').stat().st_mtime_ns
    summary={a['model']:dict(correct=True,nonfinite=0,finite_wrong=0,R_correct=0,R_count=0) for a in assets}
    item_counts={a['model']:0 for a in assets}
    for file in ds['files']:
        T=file['T'];assert sha(R/file['path'])==file['SHA256'];assert (R/file['path']).stat().st_mtime_ns>=freeze_ns
        with np.load(R/file['path']) as z:x=z['symbols']
        y=sequential_labels(x);meta=read(f'main_long_data/T{T}_metadata.json')
        for a in assets:
            mid=a['model'];record=read(f'evaluation_records/{mid}.json');old={r['id']:r for r in record['items'] if r['T']==T};path=R/'saved_predictions'/f'{mid}_T{T}.npz'
            with np.load(path) as z:bits=np.unpackbits(z['bits'],axis=1,count=T,bitorder='little');bad=np.unpackbits(z['invalid'],axis=1,count=T,bitorder='little').astype(bool)
            assert bits.shape==bad.shape==x.shape
            prediction_hash=sha(path)
            for j,md in enumerate(meta):
                row=old[md['id']];assert prediction_hash==row['prediction_SHA256'];mask=x[j]==3;wrong=mask&((bits[j]!=y[j])|bad[j]);ix=np.flatnonzero(wrong);ni=np.flatnonzero(bad[j]);ret=np.arange(T)>=9;imm=np.zeros(T,bool);imm[1:]=(x[j,:-1]<2)&mask[1:]
                expected=dict(R_correct=int((mask&~wrong).sum()),R_count=int(mask.sum()),sequence_exact=not bool(wrong.any()),first_error_position=int(ix[0]) if len(ix) else None,nonfinite_positions=int(bad[j].sum()),first_nonfinite_position=int(ni[0]) if len(ni) else None,finite_wrong_R=int((mask&(bits[j]!=y[j])&~bad[j]).sum()),nonfinite_R=int((mask&bad[j]).sum()),retention_R_correct=int((mask&ret&~wrong).sum()),retention_R_count=int((mask&ret).sum()),immediate_rewrite_R_correct=int((mask&imm&~wrong).sum()),immediate_rewrite_R_count=int((mask&imm).sum()))
                assert all(row[k]==v for k,v in expected.items()),(mid,T,md['id'])
                regraded+=1;item_counts[mid]+=1;ss=summary[mid];ss['nonfinite']+=expected['nonfinite_positions'];ss['finite_wrong']+=expected['finite_wrong_R']
                if T in [2048,8192,65000]:ss['correct']=ss['correct'] and not bool(wrong.any());ss['R_correct']+=expected['R_correct'];ss['R_count']+=expected['R_count']
                if len(ix):
                    p=int(ix[0]);last=np.flatnonzero(x[j,:p+1]<2)
                    failures.append(dict(model=mid,architecture=a['architecture'],L=a['L'],seed=a['seed'],T=T,id=md['id'],category=md['category'],position=p,correct_value=int(y[j,p]),prediction=int(bits[j,p]),nonfinite_logit=bool(bad[j,p]),last_write=int(last[-1]) if len(last) else None,local_symbols=x[j,max(0,p-8):min(T,p+9)].tolist(),saved_prediction=path.relative_to(R).as_posix()))
    for a in assets:
        mid=a['model'];assert item_counts[mid]==7*68;assert sha(R/a['path'])==a['SHA256'];d=read(f'diagnostic_records/{mid}.json')
        assert sha(R/'diagnostic_records'/f'{mid}.json')==frozen['diagnostic_record_SHA256'][mid]
        assert d['A']==all(all(r['converged'] and r['spectral_radius'] is not None and r['spectral_radius']<1 and r['readout']==value for r in d['details'] if r['value']==value and r['neutral']==neutral) for neutral in ['N','R'] for value in [0,1])
        conv=True;nonfinite=False
        for neutral in ['N','R']:
            with np.load(R/'diagnostic_traces'/f'{mid}_{neutral}.npz') as z:
                conv=conv and bool(np.isfinite(z['endpoints']).all() and np.isfinite(z['movements'][-100:]).all() and z['movements'][-100:].max()<1e-6)
                nonfinite=nonfinite or not all(np.isfinite(z[k]).all() for k in ['endpoints','movements','margins']) or not bool(z['state_finite'].all() and z['logit_finite'].all())
        assert d['D']==(conv and not nonfinite)
        s=summary[mid];models.append(dict(model=mid,architecture=a['architecture'],L=a['L'],seed=a['seed'],actual_success=s['correct'],classification='発散型' if s['nonfinite'] else ('成功型' if s['correct'] else '有限誤答型'),nonfinite_logit_positions=s['nonfinite'],finite_wrong_R_all_lengths=s['finite_wrong'],main_R_correct=s['R_correct'],main_R_count=s['R_count'],**{k:d[k] for k in ['A','B','C_all_success','C_all_failure','D','all16_converged','diagnostic_nonfinite']}))
    assert regraded==len(assets)*7*68
    csvout('g1_failure_cases.csv',failures);csvout('g1_results.csv',models)
    # Separate exact-test recomputation via integer combinatorics, not the reporting helper.
    import math
    nonlinear=[m for m in models if m['architecture']!='linear'];b=sum(m['A']==m['actual_success'] and m['D']!=m['actual_success'] for m in nonlinear);c=sum(m['D']==m['actual_success'] and m['A']!=m['actual_success'] for m in nonlinear)
    n=b+c;lower=sum(math.comb(n,i) for i in range(min(b,c)+1));exact=min(1.,2*lower/(2**n)) if n else 1.
    dump('SAVED_OUTPUT_AUDIT.json',dict(utc=now(),passed=True,models=len(assets),independent_regraded_items=regraded,registered_expected_items=len(assets)*7*68,diagnostic_A_D_recomputed=True,frozen_predictions_unchanged=True,selected_checkpoints_SHA_verified=True,sequence_generation_after_all_freeze_verified=True,independent_P1_McNemar=dict(A_only_correct=b,D_only_correct=c,p=exact),wall_seconds=time.perf_counter()-started))
    print('INDEPENDENT SAVED OUTPUT AUDIT PASS',regraded,flush=True)
