"""R4 from packed outputs and input symbols, no model loading or inference."""
from common import *
import time
T=[32,64,128,512,2048,8192,65000];MAIN=[2048,8192,65000]
def labels(x):
    # Last preceding write index, independent of the original evaluator.
    idx=np.where(x<2,np.arange(x.shape[1])[None,:],-1)
    idx=np.maximum.accumulate(idx,axis=1)
    return np.where(idx>=0,np.take_along_axis(x,np.maximum(idx,0),axis=1),0).astype(np.uint8)
def recompute():
    c=Comparisons();models=[];details=[];audit=[];groupstats={}
    for study in ['G1','FF1','FF1R']:
        root=W/('IronMath_G1' if study=='G1' else 'IronMath_FF1' if study=='FF1' else 'IronMath_FF1R')
        if study=='G1':frozen=load(root/'main_PREDICTIONS_FROZEN.json')['models'];assets={r['model']:load(root/'assets'/f"{r['model']}.json") for r in frozen}
        else:
            aa=load(root/'PREDICTIONS_A_FROZEN.json')['models'];bb={r['model']:r for r in load(root/'PREDICTIONS_B_FROZEN.json')['models']}
            frozen=[dict(r,A=r['predict_success'],B=bb[r['model']]['predict_success'],architecture='residual_gru') for r in aa]
            assets=load(root/'CHECKPOINTS_FROZEN.json')['models']
        inputs={}
        for t in T:
            p=root/('main_long_data' if study=='G1' else 'long_data')/f'T{t}.npz'
            with np.load(p,allow_pickle=False) as z:x=z['symbols']
            inputs[t]=(x==3,labels(x),len(x));assert len(x)==68
        dd=None if study=='G1' else load(root/'diagnostic_details.json')
        # Some studies wrap the list in a named record.
        if isinstance(dd,dict):dd=dd.get('details',dd.get('records',dd))
        for r in frozen:
            mid=r['model'];asset=assets[mid];actualsha=sha(root/asset['path'])
            audit.append(dict(study=study,model=mid,checkpoint_SHA256=actualsha,expected_SHA256=asset['SHA256'],match=actualsha.lower()==asset['SHA256'].lower()))
            assert audit[-1]['match']
            selected=load(root/'diagnostic_records'/f'{mid}.json')['details'] if study=='G1' else [d for d in dd if d['model']==mid]
            assert len(selected)==16,(mid,len(selected))
            D=all(d['converged'] for d in selected);finite=True
            for neutral in ['N','R']:
                with np.load(root/'diagnostic_traces'/f'{mid}_{neutral}.npz',allow_pickle=False) as z:
                    finite &= all(np.isfinite(z[k]).all() for k in ['endpoints','movements','margins'])
                    if 'state_finite' in z:finite &= bool(z['state_finite'].all() and z['logit_finite'].all())
                    D &= bool(np.max(z['movements'][-100:])<1e-6)
            D=bool(D and finite)
            if 'D' in r:c.add(study,mid+'/D',r['D'],D,'frozen predictions','diagnostic traces and convergence')
            m=dict(study=study,model=mid,architecture=r['architecture'],L=r['L'],seed=r['seed'],A=bool(r['A']),B=bool(r['B']),D=D,lengths={},nonfinite=False)
            for t in T:
                mask,y,n=inputs[t]
                with np.load(root/'saved_predictions'/f'{mid}_T{t}.npz',allow_pickle=False) as z:
                    bits=np.unpackbits(z['bits'],axis=1,count=t,bitorder='little');invalid=np.unpackbits(z['invalid'],axis=1,count=t,bitorder='little').astype(bool)
                wrong=((bits!=y)|invalid)&mask
                finitewrong=(bits!=y)&~invalid&mask
                m['lengths'][str(t)]=not bool(wrong.any());m['nonfinite'] |= bool(invalid.any())
                details.append(dict(study=study,model=mid,length=t,unique_sequences=n,total_steps=n*t,R_positions=int(mask.sum()),wrong_R=int(wrong.sum()),finite_wrong_R=int(finitewrong.sum()),nonfinite_positions=int(invalid.sum()),success=m['lengths'][str(t)]))
            m['actual3']=all(m['lengths'][str(t)] for t in MAIN);m['actual7']=all(m['lengths'].values())
            m['finite_failure']=not m['actual3'] and not m['nonfinite'];models.append(m)
            print(study,mid,'rescored',flush=True)
        subset=[m for m in models if m['study']==study]
        groups=(['nonlinear','residual_gru','gru','lstm','rnn','linear'] if study=='G1' else ['all'])
        old=load(root/('g1_metrics.json' if study=='G1' else 'metrics.json' if study=='FF1' else 'ff1r_metrics.json'))
        for g in groups:
            ms=[m for m in subset if g=='all' or (g=='nonlinear' and m['architecture']!='linear') or m['architecture']==g]
            for definition in (['actual3','actual7'] if study=='FF1R' else ['actual3']):
                y=np.array([m[definition] for m in ms]);preds={k:np.array([m[k] for m in ms]) for k in ['A','B','D']};preds.update(C_all_success=np.ones(len(ms),bool),C_all_failure=np.zeros(len(ms),bool))
                out={k:score(p,y) for k,p in preds.items()};out['paired_AD']=paired(preds['A'],preds['D'],y,81000)
                out['actual_success']=int(y.sum());out['A_success_predictions']=int(preds['A'].sum());ff=[i for i,m in enumerate(ms) if not y[i] and not m['nonfinite']]
                out['finite_failures']=len(ff);out['finite_correct_A']=sum(not ms[i]['A'] for i in ff);out['finite_correct_D']=sum(not ms[i]['D'] for i in ff)
                out['H2']='判定不能' if len(ff)<5 else '支持' if out['finite_correct_A']>out['finite_correct_D'] else '不支持'
                ind=len(set(y.tolist()))<2 or min(out['A_success_predictions'],len(ms)-out['A_success_predictions'])<(10 if study=='G1' else 5)
                hits={k:v['hits'] for k,v in out.items() if isinstance(v,dict) and 'hits' in v}
                out['H1']='判定不能' if ind else ('支持' if hits['A']>=max(hits[k] for k in ['B','C_all_success','C_all_failure'])+2 else '不支持') if study=='FF1' else ('支持' if all(hits['A']>hits[k] for k in ['B','C_all_success','C_all_failure','D']) else '不支持')
                groupstats[f'{study}/{g}/{definition}']=out
                if study=='G1':
                    ref=old['pooled_nonlinear'] if g=='nonlinear' else old['by_architecture'][g]
                    for method in preds:
                        rr=next(z for z in ref['methods'] if z['method']==method)
                        for key in ['hits','TP','FP','TN','FN']:c.add(study,f'{g}/{method}/{key}',rr[key],out[method][key],'g1_metrics.json', 'input symbols + packed bits + invalid masks')
                    for ok,nk in [('A_only_correct','A_only_correct'),('D_only_correct','other_only_correct'),('McNemar_exact_two_sided_p','McNemar_exact_two_sided_p'),('accuracy_difference','accuracy_difference')]:c.add(study,g+'/'+ok,ref[ok],out['paired_AD'][nk],'g1_metrics.json','independent paired calculation')
                    c.add(study,g+'/bootstrap95_CI',ref['bootstrap95_CI'],[out['paired_AD']['CI_low'],out['paired_AD']['CI_high']],'g1_metrics.json','paired bootstrap seed81000')
                elif study=='FF1':
                    for k in ['A','B','C_all_success','C_all_failure']:c.add(study,k+'/hits',old[dict(A='A_hits',B='B_hits',C_all_success='C_always_success_hits',C_all_failure='C_always_failure_hits')[k]],out[k]['hits'],'metrics.json','packed outputs')
                    c.add(study,'H1',old['verdict'],out['H1'],'metrics.json','original FF1 decision, no registered H2')
                    dr=rows(W/'IronMath_FF1A/ff1a_predictor_comparison.csv');c.add(study,'D/hits',next(int(z['hits']) for z in dr if z['method']=='D'),out['D']['hits'],'FF1A predictor comparison','raw diagnostic traces')
                else:
                    for k in ['H1','H2','actual_success','finite_failures']:c.add(study,definition+'/'+k,old[k],out[k],'ff1r_metrics.json (historical all7; 3-length comparison)', 'packed outputs + frozen predictions')
        if study=='G1':
            pooled=groupstats['G1/nonlinear/actual3'];p1=pooled['H1']=='支持' and pooled['paired_AD']['McNemar_exact_two_sided_p']<.05 and all(groupstats[f'G1/{g}/actual3']['D']['hits']<=groupstats[f'G1/{g}/actual3']['A']['hits'] for g in groups[1:-1])
            p2=all(not m['actual3'] and not m['A'] for m in subset if m['architecture']=='linear')
            c.add(study,'P1',old['P1'],'支持' if p1 else '不支持','g1_metrics.json','independent decision');c.add(study,'P2',old['P2'],'支持' if p2 else '不支持','g1_metrics.json','linear raw outputs')
            ts=[]
            for m in subset:
                if m['architecture']=='linear':
                    with np.load(root/'linear_analytic'/f"{m['model']}_N.npz",allow_pickle=False) as z:ts.extend((-1/np.log(z['lambdas'].astype(np.float64))).tolist())
            dump('linear_tau_recomputed.json',dict(components=len(ts),median=float(np.median(ts)),maximum=max(ts)))
    table('r4_memory_recomputation.csv',c.data);table('memory_length_scores.csv',details);table('checkpoint_hash_checks.csv',audit);dump('memory_models.json',models);dump('memory_group_statistics.json',groupstats)
if __name__=='__main__':recompute()
