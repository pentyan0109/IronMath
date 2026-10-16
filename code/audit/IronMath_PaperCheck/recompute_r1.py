"""Post hoc baselines from independently rescored stored outputs only."""
from common import *
def run():
    rules=load(R/'R1_RULES_SHA256.json');assert sha(R/'R1_RULES.md')==rules['SHA256']
    models=load(R/'memory_models.json');baselines=[];comparisons=[];conclusions=[];steps=[]
    for study in ['FF1','FF1R','G1']:
        groups=['all'] if study!='G1' else ['nonlinear','residual_gru','gru','lstm','rnn','linear']
        for group in groups:
            ms=[m for m in models if m['study']==study and (group=='all' or (group=='nonlinear' and m['architecture']!='linear') or m['architecture']==group)]
            y=np.array([m['actual3'] for m in ms]);a=np.array([m['A'] for m in ms])
            for ell in [32,64,128,512]:
                lengths=[t for t in [32,64,128,512] if t<=ell]
                for method in ['B_cum','B_only']:
                    used=lengths if method=='B_cum' else [ell]
                    p=np.array([all(m['lengths'][str(t)] for t in used) for m in ms])
                    s=score(p,y);baselines.append(dict(experiment=study,structure=group,ell=ell,method=method,**s,predicted_success=int(p.sum()),actual_success=int(y.sum()),steps_per_model=68*sum(used),total_group_steps=len(ms)*68*sum(used),post_hoc=True))
                    comparison=paired(a,p,y,82000);comparisons.append(dict(experiment=study,structure=group,ell=ell,baseline=method,A_hits=int((a==y).sum()),baseline_hits=s['hits'],**comparison))
                    if study=='G1' and group=='nonlinear' and method=='B_cum' and ell in [128,512]:
                        if comparison['McNemar_exact_two_sided_p']>=.05:sentence=f'診断Aと長さ{ell}までのテストは、同程度の予測力を持つ'
                        elif int((a==y).sum())>s['hits']:sentence=f'診断Aは、長さ{ell}までのテストより正確に予測した'
                        else:sentence=f'長さ{ell}までのテストは、診断Aより正確に予測した'
                        conclusions.append(dict(sentence=sentence,**comparisons[-1]))
    table('r1_baselines.csv',baselines);table('r1_comparison.csv',comparisons);dump('r1_conclusions.json',conclusions)
    dump('R1_AGGREGATION_RECORD.json',dict(rules_SHA256=rules['SHA256'],rules_mtime_ns=(R/'R1_RULES.md').stat().st_mtime_ns,source='memory_models.json from independent raw bit rescoring',new_training=False,new_inference=False,bootstrap_unit='model',bootstrap_seed=82000))
if __name__=='__main__':run()
