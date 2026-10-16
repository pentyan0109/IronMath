from common import *
def main():
    assert read(R/'E3_independent_replication/COMPLETE.json')['models']==20
    ms=rows(R/'E3_independent_replication/E3_MODEL_RESULTS.csv');test=rows(R/'E3_independent_replication/E3_TEST_RESULTS.csv');pairs=rows(R/'E3_independent_replication/E3_PAIR_RESULTS.csv');init=[]
    for seed in range(100,110):
        rr=[r for r in ms if int(r['initialization_seed'])==seed];assert len(rr)==2;rr.sort(key=lambda r:int(r['data_condition']));quality=[r for r in rr if r['training_achieved']=='True'];indicator=[int(int(r['strict_reproduced_pairs'])>0) for r in rr]
        row=dict(initialization_seed=seed,conditions=2,qualified_conditions=len(quality),strict_condition0=indicator[0],strict_condition1=indicator[1],mechanism_model_rate=np.mean(indicator),paired_qualified_reproduction=len(quality)==2 and all(indicator),condition_difference=indicator[1]-indicator[0],qualified_rate=np.mean([int(int(r['strict_reproduced_pairs'])>0) for r in quality]) if quality else None)
        for op in ['ADD','SUB']:
            for width in [8,64,256,2048]:row[f'{op}_{width}_accuracy']=float(np.mean([float(t['accuracy']) for t in test if t['model'] in [r['model'] for r in rr] and t['op']==op and int(t['width'])==width]))
            row[f'{op}_short_minus_long']=row[f'{op}_8_accuracy']-row[f'{op}_2048_accuracy']
        init.append(row)
    stats=[]
    fields=['mechanism_model_rate','condition_difference','qualified_rate']+[f'{op}_{w}_accuracy' for op in ['ADD','SUB'] for w in [8,64,256,2048]]+[f'{op}_short_minus_long' for op in ['ADD','SUB']]
    for field in fields:
        v=np.array([r[field] for r in init if r[field] is not None],float);ci=cluster_ci(v,99000) if len(v) else [None,None]
        stats.append(dict(metric=field,initialization_clusters=len(v),mean=float(v.mean()) if len(v) else None,sample_SD=float(v.std(ddof=1)) if len(v)>1 else None,CI95_low=ci[0],CI95_high=ci[1],bootstrap_samples=10000,bootstrap_seed=99000,unit='initialization; paired data conditions retained'))
    eligible=sum(r['qualified_conditions']==2 for r in init);n=sum(r['paired_qualified_reproduction'] for r in init);judgment='多数の独立初期化で再現' if n>=6 else '一部の初期化で再現' if n>=2 else '特定の初期化に限られる' if n==1 else '元の結果自体の再現が不十分' if eligible>=5 else '学習品質により判定不能'
    table('E3_independent_replication/E3_INITIALIZATION_RESULTS.csv',init);table('E3_independent_replication/E3_CLUSTER_STATISTICS.csv',stats);dump('E3_independent_replication/AGGREGATE.json',dict(utc=utc(),models=len(ms),qualified_models=sum(r['training_achieved']=='True' for r in ms),paired_qualified_initializations=eligible,paired_reproduced_initializations=n,judgment=judgment,shared_pairs=sum(r['shared']=='True' for r in pairs),distinct_pairs=sum(r['distinct']=='True' for r in pairs),unclassified_pairs=sum(r['both_converged']=='False' for r in pairs),strict_pairs=sum(r['strict_mechanism_reproduced']=='True' for r in pairs)))
if __name__=='__main__':
    try:main()
    except Exception as e:stage_failure('aggregate',e);raise
