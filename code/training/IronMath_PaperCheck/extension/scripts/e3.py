from core import *
from e1 import diagnose
import gzip
def freeze():
    assert read(R/'E2_causal_intervention/COMPLETE.json')['all_conditions_saved']
    if (R/'E3_independent_replication/FROZEN.json').exists():return
    md('E3_independent_replication/E3_RULES.md','''# E3：独立初期化の固定規則
初期化100〜109、各2データ提示条件c0/c1。initは両条件で同一重み、データ提示rngは101000+2*(init−100)+c。元V054の固定Train母集団・固定Validation/Testを使う。異なる提示系列であり、独立生成コーパスではない。各初期化の2条件を独立な初期化に数えない。
モデルは同じ1000334パラメータ残差GRU。AdamW lr.002 wd.01 clip1、batchADD8/SUB8、1200更新、3回の元同様の更新なしwarmup。200更新ごと等演算Validation CEの最小（同点は早い）を選択。訓練CEはADD/SUB別平均の平均。全20学習完了後のみTest評価。各新initのSHAが旧0/1/2および別initと異なり、同init内2条件では一致することを検査。
学習達成は選択checkpointのValidation整数完全一致率がADD/SUB両方95%以上。Testで達成条件やcheckpointを選ばない。固定Test8/64/256/2048は各512問、両演算。hold評価は旧V054Tのk16/64/256/1024、全16prefix/suffix文脈、12保持組の768問/モデル。first_hold_errorはhold内の最初の誤りの位置+1、prefixの誤りを別記。k間での変化と中央値/min/maxを保存。
固定点診断は元12文脈、20000step、最後100移動量max<1e-6、rho<1、2終点距離<1e-3を共有、>=1e-3を別終点。全4つの両carry保持入力対/モデルを分類。rho未計算・非収束を「共有しない」と断定せず未分類と記録する。
厳密な共有過渡機構の再現pairは、両端点収束・rho<1・距離<1e-3・少なくとも片側の終点読み出し誤り・その側のhold内誤り観測、を全て満たすもの。候補共有pair数と厳密再現pair数を分離。観測first_negativeによる時刻予測と実際のhold誤りは別の量で、k>=256の中央値との差を記録。長さ1024を越えて壊れない場合は打切り。
全20とValidation達成部分集合を別記。init内2条件平均を作り、初期化10クラスタでbootstrap10000回default_rng99000、percentile95%区間。機構率、初期化ごとの2条件差、短Test8−長Test2048の低下を効果量として報告する。
初期化の再現は達成した両条件で少なくとも1厳密再現pairを持つこと。6以上/10なら多数、2〜5は一部、1は特定、0で達成初期化5以上なら再現不十分、それ未満は学習品質により判定不能。単一条件のみの再現は別に保存。条件は結果で変更しない。
全体上限7200秒・peak2GiB。障害再開は200更新ごとのresume checkpointのモデル/optimizer/rng/torch_rngと提示履歴から。未完了分は成功扱いしない。保存済み最終runを再学習しない。追加seed・延長なし。
''')
    dump('E3_independent_replication/FROZEN.json',dict(utc=utc(),rules_SHA256=sha(R/'E3_independent_replication/E3_RULES.md'),code_SHA256=sha(Path(__file__)),initializations=list(range(100,110)),conditions=[0,1],models=20,train_data_SHA256={p.name:sha(p) for p in (W/'IronMath_V053/data').glob('*.json')}))
def loss(z,y):return (F.cross_entropy(z[:8].flatten(0,1),y[:8].flatten(),ignore_index=-100)+F.cross_entropy(z[8:].flatten(0,1),y[8:].flatten(),ignore_index=-100))/2
def train_one(d,seed,condition):
    folder=R/f'E3_independent_replication/checkpoints/i{seed}_c{condition}';folder.mkdir(exist_ok=True,parents=True)
    if (folder/'complete.json').exists():return read(folder/'complete.json')
    setup(seed);base=Residual();initial=modelhash(base);m=torch.jit.script(base);opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01);rng=random.Random(101000+2*(seed-100)+condition);step=0;best=float('inf');beststep=0;order=[];logs=[];trainwall=traincpu=0.
    if (folder/'resume.pt').exists():
        cp=torch.load(folder/'resume.pt',weights_only=False,map_location='cpu');m.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);rng.setstate(cp['rng']);torch.set_rng_state(cp['torch_rng']);step=cp['step'];best=cp['best'];beststep=cp['beststep'];order=cp['order'];logs=cp['logs'];trainwall=cp['trainwall'];traincpu=cp['traincpu']
    else:
        for _ in range(3):m.zero_grad();m(torch.zeros(16,9,3)).square().mean().backward()
        assert modelhash(m)==initial
    def save_resume():torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng.getstate(),torch_rng=torch.get_rng_state(),step=step,best=best,beststep=beststep,order=order,logs=logs,trainwall=trainwall,traincpu=traincpu),folder/'resume.pt')
    while step<1200:
        if time.time()-read(R/'audit/STARTED.json')['start_epoch']>7200:save_resume();raise RuntimeError('Fixed total resource ceiling reached; preserve unfinished model')
        rr=[]
        for op in ['ADD','SUB']:
            pool=d[f'{op}_Train']
            for _ in range(8):
                a=dict(pool[rng.randrange(len(pool))])
                if op=='ADD' and rng.randrange(2):a['a'],a['b']=a['b'],a['a']
                rr.append(a);order.append(f"{a['id']}:{a['a']}:{a['b']}")
        x,y,_=batch(rr);tt=time.perf_counter();cc=time.process_time();m.train();opt.zero_grad(set_to_none=True);l=loss(m(x),y)
        if not torch.isfinite(l):save_resume();raise RuntimeError('Nonfinite training loss; preserve and stop')
        l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();trainwall+=time.perf_counter()-tt;traincpu+=time.process_time()-cc;step+=1
        if step%200==0:
            m.eval();vl={};fit={}
            for op in ['ADD','SUB']:
                lv,pr=evaluate(m,d[f'{op}_Validation']);vl[op]=lv;fit[op]=sum(p['correct'] for p in pr)/len(pr)
            score=sum(vl.values())/2
            if score<best:best=score;beststep=step;torch.save(dict(model=m.state_dict(),seed=seed,condition=condition,selected_update=step,validation_loss=score,initial_SHA256=initial),folder/'best_validation.pt')
            logs.append(dict(seed=seed,condition=condition,update=step,minibatch_loss=float(l.detach()),validation_loss=score,Validation_ADD_exact=fit['ADD'],Validation_SUB_exact=fit['SUB'],selected_update=beststep,train_wall_seconds=trainwall,train_CPU_seconds=traincpu));save_resume();table(f'E3_independent_replication/checkpoints/i{seed}_c{condition}/training_log.csv',logs);memory();print('TRAIN',seed,condition,step,'val',round(score,6),'wall',round(trainwall,1),flush=True)
    torch.save(dict(model=m.state_dict(),seed=seed,condition=condition,update=step),folder/'final.pt');selected,cp=load_model(folder/'best_validation.pt');valid={};fits=[]
    for op in ['ADD','SUB']:
        lv,pr=evaluate(torch.jit.script(selected),d[f'{op}_Validation']);valid[op]=sum(p['correct'] for p in pr)/len(pr)
    out=dict(model=f'i{seed}_c{condition}',initialization_seed=seed,data_condition=condition,data_seed=101000+2*(seed-100)+condition,parameters=sum(p.numel() for p in m.parameters()),updates=1200,selected_update=beststep,initial_SHA256=initial,selected_checkpoint_SHA256=sha(folder/'best_validation.pt'),data_presentation_SHA256=hashlib.sha256('\n'.join(order).encode()).hexdigest(),training_achieved=all(v>=.95 for v in valid.values()),Validation_ADD_exact=valid['ADD'],Validation_SUB_exact=valid['SUB'],train_wall_seconds=trainwall,train_CPU_seconds=traincpu,peak_RAM_bytes=memory())
    dump(folder.relative_to(R)/'complete.json',out);return out
def evaluate_all(resources,d):
    summaries=[];holdrows=[];diagnostic=[];pairrows=[]
    with (W/'v054t_transition/data/probes.jsonl').open(encoding='utf-8') as f:probes=[json.loads(s) for s in f if json.loads(s)['k'] in [16,64,256,1024]]
    with gzip.open(R/'E3_independent_replication/predictions.jsonl.gz','wt',encoding='utf-8') as pf:
        for resource in resources:
            mid=resource['model'];m,_=load_model(R/f'E3_independent_replication/checkpoints/{mid}/best_validation.pt');sm=torch.jit.script(m).eval();start=time.perf_counter();cc=time.process_time()
            for width in [8,64,256,2048]:
                for op in ['ADD','SUB']:
                    lv,pr=evaluate(sm,d[f'{op}_Test{width}']);summaries.append(dict(model=mid,initialization_seed=resource['initialization_seed'],condition=resource['data_condition'],width=width,op=op,N=len(pr),correct=sum(p['correct'] for p in pr),accuracy=sum(p['correct'] for p in pr)/len(pr),loss=lv))
                    for p in pr:pf.write(json.dumps(dict(model=mid,kind='test',**p))+'\n')
            for k in [16,64,256,1024]:
                rr=[p for p in probes if p['k']==k]
                for j in range(0,len(rr),128):
                    ss=rr[j:j+128];br=[dict(id=str(r['pid']),op=r['op'],width=r['seq_len']-int(r['op']=='ADD'),a=int(r['a'],16),b=int(r['b'],16)) for r in ss];x,y,ll=batch(br)
                    with torch.no_grad():z=sm(x);pp=z.argmax(-1).numpy();vv=torch.isfinite(z).all(-1).numpy()
                    for r,a,p,v in zip(ss,br,pp,vv):
                        truth=a['a']+a['b'] if a['op']=='ADD' else a['a']-a['b'];ys=np.array([(truth>>t)&1 for t in range(r['seq_len'])]);wrong=(p[:len(ys)]!=ys)|~v[:len(ys)];within=np.flatnonzero(wrong[r['hold_start']:r['hold_end']]);value=sum(int(z)<<t for t,z in enumerate(p[:len(ys)])) if v[:len(ys)].all() else None
                        hr=dict(model=mid,op=r['op'],state=r['state'],pair=r['pair'],k=k,prefix=r['prefix'],suffix=r['suffix'],pid=r['pid'],exact=not bool(wrong.any()),prefix_error=bool(wrong[:r['hold_start']].any()),first_hold_error=int(within[0])+1 if len(within) else None,hold_censored=not len(within));holdrows.append(hr);pf.write(json.dumps(dict(model=mid,kind='hold',id=str(r['pid']),a=hex(a['a']),b=hex(a['b']),truth=hex(truth),prediction=hex(value) if value is not None else None,**{q:hr[q] for q in ['op','state','pair','k','first_hold_error']}))+'\n')
            rr,di,pa=diagnose(m,mid,'E3_independent_replication/diagnostics');diagnostic.extend(di)
            for p in pa:
                cases=[x for x in di if x['op']==p['op'] and x['pair']==p['pair']];stable=all(x['rho'] is not None and x['rho']<1 for x in cases);observed=any(x['first_hold_error'] is not None for x in holdrows if x['model']==mid and x['op']==p['op'] and x['pair']==p['pair']);p['strict_mechanism_reproduced']=p['shared'] and stable and not p['both_correct'] and observed;pairrows.append(p)
            for x in di:
                obs=[r['first_hold_error'] for r in holdrows if r['model']==mid and r['op']==x['op'] and r['pair']==x['pair'] and r['state']==x['state'] and r['k']>=256 and r['first_hold_error'] is not None];x['observed_median_first_hold']=float(np.median(obs)) if obs else None;x['timing_error']=x['first_negative_step']-float(np.median(obs)) if obs and x['first_negative_step'] is not None else None
            result=dict(resource,shared_pairs=sum(p['shared'] for p in pa),distinct_pairs=sum(p['distinct'] for p in pa),unclassified_pairs=sum(not p['both_converged'] for p in pa),strict_reproduced_pairs=sum(p['strict_mechanism_reproduced'] for p in pa),evaluation_wall_seconds=time.perf_counter()-start,evaluation_CPU_seconds=time.process_time()-cc)
            dump(f'E3_independent_replication/checkpoints/{mid}/evaluated.json',result);table('E3_independent_replication/E3_MODEL_RESULTS.csv',[read(p) for p in sorted((R/'E3_independent_replication/checkpoints').glob('*/evaluated.json'))]);table('E3_independent_replication/E3_TEST_RESULTS.csv',summaries);table('E3_independent_replication/E3_HOLD_RESULTS.csv',holdrows);table('E3_independent_replication/E3_DIAGNOSTICS.csv',diagnostic);table('E3_independent_replication/E3_PAIR_RESULTS.csv',pairrows);print('E3 EVALUATED',mid,'shared',result['shared_pairs'],'strict',result['strict_reproduced_pairs'],flush=True)
def main():
    freeze();d=data();res=[]
    for op in ['ADD','SUB']:
        keys=[]
        for split in ['Train','Validation','Test8']:
            kk={(min(r['a'],r['b']),max(r['a'],r['b'])) if op=='ADD' else (r['a'],r['b']) for r in d[f'{op}_{split}']};assert all(not kk.intersection(other) for other in keys);keys.append(kk)
    for seed in range(100,110):
        for c in [0,1]:res.append(train_one(d,seed,c));table('E3_independent_replication/TRAINING_RESULTS.csv',res)
    assert len({r['initial_SHA256'] for r in res})==10
    old={r['initial_sha256'] for r in rows(W/'IronMath_V054/cpu_memory.csv') if r['stage']=='training' and r['condition']=='C1' and r['model']=='B0'};assert not old.intersection({r['initial_SHA256'] for r in res})
    for seed in range(100,110):assert len({r['initial_SHA256'] for r in res if r['initialization_seed']==seed})==1
    dump('E3_independent_replication/TRAINING_COMPLETE.json',dict(utc=utc(),models=20,Test_used=False))
    evaluate_all(res,d);dump('E3_independent_replication/COMPLETE.json',dict(utc=utc(),models=20,additional_training=0))
if __name__=='__main__':
    try:main()
    except Exception as e:stage_failure('E3',e);raise
