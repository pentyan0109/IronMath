from core import *
def main():
    assert read(R/'E1_finite_time/COMPLETE.json')['passed']
    md('E2_causal_intervention/E2_RULES.md','''# E2固定規則
対象V054 C1 B0 seed0/1/2、E1と同じ12hold文脈。全36例を保持し、共有終点と別終点例を結果で選別しない。介入時刻はhold step1/4/8後、観測窓1024。既知V054Hの破綻前の時刻であり、新しい介入結果を見て変更しない。
正解donorは同じモデル・入力対・正解carry文脈のstep1状態（読出正解を要求）。失敗donorは同文脈step512状態（誤答を要求）。その役割を満たさないdonorは欠測NAとし、別時刻を探索しない。正解donorは短い正解過渡の状態であって、永久保持する軌道とは主張しない。carryの直接注入を含むoracle補助の介入である。
各donor差分のノルムと一致するrandom3方向、境界正/負方向、境界直交random3方向も比較。direction seed92000+seed*1000+case*20+intervention_step。正解側法線w_true−w_wrongを使用。ノルム0は無作用のまま保存。
事前予測：正解donorの直後marginは正、初回破綻は同じ固定文脈の初回Tを使ってintervention_t+(T−1)へ移動。失敗donorの直後marginは負。正/負法線の直後margin変化は±||w||*delta_norm、直交は0。randomや持続効果の優位は予め仮定しない。donor移植の時刻効果は同じ決定論的軌道を再開する性質であり、アトラクタ構造自体を変える介入ではない。
初回誤答と窓内持続誤答開始は別。介入直後/次step/4/16/64stepのmargin差、正答連続期間、baselineとのbit判定差が最後に観測されたstep、回復数を保存。窓外は打切り。
算術転移は既存Test8/Test64の先頭64問/演算/seed、固定bit位置8（0始まり）で同じhold入力・Oracle carryに一致した場合だけdonor移植。donor1/512の適格性は同じ。元の入力履歴とは異なるdonorであり、汎用推論手法や修復成功とは扱わない。対象外入力は無介入のまま、予測不変を検査。Oracleは介入割当と採点専用で、通常モデル入力には入れない。
無介入はE1再現を前提とし、E2のbaseline marginはE1の1024stepと一致。全条件・NAを保存。重み変更なし。
''')
    dump('E2_causal_intervention/FROZEN.json',dict(utc=utc(),rules_SHA256=sha(R/'E2_causal_intervention/E2_RULES.md'),checkpoints={str(s):sha(W/f'IronMath_V054/checkpoints/C1_B0_s{s}/best_validation.pt') for s in range(3)}))
    results=[];transfer=[];predictions=[];curves=[]
    for seed in range(3):
        m,_=load_model(W/f'IronMath_V054/checkpoints/C1_B0_s{seed}/best_validation.pt');before=modelhash(m);sm=torch.jit.script(m).eval();rr=selected_probes();z=np.load(R/f'E1_finite_time/cache/V054_s{seed}_trajectory.npz');states=z['states'];x=torch.from_numpy(z['inputs']);target=torch.from_numpy(z['targets']);baseline=z['margins'][:1024];rng=np.random.default_rng(92000+seed)
        for i,r in enumerate(rr):
            bv=baseline[:,i];bi=np.flatnonzero(bv<0);bt=int(bi[0]+1) if len(bi) else None
            for at in [1,4,8]:
                rng=np.random.default_rng(92000+seed*1000+i*20+at)
                cases=[('baseline',None,'none',None)]
                for donor,point in [('correct',1),('failed',512)]:
                    donorstate=states[point,i];dv=float(m.head(torch.from_numpy(donorstate)[None])[0,int(target[i])]-m.head(torch.from_numpy(donorstate)[None])[0,1-int(target[i])]);eligible=dv>0 if donor=='correct' else dv<0
                    if not eligible:results.append(dict(seed=seed,op=r['op'],state=r['state'],pair=r['pair'],at=at,condition='transplant_'+donor,status='donor unavailable; fixed time label not satisfied'));continue
                    base=states[at,i];delta=donorstate-base;norm=float(np.linalg.norm(delta));w=(m.head.weight[int(target[i])]-m.head.weight[1-int(target[i])]).detach().numpy();wn=np.linalg.norm(w);u=w/wn;cases.append(('transplant_'+donor,donorstate,donor,norm))
                    for sign in [-1,1]:cases.append((f'normal_{sign}_{donor}',base+sign*norm*u,donor,norm))
                    for j in range(3):
                        q=rng.normal(size=545).astype(np.float32);q=q/np.linalg.norm(q);qq=q-(q@u)*u;qq=qq/np.linalg.norm(qq);cases.extend([(f'random{j}_{donor}',base+norm*q,donor,norm),(f'orthogonal{j}_{donor}',base+norm*qq,donor,norm)])
                hs=torch.from_numpy(np.stack([states[at,i] if c[1] is None else np.asarray(c[1],np.float32) for c in cases]))
                with torch.no_grad():
                    immediate=m.head(hs);_,_,future,_=sm.evolution(x[i:i+1].expand(len(cases),-1),hs,target[i:i+1].expand(len(cases)),1024-at)
                for ci,(name,new,donor,norm) in enumerate(cases):
                    vs=bv.copy();vs[at-1]=float(immediate[ci,target[i]]-immediate[ci,1-target[i]]);vs[at:]=future[:,ci].numpy()
                    if name=='baseline':assert np.max(np.abs(vs-bv))<=1e-3 and np.array_equal(vs<0,bv<0)
                    first,persist,recovery=first_persistent(vs);post=vs[at-1:]>=0;ids=np.flatnonzero(~post);continuous=int(ids[0]) if len(ids) else len(post);changed=np.flatnonzero((vs<0)!=(bv<0));effect_end=int(changed[-1]+1) if len(changed) else None
                    pred_time=at+(bt-1) if name=='transplant_correct' and bt is not None else at if name=='transplant_failed' else None
                    predictions.append(dict(seed=seed,index=i,at=at,condition=name,predicted_first_error=pred_time,expected_immediate_sign='positive' if name=='transplant_correct' else 'negative' if name=='transplant_failed' else 'directional' if name.startswith('normal') else 'unspecified'))
                    row=dict(seed=seed,op=r['op'],state=r['state'],pair=r['pair'],at=at,condition=name,status='measured',donor=donor,delta_norm=norm,immediate_margin=float(vs[at-1]),immediate_margin_change=float(vs[at-1]-bv[at-1]),first_error=first+1 if first is not None else None,persistent_error_start_observed=persist+1 if persist is not None else None,first_error_censored=first is None,correct_continuation_steps=continuous,recoveries=recovery,last_classification_effect_step=effect_end,predicted_first=pred_time,time_prediction_error=first+1-pred_time if first is not None and pred_time is not None else None)
                    for offset in [1,4,16,64]:row[f'margin_change_after_{offset}']=float(vs[at-1+offset]-bv[at-1+offset])
                    results.append(row)
                    if name in ['baseline','transplant_correct','transplant_failed','normal_1_correct','orthogonal0_correct']:curves.append(dict(seed=seed,index=i,at=at,condition=name,margins=vs.tolist()))
        # Full arithmetic at a preregistered bit position; Oracle only assigns interventions.
        for op in ['ADD','SUB']:
            for width in [8,64]:
                data=read(W/f'IronMath_V053/data/{op}_Test{width}.json')[:64]
                for item in data:
                    x1,y1,ll=batch([item]);n=ll[0];baseh=torch.zeros(1,545);hist=[]
                    with torch.no_grad():
                        for t in range(n):baseh=m.advance(x1[:,t],baseh);hist.append(int(m.head(baseh).argmax(-1)))
                    for mode in ['correct','failed']:
                        h=torch.zeros(1,545);pp=[];state=0;applied=False
                        with torch.no_grad():
                            for t in range(n):
                                a=(item['a']>>t)&1;b=(item['b']>>t)&1;incoming=state;s=a+b+state if op=='ADD' else a-b-state;state=s//2 if op=='ADD' else int(s<0);h=m.advance(x1[:,t],h)
                                if t==8:
                                    ix=next((j for j,r in enumerate(rr) if r['op']==op and r['state']==incoming and r['pair']==f'{a}{b}'),None)
                                    if ix is not None:
                                        point=1 if mode=='correct' else 512;ds=torch.from_numpy(states[point,ix])[None];zz=m.head(ds);dm=float(zz[0,target[ix]]-zz[0,1-target[ix]])
                                        if (dm>0 if mode=='correct' else dm<0):h=ds.clone();applied=True
                                pp.append(int(m.head(h).argmax(-1)))
                        if not applied:assert pp==hist
                        truth=item['a']+item['b'] if op=='ADD' else item['a']-item['b'];pred=sum(v<<i for i,v in enumerate(pp));bp=sum(v<<i for i,v in enumerate(hist));transfer.append(dict(seed=seed,op=op,width=width,id=item['id'],a=hex(item['a']),b=hex(item['b']),truth=hex(truth),mode=mode,intervened=applied,baseline_correct=bp==truth,intervention_correct=pred==truth,prediction=hex(pred),baseline_prediction=hex(bp),oracle_assisted=True))
        assert modelhash(m)==before;table('E2_causal_intervention/E2_RESULTS.csv',results);table('E2_causal_intervention/E2_ARITHMETIC.csv',transfer);table('E2_causal_intervention/E2_PREDICTIONS.csv',predictions);dump('E2_causal_intervention/curves.json',curves);print('E2 seed',seed,'complete',flush=True)
    dump('E2_causal_intervention/COMPLETE.json',dict(utc=utc(),all_conditions_saved=True,weights_unchanged=True,new_training=0))
if __name__=='__main__':
    try:main()
    except Exception as e:stage_failure('E2',e);raise
