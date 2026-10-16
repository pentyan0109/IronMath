"""S1/S2 aggregation from stage2 records only (post hoc). Rules: S1S2_RULES.md (SHA256-checked)."""
import sys,json,csv,math
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;X=S.parent;G=X.parent/'IronMath_G1';sys.path.insert(0,str(S))
from s1_common import COUNTS,THRS,ARCH,conv,good,judge
N_BOOT=10000;LENS=[32,64,128,512]
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(name,rs):
    keys=list(dict.fromkeys(k for r in rs for k in r))
    with (S/name).open('w',encoding='utf-8',newline='') as f:
        q=csv.DictWriter(f,keys);q.writeheader();q.writerows(rs)
def score(p,y):
    tp=int((p&y).sum());fp=int((p&~y).sum());tn=int((~p&~y).sum());fn=int((~p&y).sum());return dict(hits=tp+tn,TP=tp,FP=fp,TN=tn,FN=fn)
def paired(a,b,y,seed=84000):
    ca=a==y;cb=b==y;oa=int((ca&~cb).sum());ob=int((cb&~ca).sum());n=oa+ob
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(oa,ob)+1))/2**n) if n else 1.
    d=ca.astype(int)-cb.astype(int);inds=np.random.default_rng(seed).integers(0,len(d),size=(N_BOOT,len(d)))
    ci=np.percentile(d[inds].mean(axis=1),[2.5,97.5]).tolist()
    return dict(A_only_correct=oa,other_only_correct=ob,McNemar_exact_two_sided_p=p,accuracy_difference=float(d.mean()),CI_low=ci[0],CI_high=ci[1],bootstrap_rng=seed)
def cond_name(c,t):return f'{c}/{t:.0e}'
def run():
    rules=load(S/'S1S2_RULES_SHA256.json');import hashlib;assert hashlib.sha256((S/'S1S2_RULES.md').read_bytes()).hexdigest()==rules['SHA256']
    rep=load(S/'s1_reproduction_check.json');assert rep['predictions_reproduced'],'stop: frozen predictions not reproduced'
    mm={m['model']:m for m in load(X/'memory_models.json') if m['study']=='G1'}
    recs={p.stem:load(p) for p in sorted((S/'s1_records').glob('*.json'))};assert len(recs)==200 and set(recs)==set(mm)
    ids=sorted(recs);arch=np.array([recs[i]['architecture'] for i in ids]);y=np.array([mm[i]['actual3'] for i in ids])
    bc={ell:np.array([all(mm[i]['lengths'][str(t)] for t in LENS if t<=ell) for i in ids]) for ell in LENS}
    groups={'nonlinear':arch!='linear',**{a:arch==a for a in ARCH},'linear':arch=='linear'}
    AD={}
    for c in COUNTS:
        for t in THRS:
            j=[judge(recs[i]['counts'][str(c)]['trajectories'],t) for i in ids];AD[c,t]=(np.array([x[0] for x in j]),np.array([x[1] for x in j]))
    A0,D0=AD[20000,1e-6];frozenA=np.array([mm[i]['A'] for i in ids]);assert np.array_equal(A0,frozenA) and np.array_equal(D0,np.array([mm[i]['D'] for i in ids]))
    # --- S1 tables
    s1=[];cmp=[]
    for (c,t),(A,D) in AD.items():
        for g,mask in groups.items():
            sa=score(A[mask],y[mask]);sd=score(D[mask],y[mask]);base=score(A0[mask],y[mask])['hits']
            s1.append(dict(condition=cond_name(c,t),iterations=c,threshold=t,group=g,N=int(mask.sum()),actual_success=int(y[mask].sum()),**{'A_'+k:v for k,v in sa.items()},**{'D_'+k:v for k,v in sd.items()},A_hits_change_from_20000_1e6=sa['hits']-base,robust_within_5=abs(sa['hits']-base)<=5,is_registered_main_condition=(c,t)==(20000,1e-6)))
            if g!='linear':
                for name,B in [('D',D),('B_cum(128)',bc[128]),('B_cum(512)',bc[512])]:
                    cmp.append(dict(condition=cond_name(c,t),iterations=c,threshold=t,group=g,comparison='A vs '+name,A_hits=int((A[mask]==y[mask]).sum()),other_hits=int((B[mask]==y[mask]).sum()),**paired(A[mask],B[mask],y[mask])))
    table('s1_conditions.csv',s1);table('s1_comparisons.csv',cmp)
    # --- 3.1 misclassification types
    ev={}
    for i in ids:
        if not y[ids.index(i)]:
            e=load(G/'evaluation_records'/f'{i}.json');bad=[it for it in e['items'] if it['T'] in (2048,8192,65000) and not it['sequence_exact']]
            first=min(bad,key=lambda it:(it['T'],it['first_error_position'] if it['first_error_position'] is not None else 10**9))
            ev[i]=dict(failed_lengths=sorted({it['T'] for it in bad}),failed_categories=sorted({it['category'] for it in bad}),first_failure_T=first['T'],first_failure_category=first['category'],first_failure_id=first['id'],first_error_position=first['first_error_position'],shortest_failed_first_error_position=min((it['first_error_position'] for it in bad if it['T']==first['T'] and it['first_error_position'] is not None),default=None))
    mis=[];mis_detail=[]
    for (c,t),(A,D) in AD.items():
        for g in ARCH+['linear']:
            cnt=dict(FN_nonconvergence=0,FN_readout_or_stability=0,FP=0)
            for k,i in enumerate(ids):
                if arch[k]!=g:continue
                tr=recs[i]['counts'][str(c)]['trajectories']
                if not A[k] and y[k]:
                    nc=[r for r in tr if not conv(r,t)];kind='FN_nonconvergence' if nc else 'FN_readout_or_stability';cnt[kind]+=1
                    bad=[r for r in tr if not good(r,t)]
                    mis_detail.append(dict(condition=cond_name(c,t),model=i,architecture=g,type=kind,nonconverged_trajectories=len(nc),readout_wrong=sum(1 for r in tr if conv(r,t) and r['readout']!=r['value']),rho_ge_1=sum(1 for r in tr if conv(r,t) and (r['spectral_radius'] is None or r['spectral_radius']>=1)),failing_trajectories=len(bad),max_last100_movement=max(r['last100_max_movement'] for r in tr)))
                elif A[k] and not y[k]:
                    cnt['FP']+=1;mis_detail.append(dict(condition=cond_name(c,t),model=i,architecture=g,type='FP',**{k2:(json.dumps(v) if isinstance(v,list) else v) for k2,v in ev[i].items()}))
            mis.append(dict(condition=cond_name(c,t),iterations=c,threshold=t,structure=g,**cnt))
    table('s1_misclassification_counts.csv',mis);table('s1_misclassification_detail.csv',mis_detail)
    # LSTM false negatives: descriptive dynamics of failing trajectories
    lstm=[]
    for (c,t),(A,D) in AD.items():
        for k,i in enumerate(ids):
            if arch[k]=='lstm' and not A[k] and y[k]:
                for r in recs[i]['counts'][str(c)]['trajectories']:
                    if good(r,t):continue
                    dd=r['movement_decay'];lstm.append(dict(condition=cond_name(c,t),model=i,neutral=r['neutral'],prefix=r['prefix'],value=r['value'],converged=conv(r,t),last100_max_movement=r['last100_max_movement'],spectral_radius=r['spectral_radius'],readout_correct=r['readout']==r['value'],window=dd['window'],movement_first=dd['first'],movement_last=dd['last'],ratio_last_over_first=dd['ratio_last_over_first'],per_step_geometric_factor=dd['per_step_geometric_factor'],log_linear_R2=dd['log_linear_R2'],c_L2=r.get('c_L2'),c_max_abs=r.get('c_max_abs'),forget_gate_mean=r.get('forget_gate_mean'),forget_gate_min=r.get('forget_gate_min'),forget_gate_max=r.get('forget_gate_max')))
    table('s1_lstm_false_negative_dynamics.csv',lstm)
    # comparison reference: LSTM converged trajectories in true-positive models at main condition
    ref=[]
    for k,i in enumerate(ids):
        if arch[k]=='lstm' and A0[k]:
            for r in recs[i]['counts']['20000']['trajectories']:ref.append(dict(model=i,c_L2=r['c_L2'],forget_gate_mean=r['forget_gate_mean'],ratio=r['movement_decay']['ratio_last_over_first'],factor=r['movement_decay']['per_step_geometric_factor'],last100=r['last100_max_movement']))
    # --- S2 timing
    bt=rows(S/'s2_btest_times.csv');assert len(bt)==800
    s2_ok=all(r['saved_predictions_bitwise_match']=='True' and r['stage1_success_match']=='True' for r in bt)
    btime={(r['model'],int(r['T'])):float(r['wall_seconds']) for r in bt}
    tA={(i,c):recs[i]['counts'][str(c)]['wall_seconds']['total_excluding_script'] for i in ids for c in COUNTS}
    tB={(i,ell):sum(btime[i,T] for T in LENS if T<=ell) for i in ids for ell in LENS}
    phase=[]
    for g,mask in groups.items():
        sel=[i for k,i in enumerate(ids) if mask[k]]
        for c in COUNTS:
            ph={p:[recs[i]['counts'][str(c)]['wall_seconds'][p] for i in sel] for p in ['prep','iterate','readout','jacobian_eig','total_excluding_script','script']}
            phase.append(dict(group=g,method='A',iterations=c,N=len(sel),**{f'{p}_mean':float(np.mean(v)) for p,v in ph.items()},total_median=float(np.median(ph['total_excluding_script'])),total_min=float(np.min(ph['total_excluding_script'])),total_max=float(np.max(ph['total_excluding_script']))))
        for ell in LENS:
            v=[tB[i,ell] for i in sel];phase.append(dict(group=g,method='B_cum',ell=ell,N=len(sel),total_excluding_script_mean=float(np.mean(v)),total_median=float(np.median(v)),total_min=float(np.min(v)),total_max=float(np.max(v)),sequence_steps_per_model=68*sum(T for T in LENS if T<=ell)))
    table('s2_time_breakdown.csv',phase)
    repeats=rows(S/'s2_repeats.csv');var=[]
    for key in sorted({(r['model'],r['method'],r['iterations'],r['ell']) for r in repeats}):
        v=[float(r['wall_seconds']) for r in repeats if (r['model'],r['method'],r['iterations'],r['ell'])==key]
        var.append(dict(model=key[0],method=key[1],iterations=key[2] or None,ell=key[3] or None,n=len(v),median=float(np.median(v)),min=float(min(v)),max=float(max(v)),range_over_median=float((max(v)-min(v))/np.median(v))))
    table('s2_repeat_variability.csv',var)
    # compute-matched points
    pts=[]
    for g,mask in groups.items():
        sel=[k for k in range(len(ids)) if mask[k]]
        for (c,t),(A,D) in AD.items():pts.append(dict(group=g,method='A',condition=cond_name(c,t),iterations=c,threshold=t,mean_seconds_per_model=float(np.mean([tA[ids[k],c] for k in sel])),hits=int((A[sel]==y[sel]).sum()),N=len(sel)))
        for ell in LENS:pts.append(dict(group=g,method='B_cum',condition=f'B_cum({ell})',ell=ell,mean_seconds_per_model=float(np.mean([tB[ids[k],ell] for k in sel])),hits=int((bc[ell][sel]==y[sel]).sum()),N=len(sel)))
    table('s2_compute_matched_points.csv',pts)
    verdict={}
    for g in groups:
        P=[p for p in pts if p['group']==g];Bp=[p for p in P if p['method']=='B_cum'];wins=[];cheaper=[]
        for a in [p for p in P if p['method']=='A']:
            within=[b for b in Bp if b['mean_seconds_per_model']<=a['mean_seconds_per_model']]
            if not within:cheaper.append(a['condition']);continue
            if a['hits']>max(b['hits'] for b in within):wins.append(dict(condition=a['condition'],A_hits=a['hits'],A_seconds=a['mean_seconds_per_model'],best_B_within=max(within,key=lambda b:b['hits'])['condition'],best_B_hits=max(b['hits'] for b in within)))
        verdict[g]=dict(A_wins_exploratory=wins,A_conditions_cheaper_than_every_B=cheaper,statement=('ある計算時間以下で、診断A のいずれかの条件が、同じ計算時間以下の B_cum(ℓ) すべてより多く当てた（探索的結果）' if wins else '同じ計算時間の範囲では、診断A がテストより正確であることは示されなかった'))
    # best-condition exploratory comparison
    nl=groups['nonlinear'];best=max(AD,key=lambda k:((AD[k][0][nl]==y[nl]).sum(),-k[0]))
    bestcmp=dict(condition=cond_name(*best),A_hits=int((AD[best][0][nl]==y[nl]).sum()),**paired(AD[best][0][nl],bc[512][nl],y[nl]),B_cum512_hits=int((bc[512][nl]==y[nl]).sum()),note='12条件から最良を事後に選んだ比較。選択による楽観的な偏りがあり、探索的結果。')
    out=dict(rules_SHA256=rules['SHA256'],reproduction=dict(models=rep['models'],predictions_reproduced=rep['predictions_reproduced'],details_bitwise_reproduced=rep['details_bitwise_reproduced']),s2_btest_reproduces_saved=s2_ok,verdict=verdict,best_condition_exploratory=bestcmp,lstm_true_positive_reference=dict(n_trajectories=len(ref),c_L2_median=float(np.median([r['c_L2'] for r in ref])) if ref else None,forget_mean_median=float(np.median([r['forget_gate_mean'] for r in ref])) if ref else None,ratio_median=float(np.median([r['ratio'] for r in ref if r['ratio'] is not None])) if ref else None,last100_median=float(np.median([r['last100'] for r in ref])) if ref else None),
             false_positive_models_main={i:ev[i] for k,i in enumerate(ids) if A0[k] and not y[k]})
    (S/'s1s2_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=1),encoding='utf-8')
    plot(pts)
    print(json.dumps(dict(verdict={g:v['statement'] for g,v in verdict.items()},best=bestcmp,s2_ok=s2_ok),ensure_ascii=False,indent=1))
def plot(pts):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams['font.family']=['Yu Gothic','Meiryo','MS Gothic','sans-serif'];SURF='#fcfcfb';INK='#2b2b2a';MUTED='#6b6a63';CA='#2a78d6';CB='#eb6834'
    def panel(ax,g,title):
        P=[p for p in pts if p['group']==g];a=[p for p in P if p['method']=='A'];b=sorted([p for p in P if p['method']=='B_cum'],key=lambda p:p['ell'])
        ax.set_facecolor(SURF);ax.grid(True,color='#e4e3df',lw=.6);ax.set_axisbelow(True)
        for s in ax.spines.values():s.set_color('#c9c8c2')
        ax.plot([p['mean_seconds_per_model'] for p in b],[p['hits'] for p in b],color=CB,lw=2,marker='s',ms=8,mec=SURF,mew=1.5,label='B_cum(ℓ) テスト',zorder=3)
        for p in b:ax.annotate(f"ℓ={p['ell']}",(p['mean_seconds_per_model'],p['hits']),xytext=(0,-14),textcoords='offset points',ha='center',fontsize=8,color=MUTED)
        mk={1e-4:'^',1e-5:'D',1e-6:'o'}
        for t in THRS:
            q=[p for p in a if p['threshold']==t];ax.scatter([p['mean_seconds_per_model']*{1e-4:.86,1e-5:1.,1e-6:1.16}[t] for p in q],[p['hits'] for p in q],marker=mk[t],s=56,color=CA,edgecolor=SURF,linewidth=1.5,zorder=4,label=f'診断A 閾値{t:.0e}')
        main=[p for p in a if p['iterations']==20000 and p['threshold']==1e-6][0];ax.annotate('20,000回・1e-6\n（事前登録の条件）',(main['mean_seconds_per_model']*1.16,main['hits']),xytext=(-40,-34),textcoords='offset points',ha='right',va='top',fontsize=8,color=INK,arrowprops=dict(arrowstyle='-',color=MUTED,lw=.8))
        ax.margins(y=.25);ax.set_xscale('log');ax.set_title(title,fontsize=11,color=INK,loc='left');ax.set_xlabel('1モデルあたりの実測合計時間（秒, 対数軸）',fontsize=9,color=MUTED);ax.set_ylabel(f"的中数（N={P[0]['N']}）",fontsize=9,color=MUTED);ax.tick_params(colors=MUTED,labelsize=8)
    fig,ax=plt.subplots(figsize=(7.5,4.8),facecolor=SURF);panel(ax,'nonlinear','G1 非線形160モデル：計算時間と的中数');ax.legend(fontsize=8,frameon=False,loc='lower right');fig.text(.01,.01,'診断A の点は、重なりを避けるため閾値ごとに横へ少しずらして描いた（1e-4 は左、1e-6 は右）。',fontsize=7,color=MUTED);fig.tight_layout(rect=(0,.03,1,1));fig.savefig(S/'s2_compute_matched_nonlinear.png',dpi=150,facecolor=SURF);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,7.5),facecolor=SURF)
    for ax,g,tt in zip(axs.flat,ARCH,['残差GRU','通常のGRU','LSTM','Elman型RNN']):panel(ax,g,f'{tt}（40モデル）')
    axs[0,0].legend(fontsize=7,frameon=False,loc='lower right');fig.text(.01,.005,'診断A の点は、重なりを避けるため閾値ごとに横へ少しずらして描いた（1e-4 は左、1e-6 は右）。',fontsize=7,color=MUTED);fig.tight_layout(rect=(0,.02,1,1));fig.savefig(S/'s2_compute_matched_by_structure.png',dpi=150,facecolor=SURF);plt.close(fig)
if __name__=='__main__':run()
