from core import *
from decimal import Decimal,localcontext
import gzip,math
THEORY=r'''# 有限時刻失敗：初等的な命題と反例
本節は既知の線形系恒等式の整理であり、新規定理の優先性を主張しない。
固定点h*が存在すれば帰納法によりh_t=h*+A^t(h0−h*)。正しいクラスへの線形マージンはm_t=m∞+w^T A^t(h0−h*)、m∞=w^Th*+beta。rho(A)<1ならA^t→0。
1. m∞<0なら、epsilon=−m∞/2として有限T以降必ずm_t<0。初め正しい場合には有限の初回誤答も持続的誤答も存在するが、同時刻とは限らない。
2. m∞>0なら十分先では正しくなるが、途中の誤答はありうる。反例m_t=.05−.9^t+2(.5)^tはm0=1.05、m1=.15、m2=−.26、極限.05。従って安定性と正しい固定点だけでは全時刻の正答を保証しない。
3. m∞=0では有限時刻交差は保証されない。m_t=2^(−t)>0は全ての有限実数時刻で正しい。回転や負固有値なら符号交替もありうる。ゼロの分類はargmax同点規則による。FPのunderflowによるゼロと実数上の境界交差は異なる。
||A^t||≤C r^t（0<r<1）が分かればD=C||w||||h0−h*||として|m_t−m∞|≤D r^t。D r^t<|m∞|は極限と同じ側にある十分条件。rhoだけからC、初期状態、読み出し境界を無視して破綻時刻は導けない。
1次元m_t=−q+alpha lambda^t、alpha>q>0、0<lambda<1の厳密負への初回時刻はfloor(log(q/alpha)/log(lambda))+1。ゼロも誤答ならceil。同点と丸めを区別する。.99,alpha1,q.1はT=230で単調減少なので持続的誤答開始も同じ。
非線形系はF(h*+e)−h*=delta+J e+r(e)。delta=F(h*)−h*は数値終点で0と仮定しない。C2かつ真の固定点近傍ならr(e)=O(||e||^2)。遠い初期状態から無条件に線形式を適用しない。局所one-step残差と累積状態/マージン誤差を別に記録する。
高精度decimal100桁は数値参照であり厳密証明ではない。有限観測窓で最後の正答後ずっと誤る時刻を「観測持続的誤答開始」と呼び、窓外の永久性は単調合成例以外で保証しない。
'''
CASES=[('scalar_wrong_limit',[['.99']],['1'],['1'],'-.1'),('diagonal_transient',[['.9','0'],['0','.5']],['-1','2'],['1','1'],'.05'),('stable_rotation',[['.8','-.4'],['.4','.8']],['1','0'],['1','0'],'.01'),('boundary_fixed_point',[['.5']],['1'],['1'],'0'),('scalar_correct_limit',[['.99']],['1'],['1'],'.1')]
def synthetic():
    md('E1_finite_time/E1_THEORY.md',THEORY)
    md('E1_finite_time/E1_RULES.md','''# E1固定規則
合成5系、b=0、t=0..4096、FP32/FP64/decimal100桁、正解クラス1でmargin<=0は誤答。初回・観測持続開始・回復・打切りを分離。状態誤差はdecimal参照、時刻誤差も参照と比較。
算術は既存3seed×12hold、20000反復、h0..h512と終点、全margin/moveを保存。元の最初512marginとの最大差<=1e-3と初回負マージン時刻の厳密一致を要求。不一致ならE2/E3停止。元t0と新規step1は対応。
局所近似はanchor0/16/64/128から64step、数値終点でのJとdeltaを使う。one-step相対残差最大<=.01を局所適合ラベルとし、全anchorの誤差・打切りを保存。結果で範囲・閾値を変更しない。
''')
    dump('E1_finite_time/FROZEN.json',dict(utc=utc(),cases=CASES,horizon=4096,hashes={p:sha(R/'E1_finite_time'/p) for p in ['E1_THEORY.md','E1_RULES.md']}))
    result=[]
    p=R/'E1_finite_time/synthetic_trajectories.csv.gz'
    with gzip.open(p,'wt',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,['case','dtype','t','margin','state_error','wrong']);wr.writeheader()
        for name,aa,hh,ww,be in CASES:
            ref=[];rm=[]
            with localcontext() as ctx:
                ctx.prec=100;a=[[Decimal(v) for v in r] for r in aa];h=list(map(Decimal,hh));w=list(map(Decimal,ww));b=Decimal(be)
                for t in range(4097):ref.append(list(h));rm.append(sum(v*z for v,z in zip(w,h))+b);h=[sum(v*z for v,z in zip(row,h)) for row in a]
            ri=[t for t,v in enumerate(rm) if v<=0];rf=ri[0] if ri else None
            for dtype in ['FP32','FP64','decimal100']:
                dt=np.float32 if dtype=='FP32' else np.float64;a=np.array(aa,dt);h=np.array(hh,dt);w=np.array(ww,dt);b=dt(be);bad=[];errors=[]
                for t in range(4097):
                    val=rm[t] if dtype=='decimal100' else w@h+b;err=0. if dtype=='decimal100' else float(np.linalg.norm(h.astype(float)-np.array([float(v) for v in ref[t]])));wrong=val<=0
                    bad.append(bool(wrong));errors.append(err);wr.writerow(dict(case=name,dtype=dtype,t=t,margin=str(val),state_error=err,wrong=wrong));h=a@h
                ids=np.flatnonzero(bad);good=np.flatnonzero(~np.array(bad));first=int(ids[0]) if len(ids) else None;persist=int(good[-1])+1 if bad[-1] and len(good) else 0 if all(bad) else None
                result.append(dict(case=name,dtype=dtype,horizon=4096,first_error=first,persistent_error_start_observed=persist,first_error_censored=first is None,persistent_censored=persist is None,recoveries=int(np.sum(np.array(bad[:-1])&~np.array(bad[1:]))),max_state_error_reference=max(errors),reference_first=rf,first_time_error=first-rf if first is not None and rf is not None else None,analytic_first=230 if name=='scalar_wrong_limit' else None))
    table('E1_finite_time/E1_RESULTS.csv',result)
def diagnose(m,label,folder):
    folder=R/folder;folder.mkdir(exist_ok=True,parents=True);rr=selected_probes();assert len(rr)==12;h0,x,y=starts(m,rr);sm=torch.jit.script(m).eval();tt=time.perf_counter()
    cache=folder/f'{label}_trajectory.npz'
    if cache.exists():
        with np.load(cache) as z:end=torch.from_numpy(z['end']);states=torch.from_numpy(z['states']);mg=torch.from_numpy(z['margins']);mv=torch.from_numpy(z['movements'])
    else:
        with torch.no_grad():end,states,mg,mv=sm.evolution(x,h0,y,20000)
        np.savez_compressed(cache,initial=h0.numpy(),inputs=x.numpy(),targets=y.numpy(),end=end.numpy(),states=states.numpy(),margins=mg.numpy(),movements=mv.numpy())
    records=[]
    for i,r in enumerate(rr):
        conv=bool(torch.isfinite(end[i]).all() and mv[-100:,i].max()<1e-6);rho=None
        if conv:
            jp=folder/f'{label}_{i}_J.npz'
            J=torch.from_numpy(np.load(jp)['jacobian']) if jp.exists() else torch.autograd.functional.jacobian(lambda z:m.advance(x[i:i+1],z[None])[0],end[i].detach());rho=float(torch.linalg.eigvals(J.double()).abs().max())
            if label.startswith('V054') and not jp.exists():np.savez_compressed(jp,jacobian=J.numpy())
        vals=mg[:,i].numpy();bad=vals<0;ids=np.flatnonzero(bad);good=np.flatnonzero(~bad);pers=int(good[-1])+2 if bad[-1] and len(good) else 1 if bad.all() else None
        records.append(dict(model=label,index=i,op=r['op'],state=r['state'],pair=r['pair'],converged=conv,last100_move=float(mv[-100:,i].max()),rho=rho,endpoint_output=int(m.head(end[i:i+1]).argmax(-1)),target=int(y[i]),endpoint_correct=int(m.head(end[i:i+1]).argmax(-1))==int(y[i]),first_negative_step=int(ids[0])+1 if len(ids) else None,persistent_negative_start=pers,recoveries=int((bad[:-1]&~bad[1:]).sum())))
    pairs=[]
    for op,pair in sorted({(r['op'],r['pair']) for r in rr}):
        ix=[i for i,r in enumerate(rr) if r['op']==op and r['pair']==pair]
        if len(ix)!=2:continue
        dist=float((end[ix[0]]-end[ix[1]]).norm());both=all(records[j]['converged'] for j in ix)
        pairs.append(dict(model=label,op=op,pair=pair,distance=dist,both_converged=both,shared=both and dist<1e-3,distinct=both and dist>=1e-3,both_correct=all(records[j]['endpoint_correct'] for j in ix)))
    dump(folder.relative_to(R)/f'{label}_diagnostic.json',dict(records=records,pairs=pairs,wall_seconds=time.perf_counter()-tt,peak_RAM_bytes=memory()));return rr,records,pairs
def arithmetic():
    gold=rows(W/'v054h_dynamics/fixed_points.csv');gt=rows(W/'v054h_dynamics/trajectory.csv');details=[];local=[];checks=[]
    for seed in range(3):
        m,_=load_model(W/f'IronMath_V054/checkpoints/C1_B0_s{seed}/best_validation.pt');label=f'V054_s{seed}';rr,records,pairs=diagnose(m,label,'E1_finite_time/cache');z=np.load(R/f'E1_finite_time/cache/{label}_trajectory.npz');st=z['states'];en=z['end'];mg=z['margins'];xx=torch.from_numpy(z['inputs']);targets=z['targets']
        for i,r in enumerate(rr):
            old=next(a for a in gold if int(a['seed'])==seed and a['op']==r['op'] and int(a['state'])==r['state'] and a['pair']==r['pair']);oldfirst=None if old['first_negative_margin_t']=='NA' else int(old['first_negative_margin_t'])+1;gs=[a for a in gt if int(a['seed'])==seed and a['op']==r['op'] and int(a['state'])==r['state'] and a['pair']==r['pair'] and int(a['t'])<512];err=max(abs(float(a['margin'])-float(mg[int(a['t']),i])) for a in gs)
            check=dict(seed=seed,op=r['op'],state=r['state'],pair=r['pair'],margin_max_abs_difference=err,first_old=oldfirst,first_replayed=records[i]['first_negative_step'],passed=err<=1e-3 and oldfirst==records[i]['first_negative_step']);checks.append(check)
            table('E1_finite_time/BASELINE_REPRODUCTION.csv',checks)
            if not check['passed']:raise RuntimeError('Baseline mismatch; dependent E2/E3 stopped')
            peer=next((j for j,b in enumerate(rr) if b['op']==r['op'] and b['pair']==r['pair'] and b['state']!=r['state']),None)
            for t in range(1,513):details.append(dict(seed=seed,op=r['op'],state=r['state'],pair=r['pair'],step=t,distance_to_endpoint=float(np.linalg.norm(st[t,i]-en[i])),between_carry_distance=float(np.linalg.norm(st[t,i]-st[t,peer])) if peer is not None else None,margin=float(mg[t-1,i]),state_move=float(np.linalg.norm(st[t,i]-st[t-1,i]))))
            jp=R/f'E1_finite_time/cache/{label}_{i}_J.npz'
            if not jp.exists():continue
            J=np.load(jp)['jacobian'];point=en[i]
            with torch.no_grad():delta=m.advance(xx[i:i+1],torch.from_numpy(point)[None])[0].numpy()-point
            w=(m.head.weight[int(targets[i])]-m.head.weight[1-int(targets[i])]).detach().numpy();beta=float((m.head.bias[int(targets[i])]-m.head.bias[1-int(targets[i])]).detach())
            for anchor in [0,16,64,128]:
                e=st[anchor,i]-point;pred=[];res=[];serr=[]
                for t in range(64):
                    etrue=st[anchor+t,i]-point;actual=st[anchor+t+1,i]-point;res.append(float(np.linalg.norm(actual-delta-J@etrue)/max(np.linalg.norm(etrue),1e-12)));e=delta+J@e;pred.append(float(w@(point+e)+beta));serr.append(float(np.linalg.norm(point+e-st[anchor+t+1,i])))
                observed=mg[anchor:anchor+64,i];pi=np.flatnonzero(np.array(pred)<0);oi=np.flatnonzero(observed<0)
                local.append(dict(seed=seed,op=r['op'],state=r['state'],pair=r['pair'],anchor=anchor,horizon=64,delta_norm=float(np.linalg.norm(delta)),max_one_step_relative_residual=max(res),locally_applicable=max(res)<=.01,max_state_error=max(serr),max_margin_error=float(np.max(np.abs(np.array(pred)-observed))),predicted_first=anchor+int(pi[0])+1 if len(pi) else None,observed_first=anchor+int(oi[0])+1 if len(oi) else None,time_error=int(pi[0]-oi[0]) if len(pi) and len(oi) else None))
        print('E1-C seed',seed,'baseline reproduced',flush=True)
    table('E1_finite_time/E1_STATE_DISTANCES.csv',details);table('E1_finite_time/E1_LINEARIZATION.csv',local);dump('E1_finite_time/COMPLETE.json',dict(utc=utc(),baseline_checks=len(checks),passed=True,readonly_checkpoint_replays=3))
if __name__=='__main__':
    try:
        if not (R/'E1_finite_time/E1_RESULTS.csv').exists():synthetic()
        arithmetic()
    except Exception as e:stage_failure('E1',e);raise
