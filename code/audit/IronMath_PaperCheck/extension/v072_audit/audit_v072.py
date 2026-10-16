"""Saved-data-only audit. No experiment imports, model creation, training or inference."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']=str(__import__('pathlib').Path(__file__).parent/'mpl_cache')
import csv,gzip,json,hashlib,math,statistics,re,shutil,subprocess,datetime
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
W=Path(__file__).resolve().parents[3]; O=Path(__file__).resolve().parent; X=O.parent
SOURCES={}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def source(p):
    p=Path(p);p=p if p.is_absolute() else W/p
    if str(p) not in SOURCES:SOURCES[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p)}
    return p
def rows(p):
    with source(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def js(p):return json.loads(source(p).read_text(encoding='utf-8-sig'))
def write(name,s):
    p=O/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8');assert p.read_text(encoding='utf-8')==s and '\ufffd' not in s
def dump(n,v):write(n,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(n,rr):
    fields=list(dict.fromkeys(k for r in rr for k in r));p=O/n;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        q=csv.DictWriter(f,fieldnames=fields);q.writeheader();q.writerows(rr)
def b(v):return str(v).lower() in ('true','1')
def f(v):return None if v in ('','NA','None',None) else float(v)
def stats(v):
    v=[float(x) for x in v if x is not None]
    return dict(n=len(v),mean=statistics.mean(v) if v else None,median=statistics.median(v) if v else None,min=min(v) if v else None,max=max(v) if v else None,SD=statistics.stdev(v) if len(v)>1 else None)
def mdtable(rr,fields):
    return '| '+' | '.join(fields)+' |\n|'+ '|'.join(['---']*len(fields))+'|\n'+''.join('| '+' | '.join('NA' if r.get(k) is None else format(r[k],'.7g') if isinstance(r.get(k),float) else str(r.get(k,'')) for k in fields)+' |\n' for r in rr)
def gkey(r):return (str(r['seed']),r['op'],str(r['state']),r['pair'])
def index_probes():
    p=source('v054t_transition/data/probes.jsonl');return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]

def e2():
    rr=rows(X/'E2_causal_intervention/E2_RESULTS.csv');probes=[r for r in index_probes() if r['k']==4096 and r['prefix']=='min' and r['suffix']=='s0'];keys={}
    baselines={gkey(r)+(str(r['at']),):r for r in rr if r['condition']=='baseline'}
    failed={gkey(r) for r in rr if r['condition']=='baseline' and r['at']=='8' and f(r['first_error']) is not None};assert len(failed)==10
    for r in rr:keys[gkey(r)+(str(r['at']),r['condition'])]=r
    result=[];curves=source(X/'audit/E2_all_curves.jsonl.gz')
    with gzip.open(curves,'rt',encoding='utf-8') as fin:
        for line in fin:
            c=json.loads(line);p=probes[c['index']];key=(str(c['seed']),p['op'],str(p['state']),p['pair']);at=c['at'];r=keys[key+(str(at),c['condition'])];base=baselines[key+(str(at),)]
            a=np.asarray(c['margins'],float);post=a[at-1:];assert len(a)==1024 and np.isfinite(a).all();wrong=post<0;inds=np.flatnonzero(wrong);first=int(inds[0]+at) if len(inds) else None;full=np.flatnonzero(a<0)
            assert (int(full[0]+1) if len(full) else None)==f(r['first_error']) and abs(post[0]-float(r['immediate_margin']))<1e-6
            bt=f(base['first_error']);recoveries=int(np.count_nonzero(wrong[:-1]&~wrong[1:]));left=bool(wrong[0]);right=not bool(len(inds));minshift=at-bt if bt is not None else None
            result.append(dict(seed=c['seed'],op=p['op'],state=p['state'],pair=p['pair'],at=at,condition=c['condition'],donor=r['donor'],baseline_failing_context=key in failed,baseline_first_error=bt,first_evaluation=at,post_first_error=first,shift=first-bt if first is not None and bt is not None else None,observable_min_shift=minshift,at_observation_floor=left and bt is not None,left_censored_latent_onset=left,right_censored_first_error=right,immediate_margin=post[0],post_recoveries=recoveries,recovered_after_any_error=recoveries>0,post_evaluations=len(post),source_status='measured'))
    assert len(result)==1350
    floor=[r for r in result if r['baseline_failing_context']];table('E2_FLOOR_AUDIT.csv',floor)
    summary=[]
    for population in ['baseline_failing_10','all_36']:
        for at in [1,4,8]:
            for condition in sorted({r['condition'] for r in result}):
                ss=[r for r in result if r['at']==at and r['condition']==condition and (population=='all_36' or r['baseline_failing_context'])]
                if not ss:continue
                s=stats([r['shift'] for r in ss]);summary.append(dict(population=population,at=at,condition=condition,measured=len(ss),shift_defined=s['n'],mean_shift=s['mean'],median_shift=s['median'],min_shift=s['min'],max_shift=s['max'],SD_shift=s['SD'],immediate_errors=sum(r['left_censored_latent_onset'] for r in ss),immediate_error_rate=sum(r['left_censored_latent_onset'] for r in ss)/len(ss),left_censored=sum(r['left_censored_latent_onset'] for r in ss),right_censored=sum(r['right_censored_first_error'] for r in ss),right_censor_rate=sum(r['right_censored_first_error'] for r in ss)/len(ss),any_censor_rate=sum(r['left_censored_latent_onset'] or r['right_censored_first_error'] for r in ss)/len(ss),recovered_cases=sum(r['recovered_after_any_error'] for r in ss),recovery_rate=sum(r['recovered_after_any_error'] for r in ss)/len(ss),conditional_recovery_rate=sum(r['recovered_after_any_error'] for r in ss)/sum(r['post_first_error'] is not None for r in ss) if any(r['post_first_error'] is not None for r in ss) else None,mean_immediate_margin=statistics.mean(r['immediate_margin'] for r in ss)))
    table('E2_ALL_CONTROLS.csv',summary);table('E2_ALL_CONTROLS_TRIALS.csv',result)
    grouped=[];within=[]
    for at in [1,4,8]:
        for donor in ['correct','failed']:
            for family in ['random','orthogonal']:
                contexts=defaultdict(list)
                for r in floor:
                    if r['at']==at and re.fullmatch(family+r'[012]_'+donor,r['condition']):contexts[(r['seed'],r['op'],r['state'],r['pair'])].append(r)
                vv=[];ranges=[]
                for key,ss in contexts.items():
                    assert len(ss)==3;v=stats([r['shift'] for r in ss]);within.append(dict(at=at,donor=donor,family=family,seed=key[0],op=key[1],state=key[2],pair=key[3],**v,direction_censored=sum(r['right_censored_first_error'] for r in ss)))
                    if v['n']==3:vv.append(v['mean']);ranges.append(v['max']-v['min'])
                ds=[r for r in summary if r['population']=='baseline_failing_10' and r['at']==at and re.fullmatch(family+r'[012]_'+donor,r['condition'])];means=[r['mean_shift'] for r in ds if r['mean_shift'] is not None];s=stats(vv)
                grouped.append(dict(at=at,donor=donor,family=family,contexts_total=len(contexts),complete_uncensored_contexts=s['n'],mean_context_direction_mean=s['mean'],SD_between_context_means=s['SD'],mean_within_context_range=statistics.mean(ranges) if ranges else None,mean_of_direction_means=statistics.mean(means) if means else None,min_direction_mean=min(means) if means else None,max_direction_mean=max(means) if means else None))
    table('E2_DIRECTION_WITHIN_CONTEXT.csv',within);table('E2_DIRECTION_GROUPED.csv',grouped)
    at8=[r for r in summary if r['population']=='baseline_failing_10' and r['at']==8];selected=[r for r in at8 if r['condition'] in ['transplant_failed','normal_-1_correct','normal_-1_failed']]
    assert all(r['immediate_errors']==10 and r['mean_shift']==-34.3 for r in selected)
    text='# E2 観測下限監査\n\n保存済み独立監査マージン1350本（1024 step）を再採点。新規推論なし。無介入で破綻した10文脈を固定し、全条件・介入時刻1/4/8を収録。CSVは文脈×条件×時刻、集計表はE2_ALL_CONTROLS.csv。利用できない失敗donor26文脈×3時刻=78件は元CSVのNAのまま保存し、方向対照を架空に補完しない。\n\n介入はstep atを処理した状態を置き換え、最初の評価は同じstep atの読み出し。介入前状態を巻き戻しての潜在的な誤答開始は観測不能。左打切りは介入直後ですでにmargin<0、右打切りは残り1024-step窓で負マージンなし。左打切りを含む時刻差は観測初回誤答の差であり、潜在効果の連続量ではない。回復は介入以後のmargin<0から>=0への遷移で定義する（元の負マージン規則を維持）。\n\n'+mdtable(selected,['condition','measured','mean_shift','immediate_errors','left_censored','right_censored','recovered_cases','mean_immediate_margin'])+'\n**Q1：はい。3条件とも10/10文脈で観測可能な最小シフト at−T に到達した。at=8の平均8−mean(T)=−34.3で、3条件の潜在的強度が等しいことを示さない。** 基準T・各最小シフトはCSVに保存。観測下限と異なる条件の差を、連続的な効果差として順位付けしない。\n'
    write('E2_FLOOR_AUDIT.md',text)
    write('E2_ALL_CONTROLS.md','# E2 全対照条件\n\n全1350測定（元結果1428行中78 NA）を使用。下表は無介入で破綻した10文脈・at8。全36文脈とat1/4もCSVに収録。初回時刻差は両時刻が観測された行のみ、打切りを1025等で代入しない。左打切りを含む平均は観測差の要約であって潜在効果ではない。率の分母はmeasured、conditional_recovery_rateのみ観測誤答ありの件数。\n\n'+mdtable(at8,['condition','measured','shift_defined','mean_shift','median_shift','min_shift','max_shift','immediate_error_rate','right_censor_rate','recovery_rate'])+'\n## 3方向の集計\n'+mdtable([r for r in grouped if r['at']==8],['donor','family','contexts_total','complete_uncensored_contexts','mean_context_direction_mean','SD_between_context_means','mean_within_context_range','min_direction_mean','max_direction_mean'])+'\n方向0/1/2は全て掲載し、代表方向を選ばない。E2_DIRECTION_WITHIN_CONTEXT.csvは各文脈内の3方向SD・範囲、E2_DIRECTION_GROUPED.csvは文脈平均の文脈間SDを区別する。打切りで3方向すべての時刻差が定まらない文脈はcomplete集計から外し件数を明記する。対照を独立な初期化の反復単位とはしない。\n\n**Q2：読み出し法線による符号付きの直後効果と時刻効果は、全方向を残しても直交・ランダム対照より大きい傾向。ただし負方向は観測下限で飽和しており、潜在的な効果量の優越は測れない。** 法線は線形headの境界を直接横断するため、これだけでアトラクタ幾何固有の因果性や永続修復を示さない。結果はat・donorノルム・基準10文脈に条件付きであり、全36での打切り率も同じCSVに提供する。\n')
    return result,summary

def e3():
    pairs=rows(X/'E3_independent_replication/E3_PAIR_RESULTS.csv');di=rows(X/'E3_independent_replication/E3_DIAGNOSTICS.csv');hold=rows(X/'E3_independent_replication/E3_HOLD_RESULTS.csv');models=rows(X/'E3_independent_replication/E3_MODEL_RESULTS.csv');assert len(pairs)==80 and len(di)==240
    pd={(r['model'],r['op'],r['pair']):r for r in pairs};hg=defaultdict(list)
    for r in hold:hg[(r['model'],r['op'],r['pair'],r['state'])].append(r)
    side=[]
    for r in di:
        key=(r['model'],r['op'],r['pair']);p=pd.get(key);hs=hg[key+(r['state'],)];ts=[f(x['first_hold_error']) for x in hs if f(x['first_hold_error']) is not None];s=stats(ts);kind='shared' if p and b(p['shared']) else 'distinct' if p and b(p['distinct']) else 'unclassified' if p else 'unpaired_input';status='failure' if ts else 'no_failure_within_window' if len(hs)==64 else 'unevaluable'
        side.append(dict(model=r['model'],initialization=int(r['model'][1:4]),data_condition=int(r['model'][-1]),op=r['op'],pair=r['pair'],state=r['state'],endpoint_class=kind,endpoint_correct=b(r['endpoint_correct']) if b(r['converged']) else None,converged=b(r['converged']),rho=f(r['rho']),endpoint_output=r['endpoint_output'],target=r['target'],outcome=status,observed_hold_trials=len(hs),failed_trials=len(ts),right_censored_trials=sum(b(x['hold_censored']) for x in hs),prefix_error_trials=sum(b(x['prefix_error']) for x in hs),first_error_min=s['min'],first_error_median=s['median'],first_error_max=s['max'],first_negative_repetition=f(r['first_negative_step'])))
    po=[]
    for p in pairs:
        ss=[r for r in side if (r['model'],r['op'],r['pair'])==(p['model'],p['op'],p['pair'])];assert len(ss)==2;ts=[r['first_error_min'] for r in ss if r['first_error_min'] is not None];kind=ss[0]['endpoint_class'];status='failure' if ts else 'no_failure_within_window' if all(r['outcome']=='no_failure_within_window' for r in ss) else 'unevaluable'
        po.append(dict(model=p['model'],initialization=ss[0]['initialization'],data_condition=ss[0]['data_condition'],op=p['op'],pair=p['pair'],endpoint_class=kind,outcome=status,distance=float(p['distance']),failing_sides=sum(r['outcome']=='failure' for r in ss),endpoint_wrong_sides=sum(r['endpoint_correct'] is False for r in ss),right_censored_side_trials=sum(r['right_censored_trials'] for r in ss),first_error_min=min(ts) if ts else None,strict_mechanism=b(p['strict_mechanism_reproduced'])))
    table('E3_ENDPOINT_OUTCOME.csv',po);table('E3_ENDPOINT_SIDE_OUTCOME.csv',side)
    cross=[]
    for kind in ['shared','distinct','unclassified']:
        ss=[r for r in po if r['endpoint_class']==kind];cross.append(dict(endpoint_class=kind,pairs=len(ss),failure=sum(r['outcome']=='failure' for r in ss),no_failure_within_window=sum(r['outcome']=='no_failure_within_window' for r in ss),unevaluable=sum(r['outcome']=='unevaluable' for r in ss),failure_rate=sum(r['outcome']=='failure' for r in ss)/len(ss) if ss else None))
    table('E3_ENDPOINT_CROSSTAB.csv',cross);d2=[]
    for scope in ['all240','paired160','shared','distinct','unclassified']:
        ss=[r for r in side if scope=='all240' or scope=='paired160' and r['endpoint_class']!='unpaired_input' or r['endpoint_class']==scope]
        for correct in [True,False,None]:
            tt=[r for r in ss if r['endpoint_correct'] is correct];d2.append(dict(scope=scope,endpoint_readout='correct' if correct else 'wrong' if correct is False else 'unavailable',n=len(tt),failure=sum(r['outcome']=='failure' for r in tt),no_failure_within_window=sum(r['outcome']=='no_failure_within_window' for r in tt),unevaluable=sum(r['outcome']=='unevaluable' for r in tt)))
    table('E3_READOUT_D2_CROSSTAB.csv',d2);bad=[r for r in side if r['endpoint_class']=='distinct' and r['endpoint_correct'] is False];table('E3_DISTINCT_WRONG_READOUT.csv',bad or [dict(status='none')]);counts=[]
    for m in models:
        ss=[r for r in po if r['model']==m['model']];counts.append(dict(model=m['model'],initialization=m['initialization_seed'],data_condition=m['data_condition'],shared=sum(r['endpoint_class']=='shared' for r in ss),distinct=sum(r['endpoint_class']=='distinct' for r in ss),unclassified=sum(r['endpoint_class']=='unclassified' for r in ss),strict=sum(r['strict_mechanism'] for r in ss)))
    table('E3_ENDPOINT_COUNTS_BY_MODEL.csv',counts)
    timing=[]
    for kind in ['shared','distinct','unpaired_input']:
        ss=[r for r in side if r['endpoint_class']==kind];s=stats([r['first_error_min'] for r in ss]);timing.append(dict(endpoint_class=kind,total_sides=len(ss),failing_sides=s['n'],no_failure_sides=sum(r['outcome']=='no_failure_within_window' for r in ss),minimum=s['min'],median=s['median'],maximum=s['max']))
    table('E3_ENDPOINT_TIMING.csv',timing)
    old=rows('v054h_dynamics/fixed_points.csv');oldagree=sum(b(r['h_star_output_correct'])==b(r['V054T_unbroken']) for r in old)
    text='# E3 終点分類と全破綻結果\n\n破綻＝凍結規則のhold内に1個以上の誤り（有限性も元評価で検査）。prefix/suffix誤りだけをhold破綻に含めない。k16/64/256/1024×16履歴=各側64試行。片側でも破綻すればpairは破綻あり。CSVは全80pair、側別CSVは全240診断側（うち対になった160側）。\n\n'+mdtable(cross,['endpoint_class','pairs','failure','no_failure_within_window','unevaluable','failure_rate'])+'\n共有候補46/80=57.5%。別終点を無限長成功と呼ばない。no_failure_within_windowの全試行はそれぞれのkで右打切り、最大1024まで。破綻例でも短いkの試行は打切り得る。全打切り数・prefix誤りは側別CSVで追跡可能。\n\n## 初期化・データ条件別\n'+mdtable(counts,['model','shared','distinct','unclassified','strict'])+'\n## D2形式（側単位）\n'+mdtable(d2,['scope','endpoint_readout','n','failure','no_failure_within_window','unevaluable'])+'\n## 破綻時刻（各側で観測された最初のhold誤り）\n'+mdtable(timing,['endpoint_class','total_sides','failing_sides','no_failure_sides','minimum','median','maximum'])+f'\n元D2は全36側（3旧モデル×12文脈）で終点読み出しとV054T窓内結果の対応が{oldagree}/36。共有終点pairの的中率ではない。E3では対応する単位はall240（20モデル×12文脈）。80pair／160側はそのうち同じ入力対で両carry側を保持できる部分集合なので36と直接分母比較しない。旧Hでは20,000反復・545次元・同構造で、E3も同じ反復・状態次元・移動量閾値を使う。ただし旧V054Tの最大k4096とE3の最大k1024は異なり、E3はrhoと距離<1e−3も厳密機構に要求。旧posthocの12pair中10共有という観察と、D2の36/36と、E3の46/80は別指標。\n\n別終点で誤読出しの全ケースはE3_DISTINCT_WRONG_READOUT.csv。終点正読出しでも過渡誤答があればD2対応は失敗するため、そのセルを隠さない。\n'
    write('E3_ENDPOINT_OUTCOME_AUDIT.md',text)
    distances=[]
    for p in po:
        z=np.load(source(X/f"E3_independent_replication/diagnostics/{p['model']}_trajectory.npz"));dd=js(X/f"E3_independent_replication/diagnostics/{p['model']}_diagnostic.json")['records'];ix=[r['index'] for r in dd if r['op']==p['op'] and r['pair']==p['pair']];assert len(ix)==2;distance=float(np.linalg.norm(z['end'][ix[0]].astype(np.float64)-z['end'][ix[1]].astype(np.float64)));assert np.isclose(distance,p['distance'],atol=1e-6,rtol=1e-6)
        distances.append(dict(**p,distance_FP64_saved_endpoints=distance,ratio_to_frozen_threshold=distance/.001,near_threshold_factor2=.0005<=distance<=.002,near_threshold_10percent=.0009<=distance<=.0011,state_dimension=545,threshold=.001))
    table('E3_ENDPOINT_DISTANCE_AUDIT.csv',distances);ss=stats([r['distance'] for r in distances]);by=[]
    for kind in ['shared','distinct']:
        a=stats([r['distance'] for r in distances if r['endpoint_class']==kind]);by.append(dict(endpoint_class=kind,**a))
    write('E3_ENDPOINT_DISTANCE_AUDIT.md','# 終点距離監査\n\n凍結済み規則：545次元のhidden、20000反復、両側収束（最後100移動の最大<1e−6）かつユークリッド距離<1e−3を共有候補、>=1e−3を別終点。既存FP32終点をFP64へ変換して距離だけを独立再計算し全80組で一致。再推論なし。\n\n'+mdtable(by,['endpoint_class','n','mean','median','min','max','SD'])+f"\n全80距離中央値{ss['median']}、最小{ss['min']}、最大{ss['max']}。閾値近傍は監査用に±10%帯とfactor2帯を併記（分類の閾値は変更しない）。±10%帯{sum(r['near_threshold_10percent'] for r in distances)}件、factor2帯{sum(r['near_threshold_factor2'] for r in distances)}件。全件距離と比率はCSV参照。\n\nこれは有限精度・有限反復の数値近接であり、数学的に同じ固定点の証明ではない。両側局所安定も大域吸引や真の固定点存在の証明ではない。距離を0と丸めたり、結果に合わせて閾値を変更したりしない。\n")
    return po,side,models

def timing():
    pp={str(r['pid']):r for r in index_probes()};raw=rows('v054t_transition/results_per_problem.csv');tt=[]
    for r in raw:
        p=pp[r['pid']];xor=int(r['prediction'],16)^int(p['answer'],16);start=int(r['hold_start']);k=int(r['k']);masked=(xor>>start)&((1<<k)-1);first=(masked&-masked).bit_length() if masked else None
        tt.append(dict(study='V054T/H',model='V054_s'+r['seed'],op=r['op'],state=r['state'],pair=r['pair'],k=k,prefix=r['prefix'],suffix=r['suffix'],first_hold_error=first,prefix_error=bool(xor&((1<<start)-1))))
    for r in rows(X/'E3_independent_replication/E3_HOLD_RESULTS.csv'):
        tt.append(dict(study='E3',model=r['model'],op=r['op'],state=r['state'],pair=r['pair'],k=int(r['k']),prefix=r['prefix'],suffix=r['suffix'],first_hold_error=f(r['first_hold_error']),prefix_error=b(r['prefix_error'])))
    table('TIMING_HISTORY_OBSERVATIONS.csv',tt);out=[];groups=defaultdict(list)
    for r in tt:groups[(r['study'],r['model'],r['op'],r['state'],r['pair'])].append(r)
    for key,rs in groups.items():
        if not any(r['first_hold_error'] is not None for r in rs):continue
        for k in sorted({r['k'] for r in rs}):
            a=[r for r in rs if r['k']==k];s=stats([r['first_hold_error'] for r in a]);pre=defaultdict(list)
            for r in a:
                if r['first_hold_error'] is not None:pre[r['prefix']].append(r['first_hold_error'])
            ps=stats([statistics.median(v) for v in pre.values()]);out.append(dict(study=key[0],model=key[1],op=key[2],state=key[3],pair=key[4],axis='across_prefixes_at_fixed_k',k=k,prefix='all',suffix='all',trials=len(a),failed_trials=s['n'],right_censored=len(a)-s['n'],min_step=s['min'],median_step=s['median'],max_step=s['max'],SD=s['SD'],range=s['max']-s['min'] if s['n'] else None,prefix_median_SD=ps['SD'],prefixes_with_errors=ps['n'],prefix_error_trials=sum(r['prefix_error'] for r in a),note='observed failures only; suffix variation also included'))
        for prefix,suffix in sorted({(r['prefix'],r['suffix']) for r in rs}):
            a=[r for r in rs if r['prefix']==prefix and r['suffix']==suffix];s=stats([r['first_hold_error'] for r in a]);out.append(dict(study=key[0],model=key[1],op=key[2],state=key[3],pair=key[4],axis='across_k_at_fixed_history',k='all',prefix=prefix,suffix=suffix,trials=len(a),failed_trials=s['n'],right_censored=len(a)-s['n'],min_step=s['min'],median_step=s['median'],max_step=s['max'],SD=s['SD'],range=s['max']-s['min'] if s['n'] else None,prefix_error_trials=sum(r['prefix_error'] for r in a),note='different k can imply different encoded prefixes; no new cases generated'))
    old_d2=[]
    for r in rows('v054h_dynamics/fixed_points.csv'):
        obs=groups[('V054T/H','V054_s'+r['seed'],r['op'],r['state'],r['pair'])];failed=any(z['first_hold_error'] is not None for z in obs);correct=b(r['h_star_output_correct']);old_d2.append(dict(seed=r['seed'],op=r['op'],state=r['state'],pair=r['pair'],endpoint_correct=correct,observed_hold_failure=failed,matching_D2=correct!=failed,trials=len(obs),max_k=max(z['k'] for z in obs)))
    assert len(old_d2)==36 and all(r['matching_D2'] for r in old_d2);table('V054H_D2_INDEPENDENT.csv',old_d2)
    table('TIMING_HISTORY_INDEPENDENCE.csv',out);sums=[]
    for study in ['V054T/H','E3']:
        for axis in ['across_prefixes_at_fixed_k','across_k_at_fixed_history']:
            aa=[r for r in out if r['study']==study and r['axis']==axis and r['failed_trials']>=2];s=stats([r['range'] for r in aa]);sums.append(dict(study=study,axis=axis,groups_with_2plus_observed=s['n'],range_median=s['median'],range_max=s['max'],groups_nonzero_range=sum(r['range']>0 for r in aa)))
    table('TIMING_HISTORY_SUMMARY.csv',sums)
    write('TIMING_HISTORY_INDEPENDENCE.md','# 破綻時刻の履歴依存性（保存データのみ）\n\nV054Tの保存整数予測と保存正答のXORからhold内の最初の誤りを独立復元。元hold_offsetは0始まりなので、本表はE3と揃えて1始まり。prefixの先行誤りがあってもhold内を別採点し、prefix_errorを残した。E3は保存済みfirst_hold_error（1始まり）を使用。新しい系列・再評価なし。\n\n'+mdtable(sums,['study','axis','groups_with_2plus_observed','range_median','range_max','groups_nonzero_range'])+'\n固定kの全prefix/suffixでの範囲・標本SDに加え、各prefix内でsuffixをまとめた中央値のprefix間SDを報告。固定prefix/suffixではk間範囲も報告するが、kにより生成された前置きの実際のビットが違う可能性があり純粋なk効果とはしない。kごとの試行数・打切り数を保存し、誤答なしを0stepと代入しない。単一観測のSDは欠測。元V054Hの選択prefix軌道は同一モデルの反復であって別の独立履歴データではない。\n\n厳密な履歴非依存なら範囲0が必要だが、観測範囲が非ゼロの群が存在する限り、一般的な厳密非依存とは表現できない。近似的な一致・小さい履歴変動が論点であり、保持入力反復での時期予測が観測手順に近いことも記述する。\n')
    return sums

def initialization(models):
    records=[];old={};evidence=[]
    for dirname,field in [('IronMath_V053','initial_sha256'),('IronMath_V054','initial_sha256'),('IronMath_V055','initial_SHA256')]:
        rr=rows(dirname+'/cpu_memory.csv');vv=[r for r in rr if r.get('stage')=='training' and r.get(field)];old[dirname]=set(r[field] for r in vv);evidence.append(dict(source=dirname,records=len(vv),unique_hashes=len(old[dirname]),overlap_with_E3=len(old[dirname]&{r['initial_SHA256'] for r in models}),hash_field=field))
    for seed in range(100,110):
        a=[r for r in models if int(r['initialization_seed'])==seed];assert len(a)==2;records.append(dict(initialization=seed,c0_initial_SHA256=next(r['initial_SHA256'] for r in a if r['data_condition']=='0'),c1_initial_SHA256=next(r['initial_SHA256'] for r in a if r['data_condition']=='1'),paired_equal=len({r['initial_SHA256'] for r in a})==1,other_initializations_duplicate=sum(r['initial_SHA256']==a[0]['initial_SHA256'] and int(r['initialization_seed'])!=seed for r in models)))
    table('E3_INITIALIZATION_AUDIT.csv',records);table('E3_OLD_HASH_LOG_COMPARISON.csv',evidence)
    for p in [X/'scripts/core.py',X/'scripts/audit_independent.py',W/'IronMath_V053/run.py',W/'IronMath_V054/run.py',W/'IronMath_V055/run.py',W/'IronMath_V055/core.py',W/'IronMath_V054/model.py']:
        if p.exists():source(p)
    dump('E3_PREVIOUS_INITIALIZATION_VERIFICATION.json',js(X/'audit/INITIALIZATION_REPRODUCTION.json'))
    write('E3_INITIALIZATION_AUDIT.md','# 初期化の独立性監査\n\n'+mdtable(records,['initialization','c0_initial_SHA256','c1_initial_SHA256','paired_equal','other_initializations_duplicate'])+'\n10個のSHAは全て異なり、各初期化のc0/c1は同じ初期重み。解析単位は10初期化であり20独立初期化ではない。SHAは同一テンソル列ではないことを示すが、確率的独立性の数学的証明ではない。seed100〜109・同じ初期化手順を利用した独立乱数初期化という実験設計上の意味に限定する。\n\n## 旧モデルの保存ログ\n'+mdtable(evidence,['source','records','unique_hashes','overlap_with_E3','hash_field'])+'\nこれらはcheckpointファイルSHAではなく初期state_dictのキーと各テンソルのバイト列を順番にhashした値。V054 B0とE3は同一構造・キー・作成順序で、旧0/1/2は先行監査で初期テンソルhashを再構成して一致済み（INITIALIZATION_REPRODUCTION.json、コピーを併置）。今回はモデルを新規生成せず、その証拠を照合した。V053やV054の別条件・V055も全て同一構造と一括して扱わず、構造別の比較可能性は別のcheckpointスキーマ監査CSVに記録する。構造やキーが異なるhashの不一致は独立性の証拠としない。\n')
    return evidence

def main():
    O.mkdir(exist_ok=True);dump('RUN_SCOPE.json',dict(start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_training=False,new_inference=False,original_files_readonly=True,output=str(O),rules='original E2/E3 frozen criteria unchanged'))
    for p in [X/'E2_causal_intervention/E2_RULES.md',X/'E2_causal_intervention/FROZEN.json',X/'E3_independent_replication/E3_RULES.md',X/'E3_independent_replication/FROZEN.json',X/'audit/INDEPENDENT_AUDIT.json']:source(p)
    e2r,e2s=e2();po,side,models=e3();timing();initialization(models)
    dump('SUMMARY.json',dict(E2_at8_floor=[r for r in e2s if r['population']=='baseline_failing_10' and r['at']==8],E3_pair_cross=rows(O/'E3_ENDPOINT_CROSSTAB.csv'),E3_D2=rows(O/'E3_READOUT_D2_CROSSTAB.csv')))
    # Outputs read above are excluded from old-source manifest.
    SOURCES.update({})
    dump('INPUT_SHA256.json',{k:v for k,v in SOURCES.items() if not Path(k).is_relative_to(O)})
    changed=[p for p,v in SOURCES.items() if not Path(p).is_relative_to(O) and sha(Path(p))!=v['sha256']];assert not changed
    dump('NUMERICAL_PHASE_STATUS.json',dict(completed=True,original_input_changes=changed,new_model_calls=0,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('Numerical phase complete. Inspect SUMMARY and cross tables before revision plan.',flush=True)
if __name__=='__main__':main()
