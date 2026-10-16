"""Final local edits from saved CSVs/tensors. No model construction, inference or training."""
import os
from pathlib import Path
O=Path(__file__).resolve().parent;X=O.parent;W=O.parents[2];P=X/'v072_figures_check';A=X/'v072_audit'
os.environ['MPLCONFIGDIR']=str(O/'mpl_cache');os.environ['OMP_NUM_THREADS']='1'
import csv,json,hashlib,statistics,re,shutil,datetime
from collections import defaultdict,Counter
import numpy as np
INPUTS={}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for a in iter(lambda:f.read(1048576),b''):h.update(a)
    return h.hexdigest()
def src(p):
    p=Path(p);INPUTS.setdefault(str(p),dict(bytes=p.stat().st_size,SHA256=sha(p)));return p
def rows(p):
    with src(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(n,s):
    p=O/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(s,encoding='utf-8');assert p.read_text(encoding='utf-8')==s and '\ufffd' not in s
def dump(n,v):write(n,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(n,rr):
    p=O/n;p.parent.mkdir(exist_ok=True,parents=True);fields=list(dict.fromkeys(k for r in rr for k in r))
    with p.open('w',encoding='utf-8',newline='') as f:q=csv.DictWriter(f,fieldnames=fields);q.writeheader();q.writerows(rr)
def timing():
    rr=rows(A/'TIMING_HISTORY_OBSERVATIONS.csv');g=defaultdict(list)
    for r in rr:g[tuple(r[k] for k in ['study','model','op','state','pair','prefix','suffix'])].append(r)
    detail=[];conflicts=[]
    for key,rs in g.items():
        times=[int(float(r['first_hold_error'])) for r in rs if r['first_hold_error']];known=bool(times);t=min(times) if known else None
        if len(set(times))>1:conflicts.append(dict(kind='timing varies across k',key=str(key),observed=str(times)))
        primary={256,1024} if key[0]=='E3' else {256,1024,4096}
        for r in rs:
            k=int(r['k']);observed=int(float(r['first_hold_error'])) if r['first_hold_error'] else None;category='unknown_break_right_censored' if not known else 'k_shorter_than_known_break' if k<t else 'k_equal_to_known_break' if k==t else 'k_longer_than_known_break'
            if known and ((k<t and observed is not None) or (k>=t and observed!=t)):conflicts.append(dict(kind='censor/event mismatch',key=str(key),k=k,known_t=t,observed=observed))
            detail.append(dict(**r,known_first_hold_error=t,category=category,right_censored=observed is None,used_in_observed_time_statistics=observed is not None,used_in_uniform_long_k_comparison=known and k in primary,primary_k_gt_t=known and k in primary and k>t))
    table('TIMING_CENSOR_DETAILS.csv',detail);summary=[]
    for study in ['E3','V054T/H']:
        for k in sorted({int(r['k']) for r in detail if r['study']==study}):
            ss=[r for r in detail if r['study']==study and int(r['k'])==k];c=Counter(r['category'] for r in ss);summary.append(dict(study=study,k=k,trials=len(ss),known_failing_history_trials=sum(r['known_first_hold_error'] is not None for r in ss),short_right_censored=c['k_shorter_than_known_break'],equal_event_observed=c['k_equal_to_known_break'],long_event_observed=c['k_longer_than_known_break'],unknown_break_right_censored=c['unknown_break_right_censored'],observed_events=sum(r['used_in_observed_time_statistics'] for r in ss)))
    table('TIMING_CENSOR_SUMMARY.csv',summary);histories=[]
    for key,rs in g.items():
        primary={256,1024} if key[0]=='E3' else {256,1024,4096};ss=[r for r in rs if int(r['k']) in primary];ts=[int(float(r['first_hold_error'])) for r in ss if r['first_hold_error']]
        if not ts:continue
        valid=len(ss)==len(primary) and len(ts)==len(primary) and len(set(ts))==1 and min(primary)>max(ts);histories.append(dict(study=key[0],model=key[1],op=key[2],state=key[3],pair=key[4],prefix=key[5],suffix=key[6],comparison_k=','.join(map(str,sorted(primary))),observed_first_error=ts[0],all_observed_and_equal=valid))
        if not valid:conflicts.append(dict(kind='uniform long-k comparison failure',key=str(key),times=str(ts)))
    table('TIMING_LONG_K_HISTORY_CHECK.csv',histories);dump('TIMING_CONFLICTS.json',conflicts)
    if conflicts:raise RuntimeError('Timing inconsistency: preserve report, do not edit manuscript')
    maxima={s:max(r['known_first_hold_error'] for r in detail if r['study']==s and r['known_first_hold_error'] is not None) for s in ['E3','V054T/H']};n=Counter(r['study'] for r in histories)
    lines='\n'.join(f"|{r['study']}|{r['k']}|{r['trials']}|{r['short_right_censored']}|{r['equal_event_observed']}|{r['long_event_observed']}|{r['unknown_break_right_censored']}|" for r in summary)
    write('TIMING_CENSOR_RECHECK.md',f'''# 保持長と打切りの再確認

保存済みTIMING_HISTORY_OBSERVATIONS.csvのみ使用。元の追加監査はfirst_hold_errorが欠測の試行を右打切りとして数え、時刻の平均・SD・範囲から除外した。時刻0やk+1を代入していない。集計コードは観測時刻が存在する試行のみをstatsへ渡し、k間範囲の概要は2件以上の観測群だけを対象にしていた。短いkの欠測から、真の破綻時刻がkで変わったとも変わらないとも推定していない。

表のknown breakは同一モデル・入力対・carry・prefix・suffixで長い保持窓から観測された時刻。長さ変更後の新しい系列を生成・評価していない。元の最大観測時刻はE3={maxima['E3']}、旧={maxima['V054T/H']}（このCSVは1始まり）。全保持長で一度も壊れない履歴は真のT不明として別列にし、有限Tがあると仮定しない。

|系列|k|全試行|k<T:右打切り|k=T:最終位置で誤答|k>T:誤答観測|T不明:右打切り|
|---|---:|---:|---:|---:|---:|---:|
{lines}

E3のk16/64、旧のk8/16/64には、破綻時刻より短いため右打切りとなる試行と、すでに誤答を観測できる試行が混在する。短いkを一律に除いた集計ではなかったが、短い窓内で誤答がないことを『破綻時刻不変』の証拠として扱ってはいけない。

k=Tの場合も最終評価位置で誤りを観測できる。E3計32試行、旧計12試行でこの境界例がある。これはk<Tとは異なり欠測ではない。ただし今回の本文比較はこの境界を避け、全ての観測破綻時刻より厳密に長い共通kだけに限定する。

本文で使う一律の長さはE3={{256,1024}}、旧={{256,1024,4096}}。同一保存履歴の比較はE3 {n['E3']}群、旧{n['V054T/H']}群で、全群の時刻範囲0、全比較でk>Tを確認。5節の旧k256/1024/4096と、8節の算術k>=256の定義と矛盾しない。本文の時刻範囲そのものや旧の0始まり表示値は変更せず、比較に用いる保持長の括弧だけを限定する。前置き間の変動はこれとは別の問題であり、厳密な履歴非依存を主張しない。
''')
    return dict(maxima=maxima,compared_histories=dict(n),conflicts=0)
def initials():
    e3=rows(A/'E3_INITIALIZATION_AUDIT.csv');new={r['c0_initial_SHA256'] for r in e3};assert len(new)==10 and all(r['c0_initial_SHA256']==r['c1_initial_SHA256'] for r in e3);out=[]
    for study,key in [('IronMath_V053','initial_sha256'),('IronMath_V054','initial_sha256'),('IronMath_V055','initial_SHA256')]:
        rr=rows(W/study/'cpu_memory.csv');rr=[r for r in rr if r.get('stage')=='training' and r.get(key)]
        if study=='IronMath_V054':rr=[r for r in rr if r['model'] in ['B0','B1']]
        old={r[key] for r in rr};matches=sorted(new&old);out.append(dict(study=study,conditions='B0/B1 only' if study=='IronMath_V054' else 'all training rows',old_records=len(rr),unique_old_hashes=len(old),new_hashes=10,matches=len(matches),matching_hashes=';'.join(matches)))
    table('INITIAL_HASH_COMPARISON.csv',out)
    if any(r['matches'] for r in out):dump('INITIALIZATION_CONFLICT.json',out);raise RuntimeError('Initial hash collision: do not append no-match claim')
    return out
def norms():
    out=[]
    for mid in [f'i{s}_c{c}' for s in range(100,110) for c in [0,1]]:
        p=src(X/f'E3_independent_replication/diagnostics/{mid}_trajectory.npz');z=np.load(p,allow_pickle=False);end=z['end'];assert end.shape==(12,545) and np.isfinite(end).all();records=json.loads(src(X/f'E3_independent_replication/diagnostics/{mid}_diagnostic.json').read_text(encoding='utf-8'))['records'];v=np.linalg.norm(end.astype(np.float64),axis=1)
        for r in records:out.append(dict(model=mid,index=r['index'],op=r['op'],state=r['state'],pair=r['pair'],state_dimension=545,endpoint_L2_norm=float(v[r['index']]),source=p.name))
    assert len(out)==240;v=[r['endpoint_L2_norm'] for r in out];result=dict(population='all20 models x12 diagnostic trajectories;240 saved545-dimensional endpoints',precision='FP64 L2 norm of saved FP32 endpoint tensors',n=240,median=float(np.median(v)),minimum=min(v),maximum=max(v));table('E3_ENDPOINT_NORMS.csv',out);dump('E3_ENDPOINT_NORMS_SUMMARY.json',result);return result
def figure2():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    cc=rows(P/'FIGURE2_SOURCE.csv');table('FIGURE2_SOURCE.csv',cc);fig,ax=plt.subplots(figsize=(10.1,3.9),layout='constrained');x=np.arange(10)
    for c,offset in [('0',-.19),('1',.19)]:
        a=[next(r for r in cc if r['initialization']==str(s) and r['data_condition']==c) for s in range(100,110)];bottom=np.zeros(10)
        for field,color,label in [('shared','#dc7c24','Shared endpoint'),('distinct','#2864a0','Distinct')]:
            v=np.array([int(r[field]) for r in a]);ax.bar(x+offset,v,.34,bottom=bottom,color=color,label=label if c=='0' else None,hatch='' if c=='0' else '///',edgecolor='white',lw=.6);bottom+=v
        for xx in x+offset:ax.text(xx,4.08,'c'+c,ha='center',fontsize=8)
    ax.set_xticks(x,[str(s) for s in range(100,110)]);ax.set_xlabel('Initialization seed (paired streams c0 / c1)');ax.set_ylabel('Input pairs per trained model');ax.set_yticks([0,1,2,3,4]);ax.set_ylim(0,4.7);ax.legend(ncol=2,loc='upper center',fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    assert ax.get_legend_handles_labels()[1]==['Shared endpoint','Distinct'];fig.savefig(O/'figures/figure2_e3_initializations.pdf');fig.savefig(O/'figures/figure2_e3_initializations.png',dpi=180);plt.close(fig)
def main():
    O.mkdir(exist_ok=True);ti=timing();ini=initials();no=norms();old=src(P/'main_v0.7.2.tex');s=old.read_text(encoding='utf-8');original=s;edits=[]
    def replace(a,b,kind):
        nonlocal s
        assert s.count(a)==1,(kind,s.count(a));edits.append(dict(kind=kind,old=a,new=b,old_line=s[:s.index(a)].count('\n')+1));s=s.replace(a,b,1)
    replace(r'(E3: $k\in\{16,64,256,1024\}$; original models: $k\in\{8,16,64,256,1024,4096\}$)',r'(for all compared hold lengths longer than the break step: E3, $k\in\{256,1024\}$; original models, $k\in\{256,1024,4096\}$)','timing comparison lengths only')
    a='Distinct hashes show that the initializations are not identical; they are not a proof of statistical independence.';replace(a,a+' None of the ten E3 initial-weight hashes matches the initial-weight hashes of V053, V054 B0/B1 or V055.','initialization no-match sentence')
    med,lo,hi=[format(no[k],'.4f') for k in ['median','minimum','maximum']]
    replace(r'(Euclidean distance in the 545-dimensional hidden state; \CHECK{typical state norm is not tabulated in the saved endpoint-distance audit})',f'(Euclidean distance in the 545-dimensional hidden state, whose endpoint norms have median ${med}$, range $[{lo}, {hi}]$)','endpoint L2 norm of all240 saved vectors')
    replace(r'Labels give the operand bits $(a_t b_t)$',r'Legend entries give the operand bits $(a,b)$','figure1 caption')
    replace('and the literature search.','and carried out the literature search.','acknowledgments wording')
    # Drop unused numeric CHECK definition/example after the final usage is filled.
    replace('\\newcommand{\\CHECK}[1]{\\textcolor{red}{[CHECK: #1]}}\n','','remove unused CHECK macro')
    replace('% Markers:  \\CHECK{...} = fact or wording to verify against reports;  \\TODO{...} = missing content.','% Marker: \\TODO{...} = missing content.','remove obsolete CHECK example')
    assert '\\CHECK' not in s
    backup=O/'source_originals';backup.mkdir(exist_ok=True);shutil.copyfile(old,backup/old.name)
    for name in ['refs_v0.7.2.bib','jmlr2e.sty','natbib.sty','booktabs.sty','plainnat.bst']:shutil.copyfile(src(P/name),O/name)
    (O/'figures').mkdir(exist_ok=True)
    for name in ['figure1_arithmetic_margins.pdf','figure3_g1_abl1.pdf']:shutil.copyfile(src(P/'figures'/name),O/'figures'/name)
    write('main_v0.7.2.tex',s);dump('LOCAL_EDITS.json',edits);figure2();dump('INPUT_SHA256.json',INPUTS)
    write('CHECK_FILL_LOG_FINAL.md',f'''# v0.7.2 最終局所修正

終点ノルム：{no['population']}。保存済みFP32 endテンソルをFP64へ変換してユークリッドL2ノルムだけを集計。モデルをロード/構築/実行せず、新しい状態を作っていない。中央値{no['median']:.15g}、最小{no['minimum']:.15g}、最大{no['maximum']:.15g}。本文へは小数4桁で{med}・[{lo},{hi}]を転記。全240値はE3_ENDPOINT_NORMS.csv。距離をノルムとして代用していない。

初期化10ハッシュはV053、V054 B0/B1（T0/T1除外）、V055の保存初期ハッシュと不一致。INITIAL_HASH_COMPARISON.csv。hash不一致を確率的独立性の証明とは記述しない。

Timing括弧は共通の長いkに限定。E3最大観測時刻78、旧88のため比較kは全て厳密に大きい。短いkの打切りとk=Tの境界をTIMING_CENSOR_RECHECK.mdで明記。5節と8節の比較長に矛盾なし。科学的本文の変更は指定の括弧・初期化1文・ノルムCHECK・図1キャプション・謝辞のみ。図2は凡例だけ変更し、棒の値は不変。全CHECK使用・不要なmacro定義・例示コメントを除去。

新学習0・新推論0・新Test評価0。その他の本文・参考文献・旧図1/3は変更しない。元原稿と旧成果物を新しい保存先から参照し、原本を上書きしていない。
''')
    dump('PREPARED.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),timing=ti,initial_hash_comparisons=ini,endpoint_norms=no,CHECK_uses=0,new_training=0,new_inference=0,new_Test=0));print('Local revisions complete; endpoint norms:',no,flush=True)
if __name__=='__main__':main()
