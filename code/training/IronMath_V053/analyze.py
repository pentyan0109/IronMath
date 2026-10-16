"""Independent saved-result audit, correlation reconstruction and report. No training/inference."""
import csv,gzip,hashlib,itertools,json,math,statistics,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text())
def read(name):return list(csv.DictReader((R/name).open(encoding='utf-8-sig')))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
def write(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (R/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:('NA' if v is None else v) for k,v in r.items()} for r in rows)
def eq_float(a,b):
    if b is None:return a in ('NA','')
    return abs(float(a)-b)<1e-10
def records():
    with gzip.open(R/'predictions.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)
status=json.loads((R/'STOPPED.json').read_text());assert status['status']=='training_and_all_evaluation_completed_stopped'
for name,h in json.loads((R/'STARTED.json').read_text())['registered_SHA'].items():assert sha(R/name)==h
manifest=json.loads((R/'dataset_manifest.json').read_text());testdata={}
for name,entry in manifest['data'].items():
    assert sha(R/'data'/f'{name}.json')==entry['sha256']
    rows=json.loads((R/'data'/f'{name}.json').read_text())
    for p in rows:
        assert 0<=p['a']<(1<<p['width']) and 0<=p['b']<(1<<p['width'])
        assert p['op']=='ADD' or p['a']>=p['b']
    if '_Test' in name:testdata[(rows[0]['width'],rows[0]['op'])]={p['id']:p for p in rows}
native_saved=read('error_correlation.csv');agreement_saved=read('agreement_analysis.csv');score_saved=read('evaluation_results.csv')
saved_corr={(x['condition'],int(x['seed']),int(x['width']),x['op']):x for x in native_saved}
saved_agree={(x['condition'],int(x['seed']),int(x['width']),x['op']):x for x in agreement_saved}
saved_score={(x['condition'],int(x['seed']),int(x['width']),x['op'],x['direction']):x for x in score_saved}
counter=0;groupcount=0;audit_pairs={};phases=defaultdict(lambda:[0,0]);first_errors=Counter();invalid=Counter();examples={};numeric_reuse=Counter()
for group_key,iterator in itertools.groupby(records(),key=lambda x:(x['condition'],x['seed'],x['width'],x['op'])):
    group=list(iterator);groupcount+=1;assert len(group)==1024;condition,seed,width,op=group_key;truths=testdata[(width,op)]
    split={d:[x for x in group if x['direction']==d] for d in ('R','L')}
    assert len(split['R'])==len(split['L'])==512
    assert [x['id'] for x in split['R']]==[x['id'] for x in split['L']]
    assert {x['id'] for x in split['R']}==set(truths)
    for direction,rows in split.items():
        correct=bc=0
        for r in rows:
            counter+=1;p=truths[r['id']];n=width+int(op=='ADD')
            assert (r['a'],r['b'],r['length'])==(p['a'],p['b'],n)
            answer=r['a']+r['b'] if op=='ADD' else r['a']-r['b'];assert answer==r['answer']==p['answer']
            bb=r['prediction_bits'];assert len(bb)==n and all(v in (0,1,None) for v in bb)
            valid=all(v is not None for v in bb);pred=sum(v<<i for i,v in enumerate(bb)) if valid else None
            wrong=[i for i,v in enumerate(bb) if v!=((answer>>i)&1)];processing=wrong if direction=='R' else [n-1-i for i in wrong]
            first=min(processing) if processing else None;firstphys=(first if direction=='R' else n-1-first) if first is not None else None
            assert r['prediction']==pred and r['numeric_valid']==int(valid) and r['correct']==int(valid and pred==answer)
            assert r['wrong_positions']==wrong and r['wrong_bits']==len(wrong)
            assert r['first_error_lsb']==(min(wrong) if wrong else None) and r['first_error_processing']==first and r['first_processing_error_physical']==firstphys
            assert abs(r['bit_accuracy']-(n-len(wrong))/n)<1e-12
            assert r['nonfinite_positions']==[i for i,b in enumerate(bb) if b is None]
            h=sum(i*3<n for i in processing);mid=sum(n<=i*3<2*n for i in processing)
            assert (r['head_errors'],r['middle_errors'],r['tail_errors'])==(h,mid,len(processing)-h-mid)
            phase=[];incoming=[];state=longest=run=0
            for i in range(n):
                x=(r['a']>>i)&1;y=(r['b']>>i)&1;old=state
                state=int(x+y+state>=2) if op=='ADD' else int(x-y-state<0)
                phase.append('start' if not old and state else 'continue' if old and state else 'release' if old and not state else 'none');incoming.append(old)
                run=run+1 if state else 0;longest=max(longest,run)
            assert state==0 and longest==r['longest_propagation']
            assert dict(Counter(phase))==r['phase_opportunities'] and dict(Counter(phase[i] for i in wrong))==r['phase_errors']
            assert r['first_error_phase']==(phase[firstphys] if firstphys is not None else 'correct')
            for label in ('start','continue','release','none'):
                values=phases[(condition,direction,seed,width,op,label)];values[0]+=sum(x==label for x in phase);values[1]+=sum(phase[i]==label for i in wrong)
            if first is not None:first_errors[(condition,direction,seed,width,op,r['first_error_phase'])]+=1
            invalid[(condition,direction,seed,width,op)]+=not valid;correct+=r['correct'];bc+=n-len(wrong)
            label='success' if r['correct'] else 'failure';example_key=(condition,direction,op,label)
            if example_key not in examples or width>examples[example_key]['width']:examples[example_key]=r
        score=saved_score[(condition,seed,width,op,direction)];assert int(score['correct'])==correct and eq_float(score['bit_accuracy'],bc/(512*n))
    rr,ll=split['R'],split['L'];cr=np.array([bool(x['correct']) for x in rr]);cl=np.array([bool(x['correct']) for x in ll])
    counts=[int((cr&cl).sum()),int((cr&~cl).sum()),int((~cr&cl).sum()),int((~cr&~cl).sum())]
    same=np.array([a['prediction'] is not None and b['prediction'] is not None and a['prediction']==b['prediction'] for a,b in zip(rr,ll)])
    missing=np.array([a['prediction'] is None or b['prediction'] is None for a,b in zip(rr,ll)])
    samewrong=int((same&~cr&~cl).sum());diffwrong=int((~same&~missing&~cr&~cl).sum());misswrong=int((missing&~cr&~cl).sum())
    phi=float(np.corrcoef(cr.astype(float),cl.astype(float))[0,1]) if cr.var()>0 and cl.var()>0 else None
    overlaps=[];bothwrong_overlaps=[];expected_pairs={}
    for a,b,agree in zip(rr,ll,same):
        absent=a['prediction'] is None or b['prediction'] is None;ea=np.zeros(a['length'],bool);eb=np.zeros(a['length'],bool);ea[a['wrong_positions']]=True;eb[b['wrong_positions']]=True
        union=int((ea|eb).sum());value=float((ea&eb).sum()/union) if union and not absent else None
        if value is not None:
            overlaps.append(value)
            if not a['correct'] and not b['correct']:bothwrong_overlaps.append(value)
        lab=('Both Correct' if a['correct'] and b['correct'] else 'R Only' if a['correct'] else 'L Only' if b['correct'] else 'Both Wrong')
        expected_pairs[a['id']]=(lab,a['correct'],b['correct'],a['prediction'],b['prediction'],int(agree),value)
    stat=saved_corr[group_key]
    for field,v in zip(('both_correct','R_only','L_only','both_wrong'),counts):assert int(stat[field])==v
    assert int(stat['same_wrong'])==samewrong and int(stat['different_wrong'])==diffwrong and int(stat['both_wrong_missing'])==misswrong
    assert eq_float(stat['correctness_phi'],phi)
    assert eq_float(stat['error_overlap_mean'],statistics.mean(overlaps) if overlaps else None)
    assert eq_float(stat['error_overlap_both_wrong_mean'],statistics.mean(bothwrong_overlaps) if bothwrong_overlaps else None)
    assert eq_float(stat['oracle_selection_upper_bound'],float((cr|cl).mean()))
    ag=saved_agree[group_key];agreecount=int(same.sum());agreecorrect=int((same&cr).sum())
    assert int(ag['agreement_count'])==agreecount and int(ag['agreement_correct'])==agreecorrect and int(ag['agreed_wrong'])==samewrong
    assert eq_float(ag['agreement_accuracy'],agreecorrect/agreecount if agreecount else None)
    audit_pairs[group_key]=expected_pairs
    print('AUDIT',condition,seed,width,op,flush=True)
assert counter==110592 and groupcount==108
pair_rows=0
with gzip.open(R/'paired_predictions.jsonl.gz','rt',encoding='utf-8') as f:
    for line in f:
        r=json.loads(line);k=(r['condition'],r['seed'],r['width'],r['op']);expected=audit_pairs[k][r['id']]
        got=(r['class_label'],r['R_correct'],r['L_correct'],r['R_prediction'],r['L_prediction'],r['answers_agree'],r['error_bit_overlap'])
        assert got==expected;pair_rows+=1
assert pair_rows==55296
write('propagation_phase_errors.csv',[dict(zip(('condition','direction','seed','width','op','phase'),k),positions=v[0],error_bits=v[1],error_rate=v[1]/v[0] if v[0] else None) for k,v in phases.items()])
write('first_error_phase_counts.csv',[dict(zip(('condition','direction','seed','width','op','phase'),k),count=v) for k,v in first_errors.items()])
write('numeric_failures.csv',[dict(zip(('condition','direction','seed','width','op'),k),unanswered=v) for k,v in invalid.items()])
(R/'examples.json').write_text(json.dumps(list(examples.values()),ensure_ascii=False,indent=2),encoding='utf-8')
resources=read('cpu_memory.csv');train=[r for r in resources if r['stage']=='training'];assert len(train)==12
for seed in (0,1,2):
    rows=[r for r in train if int(r['seed'])==seed];assert len({r['initial_sha256'] for r in rows})==1
for r in train:
    assert int(r['parameters'])==1000334 and int(r['problem_presentations'])==16*int(r['updates'])
    assert int(r['training_bit_tokens'])==136*int(r['updates'])
    if r['condition']=='C1':assert int(r['updates'])==1200
    else:assert float(r['train_cpu_seconds'])>=90 and float(r['CPU_budget_overshoot'])<1
    # Reproduce the complete training ID/order hash from saved data and fixed RNG, not model state.
    import random
    rng=random.Random(50000+int(r['seed']));hasher=hashlib.sha256()
    pools={op:json.loads((R/'data'/f'{op}_Train.json').read_text()) for op in ('ADD','SUB')}
    for step in range(int(r['updates'])):
        for op in ('ADD','SUB'):
            for _ in range(8):
                p=pools[op][rng.randrange(len(pools[op]))];a,b=p['a'],p['b']
                if op=='ADD' and rng.randrange(2):a,b=b,a
                hasher.update(f"{p['id']}:{a}:{b}\n".encode())
    assert hasher.hexdigest()==r['problem_order_sha256']
cp_paths=[]
for p in (R/'checkpoints').rglob('*.pt'):
    x=torch.load(p,weights_only=False,map_location='cpu');assert sum(v.numel() for v in x['model'].values())==1000334
    assert all(torch.isfinite(v).all() for v in x['model'].values());cp_paths.append(str(p.relative_to(R)))
assert len(cp_paths)==24
# Numeric pair reuse is distinct from a duplicated input sequence with the same width.
known_numeric={(r['op'],min(r['a'],r['b']) if r['op']=='ADD' else r['a'],max(r['a'],r['b']) if r['op']=='ADD' else r['b']) for op in ('ADD','SUB') for split in ('Train','Validation') for r in json.loads((R/'data'/f'{op}_{split}.json').read_text())}
reuse=[]
for (w,op),book in testdata.items():
    n=sum((op,min(p['a'],p['b']) if op=='ADD' else p['a'],max(p['a'],p['b']) if op=='ADD' else p['b']) in known_numeric for p in book.values())
    reuse.append(dict(width=w,op=op,numeric_pair_seen_in_8bit_Train_Val=n,test_count=512,interpretation='different width means different input sequence; numeric reuse disclosed'))
write('cross_width_numeric_reuse.csv',reuse)
before=json.loads((R/'protected_sha_before.json').read_text());assert all(sha(R.parent/p)==h for p,h in before.items())
for name,h in json.loads((R/'STARTED.json').read_text())['registered_SHA'].items():assert sha(R/name)==h
for p in R.rglob('*.gz'):
    with gzip.open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):pass
audit=dict(passed=True,independent_prediction_rows=counter,independent_pair_rows=pair_rows,correlation_groups=groupcount,correlations_recomputed=True,bit_order_and_guard_verified=True,all_12_runs_complete=True,checkpoints_reloaded_finite=cp_paths,parameter_count=1000334,initial_hash_common_within_each_seed=True,training_problem_order_independently_replayed=True,source_and_configuration_SHA_unchanged=True,protected_SHA_unchanged=len(before),numeric_unanswered=sum(invalid.values()),additional_training=0,additional_model_inference=0)
(R/'audit_results.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
(R/'audit_report.md').write_text(f'# V053独立監査\n\n{counter}保存予測を整数Oracleで再採点。guard/LSB復元、全誤りbit、first error、処理順region、carry/borrow phaseを別実装で再計算。\n\n{pair_rows}対応予測と{groupcount}相関groupを独立再計算。正誤phiはNumPy相関と一致、ゼロ分母NA。全24checkpoint再読込・有限重み・parameter一致。\n\n全12run、全9幅、各演算512問。C1 update/提示一致、C2学習CPU90秒（最後の1step超過<1秒）。初期重みhash一致、問題提示/ADD交換乱数を独立再実行してhash一致。\n\n事前設定/ソースSHA不変、既存{len(before)}files SHA不変、gzip CRC合格。非有限回答は未回答扱い{sum(invalid.values())}件で、同一誤答一致に含めない。追加学習/追加model推論なし。\n',encoding='utf-8')
aggregate=read('length_generalization.csv');pairs=read('paired_seed_comparison.csv')
def summary(condition,direction,width,op):return next(r for r in aggregate if r['condition']==condition and r['direction']==direction and int(r['width'])==width and r['op']==op)
lines=['# IronMath V053 — 残差GRUの方向と誤答相関','', '## 結果の要点','', '右→左R=LSB-first、左→右L=MSB-first。両方1,000,334parameter・同一初期重み。12runと全9幅512問を完了。GAN/PPO/parity/retry/推論時Oracle選択なし。','', '|条件|演算|2048bit R 平均±標本SD|2048bit L 平均±標本SD|','|---|---|---:|---:|']
for condition in ('C1','C2'):
    for op in ('ADD','SUB'):
        rr=summary(condition,'R',2048,op);ll=summary(condition,'L',2048,op)
        lines.append(f"|{condition}|{op}|{100*float(rr['exact_mean']):.2f}±{100*float(rr['exact_sample_SD']):.2f}%|{100*float(ll['exact_mean']):.2f}±{100*float(ll['exact_sample_SD']):.2f}%|")
lines+=['','## 構造・学習・比較可能性','', 'V051 A1のLinear(3,64),GRUCell(64,545),learned scalar residual,Linear(545,2)を再現。逐次projection/headと共通TorchScriptを採用し、元vectorized演算とのFP32差<=1e-4・native checkpoint argmax一致を事前に確認。学習gradientの丸め/加算順まで旧runとbit同一とは主張しない。','入力a,bは非負、SUBはa>=b。ADDはW+1(guard0)、SUBはW。Lは実長だけを反転、paddingは後ろのまま。物理的な整数は変えない。モデルのhiddenは各問題で0。各時刻対応bitを出し、全入力を読んでから答えるモデルではない。','Train/ValはV051元8bitと同一数値/順序。Test8は旧holdout512を固定層化抽出。OODは旧256+新規補充分。カテゴリは重複属性で、全seed・モデル共通。数値pairが8bitTrainで既知でも長幅zero-paddingは別系列であり、cross_width_numeric_reuse.csvで件数を開示する。unknown numericとlength OODを混同しない。','C1は1200更新×16問題=19200提示。C2はCPU90秒、更新数/提示数は異なる。CPUはforward/backward/clip/AdamWのみ。データ・batch・Val・保存・graph warmingは別計時。dummy warm backwardではoptimizer更新0・weights hash不変。C1/C2を混合しない。','選択はequal-op Validation CE最小→最早。Testは全12学習終了後のみ。C2終了時にもVal候補を追加する規則は開始前固定。','', '## 個別seedと全9幅','', '|条件|方向|幅|ADD s0/s1/s2%|SUB s0/s1/s2%|','|---|---|---:|---|---|']
for condition in ('C1','C2'):
    for direction in ('R','L'):
        for w in C['widths']:
            vals=[]
            for op in ('ADD','SUB'):
                rows=sorted([r for r in score_saved if r['condition']==condition and r['direction']==direction and int(r['width'])==w and r['op']==op],key=lambda r:int(r['seed']))
                vals.append(' / '.join(f'{100*float(r["exact_rate"]):.2f}' for r in rows))
            lines.append(f'|{condition}|{direction}|{w}|{vals[0]}|{vals[1]}|')
lines+=['','## 誤答相関と単純一致検査','', '|条件|演算|seed|Both Correct|R Only|L Only|Both Wrong|同じ誤答|一致率%|一致時精度%|Oracle上限%|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for condition in ('C1','C2'):
    for op in ('ADD','SUB'):
        for seed in (0,1,2):
            k=(condition,seed,2048,op);s=saved_corr[k];a=saved_agree[k];acc='NA' if a['agreement_accuracy']=='NA' else f'{100*float(a["agreement_accuracy"]):.2f}'
            lines.append(f"|{condition}|{op}|{seed}|{s['both_correct']}|{s['R_only']}|{s['L_only']}|{s['both_wrong']}|{s['same_wrong']}|{100*float(a['agreement_rate']):.2f}|{acc}|{100*float(s['oracle_selection_upper_bound']):.2f}|")
lines+=['','Oracle上限はどちらかに正解がある割合で、実際のAI/選択器の正答率ではない。L-onlyがあっても現在は正しい候補を選ぶ方法がない。単純一致の精度は一致した少数ケースの条件付き値で、回答保留率と同時に読む。同じ誤答の一致を除外していない。','correctness phi、同一/異なる誤答割合（全512分母/双方誤答分母）、aligned physical error Jaccardはerror_correlation.csv。NAを0に置換しない。数値未回答のboth-wrongは同一/異なる数値誤答とは別列。','', '## Carry/Borrowと誤り位置','', 'carry_borrow_results.csvに長さbinと重複属性、propagation_phase_errors.csvにstart/continue/release/none位置の機会数・誤り数・条件付き誤り率を保存。first_error_phase_counts.csvとrawログに最初のphysical/processing位置、先頭/中央/末尾の誤りを保存。R/Lの処理順領域を物理LSB位置と混同しない。','Lでは初回ADD guardを出す時点で下位bitを読めず、同一prefixで異なる正解guardがあることを事前テストで確認した。この逐次出力制約は方向比較の対象だが、双方向モデルや全入力読取後の計算全般を否定する結果ではない。誤り位置だけで内部状態のcauseを断定しない。','', '## CPUとRAM','', '|run|更新|提示数|選択更新|学習CPU秒|学習wall秒|peak RAM MiB|','|---|---:|---:|---:|---:|---:|---:|']
for r in train:lines.append(f"|{r['condition']}_{r['direction']}_s{r['seed']}|{r['updates']}|{r['problem_presentations']}|{r['selected_update']}|{float(r['train_cpu_seconds']):.3f}|{float(r['train_wall_seconds']):.3f}|{int(r['peak_RAM_bytes'])/2**20:.2f}|")
lines+=['','RAMは同一processの生涯peakで、run単体独立peakではない。推論はbatch128スループットで、単問latencyではない。CPU/wall範囲をruntime.jsonに記載。','', '## V051参考','', '過去2048bit ADD97.01±3.16%、SUB69.92±33.87%。V053の512問と直接順位比較しない。V051_common_reference.csvは厳密に同一入力の保存native予測を再採点したもの。8bitは512、長幅は既存256の部分集合で、新規256のbaseline結果を作っていない。追加native再推論を行った値ではない。','', '## 成功例・失敗例','', '|条件|方向|演算|幅|正解?|問題ID|','|---|---|---|---:|---|---|']
for k,r in examples.items():
    # Full integers and bit sequences preserved in JSON; compact report ID labels.
    ident=r['id'] if len(r['id'])<100 else r['id'][:80]+'… (full examples.json)'
    lines.append(f"|{r['condition']}|{r['direction']}|{r['op']}|{r['width']}|{bool(r['correct'])}|{ident}|")
decisions=[]
for condition in ('C1','C2'):
    for op in ('ADD','SUB'):
        ps=sorted([r for r in pairs if r['condition']==condition and r['op']==op and int(r['width'])==2048],key=lambda r:int(r['seed']));diffs=[float(r['R_minus_L']) for r in ps]
        lonly=sum(int(saved_corr[(condition,s,2048,op)]['L_only']) for s in (0,1,2));agwrong=sum(int(saved_agree[(condition,s,2048,op)]['agreed_wrong']) for s in (0,1,2))
        decisions.append(dict(condition=condition,op=op,paired_R_minus_L_mean=statistics.mean(diffs),paired_R_minus_L_sample_SD=statistics.stdev(diffs),each_seed_R_greater=all(d>0 for d in diffs),L_only_total=lonly,agreed_wrong_total=agwrong))
lines+=['','## 研究課題への回答','']
for d in decisions:lines.append(f"- {d['condition']} {d['op']}: 2048bitの対応差R−L={100*d['paired_R_minus_L_mean']:.2f}±{100*d['paired_R_minus_L_sample_SD']:.2f}ポイント、全seed R>L={d['each_seed_R_greater']}。L-only計{d['L_only_total']}、合意した誤答計{d['agreed_wrong_total']}。")
lines+=['','RQ1は上記対応差と全幅の失敗patternで評価。RQ2は全9幅の完全一致、bit正答率、長propagation groupとseed変動で評価する。RQ3は4分類・phi/Jaccard・consensus coverage/precisionで定量化した。','L-onlyが存在するなら補完可能性の限定的な証拠だが、実行時の選択器は未実装。同じ誤答の合意があれば一致だけは正しさの証明にならない。3seed・1固定データ集合で統計的な普遍的優位を確定しない。','基礎成功（同構造・同条件・全9幅完了）と方向差の測定は成立。個々の算術成功は実測表に従い、低い条件や未回答も保全する。今後の検証器・双方向構造・追加学習は本実験で実行しない。','', '全raw予測/失敗/対応4分類をgzip、checkpoint24本を保存。既存成果のSHA不変。独立監査後に停止。']
(R/'REPORT_V053.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(R/'conclusions.json').write_text(json.dumps(dict(basic_execution_success=True,all_nine_widths_complete=True,paired_primary=decisions,additional_training=0,additional_model_inference=0),indent=2),encoding='utf-8')
(R/'figures').mkdir(exist_ok=True)
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,op in zip(axs,('ADD','SUB')):
    for condition,style in [('C1','-'),('C2','--')]:
        for direction,color in [('R','tab:blue'),('L','tab:orange')]:
            rows=sorted([r for r in aggregate if r['condition']==condition and r['direction']==direction and r['op']==op],key=lambda r:int(r['width']))
            ax.errorbar([int(r['width']) for r in rows],[100*float(r['exact_mean']) for r in rows],yerr=[100*float(r['exact_sample_SD']) for r in rows],label=condition+' '+direction,color=color,linestyle=style,marker='o',capsize=2)
    ax.set_xscale('log',base=2);ax.set_ylim(-15,115);ax.set_title(op);ax.set_xlabel('Width (bits)');ax.set_ylabel('Exact match (%)');ax.legend()
fig.suptitle('Mean +/- sample SD,3 seeds; R=LSB-first, L=MSB-first',fontsize=10);fig.savefig(R/'figures/length_generalization.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,op in zip(axs,('ADD','SUB')):
    labels=[];values=[]
    for condition in ('C1','C2'):
        for seed in (0,1,2):
            r=saved_corr[(condition,seed,2048,op)];labels.append(condition+' s'+str(seed));values.append([int(r[k])/512 for k in ('both_correct','R_only','L_only','both_wrong')])
    x=np.arange(len(labels));bottom=np.zeros(len(labels))
    for i,name in enumerate(('Both correct','R only','L only','Both wrong')):
        v=np.array([r[i] for r in values]);ax.bar(x,v,bottom=bottom,label=name);bottom+=v
    ax.set_xticks(x,labels,rotation=25);ax.set_ylim(0,1);ax.set_title(op+' 2048bit');ax.set_ylabel('Fraction');ax.legend(fontsize=8)
fig.savefig(R/'figures/error_classes_2048.png',dpi=160);plt.close(fig)
stop=dict(status='completed_audited_and_stopped',runs=12,predictions=counter,additional_training=0,additional_model_inference=0);(R/'STOPPED.json').write_text(json.dumps(stop,indent=2),encoding='utf-8')
files={str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file() and p.name!='sha256_manifest.txt'}
(R/'sha256_manifest.txt').write_text('\n'.join(f'{h}  {p}' for p,h in sorted(files.items()))+'\n',encoding='utf-8')
print(json.dumps(dict(audit_pass=True,predictions=counter,pairs=pair_rows,primary=decisions,stopped=True)));print('V053 AUDIT/REPORT COMPLETE; STOPPED')
