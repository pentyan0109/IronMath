"""Independent saved-prediction audit and analysis. No model execution or training."""
import csv,gzip,hashlib,json,math,random,statistics,time
from collections import Counter,defaultdict
from pathlib import Path
import torch
from run import R,WORK,C,csvout,dump,sha,protected,memory

def read(name):
    with (R/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def digits(value,n,radix):
    # Independent divmod conversion (not experiment encoder).
    out=[]
    for _ in range(n):
        value,r=divmod(value,radix)
        if radix==3 and r==2:r=-1;value+=1
        out.append(r)
    assert value==0
    return out
def value(dd,radix):
    if any(v is None for v in dd):return None
    v=0
    for d in reversed(dd):v=v*radix+d
    return v
def diagnostic(r,k):
    w=r['width'];radix=3 if k[0]=='T' else 2;n=r['length'];aa=digits(r['a'],n,radix);bb=digits(r['b'],n,radix)
    if k=='B1' and r['op']=='SUB':bb=[1-b for b in bb]
    if k=='T1' and r['op']=='SUB':bb=[-b for b in bb]
    carry=int(k=='B1' and r['op']=='SUB');run=longest=0;events=[];pairs=Counter()
    for a,b in zip(aa,bb):
        old=carry;pairs[f'{a},{b}']+=1
        if k=='B0' and r['op']=='SUB':carry=int(a-b-carry<0)
        elif radix==2:carry=int(a+b+carry>=2)
        else:
            s=a+(-b if k=='T0' and r['op']=='SUB' else b)+carry
            carry=1 if s>=2 else -1 if s<=-2 else 0
        events.append((old,carry));run=run+1 if carry else 0;longest=max(longest,run)
    phase=[]
    for i,(a,b) in enumerate(events):
        x='propagation_start' if not a and b else 'continue' if a and b else 'release' if a and not b else 'inactive'
        if i==0:x='computation_start'
        elif x=='inactive' and events[i-1][0] and not events[i-1][1]:x='post_release'
        phase.append(x)
    return longest,phase,dict(Counter(str(a) for a,b in events)),dict(pairs)
def bin_length(a,b,op,w):
    carry=run=longest=0
    for i in range(w+int(op=='ADD')):
        x=(a>>i)&1;y=(b>>i)&1
        carry=int(x+y+carry>=2) if op=='ADD' else int(x-y-carry<0)
        run=run+1 if carry else 0;longest=max(longest,run)
    return longest
def label(n):return '0' if n==0 else '1' if n==1 else '2-3' if n<=3 else '4-7' if n<=7 else '8-15' if n<=15 else '16+'
def mean_sd(v):return statistics.mean(v),statistics.stdev(v)

def main():
    start=time.perf_counter();cpu=time.process_time();assert json.loads((R/'training_complete.json').read_text())['runs']==24
    assert json.loads((R/'preflight.json').read_text())['passed'];before=json.loads((R/'protected_sha_before.json').read_text());assert protected()==before
    for name,h in json.loads((R/'STARTED.json').read_text())['registered_SHA'].items():assert sha(R/name)==h
    # Raw data source and splits are audited separately from prediction files.
    datasets={};seen=set();reuse=defaultdict(set)
    for name,meta in json.loads((R/'dataset_manifest.json').read_text()).items():
        path=WORK/meta['source'];assert sha(path)==meta['sha256'];rr=json.loads(path.read_text());assert len(rr)==meta['n'];datasets[name]=rr
        for r in rr:
            a,b=sorted((r['a'],r['b'])) if r['op']=='ADD' else (r['a'],r['b']);key=(r['width'],r['op'],a,b);assert key not in seen;seen.add(key)
            reuse[(r['op'],a,b)].add(r['width'])
            answer=r['a']+r['b'] if r['op']=='ADD' else r['a']-r['b'];assert answer==r['answer']
    csvout('cross_width_numeric_reuse.csv',[dict(op=k[0],a=k[1],b=k[2],widths=','.join(map(str,sorted(v))),note='different encoded length; not necessarily numeric novelty') for k,v in reuse.items() if len(v)>1])
    logs=read('training_log.csv');resources=read('cpu_memory.csv');train=[r for r in resources if r['stage']=='training'];assert len(train)==24
    checkpoints=0
    for r in train:
        cond,k,seed=r['condition'],r['model'],int(r['seed']);updates=int(r['updates']);assert int(r['ADD_presentations'])==int(r['SUB_presentations'])==updates*8
        assert updates==1200 if cond=='C1' else 90<=float(r['train_cpu_seconds'])<=91
        ll=[l for l in logs if l['condition']==cond and l['model']==k and int(l['seed'])==seed]
        best=min(ll,key=lambda l:float(l['validation_loss']));assert int(best['update'])==int(r['selected_update'])
        rng=random.Random(50000+seed);oh=hashlib.sha256()
        for _ in range(updates):
            for op in ('ADD','SUB'):
                pool=datasets[f'{op}_Train']
                for _ in range(8):
                    p=pool[rng.randrange(len(pool))];a,b=p['a'],p['b']
                    if op=='ADD' and rng.randrange(2):a,b=b,a
                    oh.update(f"{p['id']}:{a}:{b}\n".encode())
        assert oh.hexdigest()==r['problem_order_sha256']
        for version in ('best_validation','final'):
            cp=torch.load(R/f'checkpoints/{cond}_{k}_s{seed}/{version}.pt',weights_only=False,map_location='cpu');checkpoints+=1
            assert all(torch.isfinite(v).all() for v in cp['model'].values());assert sum(v.numel() for v in cp['model'].values())==int(r['parameters'])
            assert cp['update']==(int(r['selected_update']) if version=='best_validation' else updates)
    # Stream and independently regrade every saved prediction, plus categories/phases.
    lookup={name:{r['id']:r for r in rr} for name,rr in datasets.items() if '_Test' in name}
    counts=Counter();correct=Counter();dc=Counter();token=Counter();cg=Counter();cc=Counter();phasecount=Counter();phaseerror=Counter();first=Counter();distribution=Counter();inputcounts=Counter();statecounts=Counter();examples=defaultdict(list);identities=set();num=failnum=nonfinite=0;lastkey=None;cache={};predhash=hashlib.sha256();failurehash=hashlib.sha256()
    with gzip.open(R/'predictions.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            p=json.loads(line);num+=1;g=(p['condition'],p['model'],p['seed'],p['width'],p['op']);k=p['model'];radix=3 if k.startswith('T') else 2
            identity=g+(p['id'],);assert identity not in identities;identities.add(identity)
            r=lookup[f"{p['op']}_Test{p['width']}"][p['id']];assert (p['a'],p['b'],p['answer'])==(r['a'],r['b'],r['answer'])
            n=p['length'];truth=digits(p['answer'],n,radix);dd=p['prediction_digits'];assert len(dd)==n and all(v is None or v in ((-1,0,1) if radix==3 else (0,1)) for v in dd)
            vv=value(dd,radix);assert vv==p['prediction'];ok=int(vv==p['answer']);assert ok==p['correct'];wrong=[i for i,(a,b) in enumerate(zip(dd,truth)) if a!=b];assert wrong==p['wrong_positions'];assert p['digit_correct']==n-len(wrong)
            group=(k,p['width'],p['op'])
            if group!=lastkey:cache={};lastkey=group
            if p['id'] not in cache:cache[p['id']]=(diagnostic(p,k),bin_length(p['a'],p['b'],p['op'],p['width']))
            (internal,ph,sc,ic),binary=cache[p['id']]
            assert internal==p['internal_propagation_length'] and binary==p['binary_propagation_length']
            assert label(internal)==p['internal_bucket'] and label(binary)==p['binary_bucket'];assert sc==p['internal_incoming_counts'] and ic==p['input_pair_counts']
            assert dict(Counter(ph))==p['phase_opportunities'];assert dict(Counter(ph[i] for i in wrong))==p['phase_errors'];assert p['first_error_phase']==(ph[wrong[0]] if wrong else 'correct')
            counts[g]+=1;correct[g]+=ok;dc[g]+=n-len(wrong);token[g]+=n;first[g+(p['first_error_phase'],)]+=1;distribution[g+(str(vv),)]+=1;nonfinite+=vv is None
            for definition,l in [('binary',binary),('internal',internal)]:cg[g+(definition,label(l))]+=1;cc[g+(definition,label(l))]+=ok
            for cat in set(p['attributes'])|{p['category']}:cg[g+('attribute',cat)]+=1;cc[g+('attribute',cat)]+=ok
            for q,t in Counter(ph).items():phasecount[g+(q,)]+=t
            for q,t in Counter(ph[i] for i in wrong).items():phaseerror[g+(q,)]+=t
            for pair,t in ic.items():inputcounts[g+(pair,)]+=t
            for state,t in sc.items():statecounts[g+(state,)]+=t
            if len(examples[(k,p['op'],ok)])<2:examples[(k,p['op'],ok)].append(p)
            if not ok:failnum+=1;failurehash.update(line.encode())
    assert num==24*9*2*512 and len(counts)==24*9*2 and set(counts.values())=={512}
    with gzip.open(R/'failure_cases.jsonl.gz','rt',encoding='utf-8') as f:
        fc=0
        for line in f:fc+=1;failurehash_line=json.loads(line);assert not failurehash_line['correct'];predhash.update(line.encode())
    assert fc==failnum and predhash.hexdigest()==failurehash.hexdigest()
    for r in read('evaluation_results.csv'):
        g=(r['condition'],r['model'],int(r['seed']),int(r['width']),r['op']);assert counts[g]==int(r['n']) and correct[g]==int(r['correct']);assert abs(dc[g]/token[g]-float(r['digit_accuracy']))<1e-12
    for r in read('carry_borrow_results.csv'):
        g=(r['condition'],r['model'],int(r['seed']),int(r['width']),r['op'],r['definition'],r['category']);assert cg[g]==int(r['n']) and cc[g]==int(r['correct'])
    summary=[];stability=[];comparisons=[];decisions={}
    agg=read('length_generalization.csv');aggregate={(r['condition'],r['model'],int(r['width']),r['op']):r for r in agg}
    for r in agg:
        vv=[correct[(r['condition'],r['model'],s,int(r['width']),r['op'])]/512 for s in (0,1,2)];mu,sd=mean_sd(vv);assert abs(mu-float(r['exact_mean']))<1e-12 and abs(sd-float(r['exact_sample_SD']))<1e-12
        stability.append(dict(condition=r['condition'],model=r['model'],width=r['width'],op=r['op'],seed0=vv[0],seed1=vv[1],seed2=vv[2],mean=mu,sample_SD=sd,min=min(vv),max=max(vv)))
    for cond in ('C1','C2'):
        for target,base,title in [('B1','B0','binary_unification'),('T1','T0','ternary_unification'),('T1','B1','ternary_vs_binary')]:
            delta=[(correct[(cond,target,s,2048,'SUB')]-correct[(cond,base,s,2048,'SUB')])/512 for s in (0,1,2)]
            add_delta=float(aggregate[(cond,target,2048,'ADD')]['exact_mean'])-float(aggregate[(cond,base,2048,'ADD')]['exact_mean'])
            support=all(d>0 for d in delta) and (title=='ternary_vs_binary' or add_delta>=-.05)
            decisions[cond+'_'+title]=dict(status='支持' if support else '不支持',SUB_paired_deltas=delta,ADD_mean_delta=add_delta,criterion_met=support)
            for s,d in enumerate(delta):comparisons.append(dict(condition=cond,comparison=title,target=target,baseline=base,seed=s,width=2048,SUB_paired_difference=d,ADD_mean_difference=add_delta,criterion_met=support))
    csvout('representation_comparison.csv',comparisons);csvout('seed_stability.csv',stability)
    csvout('first_error_phase.csv',[dict(condition=g[0],model=g[1],seed=g[2],width=g[3],op=g[4],phase=g[5],first_error_count=n,phase_opportunities=phasecount[g],phase_errors=phaseerror[g]) for g,n in first.items()])
    csvout('phase_error_rates.csv',[dict(condition=g[0],model=g[1],seed=g[2],width=g[3],op=g[4],phase=g[5],opportunities=n,errors=phaseerror[g],error_rate=phaseerror[g]/n) for g,n in phasecount.items()])
    csvout('overlapping_category_results.csv',[dict(condition=g[0],model=g[1],seed=g[2],width=g[3],op=g[4],category=g[6],n=n,correct=cc[g],exact_rate=cc[g]/n) for g,n in cg.items() if g[5]=='attribute'])
    csvout('internal_input_distribution.csv',[dict(condition=g[0],model=g[1],seed=g[2],width=g[3],op=g[4],input_pair=g[5],digit_count=n) for g,n in inputcounts.items()])
    csvout('internal_carry_distribution.csv',[dict(condition=g[0],model=g[1],seed=g[2],width=g[3],op=g[4],incoming_carry=g[5],digit_count=n) for g,n in statecounts.items()])
    dist=[]
    for g in counts:
        rr=[(key[-1],n) for key,n in distribution.items() if key[:-1]==g];rr.sort(key=lambda q:(-q[1],q[0]));dist.append(dict(condition=g[0],model=g[1],seed=g[2],width=g[3],op=g[4],unique_answers=len(rr),most_common_answer=rr[0][0],most_common_count=rr[0][1]))
    csvout('answer_distribution.csv',dist);dump('examples.json',[p for pp in examples.values() for p in pp]);dump('conclusions.json',decisions)
    old=[]
    with (WORK/'IronMath_V053/evaluation_results.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['condition']=='C1' and r['direction']=='R':old.append(r)
    reproduction=[]
    for r in old:
        g=('C1','B0',int(r['seed']),int(r['width']),r['op']);reproduction.append(dict(seed=g[2],width=g[3],op=g[4],V053_correct=int(r['correct']),V054_B0_correct=correct[g],same_integer_set=True,correct_count_difference=correct[g]-int(r['correct'])))
    csvout('baseline_reproduction.csv',reproduction)
    # Independent factual report, figures from saved summaries only.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (R/'figures').mkdir(exist_ok=True);fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for i,cond in enumerate(('C1','C2')):
        for j,op in enumerate(('ADD','SUB')):
            ax=axs[i,j]
            for k in C['models']:
                yy=[float(aggregate[(cond,k,w,op)]['exact_mean'])*100 for w in C['widths']];ss=[float(aggregate[(cond,k,w,op)]['exact_sample_SD'])*100 for w in C['widths']]
                ax.errorbar(C['widths'],yy,yerr=ss,label=k,marker='o',capsize=2)
            ax.set_xscale('log',base=2);ax.set_ylim(-5,105);ax.set_title(cond+' '+op);ax.grid(alpha=.25);ax.set_ylabel('Integer exact match (%)');ax.set_xlabel('Integer range bits');ax.legend()
    fig.tight_layout();fig.savefig(R/'figures/length_generalization.png',dpi=150);plt.close(fig)
    lines=['# IronMath V054 — 二進数・平衡三進法による演算統一','', '24runの固定学習・全9幅評価を完了。Testは全学習終了後のみ使用。以下はValidation選択checkpointの実測値。成功基準と結果は条件別に判定した。','', '## 実測された事実','', '|予算|方式|2048bit ADD 平均±標本SD (%)|2048bit SUB 平均±標本SD (%)|SUB seed0 / seed1 / seed2 (%)|','|---|---|---:|---:|---|']
    for cond in ('C1','C2'):
        for k in C['models']:
            ar=aggregate[(cond,k,2048,'ADD')];sr=aggregate[(cond,k,2048,'SUB')];sv=[correct[(cond,k,s,2048,'SUB')]/512*100 for s in (0,1,2)]
            lines.append(f"|{cond}|{k}|{100*float(ar['exact_mean']):.2f}±{100*float(ar['exact_sample_SD']):.2f}|{100*float(sr['exact_mean']):.2f}±{100*float(sr['exact_sample_SD']):.2f}|"+' / '.join(f'{v:.2f}' for v in sv)+'|')
    lines+=['','## 仮説・成功基準の判定','']
    for name,r in decisions.items():lines.append(f"- {name}：**{r['status']}**。SUB対応差(percentage points)={', '.join(f'{100*d:+.2f}' for d in r['SUB_paired_deltas'])}、ADD平均差={100*r['ADD_mean_delta']:+.2f}pp。全seedで厳密に改善という固定基準。")
    lines+=['','## 固定仕様と公平性','', 'B0はV053と同じ残差GRUで演算flagによってADD/SUBを学習する「独立タスク」であり、別々の重みを持つ2モデルではない。B1は共通加算器に初期Carryだけを入力しSUBのb全桁を反転。B0のSUBはW桁、B1はW+1桁のため、演算flag・初期Carry入力および系列長も変わり、純粋な重み共有効果のみの対照ではない。T0とT1は同じ固定三進桁数、op flag対符号反転。B1は二進1,000,334、T1は三進1,000,880パラメータ（差546）。第三特徴はT1で常に0だが共通骨格に保持。すべてLSB-first、位置/幅/正解/中間carry入力なし。','', '各更新ADD8/SUB8、batch16、AdamW .002、wd .01、clip1、CPU2threads。C1は1200更新・各演算9600提示、C2は学習CPU90秒。二進/三進の系列長と教師桁数は異なる。C1とC2の結果を混ぜない。Validationは200更新ごと＋終了時、等演算重みの桁CE最低値、同値最早。Testによる調整なし。','', '共通整数幅8,16,32,64,128,256,512,1024,2048、各幅各演算512問。元V053のTrain/Validation/TestとIDを維持。Train26,316/演算、Validation3,290/演算。三進最大幅は1294桁、二進最大2049桁。クロス幅の同値整数再利用はcross_width_numeric_reuse.csvで開示。これは固定長外挿であり、すべて数値未見という意味ではない。','', '## CPU・RAM','']
    changed=sum(r['correct_count_difference']!=0 for r in reproduction)
    lines.insert(lines.index('## CPU・RAM'),f'V053 C1-Rと今回B0の同一整数集合での再現比較はbaseline_reproduction.csv。54幅/演算/seed集計のうち正答数が変わった集計は{changed}件。損失のサンプル抽出は旧advanced indexingから連続sliceへ変わるが等演算CE・問題順序・数学的構造は同じ。CPU90秒条件の更新数は測定依存であり過去runと同一重みを要求しない。\n')
    lines.append(f"学習部分合計CPU={sum(float(r['train_cpu_seconds']) for r in train):.3f}秒、wall={sum(float(r['train_wall_seconds']) for r in train):.3f}秒。Validation/保存/バッチ/graph暖機は別計測。ピークRAM={max(int(r['peak_RAM_bytes']) for r in resources)/2**20:.2f}MiB（同一プロセス累積peak、各run専用peakではない）。CPU timer量子化と最後の更新分のC2超過をcpu_memory.csvに保存。")
    lines+=['','|予算|方式|更新数 seed0/1/2|学習CPU平均秒|2048 ADD/SUB推論CPU平均秒 (512問)|','|---|---|---|---:|---|']
    for cond in ('C1','C2'):
        for k in C['models']:
            rr=sorted((r for r in train if r['condition']==cond and r['model']==k),key=lambda r:int(r['seed']))
            vals=[]
            for op in ('ADD','SUB'):vals.append(statistics.mean(float(r['model_cpu_seconds']) for r in resources if r['stage']=='inference' and r['condition']==cond and r['model']==k and int(r['width'])==2048 and r['op']==op))
            lines.append(f"|{cond}|{k}|{' / '.join(r['updates'] for r in rr)}|{statistics.mean(float(r['train_cpu_seconds']) for r in rr):.3f}|{vals[0]:.3f} / {vals[1]:.3f}|")
    lines+=['','## Carry/Borrow・入力分布・誤り診断','', 'carry_borrow_results.csvは二進数値基準と内部表現基準を別々に集計する。長さ0,1,2–3,4–7,8–15,16+。空カテゴリはn=0/NA。属性カテゴリは重複するので総数へ加算しない。内部三進carryには-1/0/+1を含む。B1のSUBは初期carry=1と補数入力、T1のSUBは負のtritを含み、演算統一後も入力/内部carry分布はADDと一致しない。internal_input_distribution.csvとinternal_carry_distribution.csvに実測頻度を保存。first_error_phase.csvとphase_error_rates.csvは開始・開始carry・継続・解除・解除後・非伝播を区別。失敗位置だけで神経機構の原因を断定しない。','']
    lines+=['|予算|共通加算器|演算|2048内部 incoming carry -1/0/+1 割合 (%)|最頻の最初の誤り段階|','|---|---|---|---|---|']
    for cond in ('C1','C2'):
        for k in ('B1','T1'):
            for op in ('ADD','SUB'):
                st=[sum(statecounts[(cond,k,s,2048,op,str(c))] for s in (0,1,2)) for c in (-1,0,1)];den=sum(st)
                ff=Counter()
                for g,n in first.items():
                    if g[0]==cond and g[1]==k and g[3]==2048 and g[4]==op and g[5]!='correct':ff[g[5]]+=n
                phase,n=ff.most_common(1)[0] if ff else ('none',0)
                lines.append(f"|{cond}|{k}|{op}|"+' / '.join(f'{v/den*100:.2f}' for v in st)+f'|{phase} ({n})|')
    for cond in ('C1','C2'):
        for k in C['models']:
            for definition in ('binary','internal'):
                ss=[]
                for cat in ('0','1','2-3','4-7','8-15','16+'):
                    gn=sum(cg[(cond,k,s,2048,'SUB',definition,cat)] for s in (0,1,2));gc=sum(cc[(cond,k,s,2048,'SUB',definition,cat)] for s in (0,1,2));ss.append(f'{cat}: {gc}/{gn}'+(f' ({gc/gn*100:.2f}%)' if gn else ' (NA)'))
                lines.append(f"- {cond} {k} SUB {definition} 3seed合計（seed平均と区別）："+'; '.join(ss))
    lines+=['','桁単位正答率はlength_generalization.csvに保存。二進bitと三進tritの正答率を直接順位比較しない。整数完全一致は共通整数値で判定。回答分布と最頻回答はanswer_distribution.csv、成功/失敗例はexamples.json、全失敗はfailure_cases.jsonl.gzに残す。','', '## 成功例・失敗例','']
    for group,pp in examples.items():
        p=pp[0];lines.append(f"- {p['condition']} {p['model']} seed{p['seed']} {p['op']} 幅{p['width']} {'成功' if p['correct'] else '失敗'}：{p['a']} { '+' if p['op']=='ADD' else '-' } {p['b']}、正解{p['answer']}、予測{p['prediction']}、最初の誤り={p['first_error']}。")
    lines+=['','## 独立監査・保全','', f'保存予測{num:,}件、失敗{failnum:,}件を独立divmod変換・Horner復元・整数再採点。表の完全一致/桁正答率/内部Carry/二進Carry/Borrow/phaseを再計算。checkpoint{checkpoints}本を再読込し有限性・パラメータ数・選択更新を照合。学習問題順序を別RNG再生しSHA一致。既存成果{len(before)}ファイルSHA不変。非有限予測{nonfinite}件。監査はaudit_results.json、audit_report.md。','', '## まだ証明できないこと','', '3seedのみで統計的確実性や一般的数学的推論能力、あらゆる長さでの完全加算/減算アルゴリズム獲得、未知演算への汎化は証明しない。三進系列の短縮、head差、入力分布差が残り、進数だけの因果効果とは断定しない。実測された固定分布の成功率である。決定的M1算術器を主モデルやOracle選択器として使用していない。検出された傾向は失敗層別の関係であり内部因果機構の証明ではない。','', '## 停止','', '指定24run・固定評価・保存データ独立監査を完了し停止。追加学習0、条件変更0、他プロジェクト統合0。GAN/PPO/パリティ/再試行/V055への自動移行なし。']
    (R/'REPORT_V054.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit=dict(passed=True,primary_evaluation_rows=num,failed_predictions=failnum,nonfinite_predictions=nonfinite,common_integer_groups=len(counts),group_n=512,checkpoints_reloaded=checkpoints,protected_files_SHA_unchanged=len(before),registered_source_SHA_unchanged=True,independent_numeric_regrade=True,independent_propagation_and_phase_recalculation=True,split_leakage=False,problem_order_replayed=True,ADD_SUB_equal_presentations=True,selection_Validation_only=True,extra_training=0,additional_model_inference=0,audit_CPU_seconds=time.process_time()-cpu,audit_wall_seconds=time.perf_counter()-start,audit_peak_RAM_bytes=memory())
    dump('audit_results.json',audit);(R/'audit_report.md').write_text('# V054 independent audit\n\n'+json.dumps(audit,ensure_ascii=False,indent=2)+'\n\nSource and raw data retained. Prediction arithmetic uses separate divmod/Horner implementations; all categories/phases independently rechecked. No checkpoint forward pass or optimizer used by audit.\n',encoding='utf-8')
    assert protected()==before;dump('STOPPED.json',dict(status='completed_audited_and_stopped',runs=24,additional_training=0,additional_model_inference=0,client_date='2026-10-15'))
    (R/'sha256_manifest.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(R)}' for p in sorted(R.rglob('*')) if p.is_file() and p.name!='sha256_manifest.txt')+'\n',encoding='utf-8')
    print(json.dumps(dict(audit=audit,conclusions=decisions),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
