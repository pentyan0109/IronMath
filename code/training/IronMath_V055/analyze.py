"""Independent arithmetic audit / saved-data statistics only. No neural execution."""
import gzip,hashlib,json,math,random,statistics,time
from collections import Counter,defaultdict
import torch
from utils import *

def independent(p):
    w=p['width'];n=p['sequence_length'];a=p['a'];b=p['b'];route=p['route'];is_sub=p['op']=='SUB';mask=(1<<w)-1
    if route=='P' and is_sub:bb=(((mask^b)+1)&mask);state=0
    elif route=='B1_native' and is_sub:bb=((1<<n)-1)^b;state=1
    else:bb=b;state=0
    raw=(a-b) if route=='B0_native' and is_sub else (a+bb+state)%(1<<n)
    truth=[(raw>>i)&1 for i in range(n)];events=[];ph=[];fullrun=fullmax=run=longest=0
    for i in range(n):
        x=(a>>i)&1;y=(bb>>i)&1;old=state
        state=int(x-y-state<0) if route=='B0_native' and is_sub else int(x+y+state>=2)
        events.append((old,state));fullrun=fullrun+1 if state else 0;fullmax=max(fullmax,fullrun)
        if i<w:run=run+1 if state else 0;longest=max(longest,run)
        ph.append('computation_start' if i==0 else 'start' if not old and state else 'continue' if old and state else 'release' if old and not state else 'inactive')
    ones=[(bb>>i)&1 for i in range(w)];onesrun=maxones=0
    for v in ones:onesrun=onesrun+1 if v else 0;maxones=max(maxones,onesrun)
    edge=(w+3)//4;events=events[:w];s=0;brun=bmax=0
    for i in range(w+int(not is_sub)):
        x=(a>>i)&1;y=(b>>i)&1;s=int(x-y-s<0) if is_sub else int(x+y+s>=2);brun=brun+1 if s else 0;bmax=max(bmax,brun)
    fields=dict(longest_carry=longest,longest_carry_full_sequence=fullmax,B_ones=sum(ones),B_longest_ones=maxones,B_LSB_edge_ones=sum(ones[:edge]),B_MSB_edge_ones=sum(ones[-edge:]),edge_length=edge,correct_incoming_carry_ones=sum(x for x,y in events),correct_outgoing_carry_ones=sum(y for x,y in events),carry_start=sum(not x and y for x,y in events),carry_continue=sum(x and y for x,y in events),carry_release=sum(x and not y for x,y in events),binary_propagation=bmax)
    return raw,truth,ph,fields
def bucket(n):return '0' if not n else '1' if n==1 else '2-3' if n<=3 else '4-7' if n<=7 else '8-15' if n<=15 else '16+'
def integer(dd):return sum(v<<i for i,v in enumerate(dd)) if all(v is not None for v in dd) else None
def inputkey(r):
    b=((1<<r['width'])-1)^r['b'];b=(b+1)&((1<<r['width'])-1) if r['op']=='SUB' else r['b']
    return r['width']+1,*sorted((r['a'],b))
def digest_state(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.numpy().tobytes())
    return h.hexdigest()

def main():
    start=time.perf_counter();cpu=time.process_time();assert json.loads((R/'training_complete.json').read_text())['runs']==18
    before=json.loads((R/'protected_sha_before.json').read_text());assert protected()==before
    for path,h in json.loads((R/'STARTED.json').read_text())['registered_SHA'].items():assert sha(R/path)==h
    data={}
    for name,meta in json.loads((R/'dataset_manifest.json').read_text()).items():
        path=R/f'data/{name}.json';assert sha(path)==meta['sha256'];assert sha(WORK/C['data_source']/f'{name}.json')==meta['original_SHA256'];rr=json.loads(path.read_text());assert len(rr)==meta['n'];data[name]=rr
    tr={inputkey(r) for op in ('ADD','SUB') for r in data[op+'_Train']};va={inputkey(r) for op in ('ADD','SUB') for r in data[op+'_Validation']};te={inputkey(r) for name,rr in data.items() if '_Test' in name for r in rr};assert not(tr&va or tr&te or va&te)
    # Independently reconstruct deterministic structural exclusions.
    source={name:json.loads((WORK/C['data_source']/f'{name}.json').read_text()) for name in data}
    heldout=te|va
    for op in ('ADD','SUB'):
        expected=[r['id'] for r in source[op+'_Validation'] if inputkey(r) not in te];assert expected==[r['id'] for r in data[op+'_Validation']]
        expected=[r['id'] for r in source[op+'_Train'] if inputkey(r) not in heldout];assert expected==[r['id'] for r in data[op+'_Train']]
    registry=json.loads((R/'reference_manifest.json').read_text());zc=json.loads((R/'zero_shot_complete.json').read_text());assert zc['optimizer_updates']==0 and zc['weights_unchanged']
    for meta in registry['checkpoints'].values():
        assert sha(WORK/meta['path'])==meta['sha256'];cp=torch.load(WORK/meta['path'],map_location='cpu',weights_only=False);assert digest_state(cp['model'])==meta['model_state_SHA256']
    assert sha(WORK/registry['archive_source'])==registry['archive_SHA256']
    resources=readcsv(R/'cpu_memory.csv');train=[r for r in resources if r['stage']=='training'];logs=readcsv(R/'training_log.csv');assert len(train)==18;cpcount=0
    for r in train:
        budget,cond,seed=r['budget'],r['condition'],int(r['seed']);updates=int(r['updates']);boundary=int(r['phase_boundary_update']) if r['phase_boundary_update']!='NA' else None
        assert int(r['total_presentations'])==updates*16
        if budget=='C1':assert updates==1200
        else:assert 90<=float(r['train_CPU_seconds'])<91
        if cond=='F2':
            if budget=='C1':assert boundary==600
            else:assert float(r['phase_boundary_previous_update_start_CPU'])<45<=float(r['phase_boundary_CPU_seconds'])<46
        ll=[q for q in logs if q['budget']==budget and q['condition']==cond and int(q['seed'])==seed];assert int(min(ll,key=lambda q:float(q['validation_loss']))['update'])==int(r['selected_update'])
        rng=random.Random(50000+seed);order=hashlib.sha256();ch=hashlib.sha256();add=sub=0
        for step in range(updates):
            streams=[]
            for op in ('ADD','ADD','SUB'):
                ss=[];pool=data[op+'_Train']
                for _ in range(8):
                    p=pool[rng.randrange(len(pool))];a,b=p['a'],p['b']
                    if op=='ADD' and rng.randrange(2):a,b=b,a
                    token=f"{p['id']}:{a}:{b}\n".encode();ch.update(token);ss.append((op,token))
                streams.append(ss)
            mixed=cond=='F1' or cond=='F2' and step>=boundary
            for op,token in streams[0]+streams[2 if mixed else 1]:order.update(token);add+=op=='ADD';sub+=op=='SUB'
        assert order.hexdigest()==r['order_SHA256'] and ch.hexdigest()==r['candidate_SHA256'];assert add==int(r['ADD_presentations']) and sub==int(r['SUB_presentations'])
        for version in ('best_validation','final'):
            cp=torch.load(R/f'checkpoints/{budget}_{cond}_s{seed}/{version}.pt',map_location='cpu',weights_only=False);cpcount+=1
            assert sum(v.numel() for v in cp['model'].values())==1000334 and all(torch.isfinite(v).all() for v in cp['model'].values());assert cp['update']==(int(r['selected_update']) if version=='best_validation' else updates)
    lookup={name:{r['id']:r for r in rr} for name,rr in data.items() if '_Test' in name};oldADD={tuple([8]+sorted((r['a'],r['b']))) for r in source['ADD_Train']}
    newSHA={(budget,condition,seed):sha(R/f'checkpoints/{budget}_{condition}_s{seed}/best_validation.pt') for budget in C['budgets'] for condition in C['training_conditions'] for seed in C['seeds']}
    counts=Counter();correct=Counter();bc=Counter();tokens=Counter();gc=Counter();gn=Counter();cg=Counter();cc=Counter();metrics=defaultdict(Counter);first=Counter();phaseopp=Counter();phaseerr=Counter();moments=defaultdict(lambda:[0.,0.,0.,0.,0.]);outcome=defaultdict(Counter);native=Counter();examples=defaultdict(list);baseline={};retention=defaultdict(Counter);num=failures=NFcore=NFguard=0;failurehash=hashlib.sha256();last=None;seen=set();finished=set();cache={}
    with gzip.open(R/'predictions.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            p=json.loads(line);num+=1;g=(p['budget'],p['condition'],p['seed'],p['width'],p['op']);r=lookup[f"{p['op']}_Test{p['width']}"][p['id']]
            if g!=last:
                if last is not None:assert seen==set(lookup[f'{last[4]}_Test{last[3]}']);finished.add(last)
                assert g not in finished;last=g;seen=set();cache={}
            assert p['id'] not in seen;seen.add(p['id']);assert (p['a'],p['b'],p['answer'])==(r['a'],r['b'],r['answer'])
            cpkey=f"{p['budget']}_{'B1' if p['condition']=='Z3' else 'B0'}_s{p['seed']}"
            expectedsha=newSHA[(p['budget'],p['condition'],p['seed'])] if p['condition'].startswith('F') else registry['checkpoints'][cpkey]['sha256']
            assert p['checkpoint_SHA256']==expectedsha
            # Cache independent diagnostic truth within one512-case group only.
            if p['id'] not in cache:cache[p['id']]=independent(p)
            raw,truth,ph,fields=cache[p['id']];dd=p['prediction_digits'];expected_route='B0_native' if p['condition'] in ('Z0','R_B0') else 'B1_native' if p['condition']=='Z3' else 'P'
            assert p['route']==expected_route and p['sequence_length']==p['width']+int(p['condition']!='R_B0');assert len(dd)==p['sequence_length'] and all(v in (0,1,None) for v in dd)
            resultn=p['width'] if p['op']=='SUB' else p['width']+1;assert p['result_length']==resultn;vv=integer(dd[:resultn]);rawpred=integer(dd);assert vv==p['prediction'] and rawpred==p['raw_prediction'] and raw==p['raw_truth']
            ok=int(vv==p['answer']);assert ok==p['correct'];wrong=[i for i,(a,b) in enumerate(zip(dd[:resultn],truth)) if a!=b];assert wrong==p['wrong_positions'] and p['bit_correct']==resultn-len(wrong)
            assert p['first_error']==(wrong[0] if wrong else None) and p['first_error_phase']==(ph[wrong[0]] if wrong else 'correct');assert p['phase_opportunities']==dict(Counter(ph[:resultn])) and p['phase_errors']==dict(Counter(ph[i] for i in wrong))
            for key,v in fields.items():assert p[key]==v
            assert p['carry_bucket']==bucket(fields['longest_carry']) and p['binary_bucket']==bucket(fields['binary_propagation'])
            familiar=int(p['width']==8 and (8,*sorted((p['a'],((-p['b'])&((1<<p['width'])-1)) if p['op']=='SUB' else p['b']))) in oldADD) if p['route']=='P' else None
            assert familiar==p['old_ADD_train_input_familiar'];counts[g]+=1;correct[g]+=ok;bc[g]+=resultn-len(wrong);tokens[g]+=resultn;NFcore+=vv is None
            if len(dd)>p['width']:
                guard=dd[p['width']];assert guard==p['guard_prediction'] and truth[p['width']]==p['guard_truth'];assert p['guard_correct']==int(guard==truth[p['width']]);gc[g]+=p['guard_correct'];gn[g]+=1;NFguard+=guard is None
            for key,v in fields.items():metrics[g][key]+=v
            metrics[g]['physical_positions']+=p['width'];first[g+(p['first_error_phase'],)]+=1
            for key,v in p['phase_opportunities'].items():phaseopp[g+(key,)]+=v
            for key,v in p['phase_errors'].items():phaseerr[g+(key,)]+=v
            for definition,category in [('carry',p['carry_bucket']),('binary',p['binary_bucket']),('B_ones_run',bucket(p['B_longest_ones']))]:cg[g+(definition,category)]+=1;cc[g+(definition,category)]+=ok
            if p['condition']=='Z1':cg[g+('old_input','familiar' if familiar else 'novel')]+=1;cc[g+('old_input','familiar' if familiar else 'novel')]+=ok
            for feature in ('longest_carry','B_longest_ones'):
                x=p[feature];m=moments[g+(feature,)];m[0]+=x;m[1]+=x*x;m[2]+=1-ok;m[3]+=x*(1-ok);m[4]+=1
                outcome[g+(feature,ok)]['n']+=1;outcome[g+(feature,ok)]['sum']+=x
            if p['condition']=='Z0':baseline[(p['budget'],p['seed'],p['width'],p['id'])]=ok
            if p['condition'].startswith('F') and p['op']=='ADD':
                old=baseline[(p['budget'],p['seed'],p['width'],p['id'])];retention[g]['before_correct']+=old;retention[g]['after_correct']+=ok;retention[g]['lost']+=old and not ok;retention[g]['gained']+=not old and ok
            if 'native_V054_correct' in p:native[g]+=p['native_V054_correct'];assert p['native_V054_prediction']==rawpred and p['native_V054_correct']==int(rawpred==p['answer'])
            if len(examples[(p['condition'],p['op'],ok)])<2 and p['width']<=64:examples[(p['condition'],p['op'],ok)].append(p)
            if not ok:failures+=1;failurehash.update(line.encode())
    assert seen==set(lookup[f'{last[4]}_Test{last[3]}']);assert num==304128 and len(counts)==594 and set(counts.values())=={512}
    fh=hashlib.sha256();fn=0
    with gzip.open(R/'failure_cases.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:assert json.loads(line)['correct']==0;fh.update(line.encode());fn+=1
    assert fn==failures and fh.hexdigest()==failurehash.hexdigest()
    scores=readcsv(R/'zero_shot_results.csv')+readcsv(R/'finetuning_results.csv')
    for r in scores:
        g=(r['budget'],r['condition'],int(r['seed']),int(r['width']),r['op']);assert counts[g]==int(r['n']) and correct[g]==int(r['correct']);assert abs(bc[g]/tokens[g]-float(r['bit_accuracy']))<1e-12
    aggregates=[];groups=defaultdict(list)
    for g in counts:groups[(g[0],g[1],g[3],g[4])].append(g)
    agg={}
    for k,gg in groups.items():
        gg.sort(key=lambda g:g[2]);assert [g[2] for g in gg]==[0,1,2];v=[correct[g]/512 for g in gg];row=dict(budget=k[0],condition=k[1],width=k[2],op=k[3],exact_mean=statistics.mean(v),exact_sample_SD=statistics.stdev(v),seed0=v[0],seed1=v[1],seed2=v[2],bit_accuracy_mean=statistics.mean(bc[g]/tokens[g] for g in gg),guard_accuracy_mean=statistics.mean(gc[g]/gn[g] for g in gg) if gn[gg[0]] else None);agg[k]=row;aggregates.append(row)
    csvout('length_generalization.csv',aggregates);csvout('seed_comparison.csv',aggregates)
    carryrows=[dict(budget=g[0],condition=g[1],seed=g[2],width=g[3],op=g[4],state_kind='Borrow' if g[1]=='R_B0' else 'Carry',**m,B_ones_ratio=m['B_ones']/m['physical_positions'],correct_incoming_ratio=m['correct_incoming_carry_ones']/m['physical_positions'],correct_outgoing_ratio=m['correct_outgoing_carry_ones']/m['physical_positions'],B_LSB_edge_ratio=m['B_LSB_edge_ones']/m['edge_length'],B_MSB_edge_ratio=m['B_MSB_edge_ones']/m['edge_length']) for g,m in metrics.items()];csvout('carry_distribution.csv',carryrows)
    csvout('carry_borrow_results.csv',[dict(budget=g[0],condition=g[1],seed=g[2],width=g[3],op=g[4],definition=g[5],category=g[6],n=n,correct=cc[g],exact_rate=cc[g]/n) for g,n in cg.items()])
    corr=[]
    for g,m in moments.items():
        sx,sxx,sy,sxy,n=m;den=(n*sxx-sx*sx)*(n*sy-sy*sy);rho=(n*sxy-sx*sy)/math.sqrt(den) if den>0 else None
        good=outcome[g+(1,)];bad=outcome[g+(0,)];corr.append(dict(budget=g[0],condition=g[1],seed=g[2],width=g[3],op=g[4],feature=g[5],Pearson_with_error=rho,correct_n=good['n'],incorrect_n=bad['n'],correct_mean=good['sum']/good['n'] if good['n'] else None,incorrect_mean=bad['sum']/bad['n'] if bad['n'] else None))
    csvout('error_correlations.csv',corr);csvout('error_analysis.csv',[dict(budget=g[0],condition=g[1],seed=g[2],width=g[3],op=g[4],first_error_phase=g[5],cases=n,phase_positions=phaseopp[g],phase_errors=phaseerr[g]) for g,n in first.items()]);csvout('retention_analysis.csv',[dict(budget=g[0],condition=g[1],seed=g[2],width=g[3],op=g[4],**v,retention=v['after_correct']/v['before_correct'] if v['before_correct'] else None) for g,v in retention.items()]);csvout('native_guard_effect.csv',[dict(budget=g[0],condition=g[1],seed=g[2],width=g[3],op=g[4],native_V054_correct=n,modular_V055_correct=correct[g],guard_discard_score_difference=correct[g]-n) for g,n in native.items()]);dump('examples.json',[p for pp in examples.values() for p in pp])
    decisions={};paired=[]
    for budget in C['budgets']:
        z=[correct[(budget,'Z1',s,2048,'SUB')]/512 for s in (0,1,2)];decisions[budget+'_zero_shot']=dict(status='支持' if all(v>=.9 for v in z) else '不支持',seed_rates=z)
        for cond in ('F1','F2'):
            delta=[(correct[(budget,cond,s,2048,'SUB')]-correct[(budget,'R_B0',s,2048,'SUB')])/512 for s in (0,1,2)]
            adddelta=agg[(budget,cond,2048,'ADD')]['exact_mean']-agg[(budget,'Z0',2048,'ADD')]['exact_mean'];supported=all(v>0 for v in delta) and adddelta>=-.05
            decisions[budget+'_'+cond+'_improvement']=dict(status='支持' if supported else '不支持',SUB_paired_differences=delta,ADD_mean_difference=adddelta)
            for s,v in enumerate(delta):paired.append(dict(budget=budget,condition=cond,seed=s,SUB_vs_B0=v,ADD_mean_vs_B0=adddelta,SUB_vs_F0=(correct[(budget,cond,s,2048,'SUB')]-correct[(budget,'F0',s,2048,'SUB')])/512,criterion_met=supported))
    csvout('paired_comparisons.csv',paired);dump('conclusions.json',decisions)
    # Plots and report are generated solely from audited saved statistics.
    report(agg,decisions,train,resources,carryrows,examples,num,failures,NFcore,NFguard,len(before))
    assert protected()==before
    audit=dict(passed=True,predictions_regraded=num,failures_preserved=failures,nonfinite_result_predictions=NFcore,nonfinite_guard_predictions=NFguard,groups=594,all_group_n=512,existing_checkpoints_SHA_and_weights_unchanged=12,new_checkpoints_reloaded=cpcount,protected_files_SHA_unchanged=len(before),zero_shot_optimizer_updates=0,structural_exclusions_replayed=True,derived_split_alias_overlap=0,presentation_order_and_F2_boundary_audited=True,source_SHA_unchanged=True,additional_training=0,additional_model_inference=0,audit_CPU_seconds=time.process_time()-cpu,audit_wall_seconds=time.perf_counter()-start,audit_peak_RAM_bytes=memory())
    dump('audit_results.json',audit);(R/'audit_report.md').write_text('# V055 independent saved-data audit\n\n'+json.dumps(audit,ensure_ascii=False,indent=2)+'\n\nIndependent complement uses flip+increment; prediction reconstruction weighted bits; carry/borrow threshold recurrence independently checked. New eligible splits and all training random streams replayed; all gzip streams read to EOF. No model forward/optimizer calls.\n',encoding='utf-8')
    dump('STOPPED.json',dict(status='completed_audited_and_stopped',new_training_runs=18,extra_training=0,additional_model_inference=0,Z2='not applicable',client_date='2026-10-15'))
    (R/'sha256_manifest.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(R)}' for p in sorted(R.rglob('*')) if p.is_file() and p.name!='sha256_manifest.txt')+'\n',encoding='utf-8');print(json.dumps(dict(audit=audit,conclusions=decisions),ensure_ascii=False),flush=True)

def report(agg,decisions,train,resources,carryrows,examples,num,failures,NFcore,NFguard,protected_n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (R/'figures').mkdir(exist_ok=True)
    minbar=min(100*(r['exact_mean']-r['exact_sample_SD']) for r in agg.values());maxbar=max(100*(r['exact_mean']+r['exact_sample_SD']) for r in agg.values());ylim=(min(0,minbar)-5,max(100,maxbar)+5)
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for i,budget in enumerate(C['budgets']):
        for j,op in enumerate(('ADD','SUB')):
            ax=axs[i,j];conditions=['Z0','Z3','F0','F1','F2'] if op=='ADD' else ['R_B0','Z1','Z3','F0','F1','F2']
            for cond in conditions:
                yy=[agg[(budget,cond,w,op)]['exact_mean']*100 for w in C['widths']];sd=[agg[(budget,cond,w,op)]['exact_sample_SD']*100 for w in C['widths']]
                ax.errorbar(C['widths'],yy,yerr=sd,label=cond,marker='o',capsize=2)
            ax.set_xscale('log',base=2);ax.set_ylim(*ylim);ax.set_title(budget+' '+op);ax.grid(alpha=.25);ax.axhline(100,color='gray',linestyle='--',linewidth=.6);ax.axhline(0,color='gray',linestyle='--',linewidth=.6);ax.legend(fontsize=8,ncol=2);ax.set_ylabel('Integer exact match (%)');ax.set_xlabel('Integer range bits')
    fig.suptitle('V055 mean ± sample SD (3 seeds), full error bars');fig.tight_layout();fig.savefig(R/'figures/length_generalization.png',dpi=150);plt.close(fig)
    lines=['# IronMath V055 — 既存加算器の補数減算転用','', '固定checkpointのゼロショット試験を先に完了し、結果による変更なしで事前登録済み18runを実施した。C1/C2は別条件。新規Fモデルはユーザー確認の同一seed新規初期化であり、既存モデルの微調整ではない（指定成果物名finetuning_results.csvは維持）。','', '## 実測事実：2048bit整数完全一致','', '|予算|条件|ADD 平均±標本SD (%)|SUB 平均±標本SD (%)|SUB seed0/1/2 (%)|','|---|---|---:|---:|---|']
    for budget in C['budgets']:
        for cond in ('R_B0','Z1','Z3','F0','F1','F2'):
            add=agg[(budget,'Z0' if cond in ('R_B0','Z1') else cond,2048,'ADD')];sub=agg[(budget,cond,2048,'SUB')]
            lines.append(f"|{budget}|{cond}|{add['exact_mean']*100:.2f}±{add['exact_sample_SD']*100:.2f}|{sub['exact_mean']*100:.2f}±{sub['exact_sample_SD']*100:.2f}|"+' / '.join(f"{sub['seed'+str(s)]*100:.2f}" for s in (0,1,2))+'|')
    lines+=['','R_B0は旧B0の通常SUB。Z1のADD欄は同一の固定checkpointの通常ADD条件Z0。Z3は旧B1のnative入力を保存した別モデル条件、SUBはguard除外の剰余採点であり旧V054のfull-scoreと区別する。旧full-scoreとの差はnative_guard_effect.csv。F0/F1/F2は新規学習。全幅・個別seed・bit/guard正答率は各CSV。','', '## 支持／不支持：固定基準','']
    for name,r in decisions.items():lines.append(f"- {name}：**{r['status']}**。"+('全seed90%以上を判定。seed値='+', '.join(f'{v*100:.2f}%' for v in r['seed_rates']) if 'seed_rates' in r else 'SUB対応差='+', '.join(f'{v*100:+.2f}pp' for v in r['SUB_paired_differences'])+f"、ADD平均差={r['ADD_mean_difference']*100:+.2f}pp。"))
    lines+=['','## RQ1・RQ2：転用とCarry-in','', 'Z1は固定B0のADD経路にP変換を与えた。既存B0の第3特徴はis_SUBであってCarry-inではないため、Z2は適用不可。新しい入力層、隠れ状態注入、重み更新は使用していない。通常ADDと同じW+1入力、最後は(0,0,0)、guard出力は別診断、SUB結果は下位W桁のみ。ゼロショットのoptimizer更新0、checkpoint／重みSHA不変。','', 'Z3のB1はN=W+1全桁反転、最初だけCin1、guard入力(0,1,0)という旧仕様。文字通りのW-bit C＋ゼロguardとは入力が異なる。下位W剰余は同じだが、Z1対Z3は重み・学習・guardも異なり、+1の処理方式だけの因果効果とは判定できない。RQ2の純粋な同一重み比較は未解決。','', '## 学習・データ・公平性','', '各Fモデルは1,000,334パラメータ、Linear3->64、GRUCell64->545、scalar tanh residual、Linear545->2。問題ごとhiddenゼロ、LSB-first。第三特徴は全F/P入力で0。F0全16ADD、F1毎更新8ADD+8P-SUB、F2前半ADDのみ後半混合。C1 1200更新、境界600更新。C2 90学習CPU秒、境界45秒を更新開始時に判定。C2は総更新・総提示数一致ではなく時間一致として扱う。AdamW .002、wd .01、clip1、CPU2threads。Validation等演算full9bit CE最小、200更新ごとと終了時、同値最早。FのTestは全18run終了後。','', '既存Testは全9幅各演算512問で固定し、旧ファイルを変更しなかった。P変換後の入力同一性（ADD交換対も含む）で新規Train/Validationを事前に除外した。Train ADD23,488／SUB20,659、Validation3,245／3,206。除外8,614件をsplit_exclusions.csvに保存。F三方式は同じeligible集合を使用する。Test入力IDだけによる構造的分離であり、Testの正答率・ラベルによる条件調整ではない。','', '旧B0には8bitのP-SUB入力が通常ADD Trainと同じになるケースがある。Z1のold_input familiar/novel別集計をcarry_borrow_results.csvで開示。これはゼロショット更新0を否定しないが、8bit結果を入力未見の汎化とは呼べない。長幅は固定長入力が異なる。F条件は変換後評価入力と完全分離した。B0比較では学習利用可能データと入力flagも違うため純粋な学習方法因果効果を断定しない。','', '## RQ3：入力分布と誤答','', '|予算|条件|演算|2048 Bの1割合 (%)|正解incoming Carry/Borrow1 (%)|Bの最長1連続 平均|Carry/Borrow最長連続 平均|','|---|---|---|---:|---:|---:|---:|']
    for budget in C['budgets']:
        for cond,op in [('Z0','ADD'),('Z1','SUB'),('Z3','SUB'),('F1','SUB')]:
            rr=[r for r in carryrows if r['budget']==budget and r['condition']==cond and r['width']==2048 and r['op']==op]
            lines.append(f"|{budget}|{cond}|{op}|{statistics.mean(r['B_ones_ratio'] for r in rr)*100:.2f}|{statistics.mean(r['correct_incoming_ratio'] for r in rr)*100:.2f}|{statistics.mean(r['B_longest_ones']/512 for r in rr):.2f}|{statistics.mean(r['longest_carry']/512 for r in rr):.2f}|")
    lines+=['','Bの最初／最後のceil(W/4)桁は物理LSB端／MSB端と明示して集計。Carry頻度・最長連続はguardを除くW入力桁、guard込み長も別保存。Carry開始／継続／終了、最初の誤り、正答／誤答別長、誤答とのPearson相関を保存。R_B0の状態はBorrow、変換SUBはADD Carryなので直接同一視しない。相関は内部因果機構の証明ではない。error_correlations.csv、error_analysis.csv、carry_borrow_results.csvを参照。','', '## RQ4・RQ5：限定学習とADD能力維持','', 'F1/F2の改善は固定基準（全seed SUBで旧B0を上回る＋ADD平均低下5pp以内）で判定。F0との対応差もpaired_comparisons.csvで提示するが、登録済み成功基準を置き換えない。通常ADDの同一問題IDについて学習前B0で成功したものの失敗数lost、失敗から成功gained、維持率をretention_analysis.csvに保存。平均値だけでは入れ替わりを隠さない。','', '## CPU・RAM','']
    lines.append(f"新規学習部分CPU合計{sum(float(r['train_CPU_seconds']) for r in train):.3f}秒、wall合計{sum(float(r['train_wall_seconds']) for r in train):.3f}秒。主プロセスpeak RAM={max(int(r['peak_RAM_bytes']) for r in resources)/2**20:.2f}MiB（累積peak、run個別peakではない）。Validation、warmup、データ、保存、fit別計時。cached referenceのmodel時間はV054当時の記録であり今回の新規推論時間ではない。")
    lines+=['','|予算|条件|更新数 seed0/1/2|ADD／SUB提示数 seed0/1/2|学習CPU平均秒|','|---|---|---|---|---:|']
    for budget in C['budgets']:
        for cond in ('F0','F1','F2'):
            rr=sorted((r for r in train if r['budget']==budget and r['condition']==cond),key=lambda r:int(r['seed']));lines.append(f"|{budget}|{cond}|{' / '.join(r['updates'] for r in rr)}|{' ; '.join(r['ADD_presentations']+'/'+r['SUB_presentations'] for r in rr)}|{statistics.mean(float(r['train_CPU_seconds']) for r in rr):.3f}|")
    lines+=['','## 成功・失敗例','']
    for group,pp in examples.items():
        p=pp[0];lines.append(f"- {p['budget']} {p['condition']} seed{p['seed']} {p['op']} W={p['width']} {'成功' if p['correct'] else '失敗'}：A={p['a']}, B={p['b']}、正解={p['answer']}、予測={p['prediction']}、guard予測/正解={p['guard_prediction']}/{p['guard_truth']}、最初の誤り={p['first_error']}。")
    lines+=['','## 独立監査と未解決事項','', f'保存予測{num:,}件と失敗{failures:,}件を独立再採点。非有限result予測{NFcore}件、非有限guard{NFguard}件。独立flip+increment変換、出力幅・guard、整数・桁・phase・伝播分類を再検算。12既存checkpoint SHA・重み不変、36新規checkpoint再読込、eligible分割・学習乱数列・予算・F2境界を再生確認。既存成果{protected_n}ファイルSHA不変。監査記録はaudit_report.mdとaudit_results.json。','', '3seedの結果は今回の分布についての支持／不支持であり、任意長や数学的推論能力を証明しない。ゼロショット不成功は二の補数全般の不可能性を意味しない。正解側CarryはNN隠れ状態のCarryではない。補数方式の内部失敗原因、P対C同一重み因果比較、未知演算は未解決。guard剰余処理は事前固定の数学仕様であり、正答を参照する外部修正器・選択器ではない。','', '## 停止','', '固定checkpoint評価、事前登録18run、全9幅評価、保存データ監査と報告を完了して停止。追加学習・再試行・検証器・Oracle選択・GAN/PPO・parity・4096/8192評価・V056・他プロジェクト統合なし。']
    (R/'REPORT_V055.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
