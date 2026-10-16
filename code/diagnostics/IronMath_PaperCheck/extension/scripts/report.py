from common import *
import collections,gzip,math,re
def tab(headers,rr):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join('NA' if v is None or v=='' else str(v) for v in row)+' |\n' for row in rr)+'\n'
def main():
    assert read(R/'audit/INDEPENDENT_AUDIT.json')['passed'],'Do not revise paper until independent audit passes'
    syn=rows(R/'E1_finite_time/E1_RESULTS.csv');loc=rows(R/'E1_finite_time/E1_LINEARIZATION.csv');e2=rows(R/'E2_causal_intervention/E2_RESULTS.csv');measured=[r for r in e2 if r['status']=='measured'];ms=rows(R/'E3_independent_replication/E3_MODEL_RESULTS.csv');stats=rows(R/'E3_independent_replication/E3_CLUSTER_STATISTICS.csv');ag=read(R/'E3_independent_replication/AGGREGATE.json');test=rows(R/'E3_independent_replication/E3_TEST_RESULTS.csv');di=rows(R/'E3_independent_replication/E3_DIAGNOSTICS.csv');holds=rows(R/'E3_independent_replication/E3_HOLD_RESULTS.csv')
    # Quantitative statements are derived, not typed from expected outcomes.
    donor=[r for r in measured if r['condition'] in ['transplant_correct','transplant_failed'] and r['predicted_first']];predictionhits=sum(r['time_prediction_error']=='0' for r in donor)
    bytype=[]
    baselines={(r['seed'],r['op'],r['state'],r['pair'],r['at']):r for r in measured if r['condition']=='baseline'}
    for name in sorted({r['condition'] for r in measured}):
        rr=[r for r in measured if r['condition']==name and r['at']=='8' and baselines[r['seed'],r['op'],r['state'],r['pair'],r['at']]['first_error']];finite=[r for r in rr if r['first_error']];changes=[int(r['first_error'])-int(baselines[r['seed'],r['op'],r['state'],r['pair'],r['at']]['first_error']) for r in finite]
        bytype.append(dict(condition=name,N=len(rr),uncensored=len(finite),censored=len(rr)-len(finite),mean_first_time_change=float(np.mean(changes)) if changes else None))
    table('E2_causal_intervention/E2_CONTRASTS.csv',bytype)
    localstats=dict(anchors=len(loc),locally_applicable=sum(r['locally_applicable']=='True' for r in loc),both_crossings=sum(bool(r['time_error']) for r in loc),median_abs_timing_error=float(np.median([abs(float(r['time_error'])) for r in loc if r['time_error']])) if any(r['time_error'] for r in loc) else None)
    # A prediction window starting after the old crossing is left-censored, not a new accurate failure prediction.
    la=[]
    for r in loc:
        z=np.load(R/f"E1_finite_time/cache/V054_s{r['seed']}_trajectory.npz");probes=read(R/f"E1_finite_time/cache/V054_s{r['seed']}_diagnostic.json")['records'];i=next(p['index'] for p in probes if p['op']==r['op'] and str(p['state'])==r['state'] and p['pair']==r['pair']);anchor=int(r['anchor']);already=anchor>0 and z['margins'][anchor-1,i]<0
        la.append(dict(r,already_wrong_at_anchor=bool(already),timing_valid_window=not already and bool(r['time_error'])))
    table('E1_finite_time/E1_LINEARIZATION_WINDOW_AUDIT.csv',la);valid=[r for r in la if r['timing_valid_window']];vmae=float(np.median([abs(float(r['time_error'])) for r in valid])) if valid else None
    table('E1_finite_time/E1_TIMING_SUMMARY.csv',[dict(all_anchors=len(loc),residual_applicable=localstats['locally_applicable'],not_left_censored_crossing_windows=len(valid),median_abs_timing_error_not_left_censored=vmae,median_abs_timing_error_all_crossing_windows=localstats['median_abs_timing_error'])])
    e1text='''# E1：有限時刻失敗
線形安定性だけと有限時刻の誤分類を区別した。理論はE1_THEORY.mdの初等恒等式・十分条件・反例であり、新規定理の優先性を主張しない。
\n## 合成系\n\n'''+tab(['系','精度','初回誤答','窓内持続開始','回復','初回打切り'],[[r[k] for k in ['case','dtype','first_error','persistent_error_start_observed','recoveries','first_error_censored']] for r in syn])
    e1text+='固定点が正解側でも過渡的誤答と回復があり、境界上の例では実数上のm_t=2^(-t)は全有限時刻で正だが、FP32は150、FP64は1075で0に丸められ、クラス1の同点規則により誤る。decimal100桁では4096まで誤らなかった。高精度数値参照は証明ではない。FP64の境界例の元の状態誤差0表示は、誤差ノルムのunderflowによるもので、audit/E1_HIGH_PRECISION_ERROR_AUDIT.csvでDecimalノルムを別に保存した。0を厳密一致と扱わない。\n\n'
    e1text+=f"## 既存算術への適用\n\nV054Hは状態ベクトルがなくスカラーのみだったため、3checkpointを読み取り専用再生した。36例の初回負マージン時刻は元と厳密一致、最初512ステップのマージン差は全例1e−3以内。距離・状態間距離・破綻前局所変化をE1_STATE_DISTANCES.csvへ保存。数値終点でのdeltaも含めた局所一次近似を144窓で評価した。one-step残差基準を満たすのは{localstats['locally_applicable']}/144窓。既に誤答側へ入ったanchorは左打切りとして別記し、新しい破綻を予測した成功例に数えない。未破綻anchorで両方の交差が窓内にある{len(valid)}窓の時刻絶対誤差中央値は{vmae} step。全144窓と不成立例・打切りを残した。\n\n"
    e1text+='局所近似が不適合な初期状態へ線形公式を無条件で外挿できない。観測窓の持続誤答開始は窓外永久性ではない。再生中に出力パス型のTypeErrorが発生し、計算済み状態/Jを再利用して再開した（audit/failures.jsonl、RESUME_E1.json）。科学的条件・重み・seedは変えていない。\n'
    md('E1_finite_time/E1_REPORT.md',e1text);md('E1_REPORT.md',e1text);md('E1_THEORY.md',(R/'E1_finite_time/E1_THEORY.md').read_text(encoding='utf-8')+'\n\n'+(R/'E1_finite_time/E1_THEORY_SUPPLEMENT.md').read_text(encoding='utf-8'))
    e2text=f'''# E2：推論時状態介入
既存V054 3seed×12hold文脈、介入step1/4/8、窓1024。条件と方向予測は実施前に凍結。{len(measured)}測定行と、適格な失敗donorが存在しない{len(e2)-len(measured)}行を保存した。正解donor=step1の正答過渡状態、失敗donor=step512の誤答状態。失敗しない文脈に偽の失敗donorを作らなかった。

予測時刻が定まる移植{len(donor)}件のうち{predictionhits}件が厳密一致。正解donorによる遅延は決定論的な同じ軌道の再開で説明できる。初回の時刻だけでなく持続的誤答開始、介入後の正答継続、回復、効果の最後の観測時刻を全測定行に保存した。

## normを合わせた対照（step8、元々1024以内に壊れる文脈）

'''+tab(['条件','N','観測時刻あり','打切り','初回誤答の平均変化'],[[r['condition'],r['N'],r['uncensored'],r['censored'],r['mean_first_time_change']] for r in bytype])
    e2text+='平均変化は時刻を観測できた例の記述値で、打切りを1025などに置換していない。random/orthogonalは各3方向で、同じ文脈・時刻・donorのnormに一致させた。法線正方向だけでも遅延し、負方向は誤答を早める。従って状態の出力への因果作用は確認できるが、共有アトラクタという構造自体の因果作用は分離できない。正解donorはcarry情報の直接注入を含み、計算履歴と完全に同じ文脈ではなく、永久的な修復ではない。\n\n'
    ar=rows(R/'E2_causal_intervention/E2_ARITHMETIC.csv');ars=[]
    for op in ['ADD','SUB']:
        for w in ['8','64']:
            for mode in ['correct','failed']:
                rr=[r for r in ar if r['op']==op and r['width']==w and r['mode']==mode];ars.append([op,w,mode,len(rr),sum(r['intervened']=='True' for r in rr),sum(r['baseline_correct']=='True' for r in rr),sum(r['intervention_correct']=='True' for r in rr),sum(r['baseline_correct']=='False' and r['intervention_correct']=='True' for r in rr),sum(r['baseline_correct']=='True' and r['intervention_correct']=='False' for r in rr)])
    e2text+='## 算術転移（同じ固定部分集合）\n\n'+tab(['演算','幅','移植','N','実施件数','無介入正解','介入正解','改善','悪化'],ars)
    e2text+='bit位置8でOracleのcarryとholding pairに一致した時だけ介入。8bitでは対象外も多く、無介入の入力は予測不変を確認した。Oracleは介入割当/採点だけだが、実運用で利用できる推論方式とは呼ばない。副作用と未介入ケースもE2_ARITHMETIC.csvに保存。独立監査は代表曲線しか残っていなかった点を補い、同じ1350条件を独立実装で読み取り専用再生し全時刻指標を再採点、normと線形境界方向のmargin変化を検証した。これは新条件・追加学習ではない。\n'
    md('E2_causal_intervention/E2_REPORT.md',e2text);md('E2_REPORT.md',e2text)
    e3text=f'''# E3：独立初期化複製
10初期化seed100〜109×2データ提示条件、計20モデルを完了した。全20学習終了後にのみTest評価を開始した。構造1000334パラメータ、1200更新、ADD8/SUB8、AdamW .002/wd.01/clip1。選択は200更新ごとの等演算Validation CE最小。全20モデルの初期化SHA・提示順・選択checkpointを独立再計算した。

データ条件は同じ固定Train母集団から独立乱数で得た問題提示系列で、独立に生成したコーパスではない。c0/c1は同じ分布の2実現であり、異なるデータ分布への処置効果を表すラベルではない。2条件は同じ初期化を共有し、推論単位は初期化10クラスタ。モデル20を独立な20反復と扱わない。

Validation達成（両演算95%以上）は{ag['qualified_models']}/20モデル。両条件達成は{ag['paired_qualified_initializations']}/10初期化、両条件で少なくとも1つの厳密な共有過渡機構pairが再現したのは{ag['paired_reproduced_initializations']}/10。固定分類は「{ag['judgment']}」。共有候補{ag['shared_pairs']}pair、別終点{ag['distinct_pairs']}pair、未分類{ag['unclassified_pairs']}pair、厳密再現{ag['strict_pairs']}pair（最大80pair）だった。

'''+tab(['モデル','Validation達成','共有候補','別終点','未分類','厳密再現pair'],[[r['model'],r['training_achieved'],r['shared_pairs'],r['distinct_pairs'],r['unclassified_pairs'],r['strict_reproduced_pairs']] for r in ms])
    e3text+='## 初期化クラスタの効果量\n\n'+tab(['指標','初期化数','平均','標本SD（初期化平均）','95%区間'],[[r['metric'],r['initialization_clusters'],r['mean'],r['sample_SD'],f"[{r['CI95_low']}, {r['CI95_high']}]" ] for r in stats])
    e3text+='両データ条件を同じクラスタ内に保ち10,000回bootstrap（seed99000）。条件差はc1−c0、short_minus_longはTest8−Test2048。qualified_rateは達成モデルがある初期化の平均で、全モデルのmechanism_model_rateとは分母が異なりうる。区間が0幅の場合も、母集団で不確実性が0だとは解釈しない。\n\n'
    err=[float(r['timing_error']) for r in di if r['timing_error']]
    e3text+=f"hold長16/64/256/1024の16文脈を全モデルで評価した。予測負マージン時刻とk>=256の観測中央値が両方ある{len(err)}ケースの絶対誤差中央値は{float(np.median(np.abs(err))) if err else None} step。これは観測可能なケースの記述値で、全ケースの精度に拡大しない。予測欠測・打切り・prefix誤答・k別の誤り位置はE3_HOLD_RESULTS.csv、E3_DIAGNOSTICS.csvに残す。\n\n"
    e3text+='再現しないモデル、別終点を持つモデル、収束判定未達の未分類を削除していない。機構が見られないことと学習未達を区別した。共有判定は固定した2つの出発状態の数値終点が近いという操作的分類で、全吸引領域の探索や数学的な同一固定点の証明ではない。同じLSB-first、同じGRU構造、同じ8bit母集団に限った複製であり、全RNN・別課題・別コーパスへの一般性は未検証。\n'
    md('E3_independent_replication/E3_REPLICATION_REPORT.md',e3text);md('E3_REPLICATION_REPORT.md',e3text)
    audit=read(R/'audit/INDEPENDENT_AUDIT.json')
    md('AUDIT_REPORT.md',f'''# 統合独立監査
独立監査passed={audit['passed']}。評価モジュールをimportしないaudit_independent.pyで、E2全1350測定条件を独立の標準GRUCell定義で再生し初回/持続時刻を再採点。norm・法線marginの代数関係も照合した。
E3のTest全{audit['E3_test_groups']}群の整数をPythonで採点、hold {audit['E3_hold_predictions']}件の最初の誤りをXORと独立bit位置抽出で再算出。全20初期化/提示順を再構成し、Validation最良選択と1200更新を検査。80pairの終点距離と収束から幾何分類を再計算し、誤読側と観測誤り側が同じという厳密機構条件も独立確認。初期化クラスタの区間を別実装で再計算した。
入力SHA不変={audit['all_input_SHA_unchanged']}。既存G1・ABL1・v0.6・checkpointは変更なし。E1出力パスのTypeErrorとcache再開をfailures.jsonlへ保存。FP64の境界例の状態ノルム誤差表示0はunderflowとして原因を同定し、Decimalノルムを別資料へ保存。破綻時刻は変えていない。
収束終点のJacobianとrhoはGRUゲート偏微分式をNumPy FP64で独立実装して確認した。未収束点のrhoは補完しない。全学習の再実行はしていない。評価コードの局所線形化は保存Jとdeltaを使用した経験近似で、理論保証とは区別。監査対象の範囲を越える再現性を主張しない。
''')
    # Scientific figures, generated from measurements; no new model computation.
    os.environ['MPLCONFIGDIR']=str(R/'audit/matplotlib_cache')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    for folder in ['E1_finite_time/E1_FIGURES','E2_causal_intervention/E2_FIGURES','E3_independent_replication/E3_FIGURES']:(R/folder).mkdir(exist_ok=True)
    with gzip.open(R/'E1_finite_time/synthetic_trajectories.csv.gz','rt',encoding='utf-8') as f:tr=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(7,3.5))
    for name in ['scalar_wrong_limit','diagonal_transient','stable_rotation']:
        rr=[r for r in tr if r['case']==name and r['dtype']=='FP64' and int(r['t'])<=300];ax.plot([int(r['t']) for r in rr],[float(r['margin']) for r in rr],label=name)
    ax.axhline(0,color='black',lw=.7);ax.set_xlabel('Recurrence step');ax.set_ylabel('Correct-class margin');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(R/'E1_finite_time/E1_FIGURES/synthetic_margins.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.5));rr=[r for r in bytype if r['condition'] in ['transplant_correct','transplant_failed','random0_correct','orthogonal0_correct','normal_1_correct']];ax.bar([r['condition'].replace('_','\n') for r in rr],[r['mean_first_time_change'] for r in rr]);ax.axhline(0,color='black',lw=.7);ax.set_ylabel('Mean shift in first error (step 8)');ax.tick_params(axis='x',labelsize=7);fig.tight_layout();fig.savefig(R/'E2_causal_intervention/E2_FIGURES/first_error_shifts.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8,3.4))
    for c,marker in [('0','o'),('1','x')]:
        rr=[r for r in ms if r['data_condition']==c];axs[0].plot([int(r['initialization_seed']) for r in rr],[int(r['strict_reproduced_pairs']) for r in rr],marker=marker,label='condition '+c)
        for op in ['ADD','SUB']:
            ts=[t for t in test if t['op']==op and t['width']=='2048' and t['condition']==c];axs[1].plot([int(t['initialization_seed']) for t in ts],[float(t['accuracy']) for t in ts],marker=marker,label=op+' c'+c)
    axs[0].set_ylabel('Strict reproduced pairs / 4');axs[1].set_ylabel('2048-bit integer accuracy')
    for ax in axs:ax.set_xlabel('Initialization seed');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(R/'E3_independent_replication/E3_FIGURES/replication.pdf');plt.close(fig)
    # Only now create the new English draft, preserving the v0.6 archive byte-for-byte.
    v06=(R/'paper/v06/main.tex').read_text(encoding='utf-8');new=r'''
\section{Finite-time failure: conditions and counterexamples}
\label{sec:e1extension}
The following extension was designed after the original experiments and ABL1. Its rules were frozen before its own measurements; it is not part of the historical G1 preregistration. For an affine recurrence and a linear correct-class margin,
\[h_t=h^*+A^t(h_0-h^*),\qquad m_t=m_\infty+w^TA^t(h_0-h^*).\]
These are elementary identities, not a new theorem. If $\rho(A)<1$ and $m_\infty<0$, eventual sustained error follows. A positive limiting margin does not exclude transient errors. A zero limiting margin does not imply any finite-time error: $m_t=2^{-t}>0$ at every finite real-valued step. Bounds require the initial displacement, readout geometry, and a bound on powers of $A$, rather than the spectral radius alone.
In five fixed synthetic systems we separated the first error, the start of an error suffix within the observation window, recovery, and right censoring. The monotone scalar example failed at step 230 in FP32, FP64 and the 100-digit decimal reference. In the boundary example with class 1 as the target, floating-point underflow produced a tie and an error at step 150 in FP32 and 1075 in FP64, while the decimal reference remained correct through 4096 steps. The reference is numerical evidence, not a proof. The positive-limit diagonal and stable rotation examples recovered from transient errors.
For the frozen arithmetic checkpoints, 36 replayed contexts reproduced the original first negative-margin steps. We tested local linearization with the numerical fixed-point defect retained, $e_{t+1}\approx\delta+Je_t$. @LOCAL@ of 144 fixed windows met the registered one-step residual criterion. Windows already on the wrong side were distinguished from predictions of a new crossing. The accumulated state error is $\sum_{k=0}^{t-1}J^{t-1-k}r_k$ when $r_k$ is the one-step remainder. A small measured relative remainder is not a uniform error guarantee, and $\rho(J)<1$ does not imply a contractive operator norm. These observations do not license extrapolating a local approximation from arbitrary states.
\begin{figure}[t]\centering\includegraphics[width=.85\linewidth]{../../E1_finite_time/E1_FIGURES/synthetic_margins.pdf}\caption{Fixed synthetic examples distinguish eventual behavior from transient error.}\end{figure}

\section{Inference-time state interventions}
\label{sec:e2extension}
We froze three existing checkpoints, all 36 hold contexts, intervention steps 1, 4 and 8, and a 1024-step observation window. Donors were fixed at step 1 for a correct transient state and step 512 for a failed state, requiring the stated readout at that time; unavailable donors were not substituted. Norm-matched random, readout-normal and orthogonal perturbations were evaluated under the same conditions. All @E2N@ measured cases and @NA@ unavailable-donor cases were retained. First-error predictions were exact in @HITS@ of @DONORN@ donor cases with an observed prediction. Effects were finite: refreshing a state restarts a transient rather than changing the update map.
The donor measurements share ten failing contexts and three existing checkpoints; they are not independent trials for each intervention time. Normal perturbations alone could also delay or advance failure. The interventions demonstrate an effect of state on output, but do not isolate the causal role of shared-attractor geometry. Correct-donor transplantation injects carry-related information. Oracle-assisted assignment on fixed arithmetic subsets was reported separately, including adverse changes and unaffected cases. It is not an inference algorithm that has access to a hidden correct carry. The E3 models below replicate the observational mechanism, not the full intervention experiment.
\begin{figure}[t]\centering\includegraphics[width=.85\linewidth]{../../E2_causal_intervention/E2_FIGURES/first_error_shifts.pdf}\caption{Observed, uncensored first-error shifts for step-8 interventions; unavailable and censored cases are retained in the accompanying tables.}\end{figure}

\section{Independent-initialization arithmetic replication}
\label{sec:e3extension}
We trained 20 models: ten new initializations (100--109), each with two independently sampled presentation streams from the original fixed training population. The streams are not independent corpora. Architecture, 1200-update budget, optimizer, and Validation-only checkpoint selection matched the original binary model. All training finished before Test evaluation. @QUAL@ of 20 met the fixed Validation criterion of at least 95\% integer accuracy on both operations.
The strict mechanism required converged and locally stable endpoints within $10^{-3}$ of one another, an incorrect endpoint readout, and an observed hold error on that same side. @REPRO@ of ten initializations reproduced at least one such pair under both data conditions. Of 80 pair opportunities, @STRICT@ met the strict criterion, @SHARED@ were shared-endpoint candidates, @DISTINCT@ had distinct converged endpoints, and @UNCLASS@ were not classified as converged pairs. Non-replications were retained.
At 2048 bits, the initialization-averaged ADD accuracy was @ADD@ and SUB accuracy was @SUB@. Intervals resampled ten initialization clusters, retaining the two data conditions together (10000 percentile bootstrap replicates, seed 99000). All-model and Validation-qualified results are separate. These results address this architecture, task format and training population, not all RNNs.
\begin{figure}[t]\centering\includegraphics[width=.95\linewidth]{../../E3_independent_replication/E3_FIGURES/replication.pdf}\caption{Mechanism diversity and length-generalization accuracy across the fixed initializations and paired presentation conditions.}\end{figure}

'''
    stat={r['metric']:r for r in stats}
    fields={'LOCAL':str(localstats['locally_applicable']),'E2N':str(len(measured)),'NA':str(len(e2)-len(measured)),'HITS':str(predictionhits),'DONORN':str(len(donor)),'QUAL':str(ag['qualified_models']),'REPRO':str(ag['paired_reproduced_initializations']),'STRICT':str(ag['strict_pairs']),'SHARED':str(ag['shared_pairs']),'DISTINCT':str(ag['distinct_pairs']),'UNCLASS':str(ag['unclassified_pairs'])}
    for op in ['ADD','SUB']:
        s=stat[op+'_2048_accuracy'];fields[op]=f"{100*float(s['mean']):.2f}\\% (95\\% interval [{100*float(s['CI95_low']):.2f}, {100*float(s['CI95_high']):.2f}])"
    for k,v in fields.items():new=new.replace('@'+k+'@',v)
    paper=v06.replace(v06.splitlines()[0],'% IronMath paper -- extension draft v0.7 (E1-E3; exploratory extension)')
    added_graphics='\\usepackage{graphicx}' not in paper
    if added_graphics:paper=paper.replace('\\usepackage{booktabs}','\\usepackage{booktabs}\n\\usepackage{graphicx}')
    assert '\\section{Negative results}' in paper;paper=paper.replace('\\section{Negative results}',new+'\\section{Negative results}')
    paper=paper.replace('The confirmatory phase consists of','The historical experiment series includes')
    md('paper/v07/main.tex',paper)
    for n in ['refs.bib','jmlr2e.sty']:(R/'paper/v07'/n).write_bytes((R/'paper/v06'/n).read_bytes())
    # Preserve old numerical blocks: the inserted sections are exactly removable.
    recovered=paper.replace(new,'').replace('% IronMath paper -- extension draft v0.7 (E1-E3; exploratory extension)',v06.splitlines()[0]).replace('The historical experiment series includes','The confirmatory phase consists of')
    if added_graphics:recovered=recovered.replace('\\usepackage{booktabs}\n\\usepackage{graphicx}','\\usepackage{booktabs}')
    assert recovered==v06
    # Citation names, labels, and figure paths are checked without claiming a TeX build.
    bib=(R/'paper/v07/refs.bib').read_text(encoding='utf-8');keys=set(re.findall(r'@\w+\s*\{\s*([^,]+)',bib));cited={s.strip() for block in re.findall(r'\\cite\w*\s*(?:\[[^\]]*\])*\{([^}]+)\}',paper) for s in block.split(',')};missing=sorted(cited-keys)
    dump('paper/PAPER_SOURCE_CHECK.json',dict(utc=utc(),old_numerical_and_text_blocks_preserved=True,missing_citation_keys=missing,compiled=False,compilation_note='pdflatex not available in current shell; source-only draft',v06_SHA256=sha(R/'paper/v06/main.tex'),v07_SHA256=sha(R/'paper/v07/main.tex'),numbers=fields));assert not missing,missing
    md('PAPER_CHANGELOG_v07.md','''# v0.7変更履歴
既存v0.6 main (7).texをpaper/v06にSHA付き保全。既存のG1数値、ABL1の140対145という否定的結果、元のNegative resultsを保持した。新しいE1/E2/E3の英語節と実測図をNegative resultsの前へ追加し、過去の登録実験とは異なる拡張と明記。表題のWhy and Whenは変更しない。初等線形恒等式に新規定理の優先性を主張せず、状態介入は共有アトラクタ構造の因果証明ではないと明記。E3は2データ提示系列であり独立コーパスとは呼ばない。古い末尾のconfirmatory phaseという一括表記をhistorical experiment seriesへ限定した。
追加節・graphicx・冒頭版名とこの文言変更を逆変換すると元v0.6と全文一致することを検査。既存の原稿ファイルは書き換えていない。参考文献は元版を維持し、新しい世界初・文献優先性の主張を追加しない。pdflatexがなく、PDF組版の検証は未実施。投稿可能との判断は研究主任へ委ねる。
''')
    md('ALL_NUMBERS_CHECK_v03.md',f'''# v0.7追加数値の照合
E1の230/150/1075/4096はE1_RESULTS.csvと高精度監査に一致。E1の36再現と局所適合{fields['LOCAL']}/144はBASELINE_REPRODUCTION.csvとE1_LINEARIZATION.csv。E2の{fields['E2N']}測定・{fields['NA']}欠測と{fields['HITS']}/{fields['DONORN']}時刻一致はE2_RESULTS.csvを集計し、独立再生で確認。E3の20モデル・10初期化・1200更新・1000334パラメータ・Validation達成{fields['QUAL']}、再現初期化{fields['REPRO']}、strict pair{fields['STRICT']}、shared候補{fields['SHARED']}、distinct{fields['DISTINCT']}、未分類{fields['UNCLASS']}はE3_MODEL_RESULTS/E3_PAIR_RESULTS/AGGREGATEに一致。ADD/SUB2048平均と区間はE3_CLUSTER_STATISTICS.csvを新規節へ自動転記し独立統計監査に一致。
元v0.6は追加節等を逆変換して全文一致。従って旧G1・ABL1の数値は一字も変更していない。本照合は追加節の数値と原稿保全の照合で、元v0.6の全主張を改めて正しいと認定したものではない。新しい数値の不一致0、未検証はTeX組版と新規文献優先性。精密値・SHAと論文転記の対応はpaper/PAPER_SOURCE_CHECK.json。
''')
    master=f'''# E1〜E3 統合報告
完了：{utc()}。全20モデル・指定評価・独立監査を完了し、新規v0.7ドラフトを作成して停止する。元のG1 P1/P2、ABL1、v0.6は不変。

1. Q1：線形系の有限時刻条件と境界/過渡反例を明示したが、初等結果であり新規定理とは主張しない。合成230stepは一致、境界は精度で150/1075へ変化。非線形局所近似の適合は{fields['LOCAL']}/144窓に限られた。
2. Q2：同文脈の状態移植の時刻予測は{fields['HITS']}/{fields['DONORN']}一致。効果は有限で、持続的修復ではない。
3. Q3：norm一致random/orthogonalと区別した効果は観測されたが、法線方向だけでも遅延・早期化し、carry直接注入もある。共有アトラクタ構造自体の特異的因果効果とは結論できない。
4. Q4：E3固定分類「{ag['judgment']}」。両データ条件での再現は{ag['paired_reproduced_initializations']}/10初期化。全20と学習達成{ag['qualified_models']}/20を別に報告。
5. Q5：安定性・極限読み出し・初期状態・境界位置・過渡・数値精度が条件となる。非収束・別終点・学習未達を同じ原因としない。
6. Q6：追加の事実は、この算術器群での条件付き時刻対応、対照状態介入、独立初期化＋提示系列の複製。体系的な文献優先性調査はしておらず、世界初とは主張しない。
7. Q7：記述的なWhy/Whenと限定的な状態の因果作用・同構造内再現性を拡張できる。一方、ABL1の単純読み出し優位、固定点条件固有の予測増分未確認は保持する。
8. Q8：吸引構造そのものの因果同定、別アーキテクチャ/課題/独立コーパスへの一般性、局所近似の広域適用、窓外の永久性、TeX組版は未解決。

新規学習は20研究モデル（別に資源試験scratch10更新、性能評価なし）。既存モデルのE1/E2再生とE2監査再生を区別して保存した。CPU2threads・peak2GiB上限。学習CPU合計{sum(float(m['train_CPU_seconds']) for m in ms):.3f}秒、学習wall合計{sum(float(m['train_wall_seconds']) for m in ms):.3f}秒。これはforward/backward/clip/optimizer部分で、Validation/保存/E1/E2は含まない。モデル別資源はE3_MODEL_RESULTS.csv。全体実経過は準備・実装・監査を含むため学習費用と同一視しない。
E1の出力パスエラーを保存し、計算済みcacheから再開した。原稿保全・SHA・数値再採点はAUDIT_REPORT.md。全報告書UTF-8読み直し。投稿可否・追加研究は研究主任が判断する。
'''
    md('EXTENSION_MASTER_REPORT.md',master)
    process_snapshot('after')
    utf=[]
    for p in R.rglob('*.md'):
        s=p.read_text(encoding='utf-8');assert '\ufffd' not in s;utf.append(dict(path=p.relative_to(R).as_posix(),SHA256=sha(p),UTF8_readback=True))
    dump('audit/UTF8_CHECKS.json',dict(utc=utc(),files=utf));dump('audit/FINAL_SHA256.json',dict(utc=utc(),files={p.relative_to(R).as_posix():dict(bytes=p.stat().st_size,SHA256=sha(p)) for p in R.rglob('*') if p.is_file() and p.name!='FINAL_SHA256.json'},self_excluded=True,inputs_unchanged=True))
    print('ALL REPORTS AND v0.7 COMPLETE; STOP',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:stage_failure('reports',e);raise
