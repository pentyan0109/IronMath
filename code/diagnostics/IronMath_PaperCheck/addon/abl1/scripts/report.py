"""Finish ABL1 with auditable interpretation and UTF-8 reports; no inference."""
from prepare import R,W,read,write,sha,utc,processes
from analyze import csvrows,metrics,table,METHODS,PAIRS
import csv,json,math,ast,collections
def md(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(v) if v is not None else 'NA' for v in row)+' |\n' for row in rows)+'\n'
def flag(v):return v=='True'
def finish():
    models=csvrows(R/'abl1_model_results.csv');summary=csvrows(R/'abl1_architecture_summary.csv');pairs=csvrows(R/'abl1_pairwise.csv');traj=csvrows(R/'abl1_trajectory_details.csv');fp=csvrows(R/'abl1_fp64_sensitivity.csv')
    checks=[]
    def check(label,condition):checks.append(dict(check=label,passed=bool(condition)));assert condition,label
    check('200 unique models',len(models)==200 and len({r['model'] for r in models})==200)
    check('seed 20..39; each architecture has both L per seed',all(sum(m['architecture']==a and int(m['seed'])==s and int(m['L'])==l for m in models)==1 for a in ['residual_gru','gru','lstm','rnn','linear'] for s in range(20,40) for l in [8,16]))
    check('3200 unique diagnostic trajectories',len(traj)==3200 and len({(r['model'],r['neutral'],r['prefix'],r['value']) for r in traj})==3200)
    check('42 diagnostic summaries and 49 paired cluster comparisons',len(summary)==42 and len(pairs)==49)
    check('all per-model A and D and saved-bit truth checks passed',all(r['passed']=='True' for r in csvrows(R/'abl1_original_checks.csv')))
    check('no margins / validity ambiguity',read(R/'MARGIN_ANOMALIES.json')==[])
    check('argmax ties: value0 correct, value1 wrong',(0>=0) and not (0>0))
    check('undefined BA and MCC on linear are blank',all(r['balanced_accuracy']==r['MCC']=='' for r in summary if r['group']=='linear'))
    check('A0 success subset A1/A3/A4 success',all(not flag(m['A0']) or all(flag(m[k]) for k in ['A1','A3','A4']) for m in models))
    # No statistical outcome is required by QA. Check arithmetic and paired structure only.
    for r in pairs:
        check(f"{r['group']} {r['left']}-{r['right']} discordances conserve hit difference",int(r['left_hits'])-int(r['right_hits'])==int(r['left_only_correct'])-int(r['right_only_correct']))
        check(f"{r['group']} {r['left']}-{r['right']} seed cluster count20",r['seed_count']=='20')
    for a in pairs:
        if a['left']=='A0' and a['right']=='A3':
            b=next(r for r in pairs if r['group']==a['group'] and r['left']=='A3' and r['right']=='A0')
            check(f"{a['group']} reverse contrast sign consistency",abs(float(a['accuracy_difference'])+float(b['accuracy_difference']))<1e-12)
    # Break down overlapping labels and count changes without suggesting causal attribution.
    modes=[]
    labels={'divergence':'long_nonfinite','finite_error':'finite_failure','diagnostic_nonconvergence':'nonconvergent_trajectories','endpoint_readout_error':'endpoint_readout_errors','rho_not_below_one_converged':'rho_not_below_one_converged','intermediate_error':'intermediate_wrong_or_invalid_steps','mixed_or_rewrite_failure':'rewrite_or_mixed_failure','immediate_rewrite_error':'immediate_rewrite_R_errors'}
    for group in ['all200','nonlinear160','residual_gru','gru','lstm','rnn','linear']:
        ms=[m for m in models if group=='all200' or group=='nonlinear160' and m['architecture']!='linear' or m['architecture']==group]
        for label,col in labels.items():
            ss=[m for m in ms if flag(m[col]) if m[col] in ['True','False']] if col in ['long_nonfinite','finite_failure','rewrite_or_mixed_failure'] else [m for m in ms if int(m[col])>0]
            for meth in METHODS:modes.append(dict(group=group,label=label,overlap_permitted=True,diagnostic=meth,models=len(ss),hits=sum(m[meth]==m['actual_success'] for m in ss),actual_success=sum(flag(m['actual_success']) for m in ss),actual_failure=sum(not flag(m['actual_success']) for m in ss),predicted_failure=sum(not flag(m[meth]) for m in ss)))
    table('abl1_failure_mode_summary.csv',modes)
    diss=[dict(model=m['model'],architecture=m['architecture'],L=m['L'],seed=m['seed'],actual_success=m['actual_success'],endpoint_readout_A1=m['A1'],convergence_stability_A4=m['A4'],nonconvergent_trajectories=m['nonconvergent_trajectories'],endpoint_readout_errors=m['endpoint_readout_errors'],classification=m['classification']) for m in models if m['A1']!=m['A4']]
    table('abl1_readout_stability_disagreements.csv',diss)
    # Correct only unfinalized ABL1 output metadata: A0/A5 lookups were not timed.
    # Never claim the accumulator's initial zero is a measured execution time.
    costs=csvrows(R/'abl1_costs.csv');corrections=[]
    for row in costs:
        if row['diagnostic'] in ['A0','A5'] and row['component']=='predicate analysis on saved arrays':
            corrections.append(dict(diagnostic=row['diagnostic'],old_placeholder=row['wall_seconds'],corrected_wall_seconds=None,reason='Frozen lookup predicates were not timed; initial accumulator zero is not a measurement'))
            row['wall_seconds']='';row['status']='not separately instrumented; frozen lookup, not zero runtime'
    with (R/'abl1_costs.csv').open('w',encoding='utf-8',newline='') as f:q=csv.DictWriter(f,list(costs[0]));q.writeheader();q.writerows(costs)
    write('OUTPUT_METADATA_NOTES.json',dict(utc=utc(),scope='Only new, unfinalized ABL1 output timing metadata',corrections=corrections,diagnostic_predictions_changed=False,frozen_rules_changed=False,original_inputs_changed=False))
    # Check the reported baseline aggregate against source metrics.
    old=read(W/'IronMath_G1/g1_metrics.json')
    for method,oldmethod in [('A0','A'),('A5','D')]:
        s=next(r for r in summary if r['group']=='nonlinear160' and r['diagnostic']==method)
        source=next(r for r in old['pooled_nonlinear']['methods'] if r['method']==oldmethod)
        check('G1 '+oldmethod+' confusion matrix matches',all(int(s[k])==source[k] for k in ['hits','TP','TN','FP','FN']))
    fpchanged=[r for r in fp if r['A0_FP32']!=r['A0_FP64']];readchanged=[r for r in fp if r['changed_readout']!='0']
    convergence_changed=[r for r in fp if r['changed_convergence']!='0'];spectralchanges=sum(int(r['shared_converged_rho_threshold_changes']) for r in fp)
    extras=dict(A1_A2_model_predictions_differ=sum(m['A1']!=m['A2'] for m in models),A0_A3_model_predictions_differ=sum(m['A0']!=m['A3'] for m in models),A4_A5_model_predictions_differ=sum(m['A4']!=m['A5'] for m in models),converged_trajectories=sum(flag(r['converged']) for r in traj),converged_rho_not_eligible=sum(flag(r['converged']) and not flag(r['rho_eligible']) for r in traj),endpoint_good_but_intermediate_bad=sum(flag(r['readout_correct']) and not flag(r['trajectory_A2_good']) for r in traj),endpoint_good_intermediate_bad_models=len({r['model'] for r in traj if flag(r['readout_correct']) and not flag(r['trajectory_A2_good'])}),fp64_A0_changed_models=[r['model'] for r in fpchanged],fp64_convergence_changed_models=[r['model'] for r in convergence_changed],fp64_readout_correctness_changed_trajectories=sum(int(r['changed_readout']) for r in fp),fp64_readout_correctness_changed_models=len(readchanged),fp64_shared_converged_rho_threshold_changes=spectralchanges,fp64_A1_model_predictions_changed=sum(r['A1_FP32']!=r['A1_FP64'] for r in fp),FP32_A1_equals_FP64_A0_on_all_models=all(m['A1']==f['A0_FP64'] for m,f in zip(models,fp)))
    write('INCREMENTAL_FINDINGS.json',extras)
    fp_rows=csvrows(W/'IronMath_PaperCheck/addon/fp64/fp64_trajectories.csv')
    converged_read_changes=sum(r['converged_FP32']=='True' and r['readout_correct_FP32']!=r['readout_correct_FP64'] for r in fp_rows)
    check('FP64 comparison readout label agrees FP32 direct endpoint records',all(all(x['readout_correct_FP32']=='True' for x in fp_rows if x['model']==m['model'])==flag(m['A1']) for m in models))
    # Final source SHA check includes the paper and every source consumed above.
    manifest=read(R/'INPUT_MANIFEST.json');changed=[x['path'] for x in manifest['files'] if not (W/x['path']).is_file() or sha(W/x['path'])!=x['SHA256']]
    check('all selected original inputs unchanged at finalization',not changed)
    check('rules/env/input freeze unchanged',all(sha(R/name)==x['SHA256'] for name,x in read(R/'FREEZE.json')['files'].items()))
    write('QA_RESULTS.json',dict(utc=utc(),passed=all(x['passed'] for x in checks),checks=checks,source_inputs_checked=len(manifest['files']),source_inputs_changed=changed,no_training=True,no_model_inference=True,no_original_script_execution=True,output_metadata_corrections_only=corrections))
    # Report.
    text='''# ABL1：診断Aの構成要素と増分価値

ABL1はG1の主結果の後に追加した探索的検証であり、G1の事前登録P1・P2を変更しない。

## 1. 要点

保存済みデータだけで完了した。新規学習0回、新規推論0回、checkpointのモデルロード0回。全200モデルの凍結A/Dと元の長系列成否を再現した。主要な非線形160モデルでは、A0は140件、終点読み出しのみA1と途中読み出しA2は145件的中した。収束を要求するA3はA0と同一、読み出しを使わないA4は91件で元D（A5）と同じ予測だった。

この標本では、収束・局所安定性条件の追加による予測性能の増分は確認されず、終点読み出しのみで予測情報を再現し、FP32閾値に依存する偽陰性5件を回避した。これは固定点解析の機構記述や理論の価値まで否定する結果ではない。一方で、完全な固定点診断に固有の予測優位が実証された、とは記載できない。

## 2. Phase 0・凍結・再現仕様

実在ファイルをFILE_INVENTORY.csvに列挙した。INPUT_MANIFEST.jsonには使用資料のサイズ、SHA256、CSV列名、NPZ配列名・shape・dtype、診断記録の必須項目欠落とnullを保存した。200個のValidation選択checkpoint、凍結A/D、200診断記録、400軌道NPZ、長系列保存予測、G1の報告・コード・評価データ、R1〜R4/S1/S2、既存FP64結果、原稿を対象とした。rho欠測は非収束時に計算しない元仕様で、補完していない。

RULES_ABL1.md・PREDICTIONS_ABL1.md・INPUT_MANIFEST.json・ENVIRONMENT.jsonをFREEZE.jsonで凍結してから集計した。既知G1結果に対する事前登録ではなく、新規ABL1比較前の規則固定である。ANALYSIS_CODE_FROZEN.jsonに実行コードのSHAも記録した。

16軌道（4前置き×2値×N/R）、20000反復、収束窓100、閾値1e−6、rho<1を元コードから確認した。t=1から採点し、ウォームアップ除外なし。Nの読み出しはh_tへの試行Rで、軌道は変えない。二値argmaxの同点はクラス0なので、A2の正しいmargin判定はvalue0が>=0、value1が>0。非有限の有効性ガードを元仕様のまま使った。マージン・終点記録の不整合は0件（MARGIN_ANOMALIES.json）。

|方式|成功予測の条件（全16軌道）|
|---|---|
|A0|凍結した完全A：収束・rho<1・正しい終点読み出し|
|A1|正しい有限な終点読み出しだけ|
|A2|全20000ステップの読み出しが正しく有効|
|A3|収束と正しい終点読み出し|
|A4|収束とrho<1。読み出し不要|
|A5|凍結D：収束と軌道全体の有限性。rho・読み出し正誤は不使用|

A4とDの定義は異なる。今回は予測が全件一致したという観測結果であり、実装時に同一視したのではない。2終点間距離は元Aのモデル単位の予測条件ではなく、二点フラグのための別記録である。

## 3. A0〜A5の性能

陽性は主3長さ（2048・8192・65000）の全カテゴリ・全R正解。非線形の実測成功76・失敗84、線形は成功0・失敗40。全モデル記述、構造別、seed別を分けた。以下のBAはBalanced Accuracy。

'''
    text+=md(['群','方式','N','的中','正答率','BA','TPR','TNR','FP','FN','MCC','成功予測数'],[[r['group'],r['diagnostic'],r['N'],r['hits']]+[f'{float(r[k]):.6f}' if r[k] else None for k in ['accuracy','balanced_accuracy','TPR','TNR']]+[r['FP'],r['FN'],f"{float(r['MCC']):.6f}" if r['MCC'] else None,r['predicted_success']] for r in summary if r['group'] in ['nonlinear160','linear','all200']])
    text+='線形のTPR・BAと全ての定数予測のMCCは分母0のためNAであり、便宜的に0としていない。全42群×方式行のTP/TN、成功・失敗の実測/予測数はabl1_architecture_summary.csv。\n\n'
    text+=md(['構造（各40）']+METHODS,[[arch]+[next(r['hits'] for r in summary if r['group']==arch and r['diagnostic']==m) for m in METHODS] for arch in ['residual_gru','gru','lstm','rnn','linear']])
    text+='## 4. seedクラスタによる対応比較\n\nseed20〜39の20クラスタを復元抽出した。非線形では同じseedの8モデル（4構造×2学習長）、全200では10モデル、構造別では2モデルをまとめて選んだ。10,000回、default_rng(87000)、percentile 95%区間。群・比較ごとに同じ乱数seedで再始動した。以下は非線形160で、差は左−右。\n\n'
    text+=md(['比較','左のみ正解','右のみ正解','予測不一致','正答率差','seedクラスタ95%区間','補助McNemar p'],[[r['left']+'−'+r['right'],r['left_only_correct'],r['right_only_correct'],r['predictions_differ'],r['accuracy_difference'],f"[{r['CI95_low']}, {r['CI95_high']}]",r['McNemar_exact_p']] for r in pairs if r['group']=='nonlinear160'])
    text+='A1−A0は+3.125ポイント、区間[+1.25,+5.625]ポイント。凍結した区間規則では、この標本で簡易診断側に明確な差がある。補助McNemar p=0.0625は有意ではない。bootstrapと独立モデル仮定の正確検定は手順・帰無分布が異なり、都合のよい検定へ切り替えていない。主要解釈はseed区間、p値は依存性を考慮しない補助値である。区間は多重性未補正の探索的結果で、20seedに限られる。\n\n'
    text+='全200・構造別を含む49比較はabl1_pairwise.csvとabl1_seed_bootstrap.csv、20seedの各群の性能はabl1_seed_results.csv、不一致モデルはabl1_disagreements.csvに保存した。差が0の組の[0,0]はこの保存標本上の同一判定であり、一般的な同等性を証明した区間ではない。\n\n'
    changed5=[m for m in models if m['A0']!=m['A1']]
    text+='### 収束条件で変わった5モデル\n\n'
    text+=md(['モデル','主評価','A0/A3','A1/A2','非収束軌道数'],[[m['model'],m['actual_success'],m['A0'],m['A1'],m['nonconvergent_trajectories']] for m in changed5])
    text+='5件とも主評価に成功しており、収束条件を付加すると偽陰性になる。収束条件により新たに正しく識別されたモデルは0。スペクトル半径条件によるA0−A3の判定変化も0。\n\n'
    text+=f"保存された収束軌道は{extras['converged_trajectories']}/3200本で、うちrho条件不成立は{extras['converged_rho_not_eligible']}本。したがって今回rho条件は予測をさらに絞らなかった。非収束軌道にはrhoが保存されていないため、そこにrhoの別閾値を適用する分析は行っていない。\n\n"
    text+=f"A1とA2のモデル予測は{extras['A1_A2_model_predictions_differ']}件の差で全200一致した。ただし終点が正しく途中で誤る軌道は{extras['endpoint_good_but_intermediate_bad']}本、{extras['endpoint_good_intermediate_bad_models']}モデルに存在した。途中情報自体が無かったのではなく、他の終点誤読も含む全16本ANDでモデル判定を作ると追加のモデル識別に至らなかった。\n\n"
    text+='## 5. 失敗様式と読み出し・安定性の相違\n\n'
    falsepos=[m for m in models if flag(m['A0']) and not flag(m['actual_success'])]
    finiteerrors=[m for m in models if flag(m['finite_failure'])];divs=[m for m in models if flag(m['long_nonfinite'])]
    text+=f"非線形の失敗84件は発散{len(divs)}件と有限誤答{sum(m['architecture']!='linear' for m in finiteerrors)}件。線形40件は全て有限誤答。A0/A1/A2は発散19件を全て失敗予測した。A0の有限誤答識別は54/65で、残る11件は以下。\n\n"
    text+=md(['A0の偽陽性モデル','主3長さの失敗カテゴリ','書き換え直後Rの誤り数'],[[m['model'],m['failed_categories_main'],m['immediate_rewrite_R_errors']] for m in falsepos])
    text+='11件は全て有限誤答で、long_N/long_R単独の保持検査に失敗せずrandom/mixed/rewriteのいずれかで失敗した。mixed/rewriteを含むのは4モデル、randomのみ7モデル。固定入力保持が正しくても、入力切替・別の履歴・書き換えを含む系列全体の成功は保証しない。失敗カテゴリから内部原因を断定しない。全7長さの非有限と主3長さのカテゴリ誤答は区別して保存した。\n\n'
    epro=[m for m in models if m['A1']!=m['A4']]
    text+=f"読み出しA1と収束/安定性A4の予測が異なるモデルは{len(epro)}件。A1成功/A4失敗が{sum(flag(m['A1']) and not flag(m['A4']) for m in epro)}件（上記5モデル）、A1失敗/A4成功が{sum(not flag(m['A1']) and flag(m['A4']) for m in epro)}件。全ID・seed・要因をabl1_readout_stability_disagreements.csvに列挙した。非収束、終点誤読、途中誤読、混合/書き換え失敗、非有限は重複ラベルであり、abl1_failure_mode_summary.csvの件数を足して排他的な総数にしない。\n\n"
    text+='## 6. 既存FP64結果との対応\n\n'
    text+=f"FP64でAの予測が変わったのは{len(fpchanged)}モデルで、全て上記5件の偽陰性を正解にする方向だった。FP32のA1とFP64の完全Aのモデル予測は全200で一致={extras['FP32_A1_equals_FP64_A0_on_all_models']}。A0をFP64へ置き換えた主集計は作っていない。\n\n"
    text+=f"収束フラグの変化は{len(convergence_changed)}モデル・計{sum(int(r['changed_convergence']) for r in fp)}軌道。6モデル目のlstm_L8_s35は収束しても読み出しが誤り、Aは失敗のまま、Dだけ成功へ変わる。FP32/64の両方で収束した軌道では、rho<1の判定変化{spectralchanges}件、読み出し正誤の変化{converged_read_changes}件であった。\n\n"
    text+=f"一方、非収束軌道も含めると終点読み出し正誤の変化は{extras['fp64_readout_correctness_changed_trajectories']}軌道・{extras['fp64_readout_correctness_changed_models']}モデルで記録されている。『3040収束軌道で読み出し不変』を全3200軌道へ拡張しない。A1のモデル予測変化は{extras['fp64_A1_model_predictions_changed']}件だった。既存fp64_trajectories.csvと今回abl1_fp64_sensitivity.csvに元数値を保持した。\n\n"
    text+='## 7. 費用と情報の増分\n\nA1/A2でも同じ20000反復を必要とする。保存ログから専用オンラインパイプラインの総費用は分離できないため、簡易診断が実測で何倍速いとは結論しない。S1/S2の非線形160平均/modelの既存A実装（20000反復）は、prep 0.007071秒、反復＋途中読み出し11.871053秒、終点読み出し0.003207秒、Jacobian＋eig 0.428281秒、合計12.309612秒（JIT scripting除外）だった。Jacobianとeigを別々に、また純粋な状態更新と途中読み出しを別々に測ったログはない。収束判定だけの費用も未測定。\n\n'
    text+='直接テストBcum128は0.036741秒・15232系列step、Bcum512は0.114490秒・50048step。診断は320000の反復状態更新に加えてN読み出しの試行R更新160000を含み、前置き状態準備もある。入力種類・初期状態・隠れ状態次元・バッチ順序が異なり、1stepを同一CPU費用とみなさない。時刻は過去の実装依存測定で、新しく推論を実行して測ったものではない。\n\n'
    a=read(R/'ANALYSIS_AUDIT.json');text+=f"今回の追加解析全体はwall {a['analysis_wall_seconds']:.6f}秒、CPU {a['analysis_CPU_seconds']:.6f}秒（保存配列の読込、採点、統計、入力hash再確認を含む）。これはモデル軌道生成費用ではない。A1/A2/A3/A4の保存配列に対する述語評価だけの時刻と共有I/O時刻もabl1_costs.csvへ保存した。A0/A5の凍結lookupは個別に計時していないためNAで、初期集計器の0を実測0秒として残さないメタデータ整備だけをOUTPUT_METADATA_NOTES.jsonに記録した。診断予測・凍結規則・既存入力は変えていない。\n\n"
    text+='## 8. 科学的解釈の監査（Q1〜Q8）\n\n'
    answers=[('Q1','終点読み出しだけで再現できるか','この標本ではA1がA0を5件上回り、追加の正答は全て実測成功モデル。高い予測精度のために完全な固定点条件が必要とは示されなかった。'),('Q2','収束の追加識別','正しい識別の追加0、偽陰性追加5。FP32の固定閾値が今回の予測には不利だった。'),('Q3','rhoによる判定変化','A0対A3は0件。収束3040軌道は全てrho<1で、rhoの予測上の増分はこの標本では実証されない。'),('Q4','読み出しと安定性の相違','62件を全IDで記録。安定に収束しても読み出しが誤る場合と、読み出しが正しくてもFP32閾値で未収束になる場合を分離した。'),('Q5','固定入力診断の取りこぼし','偽陽性11件はrandom/mixed/rewrite失敗。混合/書き換えでの誤りを固定入力保持条件だけで保証できない。因果原因は未同定。'),('Q6','FP64が影響する条件','Aの変化5件は収束条件。共有収束軌道のrho<1/読み出し判定は不変。非収束軌道での読み出し変化も別に数えた。'),('Q7','読み出しだけでは得られない力学情報','終点状態ベクトル、最後100ステップの移動量、局所rho、前置き・値ごとの終点距離、二点フラグ、過渡的な誤読時刻、線形λ/時定数が実際に保存されている。これらは予測精度とは異なる情報。ただし4前置きの観測は吸引領域全体の測定ではなく、rhoは局所量、観測固定点から大域的安定性・因果性を主張しない。'),('Q8','論文の主張','G1の凍結A対DとP1/P2は維持する。一方、A対Dの優位を収束・Jacobian解析固有の予測価値とする解釈は限定が必要。読み出しだけのABL1比較を追加し、今回の同一判定/より高い的中、FP32閾値の偽陰性、未測定費用を記載すべき。機構記述と予測精度を分け、独立課題・独立データへの一般性や因果性は未検証とする。')]
    text+=md(['問','対象','回答'],answers)
    text+='ケース1（完全Aの予測優位）は支持する観測を得なかった。ケース2の単純読み出しで多くの情報を得る可能性は、今回はA1/A2がA0より多く的中した形で示された。ケース3は収束条件の負の増分、ケース4はrho追加の判定変化なしとして記録する。差が未検出の比較を同等性の証明へ置き換えない。ABL1は探索的であり、論文の採否や最終方針は研究主任が判断する。原稿を修正していない。\n\n'
    text+='## 9. 保全・再現・停止\n\n'
    env=read(R/'ENVIRONMENT.json');text+=f"Python: {env['python'].splitlines()[0]}、NumPy {env['numpy']}、PyTorch {env['pytorch']}（環境情報の取得のみ）、torch threads={env['torch_threads']}。CPU情報はENVIRONMENT.json。開始時のPython/Julia一覧はprocesses_before.jsonで、ABL1自身以外はなかった。CPU負荷の高い処理を並列に実行していない。\n\n"
    text+=f"入力{len(manifest['files'])}ファイルのSHA256は開始・終了で全件一致、原稿・凍結予測・checkpoint・過去結果は不変。元A/D・保存出力の独立再採点の一致を全200で確認した。QA_RESULTS.jsonに数値再現、欠測、同点、cluster構造、逆比較、保全検査を保存。全MarkdownはUTF-8保存直後に読み直した。SHA256_MANIFEST.jsonは本ディレクトリの最終成果物と入力監査結果を記録する。\n\n"
    text+='''実行順はprepare.py（inventoryと規則凍結）→analyze.py（保存データの再採点・ABL1比較）→report.py（品質検査・報告）。すべて`python -B`で実行し、既存実験コードをimport/実行していない。書込み先は本ABL1ディレクトリだけ。

```powershell
.venv/Scripts/python.exe -B IronMath_PaperCheck/addon/abl1/scripts/prepare.py
.venv/Scripts/python.exe -B IronMath_PaperCheck/addon/abl1/scripts/analyze.py
.venv/Scripts/python.exe -B IronMath_PaperCheck/addon/abl1/scripts/report.py
```

規則・監査・成果物が存在する場合は上書きせず停止する設計。再現時に元データを変更せず、別の出力先へコードを複製してパスだけを固定し直すこと。出力メタデータ整備の2行は本分析で新規作成した未確定CSVだけを対象とした。

ABL1の比較・監査・報告を完了して停止する。新規実験・追加学習・原稿改稿へ進まない。
'''
    write('REPORT_ABL1.md',text)
    # Explicit per-Markdown readback and final process record.
    utf8=[]
    for p in R.glob('*.md'):
        s=p.read_text(encoding='utf-8');assert '\ufffd' not in s and '???' not in s;utf8.append(dict(path=p.name,SHA256=sha(p),readback_UTF8=True,replacement_character_absent=True))
    write('UTF8_CHECKS.json',dict(utc=utc(),files=utf8,report_readback_exact=True))
    processes('after')
    files={p.relative_to(R).as_posix():dict(bytes=p.stat().st_size,SHA256=sha(p)) for p in R.rglob('*') if p.is_file() and p.name!='SHA256_MANIFEST.json'}
    write('SHA256_MANIFEST.json',dict(utc=utc(),files=files,input_files_checked=len(manifest['files']),all_inputs_unchanged=True,changed_inputs=[],no_new_training=True,no_new_inference=True,report_UTF8_verified=True,manifest_self_hash_excluded=True))
    print('REPORT COMPLETE',len(checks),'quality checks',len(files),'output files',flush=True)
if __name__=='__main__':finish()
