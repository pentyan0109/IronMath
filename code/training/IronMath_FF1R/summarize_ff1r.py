"""FF1-R hypothesis analysis of independently rescored saved predictions."""
import math,statistics,collections
from common import *
def summarize():
    import numpy as np
    registrations_ok();assert read('OUTPUT_AUDIT.json')['passed']
    assert read('WORKSPACE_COMPARISON.json')['original_files_unchanged']
    def records(n):
        with (R/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    def b(x):return x is True or x=='True'
    predictions=records('ff1r_predictions.csv');items=records('long_item_results.csv');details=read('diagnostic_details.json')
    A={a['model']:a for a in read('PREDICTIONS_A_FROZEN.json')['models']};D={a['model']:a for a in read('PREDICTIONS_D_FROZEN.json')['models']}
    primary_lengths=read('PREREGISTRATION.json')['primary_lengths'];models=[];diagnostic=[]
    for p in predictions:
        mid=p['model'];ii=[i for i in items if i['model']==mid];ss=[i for i in ii if int(i['T']) in primary_lengths]
        actual=all(int(i['R_correct'])==int(i['R_count']) for i in ss);nf=sum(int(i['nonfinite_positions']) for i in ii)
        kind='発散型' if nf else ('成功型' if actual else '有限誤答型')
        models.append(dict(model=mid,H=int(p['H']),L=int(p['L']),seed=int(p['seed']),actual_success=actual,classification=kind,nonfinite_logit_positions_all_lengths=nf,A=b(p['A']),B=b(p['B']),C_all_success=True,C_all_failure=False,D=b(p['D']),all16_converged=D[mid]['all16_converged'],diagnostic_nonfinite=D[mid]['diagnostic_nonfinite'],two_attractors_N=A[mid]['two_attractors']['N'],two_attractors_R=A[mid]['two_attractors']['R'],main_R_correct=sum(int(i['R_correct']) for i in ss),main_R_count=sum(int(i['R_count']) for i in ss)))
    comparisons=[]
    for subset,mm in [('all40',models),('finite_failures',[m for m in models if m['classification']=='有限誤答型']),('nondivergent',[m for m in models if not m['nonfinite_logit_positions_all_lengths']])]:
        for method in ['A','B','C_all_success','C_all_failure','D']:
            tp=sum(m[method] and m['actual_success'] for m in mm);fp=sum(m[method] and not m['actual_success'] for m in mm);tn=sum(not m[method] and not m['actual_success'] for m in mm);fn=sum(not m[method] and m['actual_success'] for m in mm)
            comparisons.append(dict(subset=subset,method=method,N=len(mm),hits=tp+tn,TP=tp,FP=fp,TN=tn,FN=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None))
    main={r['method']:r for r in comparisons if r['subset']=='all40'}
    reasons=[];actual_success=sum(m['actual_success'] for m in models);positive=sum(m['A'] for m in models)
    if actual_success in [0,40]:reasons.append('識別不能：全モデルの実際の結果が同じ')
    if min(positive,40-positive)<5:reasons.append('件数不足：A予測の片方が5件未満')
    H1='判定不能' if reasons else ('支持' if all(main['A']['hits']>main[n]['hits'] for n in ['B','C_all_success','C_all_failure','D']) else '不支持')
    finite=[m for m in models if m['classification']=='有限誤答型'];fa=sum(not m['A'] for m in finite);fd=sum(not m['D'] for m in finite)
    H2='判定不能' if len(finite)<5 else ('支持' if fa>fd else '不支持')
    aonly=sum((m['A']==m['actual_success']) and (m['D']!=m['actual_success']) for m in models);donly=sum((m['D']==m['actual_success']) and (m['A']!=m['actual_success']) for m in models)
    n=aonly+donly;pvalue=min(1.,2*sum(math.comb(n,k) for k in range(min(aonly,donly)+1))/2**n) if n else 1.
    differences=np.array([int(m['A']==m['actual_success'])-int(m['D']==m['actual_success']) for m in models])
    bs=np.random.default_rng(80000).choice(differences,size=(10000,40),replace=True).mean(1);CI=np.percentile(bs,[2.5,97.5]).tolist()
    disagree=[]
    for m in models:
        mid=m['model'];dd=[d for d in details if d['model']==mid]
        diagnostic.extend({**d,'eigenvalues':json.dumps(d['eigenvalues'])} for d in dd)
        if m['A']!=m['D']:
            pp=[x for x in records('attractor_pairs.csv') if x['model']==mid]
            disagree.append(dict(model=mid,A=m['A'],D=m['D'],actual_success=m['actual_success'],classification=m['classification'],all16_converged=all(d['converged'] for d in dd),wrong_finite_readouts=sum(d['readout_finite'] and d['readout']!=d['value'] for d in dd),nonfinite_readouts=sum(not d['readout_finite'] for d in dd),unstable_converged_endpoints=sum(d['converged'] and d['spectral_radius'] is not None and d['spectral_radius']>=1 for d in dd),prefix_pairs_distance_ge_1e_minus3=sum(float(x['distance'])>=1e-3 for x in pp if x['distance']),prefix_pairs_total=len(pp)))
    csvout('ff1r_model_results.csv',models);csvout('ff1r_predictor_comparison.csv',comparisons);csvout('ff1r_finite_failures.csv',finite);csvout('ff1r_diagnostic_details.csv',diagnostic);csvout('ff1r_disagreements.csv',disagree)
    metrics=dict(H1=H1,H1_indeterminate_reasons=reasons,H2=H2,actual_success=actual_success,A_predict_success=positive,finite_failures=len(finite),finite_failure_correct_A=fa,finite_failure_correct_D=fd,A_only_correct=aonly,D_only_correct=donly,McNemar_two_sided_exact_p=pvalue,paired_accuracy_difference=float(differences.mean()),paired_bootstrap95_CI=CI,statistically_significant_at_005=pvalue<.05,bootstrap_seed=80000,bootstrap_resamples=10000)
    dump('ff1r_metrics.json',metrics)
    # Freeze/order audit and original FF1 data comparison; no model execution.
    allfreeze=read('ALL_PREDICTIONS_FROZEN.json');freeze_time=(R/'ALL_PREDICTIONS_FROZEN.json').stat().st_mtime_ns
    for name,h in allfreeze['SHA256'].items():assert sha(R/f'PREDICTIONS_{name}_FROZEN.json')==h
    assert sha(R/'ff1r_predictions.csv')==allfreeze['prediction_csv_SHA256']
    ds=read('LONG_DATA_FROZEN.json');data_comparison=[]
    for f in ds['files']:
        p=R/f['path'];old=W/'IronMath_FF1'/f['path'];assert p.stat().st_mtime_ns>=freeze_time
        with np.load(p) as z: x=z['symbols']
        with np.load(old) as z:y=z['symbols']
        data_comparison.append(dict(T=f['T'],new_SHA256=sha(p),old_SHA256=sha(old),file_SHA_equal=sha(p)==sha(old),symbols_equal=bool(np.array_equal(x,y)),generated_file_mtime_ns=p.stat().st_mtime_ns))
        assert data_comparison[-1]['symbols_equal']
    assert (R/'TRAINING_STARTED.json').stat().st_mtime_ns>(R/'PREREG_FF1R.md').stat().st_mtime_ns
    assert (R/'PREDICTIONS_D_FROZEN.json').stat().st_mtime_ns<=(R/'PREDICTIONS_B_FROZEN.json').stat().st_mtime_ns<=freeze_time
    assert (R/'LONG_EVALUATION_STARTED.json').stat().st_mtime_ns>=(R/'LONG_DATA_FROZEN.json').stat().st_mtime_ns
    for asset in read('CHECKPOINTS_FROZEN.json')['models'].values():assert sha(R/asset['path'])==asset['SHA256']
    source=read('SOURCE_COPY_AUDIT.json');source_checks=[]
    for f in source['files']:
        assert sha(W/'IronMath_FF1'/f['file'])==f['source_SHA256'];assert sha(R/f['file'])==f['copy_SHA256'];source_checks.append(f)
    audit=dict(utc=now(),passed=True,registration_SHA256=sha(R/'PREREGISTRATION.json'),PREREG_FF1R_SHA256=sha(R/'PREREG_FF1R.md'),copied_source=source_checks,selected_checkpoint_count=40,independent_regrade=read('OUTPUT_AUDIT.json'),preservation=read('WORKSPACE_COMPARISON.json'),data_comparison=data_comparison,prediction_freeze=allfreeze,prediction_freeze_mtime_ns=freeze_time,stage_order_verified=True,A_success_subset_D_success=all(not m['A'] or m['D'] for m in models),new_training_after_freeze=False)
    dump('ff1r_audit.json',audit)
    report='# IronMath FF1-R\n\n新規seed10〜19での独立再現。フリップフロップ課題に限定し、因果・他課題への一般性は主張しない。FF1・FF1-A・D11の結果を条件へ反映していない。\n\n'
    report+=f"## 主結果\n\nH1：**{H1}**。H2：**{H2}**。判定不能理由：{reasons}。主判定長さ：{primary_lengths}。\n\n|方式|的中/40|TP|FP|TN|FN|適合率|再現率|\n|---|---|---|---|---|---|---|---|\n"
    for method,r in main.items():report+='|'+ '|'.join(str(x) for x in [method,r['hits'],r['TP'],r['FP'],r['TN'],r['FN'],r['precision'],r['recall']])+'|\n'
    report+=f"\n実際の成功{actual_success}/40、有限誤答型{len(finite)}。有限誤答型での正しい失敗予測A={fa}、D={fd}。発散型={sum(m['classification']=='発散型' for m in models)}。分類は全7長さの非R位置も含むlogit非有限マスクを使用。H2は事前登録された結果依存の部分集合分析。\n\n"
    report+=f"Aのみ正解{aonly}、Dのみ正解{donly}。McNemar正確両側p={pvalue:.8g}。的中率差A−D={differences.mean():.3f}、対応ブートストラップ95% CI={CI}。有意水準.05で{'有意' if pvalue<.05 else '有意ではない'}。H1は検定ではなく登録した的中数規則で判定する。モデル40個を反復単位とし、評価項目やR位置を独立標本として水増ししない。同じseedを条件間で使用する設計上の依存の可能性と40モデルという範囲を留保する。\n\n"
    report+='## H3：AとDの差\n\n|モデル|A|D|真値|全収束|有限誤読|非有限読出|不安定終点|距離条件成立ペア|\n|---|---|---|---|---|---|---|---|---|\n'
    for d in disagree:report+='|'+ '|'.join(str(d[k]) for k in ['model','A','D','actual_success','all16_converged','wrong_finite_readouts','nonfinite_readouts','unstable_converged_endpoints','prefix_pairs_distance_ge_1e_minus3'])+'|\n'
    report+='\n距離は2点フラグのみに使用し、Aモデル予測の条件ではない。上表は判定要素の分解であり、読み出し誤りや引き込み点が長系列失敗の原因であると示す介入ではない。Dは非有限マージンの有無だけを確認し、符号・正誤・距離は予測入力にしない。\n\n'
    report+='## 全モデル\n\n|モデル|A|B|D|実際|分類|\n|---|---|---|---|---|---|\n'
    for m in models:report+='|'+ '|'.join(str(m[k]) for k in ['model','A','B','D','actual_success','classification'])+'|\n'
    report+='\n## FF1との比較と範囲\n\nFF1の登録結果A38/B7/C失敗33、事後D29は変更しない。今回のseedは10〜19、旧seed0〜9のcheckpointを再利用しない。長系列の生成規則・seedを維持し、同一問題で新モデルを評価した。集合のファイルSHA一致と内容一致はff1r_audit.jsonに記録。新seedの再現であり、新しい課題・未知アーキテクチャでの再現ではない。\n\n'
    report+='FF1のコピーanalyze.pyが出力するmetrics.jsonは旧FF1規則の補助集計に限る。FF1-Rの正式なH1/H2判定はff1r_metrics.jsonおよび本報告書を用いる。全7長さのカテゴリ別正答、書き換え後回復、失敗例、保持予測と破綻時点はcategory_results.csv、failure_cases.csv、retention_units.csv、break_step_analysis.csvに保存。\n\n'
    report+='## 学習・資源・監査\n\nLinear4→16、GRUCell H32/64、スカラー残差係数g初期0、Linear H→2。パラメータ4947/15955。AdamW lr.002、weight decay.01、batch32、clip1、1500更新、R位置CE、CPU2threads、FP32。Validation512系列を100更新ごとに測り厳密最小、同値は早期。seed別履歴history/、選択checkpointの更新数とSHAはCHECKPOINTS_FROZEN.json。事前登録JSON/MarkdownのSHA、コード同一性、予測凍結、データ生成、独立再採点、実験前後の全workspace監査が合格。新規最上位ディレクトリは名前・作成時刻だけを記録し作成元を断定しない。\n\n'
    for name in ['training','diagnostics_A','diagnostics_D','short_B','long_evaluation','independent_output_audit','workspace_after']:
        rr=read(f'stage_resources/{name}.json');report+=f"- {name}: wall {rr['wall_seconds']:.3f}s、CPU {rr['CPU_seconds']:.3f}s、process peak RAM {rr['peak_RAM_bytes']/1048576:.2f} MiB。\n"
    report+='\n順序のUTC時刻と更新時刻：\n\n'
    for n in ['PREREGISTRATION.json','TRAINING_STARTED.json','TRAINING_COMPLETE.json','PREDICTIONS_A_FROZEN.json','PREDICTIONS_D_FROZEN.json','PREDICTIONS_B_FROZEN.json','ALL_PREDICTIONS_FROZEN.json','LONG_DATA_FROZEN.json','LONG_EVALUATION_STARTED.json','LONG_EVALUATION_COMPLETE.json']:
        a=read(n);report+=f"- {n}: {a.get('utc',a.get('registered_utc',a.get('frozen_utc',a.get('generated_utc'))))}; mtime_ns={(R/n).stat().st_mtime_ns}\n"
    report+='\n再現コマンド：`.venv/Scripts/python.exe -B IronMath_FF1R/run_ff1r.py`。完了後の再学習は禁止するガードを設置済み。同一出力先で再実行しない。全成果物を保存し停止。追加条件、追加seed、D11への反映はしない。\n'
    (R/'REPORT_FF1R.md').write_text(report,encoding='utf-8')
    dump('FINAL_STATUS.json',dict(utc=now(),status='completed_audited_stopped',H1=H1,H2=H2,models=40,no_additional_experiments=True))
    paths=sorted(p for p in R.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
    (R/'SHA256SUMS.txt').write_text(''.join(sha(p)+'  '+p.relative_to(R).as_posix()+'\n' for p in paths),encoding='utf-8')
    print('FF1R COMPLETE',metrics,flush=True)
