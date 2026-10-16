from g1_utils import *
import csv,platform,importlib.metadata

def summarize(assets,all_assets):
    registrations_ok();assert read('SAVED_OUTPUT_AUDIT.json')['passed'];assert read('WORKSPACE_COMPARISON.json')['original_files_unchanged']
    with (R/'g1_results.csv').open(encoding='utf-8-sig',newline='') as f:models=list(csv.DictReader(f))
    for m in models:
        for k in ['actual_success','A','B','C_all_success','C_all_failure','D','all16_converged','diagnostic_nonfinite']:m[k]=m[k]=='True'
    nonlinear=[m for m in models if m['architecture']!='linear'];linear=[m for m in models if m['architecture']=='linear']
    pooled=scientific_metrics(nonlinear);by_arch={a:scientific_metrics([m for m in models if m['architecture']==a]) for a in ARCHS};hits={r['method']:r['hits'] for r in pooled['methods']}
    reasons=[];success=sum(m['actual_success'] for m in nonlinear);positive=sum(m['A'] for m in nonlinear)
    if success in [0,len(nonlinear)]:reasons.append('識別不能：非線形モデルの実際の成否が全て同じ')
    if min(positive,len(nonlinear)-positive)<10:reasons.append('件数不足：Aの成功または失敗予測が10件未満')
    all_better=all(hits['A']>hits[n] for n in ['B','C_all_success','C_all_failure','D'])
    no_arch_loss=all(next(r['hits'] for r in by_arch[a]['methods'] if r['method']=='A')>=next(r['hits'] for r in by_arch[a]['methods'] if r['method']=='D') for a in ARCHS if a!='linear')
    P1='判定不能' if reasons else ('支持' if all_better and pooled['McNemar_exact_two_sided_p']<.05 and no_arch_loss else '不支持')
    P2='支持' if all(not m['actual_success'] and not m['A'] for m in linear) else '不支持'
    assert pooled['McNemar_exact_two_sided_p']==read('SAVED_OUTPUT_AUDIT.json')['independent_P1_McNemar']['p']
    comparisons=[]
    for group,stats in [('nonlinear_pooled',pooled)]+list(by_arch.items()):
        for r in stats['methods']:comparisons.append(dict(group=group,**r,A_only_correct=stats['A_only_correct'],D_only_correct=stats['D_only_correct'],McNemar_exact_two_sided_p=stats['McNemar_exact_two_sided_p'],A_minus_D=stats['accuracy_difference'],CI_low=stats['bootstrap95_CI'][0],CI_high=stats['bootstrap95_CI'][1]))
    csvout('g1_architecture_comparison.csv',comparisons)
    manifest=[]
    for a in all_assets:manifest.append({**a,'included_in_final_evaluation':a['seed'] in read('BUDGET_DECISION.json')['seeds']})
    csvout('g1_model_manifest.csv',manifest)
    predictions=read('main_PREDICTIONS_FROZEN.json')['models'];csvout('g1_predictions.csv',predictions)
    details=[];pairs=[];linear_rows=[];components=[];disagreements=[];resources=[]
    for a in all_assets:
        mid=a['model'];d=read(f'diagnostic_records/{mid}.json');details.extend(d['details']);pairs.extend(d['pairs']);linear_rows.extend(d['linear_analytic'])
        if a['architecture']=='linear':
            with np.load(R/'linear_analytic'/f'{mid}_N.npz') as z:
                for i,(lam,tau) in enumerate(zip(z['lambdas'],z['taus'])):components.append(dict(model=mid,component=i,lambda_i=float(lam),tau_i=float(tau)))
        resources.append(dict(model=mid,architecture=a['architecture'],L=a['L'],seed=a['seed'],stage='training',wall_seconds=a['train_wall_seconds'],CPU_seconds=a['train_CPU_seconds'],peak_RAM_bytes=a['peak_RAM_bytes'],parameters=a['parameters']))
        resources.append(dict(model=mid,architecture=a['architecture'],L=a['L'],seed=a['seed'],stage='diagnostics_and_B',wall_seconds=d['diagnostic_wall_seconds']+d['short_B_wall_seconds'],CPU_seconds=d['diagnostic_CPU_seconds']+d['short_B_CPU_seconds'],peak_RAM_bytes=d['peak_RAM_bytes'],parameters=a['parameters']))
        if (R/f'evaluation_records/{mid}.json').exists():
            e=read(f'evaluation_records/{mid}.json');resources.append(dict(model=mid,architecture=a['architecture'],L=a['L'],seed=a['seed'],stage='long_evaluation',wall_seconds=e['wall_seconds'],CPU_seconds=e['CPU_seconds'],peak_RAM_bytes=e['peak_RAM_bytes'],parameters=a['parameters']))
        if d['A']!=d['D']:
            disagreements.append(dict(model=mid,architecture=a['architecture'],L=a['L'],seed=a['seed'],A=d['A'],D=d['D'],all16_converged=d['all16_converged'],wrong_finite_readouts=sum(x['readout_finite'] and x['readout']!=x['value'] for x in d['details']),nonfinite_readouts=sum(not x['readout_finite'] for x in d['details']),unstable_converged_endpoints=sum(x['converged'] and x['spectral_radius'] is not None and x['spectral_radius']>=1 for x in d['details']),two_attractors_N=d['two_attractors']['N'],two_attractors_R=d['two_attractors']['R']))
    csvout('g1_diagnostic_details.csv',details);csvout('g1_attractor_pairs.csv',pairs);csvout('g1_linear_dynamics.csv',linear_rows);csvout('g1_linear_components.csv',components);csvout('g1_disagreements.csv',disagreements);csvout('g1_resource_usage.csv',resources)
    checkpoint_rows=[]
    for path in sorted((R/'checkpoints').rglob('*.pt')):checkpoint_rows.append(dict(path=path.relative_to(R).as_posix(),bytes=path.stat().st_size,SHA256=sha(path)))
    csvout('g1_all_checkpoint_SHA256.csv',checkpoint_rows)
    old_data=[]
    for f in read('main_LONG_DATA_FROZEN.json')['files']:
        p=W/'IronMath_FF1/long_data'/f"T{f['T']}.npz";old_data.append(dict(T=f['T'],G1_SHA256=f['SHA256'],FF1_SHA256=sha(p),file_SHA_equal=f['SHA256']==sha(p)))
    prereg=read('PREREGISTRATION.json')
    assert sha(R/'PREREG_G1.md')==read('PREREG_G1_SHA256.json')['SHA256'];assert sha(R/'THEORY_G1.md')==read('THEORY_G1_SHA256.json')['SHA256']
    for p in ['PREREG_G1.md','THEORY_G1.md']:text=(R/p).read_text(encoding='utf-8',errors='strict');assert '\ufffd' not in text and '???' not in text
    freeze_ns=(R/'main_PREDICTIONS_FROZEN.json').stat().st_mtime_ns
    assert (R/'TRAINING_COMPLETE.json').stat().st_mtime_ns<=(R/'g1_diagnostic_audit.json').stat().st_mtime_ns<=freeze_ns<=(R/'main_LONG_DATA_FROZEN.json').stat().st_mtime_ns
    assert (R/'THEORY_G1.md').stat().st_mtime_ns<(R/'TRAINING_STARTED.json').stat().st_mtime_ns and (R/'PREREG_G1.md').stat().st_mtime_ns<(R/'TRAINING_STARTED.json').stat().st_mtime_ns
    result=dict(P1=P1,P1_indeterminate_reasons=reasons,P1_all_hit_superior=all_better,P1_no_architecture_D_superior=no_arch_loss,P2=P2,nonlinear_N=len(nonlinear),linear_N=len(linear),nonlinear_actual_success=success,A_predict_success=positive,pooled_nonlinear=pooled,by_architecture=by_arch,budget=read('BUDGET_DECISION.json'))
    dump('g1_metrics.json',result)
    audit=dict(utc=now(),passed=True,prereg_SHA256=sha(R/'PREREG_G1.md'),theory_SHA256=sha(R/'THEORY_G1.md'),all_checkpoint_SHA_manifest_SHA256=sha(R/'g1_all_checkpoint_SHA256.csv'),checkpoint_count=len(checkpoint_rows),prediction_freeze_SHA256=sha(R/'main_PREDICTIONS_FROZEN.json'),sequence_order_verified=True,no_early_pilot_evaluation=True,workspace=read('WORKSPACE_COMPARISON.json'),independent_saved_output_audit=read('SAVED_OUTPUT_AUDIT.json'),diagnostic_audit=read('g1_diagnostic_audit.json')['passed'],FF1_dataset_SHA_comparison=old_data,UTF8_readback_verified=True,no_extra_seeds_or_architectures=True)
    dump('g1_integrity_audit.json',audit)
    env=dict(utc=now(),python=sys.version,platform=platform.platform(),torch=importlib.metadata.version('torch'),numpy=importlib.metadata.version('numpy'),CPU_threads=2,precision='FP32',device='CPU');dump('ENVIRONMENT.json',env)
    report='# G1 最終報告\n\nフリップフロップ記憶課題で再帰構造を比較した。因果介入、算術以外一般、Mamba/RWKVへの適用は検証していない。\n\n'
    report+=f"## 判定\n\nP1：**{P1}**。P2：**{P2}**。判定不能理由：{reasons}。非線形の主判定対象{len(nonlinear)}モデル、線形{len(linear)}モデル。学習した200モデルは全て保存。主判定は2048・8192・65000の全カテゴリ・全R位置、残り4長さは副次。\n\n"
    report+='|構造|N|成功|発散型|有限誤答型|A|B|C全成功|C全失敗|D|Aのみ正解|Dのみ正解|McNemar p|A−D 95% CI|\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n'
    for arch in ARCHS:
        mm=[m for m in models if m['architecture']==arch];st=by_arch[arch];hh={r['method']:r['hits'] for r in st['methods']};report+='|'+ '|'.join(str(x) for x in [arch,len(mm),sum(m['actual_success'] for m in mm),sum(m['classification']=='発散型' for m in mm),sum(m['classification']=='有限誤答型' for m in mm),hh['A'],hh['B'],hh['C_all_success'],hh['C_all_failure'],hh['D'],st['A_only_correct'],st['D_only_correct'],st['McNemar_exact_two_sided_p'],st['bootstrap95_CI']])+'|\n'
    report+=f"\n非線形合算：A={hits['A']}、B={hits['B']}、C全成功={hits['C_all_success']}、C全失敗={hits['C_all_failure']}、D={hits['D']}。A−D差={pooled['accuracy_difference']}、95% CI={pooled['bootstrap95_CI']}、McNemar正確両側p={pooled['McNemar_exact_two_sided_p']}。有意水準.05で{'有意' if pooled['McNemar_exact_two_sided_p']<.05 else '有意ではない'}。構造ごとの効果が偏る場合、全構造への一般性を主張しない。混同行列・適合率・再現率はg1_architecture_comparison.csv。ブートストラップはモデル単位10,000回、seed81000。\n\n"
    report+='## 理論と対角線形再帰\n\nTHEORY_G1.mdのG1-Tは実数上のアフィン写像についての証明。数値実験は証明の代用ではない。P2は有限長の事前予測であり、定理から必然ではない。P2不支持でも定理の反証ではない。λ・τはg1_linear_components.csv、各中立入力・前置き・値の解析的固定点と収束時の数値終点との差はg1_linear_dynamics.csv、全ベクトルはlinear_analytic/に保存。非収束終点の差は欠測。λが1に近いと有限反復非収束や長い有限記憶があり得る。\n\n'
    report+='## 実装・学習・失敗分析\n\n入力one-hot4→Linear4→16、hidden64。LSTMはh,c連結128次元を収束・距離・ヤコビアン全てに使用。読み出しはhから行う。Aは4保持単位を全て満たす場合のみ成功予測し、距離は2点フラグのみ。Dは非収束・非有限だけを使う。NのR読み出し試行は反復軌道を変えない。構造間でhidden次元は同じだがパラメータ数・計算量は揃っていない。\n\n'
    report+='AdamW lr.002、weight decay.01、batch32、1500更新、clip1、R位置CE。Validation512系列を100更新ごとに測り厳密最小を選択、同値は早期。モデルseed20〜39、Train10000+seed、Validation20000+seed。選択更新数・パラメータ数・CPU/wall時間はg1_model_manifest.csv。seed別履歴と全checkpointを保存。\n\n'
    report+='必須診断監査は5構造×2学習済みモデルで実施。FP32出力の完全一致、N/R・8出発点ごとのε=1e−4中央差分と自動微分Jacobianの相対Frobenius誤差≤1e−2を検証。詳細はg1_diagnostic_audit.json。発散型は7評価長の全位置でlogit非有限が1件でもあるモデル。有限誤答型と診断非収束は別項目。失敗例・最初の位置・周辺記号はg1_failure_cases.csv、予測bit列と非有限マスクを保存。A/D不一致の読み出し・局所安定・2点判定をg1_disagreements.csvに保存し、機構の因果証明としない。\n\n'
    report+='## 順序・予算・資源\n\nユーザー指定により先行10モデルだけの評価例外を行わず、全200学習→10モデルで構造監査→全200診断と短系列B→全予測凍結→長系列集合生成→最初10モデル評価と時間だけによる見積もり→残り評価、の順序を遵守した。縮小しても既存200モデルの学習費用を計上し、除外モデルを削除しない。これは事前登録済みの順序上の取扱いである。\n\n'
    bd=read('BUDGET_DECISION.json');report+=f"全規模見積もり{bd['full_estimated_seconds']/3600:.3f}時間、採用見積もり{bd['chosen_estimated_seconds']/3600:.3f}時間。縮小適用={bd['reduction_applied_once']}。\n\n"
    for stage in ['training','diagnostics_and_B','long_evaluation']:
        rr=[r for r in resources if r['stage']==stage];report+=f"- {stage}: wall合計{sum(r['wall_seconds'] for r in rr):.3f}s、CPU合計{sum(r['CPU_seconds'] for r in rr):.3f}s、process peak最大{max(r['peak_RAM_bytes'] for r in rr)/1048576:.2f}MiB。\n"
    report+='\nSHA監査・実装監査・集計時間はstage_resources/およびWORKSPACE_BEFORE.jsonに別計測。時刻はUTC、クライアント日付は2026年10月16日（日本時間）。\n\n'
    for name in ['PREREG_G1.md','THEORY_G1.md','TRAINING_STARTED.json','TRAINING_COMPLETE.json','g1_diagnostic_audit.json','main_PREDICTIONS_FROZEN.json','main_LONG_DATA_FROZEN.json','LONG_EVALUATION_COMPLETE.json']:
        report+=f"- {name}: mtime_ns={(R/name).stat().st_mtime_ns}\n"
    report+='\n## 監査・FF1-Rの別集計・終了\n\n既存workspaceの実験前後SHA一致を確認。新規最上位ディレクトリは名前と作成時刻のみ記録し作成元を推測しない。全checkpointと凍結予測のSHA、別スクリプトによる長系列再採点、McNemar再計算を確認。FF1と評価集合がファイルSHAまで一致したかはg1_integrity_audit.jsonに保存。\n\nFF1-Rの3長さと全7長さによる別集計はff1r_reaggregation.jsonに保存し、元のFF1-Rは変更しない。的中数・H1・H2は両定義で変化しなかった。G1の判定とは混ぜない。\n\n報告書・事前登録・理論文書はUTF-8で保存し、書き出し後に日本語を読み直して一致・置換文字なしを確認。再現入口はrun_g1.py、登録済み完了先での再学習はガードで禁止。G1をもって無印IronMathの新規学習を終了する。追加seed・追加構造・条件変更・再学習・次実験への自動移行を行わず停止。\n'
    report_sha=write_utf8('REPORT_G1.md',report);dump('REPORT_UTF8_CHECK.json',dict(utc=now(),SHA256=report_sha,UTF8_roundtrip_exact=True))
    dump('FINAL_STATUS.json',dict(utc=now(),status='completed_audited_stopped',P1=P1,P2=P2,trained_models=200,evaluated_models=len(assets),no_further_training=True))
    paths=sorted(p for p in R.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
    (R/'SHA256SUMS.txt').write_text(''.join(sha(p)+'  '+p.relative_to(R).as_posix()+'\n' for p in paths),encoding='utf-8')
    print('G1 COMPLETE',dict(P1=P1,P2=P2,N=len(assets),pooled=pooled),flush=True)
