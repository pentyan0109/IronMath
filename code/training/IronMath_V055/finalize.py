"""Artifact closure and transparent startup-failure record. No model execution."""
import ast,json
from utils import *
def main():
    audit=json.loads((R/'audit_results.json').read_text());assert audit['passed']
    backup=R/'preflight_attempt_1';restart=json.loads((R/'preflight_restart_record.json').read_text());assert restart['new_training_updates_before_fix']==restart['Z1_inference_before_fix']==0 and not restart['conditions_changed']
    unchanged=[]
    for name in ('config.json','SPEC_COMPLEMENT.md','PREREGISTRATION.md','core.py','data.py','utils.py','analyze.py','probe.py'):
        assert sha(R/name)==sha(backup/name);unchanged.append(name)
    def functions(path):return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(node,ast.FunctionDef)}
    old,new=functions(backup/'run.py'),functions(R/'run.py')
    for name in ('fit','record','evaluate','score','train','references','modelhash_state'):assert old[name]==new[name]
    registry=json.loads((R/'reference_manifest.json').read_text())['checkpoints']
    csvout('Z2_availability.csv',[dict(budget=b,seed=s,condition='Z2',status='not applicable',reason='B0 third feature is operation flag; no Carry-in path',checkpoint_SHA256=registry[f'{b}_B0_s{s}']['sha256'],new_inference=0,optimizer_updates=0) for b in C['budgets'] for s in C['seeds']])
    manifest=json.loads((R/'dataset_manifest.json').read_text());data={name:json.loads((R/f'data/{name}.json').read_text()) for name in manifest}
    def key(r):return r['width']+1,*sorted((r['a'],(-r['b'])%(1<<r['width']) if r['op']=='SUB' else r['b']))
    add={key(r) for r in data['ADD_Train']};sub={key(r) for r in data['SUB_Train']}
    inputcounts=dict(ADD_Train_rows=len(data['ADD_Train']),P_SUB_Train_rows=len(data['SUB_Train']),unique_ADD_Train_inputs=len(add),unique_P_SUB_Train_inputs=len(sub),common_ADD_and_P_SUB_Train_inputs=len(add&sub),unique_union_inputs=len(add|sub),definition='unordered converted addition input; legal within-Train aliases, no cross-split aliases')
    dump('effective_input_counts.json',inputcounts)
    report=(R/'REPORT_V055.md').read_text(encoding='utf-8');section='''## 開始前不具合の保全・追加監査

初回の開始前処理は、入口がscore/evaluate/train定義より前に置かれていたNameErrorで停止した。新規学習更新0、Z1推論0、予測gzipは空、checkpoint未作成を確認した。初回のコード・設定・データ・STOPPED・SHAをpreflight_attempt_1/へ保全。呼び出し順を修正し、再発検出テストを加えた7品質テストと12旧checkpoint互換検査に再合格後、同じ条件で実験を開始した。

初回と最終のconfig、spec、preregistration、モデル、データ、監査コードなどのSHAは一致。fit/record/evaluate/score/train等の学習・予測関数ASTも一致し、変更は入口と学習前再開検査、品質テストのみ。学習開始後の修正・再学習はない。preflight_restart_record.jsonとfinal_artifact_audit.jsonに記録。

Z2_availability.csvに全budget/seedの適用不可を保存。Z3の旧guard入力と文字通りのCのguardは異なるが、下位W桁の入力prefixは同じである。LSB-firstの因果的逐次出力では未来のguard入力は下位W出力へ影響しない。それでもZ1対Z3は重み・学習履歴が異なるため、P/C処理方式のみの効果と断定しない。

effective_input_counts.jsonで、Train内の通常ADDとP-SUBの同値入力、固有入力数を開示した。同値入力を独立した新問題数とは数えない。F0/F1/F2は同じ候補poolを使うが、提示比率や固有入力の使用頻度が異なる。分割間の同値入力は除外済み。

今回のF条件はユーザー確認の新規初期化である。retention_analysis.csvのbeforeは既存の学習済みB0、afterは新規Fモデルの比較なので、既存性能水準への一致・不一致を測る指標であり、同じモデルを継続学習したときの破滅的忘却を測ったものではない。旧成功問題の新規モデルでの失敗化／成功化は問題単位比較として保存する。Z1は重み不変の転用なので、元の通常ADD経路自体は変更されていない。

'''
    if '## 開始前不具合の保全・追加監査' not in report:report=report.replace('## 停止',section+'## 停止');(R/'REPORT_V055.md').write_text(report,encoding='utf-8')
    before=json.loads((R/'protected_sha_before.json').read_text());assert protected()==before
    required=['REPORT_V055.md','SPEC_COMPLEMENT.md','PREREGISTRATION.md','zero_shot_results.csv','finetuning_results.csv','length_generalization.csv','carry_distribution.csv','error_analysis.csv','seed_comparison.csv','cpu_memory.csv','predictions.jsonl.gz','failure_cases.jsonl.gz','audit_report.md','figures/length_generalization.png']
    assert all((R/p).is_file() and (R/p).stat().st_size>0 for p in required);assert len(list((R/'checkpoints').rglob('*.pt')))==36
    final=dict(passed=True,required_artifacts=len(required),new_checkpoints=36,existing_checkpoints=12,protected_files_SHA_unchanged=len(before),startup_failure_preserved=True,startup_fix_before_learning_and_Z1=True,conditions_and_learning_prediction_functions_unchanged=True,Z2_availability_recorded=True,effective_input_counts=inputcounts,extra_training=0,additional_model_inference=0)
    dump('final_artifact_audit.json',final)
    with (R/'audit_report.md').open('a',encoding='utf-8') as f:f.write('\n## Supplemental artifact closure\n\n'+json.dumps(final,ensure_ascii=False,indent=2)+'\n')
    (R/'sha256_manifest.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(R)}' for p in sorted(R.rglob('*')) if p.is_file() and p.name!='sha256_manifest.txt')+'\n',encoding='utf-8');print(json.dumps(final,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
