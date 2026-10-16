"""Supplemental saved-data audits and artifact closure. No neural execution."""
import csv,gzip,json
from collections import Counter
from pathlib import Path
from run import R,WORK,C,csvout,dump,sha,protected

def read(name):
    with (R/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    audit=json.loads((R/'audit_results.json').read_text());assert audit['passed']
    assert sha(R/'analyze.py')==json.loads((R/'ANALYSIS_REGISTERED.json').read_text())['analysis_SHA256']
    overlap=json.loads((R/'transformed_input_split_audit.json').read_text());assert len(overlap)==12 and all(r['exact_transformed_input_overlap']==0 for r in overlap)
    rows=read('internal_carry_distribution.csv')
    states={(r['condition'],r['model'],int(r['seed']),int(r['width']),r['op'],int(r['incoming_carry'])):int(r['digit_count']) for r in rows}
    terncontexts=binarycontexts=0
    for cond in C['conditions']:
        for seed in C['seeds']:
            for width in C['widths']:
                for op in ('ADD','SUB'):
                    for state in (-1,0,1):assert states.get((cond,'T0',seed,width,op,state),0)==states.get((cond,'T1',seed,width,op,state),0)
                    terncontexts+=1
                assert states.get((cond,'B1',seed,width,'SUB',0),0)==states.get((cond,'B0',seed,width,'SUB',1),0)
                assert states.get((cond,'B1',seed,width,'SUB',1),0)==states.get((cond,'B0',seed,width,'SUB',0),0)+512
                binarycontexts+=1
    relations=dict(ternary_same_teacher_transition_distribution_contexts=terncontexts,binary_complementary_carry_borrow_contexts=binarycontexts,SUB_extra_guard_in_B1=512,scope='correct arithmetic diagnostic states, not neural hidden state claims',passed=True)
    dump('diagnostic_relations.json',relations)
    scores=read('evaluation_results.csv');numeric=Counter();first=Counter()
    if audit['nonfinite_predictions']:
        with gzip.open(R/'predictions.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                p=json.loads(line)
                if p['prediction'] is None:
                    g=(p['condition'],p['model'],p['seed'],p['width'],p['op']);numeric[g]+=1;first[g+(next(i for i,x in enumerate(p['prediction_digits']) if x is None),)]+=1
    assert sum(numeric.values())==audit['nonfinite_predictions']
    csvout('numeric_failures.csv',[dict(condition=r['condition'],model=r['model'],seed=r['seed'],width=r['width'],op=r['op'],n=r['n'],nonfinite_predictions=numeric[(r['condition'],r['model'],int(r['seed']),int(r['width']),r['op'])]) for r in scores])
    if first:csvout('first_nonfinite_position.csv',[dict(condition=g[0],model=g[1],seed=g[2],width=g[3],op=g[4],position=g[5],n=n) for g,n in first.items()])
    report=(R/'REPORT_V054.md').read_text(encoding='utf-8')
    learning_width=[r for r in scores if int(r['width'])==8];assert len(learning_width)==48
    width_note=f"8bit Testの全48run/演算集計の整数完全一致率は最小{100*min(float(r['exact_rate']) for r in learning_width):.2f}%、最大{100*max(float(r['exact_rate']) for r in learning_width):.2f}%。全幅・seed別値はevaluation_results.csv。長幅の精度と混同しない。\n\n"
    section='''## 原因切り分けの到達点・追加独立監査

T0/T1の正解側Carry遷移分布は全108評価文脈で一致した。三進符号反転でCarry依存そのものは消えない。T1対T0の差は、今回の入力変換・演算flag変更・学習条件の効果であり、Carry列が短くなったことでは説明できない。

B1のSUBにおける正解CarryはB0の正解Borrowと補完関係にある。54文脈でCarry=0の位置数がBorrow=1と一致し、Carry=1はBorrow=0に追加guard512位置を加えた数と一致した。Borrowなし問題でもB1ではCarry=1を保持する長い系列になる。これらは正しい算術側の状態についての事実であり、NNが同じ情報を保持・因果利用したという証明ではない。diagnostic_relations.jsonに検証結果を保存。

全4方式・3組の分割比較で、変換後入力の重複は0。TrainのADD交換拡張を含めた独立検査。T1のA+0/A-0の同値入力も分割をまたいでいない。transformed_input_split_audit.jsonと読み取り監査コードを保存。この追加検査は学習後に実施し、データや学習条件の変更はしていない。

共通整数Testには二進Carry/Borrow由来の構造問題が含まれる。任意の長い三進Carry列を網羅したTestではない。空の三進長伝播カテゴリについて能力を推定しない。C2の更新数はCPU速度・熱状態・周辺負荷の影響も受ける。DESIGN_NOTES.mdはTest結果を見る前に保存した解釈上の制約。

numeric_failures.csvに各文脈の非有限予測件数を保存。通常の有限な桁誤りと区別する。桁単位正答率が高くても整数完全一致は保証されない。失敗分類の関連だけから、Borrow保持失敗・演算分離・表現非対称性のどれか一つを唯一の内部原因とは断定できない。内部状態への因果介入は今回実施していない。

'''
    if '## 原因切り分けの到達点・追加独立監査' not in report:report=report.replace('## 停止',section+width_note+'## 停止');(R/'REPORT_V054.md').write_text(report,encoding='utf-8')
    before=json.loads((R/'protected_sha_before.json').read_text());assert protected()==before
    mandatory=['REPORT_V054.md','SPEC_REPRESENTATION.md','PREREGISTRATION.md','training_results.csv','length_generalization.csv','representation_comparison.csv','carry_borrow_results.csv','seed_stability.csv','cpu_memory.csv','predictions.jsonl.gz','failure_cases.jsonl.gz','audit_report.md','config.json','figures/length_generalization.png']
    assert all((R/p).is_file() and (R/p).stat().st_size>0 for p in mandatory)
    assert len(list((R/'checkpoints').rglob('*.pt')))==48
    closure=dict(passed=True,mandatory_artifacts=len(mandatory),checkpoints=48,exact_transformed_input_split_comparisons=12,all_split_overlaps_zero=True,diagnostic_relations=relations,analysis_registered_SHA_unchanged=True,protected_files_SHA_unchanged=len(before),additional_model_inference=0,additional_training=0)
    dump('final_artifact_audit.json',closure)
    with (R/'audit_report.md').open('a',encoding='utf-8') as f:f.write('\n## Supplemental saved-data closure\n\n'+json.dumps(closure,ensure_ascii=False,indent=2)+'\n')
    dump('STOPPED.json',dict(status='completed_audited_and_stopped',runs=24,additional_training=0,additional_model_inference=0,client_date='2026-10-15'))
    (R/'sha256_manifest.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(R)}' for p in sorted(R.rglob('*')) if p.is_file() and p.name!='sha256_manifest.txt')+'\n',encoding='utf-8')
    print(json.dumps(closure,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
