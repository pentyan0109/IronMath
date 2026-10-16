"""Independent saved-file checks only. Never imports or invokes the model."""
import csv,json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'IronMath_FF1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
checked=[]
for line in (S/'SHA256SUMS.txt').read_text(encoding='utf-8-sig').splitlines():
    h,p=line.split('  ',1);assert sha(S/p)==h,p;checked.append(p)
model=rows(R/'ff1a_model_classification.csv');assert len(model)==40
old={r['model']:r for r in rows(S/'model_comparison.csv')}
assert all(r['actual_success']==old[r['model']]['actual_success'] and r['A_success']==old[r['model']]['A_success'] for r in model)
assert all(r['D_success']=='True' for r in model if r['A_success']=='True')
assert all(int(r['nonfinite_positions'])==0 for r in model if r['actual_success']=='True')
diff=rows(R/'ff1a_disagreements.csv');assert len(diff)==9 and all(r['A_only_correct']=='True' for r in diff)
report=R/'REPORT_FF1A.md'
with report.open('a',encoding='utf-8') as f:
    f.write('\n## 有限誤答型と予測差の内訳\n\n|モデル|Nの2点判定|Rの2点判定|A|D|\n|---|---|---|---|---|\n')
    for r in model:
        if r['classification']=='有限誤答型':f.write('|'+ '|'.join(r[k] for k in ['model','two_attractors_N','two_attractors_R','A_success','D_success'])+'|\n')
    f.write('\nAとDが異なる9モデルはすべてAのみ正解。原因の保存ログ上の内訳：\n\n')
    for r in diff:f.write(f"- {r['model']}: {r['reasons']}\n")
    detail=json.loads((S/'diagnostic_details.json').read_text(encoding='utf-8'))
    dd=[r for r in detail if r['model'] in {x['model'] for x in diff}]
    wrong=sum(r['readout_finite'] and r['readout']!=r['value'] for r in dd)
    bad=sum(not r['readout_finite'] for r in dd)
    unstable=sum(r['spectral_radius'] is not None and r['spectral_radius']>=1 for r in dd)
    f.write(f'\n差のある9モデルの144終点中、有限で誤った読み出しは{wrong}終点、非有限読み出し{bad}終点、rho>=1は{unstable}終点。したがって今回確認された追加情報は、単なる収束に対する「正しい値の読み出し」の情報である。引き込み点の多重性や局所安定性そのものが差を生んだと分離して証明した結果ではない。\n')
    f.write('\n独立監査でFF1最終マニフェスト全'+str(len(checked))+'ファイルのSHA一致を確認。元のモデル単位の真値・A予測と今回の再採点結果が一致。優先度変更の最初の試行はctypesのハンドル型指定不備で失敗したため修正し、分析のみ再実行。学習・推論は実行していない。\n')
audit=json.loads((R/'ff1a_audit.json').read_text(encoding='utf-8'))
audit['independent_audit']={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'entire_FF1_manifest_files':len(checked),'all_SHA_match':True,'model_labels_match_original':True,'nine_disagreements_all_A_only_correct':True,'no_success_has_nonfinite_logits':True}
(R/'ff1a_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
files=sorted(p for p in R.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
(R/'SHA256SUMS.txt').write_text(''.join(sha(p)+'  '+p.relative_to(R).as_posix()+'\n' for p in files),encoding='utf-8')
print(json.dumps(audit['independent_audit']))
