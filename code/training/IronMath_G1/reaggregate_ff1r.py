"""Saved-file reaggregation only, no model inference or old-module imports."""
import csv,json
from common import R,W,sha,dump,now
S=W/'IronMath_FF1R'
files=['ff1r_predictions.csv','long_item_results.csv','ff1r_metrics.json']
before={n:sha(S/n) for n in files}
def rows(n):
    with (S/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
pred=rows('ff1r_predictions.csv');items=rows('long_item_results.csv');out=[]
for lengths in [[32,64,128,512,2048,8192,65000],[2048,8192,65000]]:
    models=[]
    for p in pred:
        ii=[r for r in items if r['model']==p['model']]
        actual=all(int(r['R_correct'])==int(r['R_count']) for r in ii if int(r['T']) in lengths)
        nf=any(int(r['nonfinite_positions'])>0 for r in ii)
        models.append(dict(model=p['model'],actual_success=actual,nonfinite=nf,**{n:p[n]=='True' for n in ['A','B','D']},C_all_success=True,C_all_failure=False))
    hits={n:sum(m[n]==m['actual_success'] for m in models) for n in ['A','B','C_all_success','C_all_failure','D']}
    positive=sum(m['A'] for m in models);success=sum(m['actual_success'] for m in models)
    H1='判定不能' if success in [0,40] or min(positive,40-positive)<5 else ('支持' if all(hits['A']>hits[n] for n in ['B','C_all_success','C_all_failure','D']) else '不支持')
    finite=[m for m in models if not m['actual_success'] and not m['nonfinite']];fa=sum(not m['A'] for m in finite);fd=sum(not m['D'] for m in finite)
    H2='判定不能' if len(finite)<5 else ('支持' if fa>fd else '不支持')
    out.append(dict(lengths=lengths,actual_success=success,hits=hits,H1=H1,H2=H2,finite_failures=len(finite),finite_failure_correct_A=fa,finite_failure_correct_D=fd,models=models))
assert all(sha(S/n)==h for n,h in before.items())
original=json.loads((S/'ff1r_metrics.json').read_text(encoding='utf-8'))
assert original['H1']==out[0]['H1'] and original['H2']==out[0]['H2']
dump('ff1r_reaggregation.json',dict(utc=now(),analysis='FF1-R saved-output reaggregation, separate from G1',source_SHA256=before,source_files_unchanged=True,no_training_or_inference=True,original_FF1R_results_unchanged=True,definitions=out,hit_counts_changed=out[0]['hits']!=out[1]['hits'],H1_changed=out[0]['H1']!=out[1]['H1'],H2_changed=out[0]['H2']!=out[1]['H2']))
print('FF1R reaggregation complete; hits/H1/H2 changed:',out[0]['hits']!=out[1]['hits'],out[0]['H1']!=out[1]['H1'],out[0]['H2']!=out[1]['H2'])
