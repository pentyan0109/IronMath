"""Saved-table diagnostics only; no model inference or training."""
import csv,gzip,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parent
def read(n):return list(csv.DictReader((R/n).open(encoding='utf-8-sig')))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
corr=read('error_correlation.csv');agreement=read('agreement_analysis.csv');pairs=read('paired_seed_comparison.csv');cats=read('carry_borrow_results.csv');resource=read('cpu_memory.csv')
buckets=defaultdict(dict)
for r in cats:buckets[(r['condition'],r['seed'],r['width'],r['op'],r['category'])][r['direction']]=r
reversals=[]
for k,v in buckets.items():
    if float(v['L']['exact_rate'])>float(v['R']['exact_rate']):
        reversals.append(dict(condition=k[0],seed=k[1],width=k[2],op=k[3],category=k[4],n=v['R']['n'],R_exact=v['R']['exact_rate'],L_exact=v['L']['exact_rate']))
with (R/'category_reversals.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['condition','seed','width','op','category','n','R_exact','L_exact']);w.writeheader();w.writerows(reversals)
same_groups=[r for r in corr if int(r['same_wrong'])];same_examples=[]
with gzip.open(R/'paired_predictions.jsonl.gz','rt',encoding='utf-8') as f:
    for line in f:
        r=json.loads(line)
        if r['class_label']=='Both Wrong' and r['answers_agree']:same_examples.append(r)
assert len(same_examples)==sum(int(r['same_wrong']) for r in corr)
(R/'same_wrong_examples.json').write_text(json.dumps(same_examples,indent=2),encoding='utf-8')
lines=['','## 全幅・カテゴリ逆転と二重計算の限界','',f"全幅の対応seed平均問題集合でL>Rとなったgroupは{sum(float(r['R_minus_L'])<0 for r in pairs)}。一方、重複カテゴリ内でL>Rとなった条件/seed/幅/演算/categoryは{len(reversals)}group（同じ問題が複数カテゴリに入るため独立試験数ではない）。全行はcategory_reversals.csv。",'','|条件|seed|幅|演算|カテゴリ|母数|R%|L%|','|---|---:|---:|---|---|---:|---:|---:|']
for r in reversals[:20]:lines.append(f"|{r['condition']}|{r['seed']}|{r['width']}|{r['op']}|{r['category']}|{r['n']}|{100*float(r['R_exact']):.2f}|{100*float(r['L_exact']):.2f}|")
lines+=['',f'全108相関groupで、同じ数値誤答の一致は{len(same_examples)}件。2048bitでは0件だったが、他幅で実例があり、一致は数学的正しさの証明ではない。same_wrong_examples.jsonに全該当例を保存。']
for x in same_examples:lines.append(f"- {x['condition']} seed{x['seed']} {x['op']} {x['width']}bit / {x['id']}: R=L={x['R_prediction']}（両方誤答）。")
lines+=['','|2048bit条件/演算|一致率平均%|一致時精度 s0/s1/s2%|保留率平均%|Oracle上限平均%|','|---|---:|---|---:|---:|']
for condition in ('C1','C2'):
    for op in ('ADD','SUB'):
        aa=sorted([x for x in agreement if x['condition']==condition and x['op']==op and x['width']=='2048'],key=lambda x:int(x['seed']));cc=[x for x in corr if x['condition']==condition and x['op']==op and x['width']=='2048']
        accuracy=' / '.join('NA' if x['agreement_accuracy']=='NA' else f"{100*float(x['agreement_accuracy']):.2f}" for x in aa)
        lines.append(f"|{condition} {op}|{100*statistics.mean(float(x['agreement_rate']) for x in aa):.2f}|{accuracy}|{100*statistics.mean(float(x['abstention_rate']) for x in aa):.2f}|{100*statistics.mean(float(x['oracle_selection_upper_bound']) for x in cc):.2f}|")
lines+=['','減算の2048bit L-onlyはC1で6件、C2で4件で、それぞれseed0だけ。他seedでは0件。補完が全seedに再現したとは言えない。一致した減算回答の精度が100%でもcoverageは約1〜2%であり、ほとんどを保留する。加算の2048bit一致は0件、条件付き精度はNA。','',f"新規学習CPU合計 {sum(float(x['train_cpu_seconds']) for x in resource if x['stage']=='training'):.3f}秒、学習wall合計 {sum(float(x['train_wall_seconds']) for x in resource if x['stage']=='training'):.3f}秒。全process peak {max(int(x['peak_RAM_bytes']) for x in resource)/2**20:.2f}MiB。この値に独立監査processのRAMは含まない。",'長さ汎化: Rは8bitから2048bitの加算で高い完全一致を維持したが、減算はseed差が大きく完全なアルゴリズム獲得とは言えない。Lは同じ逐次出力制約で長幅加算が0%、減算も低い。これは全入力を読み終えてから計算するMSB手法や双方向モデル一般の否定ではない。','次段階の検証器導入を直ちに成功見込みと判断する根拠は弱い。まずL-onlyが現れる条件の再現性と、同じ誤答を一致させる失敗例を別途事前登録して調べる候補とする。追加学習・検証器学習は今回実施せず停止した。']
with (R/'REPORT_V053.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
before=json.loads((R/'protected_sha_before.json').read_text());assert all(sha(R.parent/p)==h for p,h in before.items())
for name,h in json.loads((R/'STARTED.json').read_text())['registered_SHA'].items():assert sha(R/name)==h
(R/'final_artifact_audit.json').write_text(json.dumps(dict(passed=True,category_reversal_groups=len(reversals),same_wrong_examples=len(same_examples),protected_SHA_unchanged=len(before),source_SHA_unchanged=True,additional_training=0,additional_inference=0),indent=2),encoding='utf-8')
manifest={str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file() and p.name!='sha256_manifest.txt'}
(R/'sha256_manifest.txt').write_text('\n'.join(f'{h}  {p}' for p,h in sorted(manifest.items()))+'\n',encoding='utf-8')
print(json.dumps(dict(category_reversals=len(reversals),same_wrong_examples=same_examples,protected=len(before),stopped=True)))
