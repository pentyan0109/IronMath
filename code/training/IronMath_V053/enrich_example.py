"""Attach existing inputs/Oracle labels to the observed same-wrong record."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
rows=json.loads((R/'same_wrong_examples.json').read_text())
for r in rows:
    source=json.loads((R/'data'/f"{r['op']}_Test{r['width']}.json").read_text())
    p=next(x for x in source if x['id']==r['id'])
    r.update(a=p['a'],b=p['b'],answer=p['answer'],longest_propagation=p['longest_propagation'])
    assert r['R_prediction']==r['L_prediction'] and r['R_prediction']!=r['answer']
(R/'same_wrong_examples.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
with (R/'REPORT_V053.md').open('a',encoding='utf-8') as f:
    for r in rows:f.write(f"\n同一誤答の完全な数値例: {r['condition']} seed{r['seed']} {r['width']}bit {r['a']} {'+' if r['op']=='ADD' else '-'} {r['b']}、正解 {r['answer']}、R=L予測 {r['R_prediction']}。一致しても誤答だった。\n")
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
files={str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file() and p.name!='sha256_manifest.txt'}
(R/'sha256_manifest.txt').write_text('\n'.join(f'{h}  {p}' for p,h in sorted(files.items()))+'\n',encoding='utf-8')
print(json.dumps(rows))
