"""事後の補助分析 (post-hoc, written after seeing stage-0 results; not preregistered).
Classifies each failure's error d by its signed-binary weight w = minimal number of +-2^i terms (NAF weight).
w==1 means d = +-2^s: a single carry/borrow state error that then propagated consistently. Such d is never 0 mod any odd m."""
import csv, json, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent


def naf_weight(d):
    d = abs(d); w = 0
    while d:
        if d & 1:
            d -= 2 - (d % 4); w += 1
        d >>= 1
    return w


def bucket(w):
    return '1' if w == 1 else '2' if w == 2 else '3-4' if w <= 4 else '5+'


g = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for r in csv.DictReader(open(HERE / 'stage0_upper_bound.csv', encoding='utf-8')):
    d = int(r['d']); b = bucket(naf_weight(d))
    for key in ('ALL', r['source'], f"{r['source']}|{r['op']}", f"{r['source']}|{r['op']}|seed{r['seed']}"):
        for kk in (b, 'w>=2' if b != '1' else None, 'total'):
            if kk:
                c = g[key][kk]; c[0] += 1; c[1] += int(r['det3'])
out = {k: {b: dict(n=n, det3=d3, det3_rate=round(d3 / n, 6)) for b, (n, d3) in sorted(v.items())} for k, v in sorted(g.items())}
json.dump(dict(note='事後の補助分析（結果を見た後に追加、事前登録外）', naf_weight_buckets=out),
          open(HERE / 'stage0_posthoc_d_shape.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
for k in ('ALL', 'V054', 'V054T', 'V054|SUB|seed0'):
    print(k, out[k])
