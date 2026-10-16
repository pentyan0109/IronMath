"""Stage 0 addendum: perfect-checker upper bound on V054E (endurance) failures. Read-only on v054e_endurance/.
Same computation as stage0.py plus NAF weight w. P0 is not re-judged."""
import csv, json, time, collections, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
E = ROOT / 'v054e_endurance'
MODS = (3, 5, 7, 9)


def naf_weight(d):  # identical to stage0_posthoc.naf_weight (copied; importing that module would rerun it)
    d = abs(d); w = 0
    while d:
        if d & 1:
            d -= 2 - (d % 4); w += 1
        d >>= 1
    return w


csv.field_size_limit(1 << 30)
t0 = time.process_time(); w0 = time.perf_counter()

problems = {}
for f in sorted((E / 'data').glob('*.json')):
    if f.name == 'manifest.json':
        continue
    for p in json.load(open(f)):
        key = (f.stem, p['id']); assert key not in problems
        a, b, ans = int(p['a'], 16), int(p['b'], 16), int(p['answer'], 16)
        assert p['op'] == 'ADD' and ans == a + b and int(p['length']) == int(p['width']) + 1
        problems[key] = (a, b, ans, int(p['width']))

rows = []; n_scanned = collections.Counter()
for r in csv.DictReader(open(E / 'results_per_problem.csv', encoding='utf-8-sig')):
    a, b, ans, w = problems[(r['set_name'], r['problem_id'])]; n = int(r['length']); assert n == w + 1
    pred = int(r['prediction_hex'], 16); n_scanned[r['set_name']] += 1
    assert int(r['exact']) == int(pred == ans) and pred < (1 << n)
    if pred == ans:
        continue
    d = pred - ans; x = pred ^ ans; s = bin(x)[2:]
    row = dict(source='V054E', set_name=r['set_name'], seed=int(r['seed']), op='ADD', width=w, id=r['problem_id'], category=r['category'],
               length=n, d_sign='+' if d > 0 else '-', abs_d_bits=abs(d).bit_length(), wrong_bits=s.count('1'),
               longest_wrong_run=max(len(t) for t in s.split('0')), naf_w=naf_weight(d))
    row['single_bit'] = int(row['wrong_bits'] == 1)
    for m in MODS:
        row[f'd_mod{m}'] = d % m; row[f'det{m}'] = int(d % m != 0)
    assert row['wrong_bits'] == int(r['wrong_bits'])
    rows.append(row)

with open(HERE / 'stage0_endurance.csv', 'w', newline='', encoding='utf-8') as f:
    wr = csv.DictWriter(f, list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)


def stats(rs):
    n = len(rs); s = dict(n=n)
    for m in MODS:
        s[f'det{m}_rate'] = round(sum(r[f'det{m}'] for r in rs) / n, 6)
    s['det3'] = sum(r['det3'] for r in rs)
    w1 = [r for r in rs if r['naf_w'] == 1]; w2 = [r for r in rs if r['naf_w'] >= 2]
    s['w1'] = len(w1); s['w1_frac'] = round(len(w1) / n, 6)
    s['det3_rate_w>=2'] = round(sum(r['det3'] for r in w2) / len(w2), 6) if w2 else None
    s['n_w>=2'] = len(w2)
    s['single_bit'] = sum(r['single_bit'] for r in rs)
    wb = sorted(r['wrong_bits'] for r in rs); lr = sorted(r['longest_wrong_run'] for r in rs)
    s['wrong_bits_median'] = wb[n // 2]; s['longest_run_median'] = lr[n // 2]; s['longest_run_max'] = lr[-1]
    s['naf_w_counts'] = dict(sorted(collections.Counter(min(r['naf_w'], 5) for r in rs).items()))
    return s


def group(kf):
    g = collections.defaultdict(list)
    for r in rows:
        g[kf(r)].append(r)
    return {k: stats(v) for k, v in sorted(g.items(), key=lambda kv: (len(kv[0]), kv[0]))}


summary = dict(note='V054E failures; separate table; P0 not re-judged', n_problems_scanned=dict(n_scanned), total=stats(rows),
               by_set=group(lambda r: r['set_name']), by_set_seed=group(lambda r: f"{r['set_name']}|seed{r['seed']}"),
               by_seed=group(lambda r: f"seed{r['seed']}"), by_category=group(lambda r: r['category']),
               overlap='main_2048 = V054 existing ADD Test2048: its failures duplicate V054 ADD-2048 failures',
               cpu_seconds=round(time.process_time() - t0, 2), wall_seconds=round(time.perf_counter() - w0, 2))
json.dump(summary, open(HERE / 'stage0_endurance_summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
for k in ('total',):
    print(k, summary[k])
for k, v in summary['by_set'].items():
    print(k, {x: v[x] for x in ('n', 'det3_rate', 'det5_rate', 'det7_rate', 'det9_rate', 'w1', 'w1_frac', 'det3_rate_w>=2', 'wrong_bits_median', 'longest_run_median', 'naf_w_counts')})
