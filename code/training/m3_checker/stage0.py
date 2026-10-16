"""Stage 0: upper bound of a perfect mod-m checker on saved failures (no training, no model execution)."""
import csv, gzip, json, time, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V054 = ROOT / 'IronMath_V054'
V054T = ROOT / 'v054t_transition'
MODS = (3, 5, 7, 9)
cfg = json.load(open(HERE / 'config.json', encoding='utf-8'))
t0 = time.process_time(); w0 = time.perf_counter()


def shape(pred, ans):
    x = pred ^ ans; s = bin(x)[2:]
    return s.count('1'), max(len(r) for r in s.split('0'))


def row(source, seed, op, width, k, pid, length, pred, ans):
    assert pred != ans and 0 <= ans < (1 << length) and 0 <= pred < (1 << length)
    d = pred - ans; wb, run = shape(pred, ans)
    r = dict(source=source, seed=seed, op=op, width=width, k=k, id=pid, length=length, d=d,
             wrong_bits=wb, longest_wrong_run=run, single_bit=int(wb == 1))
    for m in MODS:
        r[f'd_mod{m}'] = d % m; r[f'det{m}'] = int(d % m != 0)
    return r


rows = []
# --- V054 C1/B0 saved failures, cross-checked with predictions.jsonl.gz
pred_fail = set()
with gzip.open(V054 / 'predictions.jsonl.gz', 'rt') as h:
    for l in h:
        r = json.loads(l)
        if r['condition'] == 'C1' and r['model'] == 'B0' and not r['correct']:
            pred_fail.add((r['seed'], r['id']))
seen = set()
with gzip.open(V054 / 'failure_cases.jsonl.gz', 'rt') as h:
    for l in h:
        r = json.loads(l)
        if not (r['condition'] == 'C1' and r['model'] == 'B0'):
            continue
        a, b, n = int(r['a']), int(r['b']), int(r['length'])
        ans = a + b if r['op'] == 'ADD' else a - b
        assert ans == int(r['answer']) and n == int(r['width']) + (r['op'] == 'ADD')
        digits = r['prediction_digits']; pred = sum(v << i for i, v in enumerate(digits))
        assert len(digits) == n and pred == int(r['prediction'])
        seen.add((r['seed'], r['id']))
        rows.append(row('V054', r['seed'], r['op'], int(r['width']), '', r['id'], n, pred, ans))
assert seen == pred_fail, 'failure_cases and predictions disagree'

# --- V054T probe failures
probes = {}
for l in open(V054T / 'data' / 'probes.jsonl'):
    p = json.loads(l); key = (p['op'], p['pid']); assert key not in probes; probes[key] = p
n_t = 0
for r in csv.DictReader(open(V054T / 'results_per_problem.csv', encoding='utf-8-sig')):
    p = probes[(r['op'], int(r['pid']))]; n_t += 1
    assert int(r['seq_len']) == p['seq_len'] and int(r['k']) == p['k']
    a, b, ans = int(p['a'], 16), int(p['b'], 16), int(p['answer'], 16)
    assert ans == (a + b if p['op'] == 'ADD' else a - b)
    pred = int(r['prediction'], 16)
    assert int(r['exact']) == int(pred == ans)
    if pred != ans:
        rows.append(row('V054T', int(r['seed']), r['op'], p['seq_len'], p['k'], f"pid{p['pid']}", p['seq_len'], pred, ans))

fields = list(rows[0].keys())
with open(HERE / 'stage0_upper_bound.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)


def stats(rs):
    n = len(rs)
    if not n:
        return dict(n=0)
    s = dict(n=n)
    for m in MODS:
        s[f'det{m}'] = sum(r[f'det{m}'] for r in rs); s[f'det{m}_rate'] = round(s[f'det{m}'] / n, 6)
    multi = [r for r in rs if not r['single_bit']]
    s['single_bit'] = n - len(multi)
    s['det3_rate_multi_bit_only'] = round(sum(r['det3'] for r in multi) / len(multi), 6) if multi else None
    s['d_mod3_counts'] = dict(sorted(collections.Counter(r['d_mod3'] for r in rs).items()))
    wb = sorted(r['wrong_bits'] for r in rs); lr = sorted(r['longest_wrong_run'] for r in rs)
    s['wrong_bits_median'] = wb[n // 2]; s['wrong_bits_max'] = wb[-1]
    s['longest_run_median'] = lr[n // 2]; s['longest_run_max'] = lr[-1]
    return s


def group(keyf):
    g = collections.defaultdict(list)
    for r in rows:
        g[keyf(r)].append(r)
    return {'|'.join(map(str, k)) if isinstance(k, tuple) else str(k): stats(v) for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}


def bucket(n):
    return '1' if n == 1 else '2' if n == 2 else '3-4' if n <= 4 else '5-8' if n <= 8 else '9-16' if n <= 16 else '17-64' if n <= 64 else '65+'


total = stats(rows); rate = total['det3_rate']
lo, hi = 0.55, 0.80
summary = dict(
    total=total,
    P0=dict(prediction=cfg['stage0']['P0']['prediction'], observed=rate, range=[lo, hi], verdict='支持' if lo <= rate <= hi else '不支持'),
    by_source=group(lambda r: r['source']),
    by_source_op=group(lambda r: (r['source'], r['op'])),
    by_source_seed=group(lambda r: (r['source'], r['seed'])),
    by_source_seed_op=group(lambda r: (r['source'], r['seed'], r['op'])),
    by_V054_width_op=group(lambda r: (r['source'], r['op'], r['width']) if r['source'] == 'V054' else ('V054T', r['op'], 'k=' + str(r['k']))),
    by_wrong_bits_bucket=group(lambda r: bucket(r['wrong_bits'])),
    by_longest_run_bucket=group(lambda r: bucket(r['longest_wrong_run'])),
    counts=dict(V054_failures=sum(r['source'] == 'V054' for r in rows), V054T_failures=sum(r['source'] == 'V054T' for r in rows), V054T_problems_scanned=n_t),
    endurance_v054e='not used (REPORT_V054E.md absent)',
    cpu_seconds=round(time.process_time() - t0, 2), wall_seconds=round(time.perf_counter() - w0, 2))
json.dump(summary, open(HERE / 'stage0_summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(dict(total=total, P0=summary['P0'], counts=summary['counts']), ensure_ascii=False, indent=1))
