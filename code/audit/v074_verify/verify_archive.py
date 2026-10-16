"""Recompute headline numbers from the archive alone (no training, inference or test evaluation).

Usage: python verify_archive.py <archive root> <output dir>
Checks (ARCHIVE_VERIFY.md):
  a. G1 correct counts of A / D / B / B_cum(128) / B_cum(512) on 160 nonlinear models, and ABL1 A1/A2,
     recomputed from frozen predictions, saved diagnostic records/traces and saved per-length scores.
  b. E3 240-case endpoint readout x outcome cross-tabulation and 46/80 shared endpoints.
  c. E2 per-context summary (7/10, 8/10, +7.3 [4.0,10.4], +14.4 [8.3,22.1]).
  e. SHA-256 of preregistration documents against the recorded values.
  (d. figures: see render_figures.py and compare_figures.py)
"""
import csv, json, hashlib, re, sys
from pathlib import Path
import numpy as np

A = Path(sys.argv[1]).resolve(); O = Path(sys.argv[2]).resolve(); O.mkdir(parents=True, exist_ok=True)
RES, PRE = A / 'results', A / 'preregistration'
def rows(p):
    with open(p, encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
truth = lambda v: str(v).strip().lower() == 'true'
report = {}

# ---------- a. G1 ----------
g = RES / 'IronMath_G1'
res = {r['model']: r for r in rows(g / 'g1_results.csv')}
pred = {r['model']: r for r in rows(PRE / 'IronMath_G1/main_predictions.csv')}
scores = {}
for r in rows(RES / 'IronMath_PaperCheck/memory_length_scores.csv'):
    if r['study'] == 'G1': scores.setdefault(r['model'], {})[int(r['length'])] = truth(r['success'])
assert len(res) == 200 and len(pred) == 200 and len(scores) == 200
lengths = sorted(next(iter(scores.values())))
a = dict(lengths_in_scores=lengths)
nonlin = [m for m in res if res[m]['architecture'] != 'linear']
assert len(nonlin) == 160
actual = {}
for m in res:
    s = scores[m]; act = all(s[L] for L in (2048, 8192, 65000))
    assert act == truth(res[m]['actual_success']), m
    actual[m] = act
a['actual_success_nonlinear'] = sum(actual[m] for m in nonlin)
A0, A1, A2, D2 = {}, {}, {}, {}
for m in res:
    d = json.loads((g / 'diagnostic_records' / f'{m}.json').read_text(encoding='utf-8')); dd = d['details']; assert len(dd) == 16
    fin = lambda v: v is not None and np.isfinite(float(v))
    A0[m] = all(x['converged'] and fin(x['spectral_radius']) and float(x['spectral_radius']) < 1 and x['readout_finite'] and x['readout'] == x['value'] for x in dd)
    A1[m] = all(x['readout_finite'] and x['readout'] == x['value'] for x in dd)
    ok = True
    for neutral in ('N', 'R'):
        z = np.load(g / 'diagnostic_traces' / f'{m}_{neutral}.npz'); mg = z['margins']; lf = z['logit_finite']
        values = np.arange(8) % 2
        correct = np.where(values[None, :] == 0, mg >= 0, mg > 0) & ~np.isnan(mg) & lf
        ok = ok and bool(correct.all())
    A2[m] = ok
    assert A0[m] == truth(pred[m]['A']), ('A0 differs from frozen A', m)
def count(f): return sum(f(m) == actual[m] for m in nonlin)
bcum = lambda ell: (lambda m: all(scores[m][L] for L in lengths if L <= ell))
a['correct_on_160'] = {'A (frozen)': count(lambda m: truth(pred[m]['A'])), 'A0 recomputed from diagnostic records': count(lambda m: A0[m]),
                       'D (frozen)': count(lambda m: truth(pred[m]['D'])), 'B (frozen)': count(lambda m: truth(pred[m]['B'])),
                       'B_cum(128)': count(bcum(128)), 'B_cum(512)': count(bcum(512)),
                       'ABL1 A1 endpoint readout': count(lambda m: A1[m]), 'ABL1 A2 readout at every step (from traces)': count(lambda m: A2[m])}
expect_a = {'A (frozen)': 140, 'A0 recomputed from diagnostic records': 140, 'D (frozen)': 91, 'B (frozen)': 76, 'B_cum(128)': 119, 'B_cum(512)': 143,
            'ABL1 A1 endpoint readout': 145, 'ABL1 A2 readout at every step (from traces)': 145}
a['expected'] = expect_a; a['match'] = all(a['correct_on_160'][k] == v for k, v in expect_a.items())
report['a'] = a

# ---------- b. E3 ----------
e = RES / 'IronMath_PaperCheck/extension/E3_independent_replication'
pairs = rows(e / 'E3_PAIR_RESULTS.csv'); diag = rows(e / 'E3_DIAGNOSTICS.csv'); hold = rows(e / 'E3_HOLD_RESULTS.csv')
hg = {}
for r in hold: hg.setdefault((r['model'], r['op'], r['pair'], r['state']), []).append(r)
cross = {}
for r in diag:
    hs = hg.get((r['model'], r['op'], r['pair'], r['state']), [])
    fails = [x for x in hs if x['first_hold_error'] not in ('', 'None', 'NA')]
    outcome = 'broke' if fails else ('did_not_break' if len(hs) == 64 else 'not_evaluable')
    readout = ('correct' if truth(r['endpoint_correct']) else 'wrong') if truth(r['converged']) else 'unavailable'
    cross[(readout, outcome)] = cross.get((readout, outcome), 0) + 1
b = dict(cases=len(diag), crosstab={f'{k[0]} | {k[1]}': v for k, v in sorted(cross.items())},
         pairs=len(pairs), shared=sum(truth(p['shared']) for p in pairs), distinct=sum(truth(p['distinct']) for p in pairs))
b['match'] = (len(diag) == 240 and cross.get(('wrong', 'broke'), 0) == 46 and cross.get(('wrong', 'did_not_break'), 0) == 0 and cross.get(('correct', 'broke'), 0) == 0
              and cross.get(('correct', 'did_not_break'), 0) == 194 and sum(v for k, v in cross.items() if k[1] == 'not_evaluable') == 0 and b['shared'] == 46 and b['pairs'] == 80)
report['b'] = b

# ---------- c. E2 ----------
e2 = rows(RES / 'IronMath_PaperCheck/extension/E2_causal_intervention/E2_RESULTS.csv')
meas = [r for r in e2 if r['status'] == 'measured']
key = lambda r: (r['seed'], r['op'], r['state'], r['pair'])
base = {key(r): r for r in meas if r['condition'] == 'baseline' and r['at'] == '8' and r['first_error']}
cell = {(key(r), r['condition']): r for r in meas if r['at'] == '8'}
CONTROLS = [f'orthogonal{i}' for i in range(3)] + [f'random{i}' for i in range(3)]
rng = np.random.default_rng(73000); c = dict(contexts=len(base))
for donor in ('correct', 'failed'):
    diffs, wins, by_ckpt = [], 0, {}
    for k in sorted(base):
        T0 = int(base[k]['first_error'])
        sp = int(cell[(k, f'normal_1_{donor}')]['first_error']) - T0
        cs = [int(cell[(k, f'{cc}_{donor}')]['first_error']) - T0 for cc in CONTROLS]
        wins += sp > max(cs); diffs.append(sp - float(np.mean(cs))); by_ckpt.setdefault(k[0], []).append(diffs[-1])
    v = np.asarray(diffs); s = v[rng.integers(0, len(v), size=(10000, len(v)))]
    c[donor] = dict(normal_gt_all6=int(wins), mean_diff=float(v.mean()), CI95=[float(x) for x in np.percentile(s.mean(1), [2.5, 97.5])],
                    median_diff=float(np.median(v)), checkpoint_means={k: float(np.mean(x)) for k, x in sorted(by_ckpt.items())})
    np.percentile(np.median(s, 1), [2.5, 97.5])  # keeps the generator sequence identical to aggregate_v073.py
r1 = lambda x: round(x, 1)
c['match'] = (c['contexts'] == 10 and c['correct']['normal_gt_all6'] == 7 and c['failed']['normal_gt_all6'] == 8
              and r1(c['correct']['mean_diff']) == 7.3 and [r1(x) for x in c['correct']['CI95']] == [4.0, 10.4]
              and r1(c['failed']['mean_diff']) == 14.4 and [r1(x) for x in c['failed']['CI95']] == [8.3, 22.1])
report['c'] = c

# ---------- e. preregistration SHA-256 ----------
fm = {r['original_path']: r for r in rows(A / 'FILE_MAP.csv')}
checks = []
for r in rows(PRE / 'IronMath_ALL/ALL_SEEDS_HASHES.csv') if (PRE / 'IronMath_ALL/ALL_SEEDS_HASHES.csv').exists() else rows(RES / 'IronMath_ALL/ALL_SEEDS_HASHES.csv'):
    paths = [p.strip().strip('`') for p in re.split(r';\s*', r['事前登録ファイル']) if '/' in p]
    hashes = [h.strip() for h in re.split(r';\s*', r['事前登録SHA256（本作業開始時の値）']) if h.strip()]
    for i, p in enumerate(paths):
        rec = hashes[i] if i < len(hashes) else ''
        m = fm.get(p)
        if m is None: checks.append(dict(source='ALL_SEEDS_HASHES', experiment=r['実験ID'], file=p, recorded=rec, status='file not in archive')); continue
        got = sha(A / m['archive_path'])
        st = 'match' if rec and got.startswith(rec.replace('…', '')) and len(rec) >= 8 else ('no recorded value' if not rec else 'MISMATCH')
        checks.append(dict(source='ALL_SEEDS_HASHES', experiment=r['実験ID'], file=p, recorded=rec, archive_SHA256=got, recorded_length=len(rec), status=st))
# self-recorded hash files (e.g. PREREGISTRATION_SHA256.json): every 64-hex value whose key or sibling names an archived file
for p in sorted(PRE.rglob('*SHA256*.json')):
    try: d = json.loads(p.read_text(encoding='utf-8-sig'))
    except Exception: continue
    def walk(o, ctx=''):
        if isinstance(o, dict):
            names = [v for k, v in o.items() if isinstance(v, str) and re.search(r'\.(md|json|csv|txt|py)$', v)]
            for k, v in o.items():
                if isinstance(v, str) and re.fullmatch(r'[0-9a-f]{64}', v):
                    target = None
                    cand = [k] + names if re.search(r'\.(md|json|csv|txt|py)$', k) else names
                    for nm in cand:
                        q = (p.parent / nm)
                        if q.exists(): target = q; break
                    if target is None:
                        stem = re.sub(r'(?i)_?SHA256$', '', p.stem)
                        for ext in ('.md', '.json', '.csv'):
                            q = p.parent / (stem + ext)
                            if q.exists() and q != p and k.lower() in ('sha256', 'sha', 'hash', 'registered_sha256', 'preregistration_sha256'): target = q; break
                    if target is not None:
                        got = sha(target); checks.append(dict(source=p.relative_to(A).as_posix(), experiment='', file=target.relative_to(A).as_posix(), recorded=v, archive_SHA256=got, recorded_length=64, status='match' if got == v else 'MISMATCH'))
                walk(v, k)
        elif isinstance(o, list):
            for v in o: walk(v, ctx)
    walk(d)
# self-recorded config hashes kept as text (V054T/V054H/V054R/D10-F): the registered config.json of the same experiment
for p in sorted(PRE.rglob('*sha*.txt')):
    hx = re.findall(r'\b[0-9a-f]{64}\b', p.read_text(encoding='utf-8-sig', errors='replace'))
    target = PRE / p.relative_to(PRE).parts[0] / 'config.json'
    if hx and target.exists():
        got = sha(target)
        checks.append(dict(source=p.relative_to(A).as_posix(), experiment='', file=target.relative_to(A).as_posix(), recorded=hx[0], archive_SHA256=got, recorded_length=64,
                           status='match' if got in hx else 'MISMATCH'))
with open(O / 'PREREG_SHA_CHECKS.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['source', 'experiment', 'file', 'recorded', 'archive_SHA256', 'recorded_length', 'status']); w.writeheader(); w.writerows(checks)
st = {}
for x in checks: st[x['status']] = st.get(x['status'], 0) + 1
report['e'] = dict(checks=len(checks), by_status=st, mismatches=[x for x in checks if x['status'] == 'MISMATCH'], match=('MISMATCH' not in st and 'file not in archive' not in st))
json.dump(report, open(O / 'verify_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: v.get('match') for k, v in report.items()}, indent=1))
