"""Independent audit for D10-F. usage: python -B audit.py  -> data/audit_results.json (+ audit_report.md written by hand from it)"""
import csv, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
H = Path(__file__).resolve().parent; WORK = H.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(H))
import d10f  # stream reconstruction only (verified by stored order hashes)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def rcsv(name):
    with (H / name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


res = {}
# 1-2. checkpoints and data unchanged vs earlier records
ra = json.loads((WORK / 'v054r_replication' / 'asset_paths.json').read_text(encoding='utf-8'))['files']
ck = {}
for s in (0, 1, 2):
    for key in [f'V054_B0_s{s}', f'V055_F0_s{s}', f'V055_F1_s{s}', f'V055_F2_s{s}']:
        ck[key] = sha(ra[key]['path']) == ra[key]['sha256']
    rec = json.loads((WORK / 'decimal_d10' / 'checkpoints' / f's{s}' / 'training_record.json').read_text())
    ck[f'D10_s{s}'] = sha(WORK / 'decimal_d10' / 'checkpoints' / f's{s}' / 'best_validation.pt') == rec['best_checkpoint_file_SHA256']
res['checkpoints_unchanged_vs_earlier_records'] = ck
d10sums = {}
for line in (WORK / 'decimal_d10' / 'data' / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
    h, n = line.split(None, 1); n = n.strip().lstrip('*')
    if n in ('Train.json',):
        d10sums[n] = sha(WORK / 'decimal_d10' / 'data' / n) == h
res['D10_Train_sha_vs_SHA256SUMS'] = d10sums
v055m = json.loads((WORK / 'IronMath_V055' / 'dataset_manifest.json').read_text())
res['V055_Train_sha_vs_dataset_manifest'] = {k: sha(WORK / 'IronMath_V055' / 'data' / f'{k}.json') == v055m[k]['sha256'] for k in ('ADD_Train', 'SUB_Train') if k in v055m}
v054m = json.loads((WORK / 'IronMath_V054' / 'dataset_manifest.json').read_text())
res['V053_Train_sha_vs_V054_dataset_manifest'] = {k: sha(WORK / v054m[k]['source']) == v054m[k]['sha256'] for k in ('ADD_Train', 'SUB_Train')}

# 3. carry recount with the integer formula, independent of the ripple counter
freq = rcsv('training_frequency.csv'); prim = {(r['series'], int(r['seed']), r['pair']): (int(r['carry0']), int(r['carry1'])) for r in freq if r['primary'] == '1'}
recount_ok = {}
for series in d10f.SERIES:
    for seed in (0, 1, 2):
        if series == 'B0':
            rows, meta, _ = d10f.stream_B0(seed); base = 2; kinds = {'ADD'}
        elif series == 'D10':
            rows, meta, _ = d10f.stream_D10(seed); base = 10; kinds = {'ADD'}
        else:
            rows, meta, _ = d10f.stream_F(series, seed); base = 2; kinds = {'ADD', 'SUB_P'} if series in ('F1', 'F2') else {'ADD'}
        cnt = defaultdict(lambda: [0, 0])
        for r in rows:
            if r['kind'] not in kinds:
                continue
            a, b, n = r['a'], r['b'], r['width'] + 1
            for i in range(n):
                cin = ((a % base**i) + (b % base**i)) // base**i
                x = (a // base**i) % base; y = (b // base**i) % base
                if x + y == base - 1:
                    cnt[f'{x}{y}'][cin] += 1
        recount_ok[f'{series}_s{seed}'] = all(tuple(cnt[p]) == prim[(series, seed, p)] for p in d10f.PASS[base]) and meta['order_sha_match']
res['carry_recount_integer_formula_and_order_hash'] = recount_ok

# 4. pass-input definition by brute force
res['pass_definition'] = dict(binary=d10f.PASS[2], decimal=d10f.PASS[10],
                              binary_ok=d10f.PASS[2] == ['01', '10'], decimal_ok=d10f.PASS[10] == [f'{x}{9 - x}' for x in range(10)])

# 5. completeness
comp = rcsv('attractor_comparison.csv')
res['completeness'] = {s: sum(r['series'] == s for r in comp) for s in d10f.SERIES}
res['completeness_ok'] = res['completeness'] == {'B0': 6, 'F0': 6, 'F1': 6, 'F2': 6, 'D10': 30}

# 6. stream vs fixed Train separated
scopes = sorted({(r['series'], r['scope'], r['primary']) for r in freq})
res['scopes'] = scopes
res['primary_is_stream'] = all(not (p == '1' and sc.startswith('fixed')) for _, sc, p in scopes)

# 7. one/two point labels vs stored results
lab = {}
for r in comp:
    if r['stored_label_consistent'] != 'NA':
        lab[f"{r['series']}_s{r['seed']}_{r['pair']}"] = r['stored_label_consistent'] == '1'
ifp = rcsv('../v054r_replication/I_fixed_points.csv')
for r in ifp:
    if r['c'] == 'orig' and r['op'] == 'ADD' and r.get('n_points') not in (None, '', 'NA'):
        c = [x for x in comp if x['series'] == 'B0' and x['seed'] == r['seed'] and x['pair'] == r['pair']][0]
        lab[f"B0_s{r['seed']}_{r['pair']}_vs_V054R_I"] = int(r['n_points']) == int(c['n_points'])
res['one_two_point_labels_consistent'] = lab
res['one_two_point_all_consistent'] = all(lab.values())

# 8-9. F1 numerator/denominator recomputed from the CSVs
num = den = 0; per = defaultdict(lambda: [0, 0])
for r in comp:
    if r['n_points'] != '1':
        continue
    base = 10 if r['series'] == 'D10' else 2; o = int(r['output_state0_side'])
    st = (1 - o) if base == 2 else (0 if o == 9 else 1 if o == 0 else None)
    c0, c1 = prim[(r['series'], int(r['seed']), r['pair'])]
    if st is None or c0 == c1:
        continue
    maj = 0 if c0 > c1 else 1; den += 1; num += st == maj; per[r['series']][0] += st == maj; per[r['series']][1] += 1
summ = rcsv('per_model_summary.csv')
allrow = [r for r in summ if r['series'] == 'ALL'][0]
res['F1_recomputed'] = dict(numerator=num, denominator=den, per_series={k: v for k, v in per.items()})
res['F1_matches_summary'] = int(allrow['F1_matches']) == num and int(allrow['F1_denominator']) == den and all(
    int([r for r in summ if r['series'] == k][0]['F1_matches']) == v[0] and int([r for r in summ if r['series'] == k][0]['F1_denominator']) == v[1] for k, v in per.items())

# 10. criteria unchanged since registration
reg = (H / 'data' / 'config_sha_at_registration.txt').read_text().split()[0]
res['config_unchanged'] = sha(H / 'config.json') == reg
res['all_pass'] = all([all(ck.values()), all(d10sums.values()), all(res['V055_Train_sha_vs_dataset_manifest'].values()), all(res['V053_Train_sha_vs_V054_dataset_manifest'].values()),
                       all(recount_ok.values()), res['pass_definition']['binary_ok'], res['pass_definition']['decimal_ok'], res['completeness_ok'],
                       res['primary_is_stream'], res['one_two_point_all_consistent'], res['F1_matches_summary'], res['config_unchanged']])
(H / 'data' / 'audit_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: v for k, v in res.items() if k not in ('scopes',)}, ensure_ascii=False)[:3000])
