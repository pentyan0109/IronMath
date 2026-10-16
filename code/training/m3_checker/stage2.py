"""Stage 2 (inference only): learned mod-3 checker applied to a, b and the frozen model's prediction r.
Flag (Python control, ground truth never used): ADD (ra+rb-rr)%3!=0, SUB (ra-rb-rr)%3!=0."""
import csv, gzip, json, random, time, collections
import numpy as np
from m3_common import HERE, ROOT, int_bits, predict, load_checker, peak_ram_mib

csv.field_size_limit(1 << 30)
t0 = time.process_time(); w0 = time.perf_counter()


def naf_weight(d):
    d = abs(d); w = 0
    while d:
        if d & 1:
            d -= 2 - (d % 4); w += 1
        d >>= 1
    return w


# ---------- collect problems: (source, group, op, width, model_seed, id, a, b, pred, ans, n)
probs = []
with gzip.open(ROOT / 'IronMath_V054/predictions.jsonl.gz', 'rt') as h:
    for l in h:
        r = json.loads(l)
        if r['condition'] == 'C1' and r['model'] == 'B0':
            a, b = int(r['a']), int(r['b']); n = int(r['length'])
            probs.append(('V054', f"w{r['width']}", r['op'], int(r['width']), r['seed'], r['id'], a, b, int(r['prediction']), a + b if r['op'] == 'ADD' else a - b, n))
T = ROOT / 'v054t_transition'
pt = {}
for l in open(T / 'data/probes.jsonl'):
    p = json.loads(l); pt[(p['op'], p['pid'])] = p
for r in csv.DictReader(open(T / 'results_per_problem.csv', encoding='utf-8-sig')):
    p = pt[(r['op'], int(r['pid']))]
    probs.append(('V054T', f"k{p['k']}", r['op'], p['seq_len'], int(r['seed']), f"pid{p['pid']}", int(p['a'], 16), int(p['b'], 16),
                  int(r['prediction'], 16), int(p['answer'], 16), p['seq_len']))
E = ROOT / 'v054e_endurance'; pe = {}
for f in (E / 'data').glob('*.json'):
    if f.name != 'manifest.json':
        for p in json.load(open(f)):
            pe[(f.stem, p['id'])] = p
for r in csv.DictReader(open(E / 'results_per_problem.csv', encoding='utf-8-sig')):
    p = pe[(r['set_name'], r['problem_id'])]
    probs.append(('V054E', r['set_name'], 'ADD', int(p['width']), int(r['seed']), r['problem_id'], int(p['a'], 16), int(p['b'], 16),
                  int(r['prediction_hex'], 16), int(p['answer'], 16), int(r['length'])))
for P in probs:
    assert P[9] == (P[6] + P[7] if P[2] == 'ADD' else P[6] - P[7]) and max(P[6], P[7], P[8], P[9]) < (1 << P[10])

# ---------- select: all failures + up to 512 correct per (source, group, op, model seed), Random(94000)
groups = collections.defaultdict(list)
for P in probs:
    if P[8] == P[9]:
        groups[(P[0], P[1], P[2], P[4])].append(P)
selected = [P for P in probs if P[8] != P[9]]
n_fail = len(selected)
for k in sorted(groups):
    g = sorted(groups[k], key=lambda P: P[5])
    selected += random.Random(94000).sample(g, min(512, len(g)))
print('failures', n_fail, 'correct sampled', len(selected) - n_fail, flush=True)

# ---------- unique sequences (value, length)
uniq = sorted({(v, P[10]) for P in selected for v in (P[6], P[7], P[8])}, key=lambda t: (t[1], t[0]))
index = {u: i for i, u in enumerate(uniq)}
print('unique sequences', len(uniq), 'total bit-steps', sum(n for _, n in uniq), flush=True)
seqs = [int_bits(v, n) for v, n in uniq]

res = {}; timing = {}
for cs in (0, 1, 2):
    m, _ = load_checker(cs); ts = time.perf_counter()
    out = predict(m, seqs, batch=512, chunk=4096)
    res[cs] = np.array([int(o[-1]) for o in out], np.int8); timing[cs] = round(time.perf_counter() - ts, 1)
    print('checker seed', cs, 'wall', timing[cs], flush=True)
truth = np.array([v % 3 for v, _ in uniq], np.int8)

rows = []
for P in selected:
    src, grp, op, width, ms, pid, a, b, pred, ans, n = P
    fail = pred != ans; d = pred - ans
    w = naf_weight(d) if fail else ''
    for cs in (0, 1, 2):
        ia, ib, ir = index[(a, n)], index[(b, n)], index[(pred, n)]
        ra, rb, rr = int(res[cs][ia]), int(res[cs][ib]), int(res[cs][ir])
        flag = int(((ra + rb - rr) if op == 'ADD' else (ra - rb - rr)) % 3 != 0)
        ea, eb, er = int(ra != truth[ia]), int(rb != truth[ib]), int(rr != truth[ir])
        rows.append(dict(source=src, group=grp, op=op, width=width, model_seed=ms, checker_seed=cs, id=pid, failure=int(fail),
                         naf_w=w, upper_det3=int(d % 3 != 0) if fail else '', ra=ra, rb=rb, rr=rr, flag=flag,
                         err_a=ea, err_b=eb, err_r=er, checker_any_error=int(ea or eb or er)))
with open(HERE / 'integration_results.csv', 'w', newline='') as f:
    wr = csv.DictWriter(f, list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)


def stats(rs):
    F = [r for r in rs if r['failure']]; C = [r for r in rs if not r['failure']]
    s = dict(n_fail=len(F), n_correct=len(C))
    if F:
        s['detection'] = round(sum(r['flag'] for r in F) / len(F), 6); s['upper_bound'] = round(sum(r['upper_det3'] for r in F) / len(F), 6)
        s['gap_points'] = round(100 * (s['detection'] - s['upper_bound']), 3)
        for lab, sub in (('w1', [r for r in F if r['naf_w'] == 1]), ('w>=2', [r for r in F if r['naf_w'] >= 2])):
            s[f'n_{lab}'] = len(sub); s[f'detection_{lab}'] = round(sum(r['flag'] for r in sub) / len(sub), 6) if sub else None
    if C:
        s['false_alarm'] = round(sum(r['flag'] for r in C) / len(C), 6)
    s['checker_errors'] = dict(a=sum(r['err_a'] for r in rs), b=sum(r['err_b'] for r in rs), r=sum(r['err_r'] for r in rs),
                               any=sum(r['checker_any_error'] for r in rs))
    return s


def group(kf):
    g = collections.defaultdict(list)
    for r in rows:
        g[kf(r)].append(r)
    return {'|'.join(map(str, k)): stats(v) for k, v in sorted(g.items(), key=lambda kv: tuple(str(x).zfill(8) for x in kv[0]))}


# I2: w=1 failures split by checker correctness
I2 = {}
for cs in (0, 1, 2):
    F1 = [r for r in rows if r['checker_seed'] == cs and r['failure'] and r['naf_w'] == 1]
    ok = [r for r in F1 if not r['checker_any_error']]; bad = [r for r in F1 if r['checker_any_error']]
    C = [r for r in rows if r['checker_seed'] == cs and not r['failure']]
    I2[f'checker_s{cs}'] = dict(w1_failures=len(F1), w1_checker_all_correct=len(ok), w1_detected_when_all_correct=sum(r['flag'] for r in ok),
                                w1_checker_any_error=len(bad), w1_missed_with_checker_error=sum(1 - r['flag'] for r in bad),
                                correct_problems=len(C), correct_checker_all_correct=sum(1 - r['checker_any_error'] for r in C),
                                false_alarms_when_all_correct=sum(r['flag'] for r in C if not r['checker_any_error']),
                                correct_checker_any_error=sum(r['checker_any_error'] for r in C),
                                false_alarms_with_checker_error=sum(r['flag'] for r in C if r['checker_any_error']))
s1 = json.load(open(HERE / 'stage1_summary.json', encoding='utf-8'))
qual = [s for s in (0, 1, 2) if s1['random'][f's{s}|65000']['final_acc'] >= 0.99]
summary = dict(n_failures=n_fail, n_correct_sampled=len(selected) - n_fail, n_unique_sequences=len(uniq),
               by_checker=group(lambda r: (r['checker_seed'],)),
               by_checker_source=group(lambda r: (r['checker_seed'], r['source'])),
               by_checker_source_op=group(lambda r: (r['checker_seed'], r['source'], r['op'])),
               by_checker_source_op_group=group(lambda r: (r['checker_seed'], r['source'], r['op'], r['group'])),
               by_checker_source_op_modelseed=group(lambda r: (r['checker_seed'], r['source'], r['op'], r['model_seed'])),
               I1=dict(qualifying_checker_seeds=qual, verdict='判定不能（前提未達）' if not qual else 'see by_checker'),
               I2=I2, checker_wall_s=timing, cpu_seconds=round(time.process_time() - t0, 1), wall_seconds=round(time.perf_counter() - w0, 1),
               peak_ram_mib=peak_ram_mib())
json.dump(summary, open(HERE / 'stage2_summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(dict(by_checker=summary['by_checker'], by_checker_source=summary['by_checker_source'], I1=summary['I1'], I2=I2), ensure_ascii=False, indent=1))
