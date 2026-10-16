"""D10-F: carry-state frequency in the actual training streams vs existing attractor results.
Read/aggregate only (no training, no inference, no fixed-point search). usage: python -B d10f.py"""
import csv, hashlib, json, random, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

H = Path(__file__).resolve().parent; WORK = H.parent
sys.dont_write_bytecode = True
import torch  # only to read stored hashes from checkpoints (torch.load); no model is built

V054 = WORK / 'IronMath_V054'; V055 = WORK / 'IronMath_V055'; D10 = WORK / 'decimal_d10'
SEEDS = (0, 1, 2); SERIES = ('B0', 'F0', 'F1', 'F2', 'D10')
CFG = json.loads((H / 'config.json').read_text(encoding='utf-8'))


def python_procs():
    out = subprocess.run(['tasklist', '/FO', 'CSV', '/NH'], capture_output=True, text=True).stdout
    return [dict(name=r[0], pid=int(r[1])) for r in csv.reader(out.splitlines()) if r and r[0].lower().startswith('python')]


def csvout(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (H / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows({k: ('NA' if v is None else v) for k, v in r.items()} for r in rows)


def jload(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def digits(v, n, base):
    out = []
    for _ in range(n):
        out.append(v % base); v //= base
    return out


def pass_pairs(base):
    """pairs (x,y) whose next carry equals the incoming carry for both carry 0 and 1 (integer arithmetic)."""
    return [f'{x}{y}' for x in range(base) for y in range(base) if all((x + y + c) // base == c for c in (0, 1))]


PASS = {2: pass_pairs(2), 10: pass_pairs(10)}
assert PASS[2] == CFG['pass_inputs']['binary_ADD'] and PASS[10] == CFG['pass_inputs']['decimal']


# ---------------------------------------------------------------- stream reconstruction
def stream_B0(seed):
    data = {op: jload(WORK / 'IronMath_V053' / 'data' / f'{op}_Train.json') for op in ('ADD', 'SUB')}
    rng = random.Random(50000 + seed); oh = hashlib.sha256(); rows = []
    for _ in range(1200):
        for op in ('ADD', 'SUB'):
            pool = data[op]
            for _ in range(8):
                r = dict(pool[rng.randrange(len(pool))])
                if op == 'ADD' and rng.randrange(2):
                    r['a'], r['b'] = r['b'], r['a']
                oh.update(f"{r['id']}:{r['a']}:{r['b']}\n".encode())
                rows.append(dict(kind=op, a=r['a'], b=r['b'], width=r['width']))
    cp = torch.load(V054 / 'checkpoints' / f'C1_B0_s{seed}' / 'best_validation.pt', map_location='cpu', weights_only=False)
    ok = oh.hexdigest() == cp['problem_order_sha256'] and cp['update'] == 1200
    return rows, dict(order_sha_match=ok, stored=cp['problem_order_sha256'], recomputed=oh.hexdigest(), checkpoint_update=cp['update']), data['ADD']


def stream_F(cond, seed):
    data = {op: jload(V055 / 'data' / f'{op}_Train.json') for op in ('ADD', 'SUB')}
    rng = random.Random(50000 + seed); order = hashlib.sha256(); cand = hashlib.sha256(); rows = []
    for step in range(1200):
        mixed = cond == 'F1' or (cond == 'F2' and step >= 600)
        streams = []
        for op in ('ADD', 'ADD', 'SUB'):
            ss = []; pool = data[op]
            for _ in range(8):
                r = dict(pool[rng.randrange(len(pool))])
                if op == 'ADD' and rng.randrange(2):
                    r['a'], r['b'] = r['b'], r['a']
                ss.append(r)
            streams.append(ss)
        for ss in streams:
            for r in ss:
                cand.update(f"{r['id']}:{r['a']}:{r['b']}\n".encode())
        used = streams[0] + streams[2 if mixed else 1]
        for r in used:
            order.update(f"{r['id']}:{r['a']}:{r['b']}\n".encode())
            w = r['width']
            if r['op'] == 'SUB':  # route P: a + ((-b) mod 2^w), flag 0, length w+1
                rows.append(dict(kind='SUB_P', a=r['a'], b=(-r['b']) % (1 << w), width=w))
            else:
                rows.append(dict(kind='ADD', a=r['a'], b=r['b'], width=w))
    cp = torch.load(V055 / 'checkpoints' / f'C1_{cond}_s{seed}' / 'best_validation.pt', map_location='cpu', weights_only=False)
    ok = order.hexdigest() == cp['order_SHA256'] and cand.hexdigest() == cp['candidate_SHA256'] and cp['update'] == 1200
    return rows, dict(order_sha_match=ok, candidate_sha_match=cand.hexdigest() == cp['candidate_SHA256'], stored=cp['order_SHA256'], recomputed=order.hexdigest(), checkpoint_update=cp['update']), data['ADD']


def stream_D10(seed):
    train = jload(D10 / 'data' / 'Train.json'); rng = random.Random(50000 + seed); order = hashlib.sha256(); rows = []
    for _ in range(1200):
        for _ in range(16):
            r = dict(train[rng.randrange(len(train))])
            if rng.randrange(2):
                r['a'], r['b'] = r['b'], r['a']
            order.update(f"{r['id']}:{r['a']}:{r['b']}\n".encode()); rows.append(dict(kind='ADD', a=r['a'], b=r['b'], width=r['width']))
    rec = jload(D10 / 'checkpoints' / f's{seed}' / 'training_record.json')
    cp = torch.load(D10 / 'checkpoints' / f's{seed}' / 'best_validation.pt', map_location='cpu', weights_only=False)
    ok = order.hexdigest() == rec['order_SHA256'] == cp['order_SHA256'] and cp['update'] == 1200
    return rows, dict(order_sha_match=ok, stored=rec['order_SHA256'], recomputed=order.hexdigest(), checkpoint_update=cp['update']), train


# ---------------------------------------------------------------- counting
def count(rows, base, kinds):
    """counts per pass pair of carry-in, same-pair runs, any-pass runs (carry constant inside)."""
    c = {p: [0, 0] for p in PASS[base]}; pair_run = {p: [0, 0, 0] for p in PASS[base]}  # max run any carry, c0, c1
    any_runs = Counter(); n_ex = 0; n_pos = 0
    for r in rows:
        if r['kind'] not in kinds:
            continue
        n = r['width'] + 1; A = digits(r['a'], n, base); B = digits(r['b'], n, base); s = 0; n_ex += 1; n_pos += n
        prev = None; run = 0; arun = 0; acarry = None
        for x, y in zip(A, B):
            p = f'{x}{y}'
            if p in c:
                c[p][s] += 1
                run = run + 1 if p == prev else 1
                pr = pair_run[p]; pr[0] = max(pr[0], run); pr[1 + s] = max(pr[1 + s], run)
                arun = arun + 1 if acarry is not None else 1; acarry = s
            else:
                if arun:
                    any_runs[(acarry, arun)] += 1
                arun = 0; acarry = None; run = 0
            prev = p if p in c else None
            s = (x + y + s) // base
        if arun:
            any_runs[(acarry, arun)] += 1
        assert s == 0
    return c, pair_run, any_runs, n_ex, n_pos


def main():
    t0 = time.perf_counter(); procs = python_procs()
    freq = []; runs = []; streams_meta = {}; majority = {}
    for series in SERIES:
        for seed in SEEDS:
            if series == 'B0':
                rows, meta, fixed = stream_B0(seed); base = 2; scopes = {'stream_ADD': {'ADD'}}
            elif series == 'D10':
                rows, meta, fixed = stream_D10(seed); base = 10; scopes = {'stream_ADD': {'ADD'}}
            else:
                rows, meta, fixed = stream_F(series, seed); base = 2; scopes = {'stream_all': {'ADD', 'SUB_P'}, 'stream_ADD': {'ADD'}}
            meta['presented_rows'] = dict(Counter(r['kind'] for r in rows)); streams_meta[f'{series}_s{seed}'] = meta
            primary = 'stream_all' if series in ('F1', 'F2') else 'stream_ADD'
            if series == 'F0':
                assert meta['presented_rows'] == {'ADD': 19200}
            fixed_rows = [dict(kind='ADD', a=r['a'], b=r['b'], width=r['width']) for r in fixed]
            scope_rows = dict((k, (rows, v)) for k, v in scopes.items()); scope_rows['fixed_Train_ADD_proxy'] = (fixed_rows, {'ADD'})
            for scope, (rr, kinds) in scope_rows.items():
                c, pr, anyr, n_ex, n_pos = count(rr, base, kinds)
                for p in PASS[base]:
                    c0, c1 = c[p]; tot = c0 + c1
                    maj = None if c0 == c1 else 0 if c0 > c1 else 1
                    freq.append(dict(series=series, seed=seed, scope=scope, primary=int(scope == primary), pair=p, carry0=c0, carry1=c1, total=tot,
                                     frac_carry0=c0 / tot if tot else None, frac_carry1=c1 / tot if tot else None,
                                     majority=maj if tot else None, tie=int(c0 == c1), examples=n_ex, positions=n_pos))
                    runs.append(dict(series=series, seed=seed, scope=scope, primary=int(scope == primary), pair=p, max_same_pair_run=pr[p][0],
                                     max_same_pair_run_carry0=pr[p][1], max_same_pair_run_carry1=pr[p][2]))
                    if scope == primary:
                        majority[(series, seed, p)] = dict(majority=maj, c0=c0, c1=c1)
                for cc in (0, 1):
                    lens = [L for (s, L), n in anyr.items() if s == cc for _ in range(n)]
                    runs.append(dict(series=series, seed=seed, scope=scope, primary=int(scope == primary), pair=f'any_pass_carry{cc}', max_same_pair_run=None,
                                     any_pass_runs=len(lens), any_pass_max_run=max(lens) if lens else 0,
                                     any_pass_runs_ge4=sum(L >= 4 for L in lens), any_pass_runs_ge8=sum(L >= 8 for L in lens),
                                     any_pass_positions=sum(lens)))
            print(series, seed, meta['order_sha_match'], meta['presented_rows'], flush=True)
    csvout('training_frequency.csv', freq); csvout('carry_run_statistics.csv', runs)
    (H / 'data' / 'stream_reconstruction.json').write_text(json.dumps(streams_meta, indent=1), encoding='utf-8')

    # ------------------------------------------------ attractors (existing results only)
    att = []
    with (WORK / 'v054h_dynamics' / 'posthoc_fixed_point_pairs.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['op'] == 'ADD':
                att.append(dict(series='B0', seed=int(r['seed']), pair=r['pair'], distance=float(r['distance_hstar_state0_vs_state1']),
                                stored_label=None, out0=int(r['output_state0_side']), out1=int(r['output_state1_side']), source='v054h_dynamics/posthoc_fixed_point_pairs.csv'))
    fp = {}
    with (WORK / 'v054r_replication' / 'R_fixed_points.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            fp[(r['model'], int(r['seed']), r['op'], int(r['state']), r['pair'])] = int(r['h_star_output'])
    with (WORK / 'v054r_replication' / 'R_pairs.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['model'].startswith('V055'):
                m = r['model']; s = int(r['seed'])
                att.append(dict(series=m[5:], seed=s, pair=r['pair'], distance=float(r['distance']), stored_label=int(r['n_points']),
                                out0=fp[(m, s, 'ADD', 0, r['pair'])], out1=fp[(m, s, 'ADD', 1, r['pair'])], source='v054r_replication/R_pairs.csv + R_fixed_points.csv'))
    with (D10 / 'pairs.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            att.append(dict(series='D10', seed=int(r['seed']), pair=r['pair'], distance=float(r['distance_c0_vs_c1']), stored_label=1 if r['same_point'] == '1' else 2,
                            out0=int(r['output_c0_side']), out1=int(r['output_c1_side']), source='decimal_d10/pairs.csv'))
    comp = []
    for a in att:
        base = 10 if a['series'] == 'D10' else 2
        npts = 1 if a['distance'] < 1e-3 else 2
        label_ok = None if a['stored_label'] is None else int(npts == a['stored_label'])
        st = None
        if npts == 1:
            o = a['out0']
            st = (1 - o) if base == 2 else (0 if o == 9 else 1 if o == 0 else None)
        mj = majority[(a['series'], a['seed'], a['pair'])]
        included = int(npts == 1 and st is not None and mj['majority'] is not None)
        comp.append(dict(series=a['series'], seed=a['seed'], pair=a['pair'], distance=a['distance'], n_points=npts, stored_label_consistent=label_ok,
                         output_state0_side=a['out0'], output_state1_side=a['out1'], one_point_output_state=st,
                         train_carry0=mj['c0'], train_carry1=mj['c1'], majority_carry=mj['majority'],
                         majority_fraction=max(mj['c0'], mj['c1']) / (mj['c0'] + mj['c1']) if mj['c0'] + mj['c1'] else None,
                         F1_included=included, F1_match=int(st == mj['majority']) if included else None, source=a['source']))
    expected = {s: (6 if s == 'B0' or s.startswith('F') else 30) for s in SERIES}
    for s in SERIES:
        assert sum(r['series'] == s for r in comp) == expected[s], (s, sum(r['series'] == s for r in comp))
    comp.sort(key=lambda r: (SERIES.index(r['series']), r['seed'], r['pair']))
    csvout('attractor_comparison.csv', comp)
    summ = []
    for s in SERIES + ('ALL',):
        rr = [r for r in comp if s == 'ALL' or r['series'] == s]; inc = [r for r in rr if r['F1_included']]
        summ.append(dict(series=s, conditions=len(rr), one_point=sum(r['n_points'] == 1 for r in rr), two_points=sum(r['n_points'] == 2 for r in rr),
                         F1_denominator=len(inc), F1_matches=sum(r['F1_match'] for r in inc), F1_rate=sum(r['F1_match'] for r in inc) / len(inc) if inc else None,
                         one_point_state0=sum(r['one_point_output_state'] == 0 for r in rr), one_point_state1=sum(r['one_point_output_state'] == 1 for r in rr),
                         excluded_one_point_no_state=sum(r['n_points'] == 1 and r['one_point_output_state'] is None for r in rr),
                         excluded_tie=sum(r['n_points'] == 1 and r['majority_carry'] is None for r in rr)))
    csvout('per_model_summary.csv', summ)
    allr = summ[-1]; rate = allr['F1_rate']
    verdict = '判定不能' if rate is None else '支持' if rate >= 0.8 else '不支持' if rate <= 0.5 else '判定不能'
    # secondary: ADD-only for F1/F2
    sec = []
    for r in comp:
        if r['series'] in ('F1', 'F2') and r['F1_included']:
            q = [x for x in freq if x['series'] == r['series'] and x['seed'] == r['seed'] and x['pair'] == r['pair'] and x['scope'] == 'stream_ADD'][0]
            sec.append(dict(series=r['series'], seed=r['seed'], pair=r['pair'], ADD_only_majority=q['majority'], state=r['one_point_output_state'], match=int(q['majority'] == r['one_point_output_state'])))
    (H / 'data' / 'analysis_summary.json').write_text(json.dumps(dict(F1_verdict=verdict, F1_rate=rate, per_series=summ, F1F2_ADD_only_secondary=sec,
                                                                      python_processes_at_start=procs, wall_seconds=time.perf_counter() - t0), ensure_ascii=False, indent=1), encoding='utf-8')
    print('F1', verdict, rate, [(r['series'], r['F1_matches'], r['F1_denominator']) for r in summ])


if __name__ == '__main__':
    main()
