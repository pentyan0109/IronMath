"""IronMath V054H: argument swap (A), mixed-pair state holding (C), hidden dynamics (B). Inference only.
usage (run with python -B so that no __pycache__ is written into V054 / v054t_transition):
  python -B v054h.py assets|generate|repro|equiv|estimate|analyze
  python -B v054h.py A SEED | C SEED | B SEED
"""
import csv, gzip, hashlib, json, random, statistics, sys, time
from collections import Counter, defaultdict
from pathlib import Path

H = Path(__file__).resolve().parent
WORK = H.parent
V054 = WORK / 'IronMath_V054'
V054T = WORK / 'v054t_transition'
DATA053 = WORK / 'IronMath_V053' / 'data'
sys.dont_write_bytecode = True
sys.path.insert(0, str(V054T))
import numpy as np
import torch
import v054t  # read-only reuse of V054T helpers (it imports IronMath_V054/model.py)
from v054t import HAND, KS, OPS, SEEDS, bits, ckpt, holding_table, load_model, modelhash, predict, ripple, seq_len, sha, step
from model import Residual

CFG = json.loads((H / 'config.json').read_text(encoding='utf-8'))
KC = CFG['C']['k']
TYPES = ('M_rand', 'M_alt12', 'M_alt13', 'M_alt23')


def dump(name, x):
    p = H / name; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')


def load(name):
    return json.loads((H / name).read_text(encoding='utf-8'))


def csvout(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (H / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows({k: ('NA' if v is None else v) for k, v in r.items()} for r in rows)


def jl_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def jl_read(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]


def load_eager(seed):
    cp = torch.load(ckpt(seed), weights_only=False, map_location='cpu')
    base = Residual('B0'); base.load_state_dict(cp['model']); base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    return base


def step_fn(base, x, h):
    """One recurrent step using V054 modules. Composition line copied verbatim from model.py Residual.forward."""
    h = base.core(base.projection(x), h) + torch.tanh(base.residual_gate) * h
    return h, base.head(h)


def make_x(op, a, b, n):
    x = np.zeros((n, 3), np.float32)
    x[:, 0] = bits(a, n); x[:, 1] = bits(b, n); x[:, 2] = int(op == 'SUB')
    return x


@torch.no_grad()
def predict_steps(base, items):
    nmax = max(n for *_, n in items); x = np.zeros((len(items), nmax, 3), np.float32)
    for i, (op, a, b, n) in enumerate(items):
        x[i, :n] = make_x(op, a, b, n)
    x = torch.from_numpy(x); h = torch.zeros(len(items), 545); outs = []
    for t in range(nmax):
        h, z = step_fn(base, x[:, t], h); outs.append(z.argmax(-1))
    p = torch.stack(outs, 1).numpy()
    return [[int(v) for v in p[i, :n]] for i, (*_, n) in enumerate(items)]


def score(r, p, extra=None):
    n = r['seq_len']; truth = bits(int(r['answer'], 16), n)
    wrong = [i for i in range(n) if p[i] != truth[i]]
    longest = run = 0; prev = None
    for i in wrong:
        run = run + 1 if prev == i - 1 else 1; longest = max(longest, run); prev = i
    fe = wrong[0] if wrong else None
    region = None if fe is None else 'prefix' if fe < r['hold_start'] else 'hold' if fe < r['hold_end'] else 'suffix'
    pred = sum(v << i for i, v in enumerate(p))
    row = dict(extra or {}); row.update(exact=int(pred == int(r['answer'], 16)), first_error=fe, first_error_region=region,
                                         hold_offset=(fe - r['hold_start']) if region == 'hold' else None,
                                         first_error_minus_hold_start=(fe - r['hold_start']) if fe is not None else None,
                                         wrong_bits=len(wrong), longest_wrong_run=longest, prediction=hex(pred))
    return row


# ------------------------------------------------------------------ prep
def stage_assets():
    files = {f'checkpoint_C1_B0_s{s}': ckpt(s) for s in SEEDS}
    files.update(model_code=V054 / 'model.py', v054_eval_code=V054 / 'run.py', predictions=V054 / 'predictions.jsonl.gz',
                 failure_cases=V054 / 'failure_cases.jsonl.gz', test2048_ADD=DATA053 / 'ADD_Test2048.json', test2048_SUB=DATA053 / 'SUB_Test2048.json',
                 v054t_probes=V054T / 'data' / 'probes.jsonl', v054t_helpers=V054T / 'v054t.py', v054t_results=V054T / 'results_per_problem.csv',
                 v054t_failure_context=V054T / 'v054_failure_context.csv')
    out = dict(files={k: dict(path=str(p), sha256=sha(p)) for k, p in files.items()},
               v054t_probes_sha_matches_v054t_record=sha(V054T / 'data' / 'probes.jsonl') == json.loads((V054T / 'data' / 'probe_meta.json').read_text(encoding='utf-8'))['probes_sha256'],
               note='V054T と同じ資産。model.py は python -B で import。v054t_transition は読み取りのみ。')
    dump('asset_paths.json', out); print(out['v054t_probes_sha_matches_v054t_record'])


def gen_prefix_suffix(rng):
    prefixes = {}
    for op in OPS:
        for s in (0, 1):
            pre = [dict(id='min', a=([] if s == 0 else ([1] if op == 'ADD' else [0])), b=([] if s == 0 else [1]))]
            while len(pre) < 4:
                a = rng.getrandbits(16); b = rng.getrandbits(16)
                if ((a + b) >> 16 if op == 'ADD' else int(a < b)) == s:
                    pre.append(dict(id=f'r{len(pre)}', a=bits(a, 16), b=bits(b, 16), a_hex=hex(a), b_hex=hex(b)))
            for p in pre:
                assert ripple(op, p['a'], p['b'])[2] == s
            prefixes[(op, s)] = pre
    suffixes = {op: [] for op in OPS}
    for op in OPS:
        for j in range(4):
            a = rng.getrandbits(16); b = rng.getrandbits(16)
            suffixes[op].append(dict(id=f's{j}', a=bits(a, 16), b=bits(b, 16), a_hex=hex(a), b_hex=hex(b)))
    return prefixes, suffixes


def stage_generate():
    # A1: swapped V054T ADD probes
    probes = jl_read(V054T / 'data' / 'probes.jsonl'); a1 = []
    for r in probes:
        if r['op'] != 'ADD':
            continue
        a, b = int(r['a'], 16), int(r['b'], 16)
        assert a + b == b + a == int(r['answer'], 16)
        q = dict(r); q['a'], q['b'] = hex(b), hex(a); q['orig_pair'] = r['pair']; q['pair'] = r['pair'][::-1]
        n = q['seq_len']; out, inc, fin = ripple('ADD', bits(b, n), bits(a, n)); assert out == bits(a + b, n)
        assert all(inc[i] == r['state'] for i in range(r['hold_start'], r['hold_end'] + 1))
        a1.append(q)
    assert len(a1) == 576
    jl_write(H / 'data' / 'A1_swapped_probes.jsonl', a1)
    # A2: swapped 2048 ADD Test
    rows = json.loads((DATA053 / 'ADD_Test2048.json').read_text()); a2 = []
    for r in rows:
        assert r['a'] + r['b'] == r['b'] + r['a']
        a2.append(dict(id=r['id'], a=hex(r['b']), b=hex(r['a']), answer=hex(r['a'] + r['b']), seq_len=2049, category=r['category']))
    assert len(a2) == 512
    jl_write(H / 'data' / 'A2_swapped_test2048.jsonl', a2)
    # C: mixed-pair holding probes
    table = holding_table(); rng = random.Random(91000); prefixes, suffixes = gen_prefix_suffix(rng); mixed = []
    for op in OPS:
        for s in (0, 1):
            pairs = table[(op, s)]
            for typ in TYPES:
                for k in KC:
                    for p in prefixes[(op, s)]:
                        for q in suffixes[op]:
                            if typ == 'M_rand':
                                hold = [pairs[rng.randrange(3)] for _ in range(k)]
                            else:
                                i, j = int(typ[-2]) - 1, int(typ[-1]) - 1
                                hold = [pairs[i] if t % 2 == 0 else pairs[j] for t in range(k)]
                            A = p['a'] + [x for x, _ in hold] + q['a']; B = p['b'] + [y for _, y in hold] + q['b']; guard = 0
                            if op == 'SUB' and ripple(op, A, B)[2] == 1:
                                A = A + [1]; B = B + [0]; guard = 1
                            L = len(A); a = sum(v << i for i, v in enumerate(A)); b = sum(v << i for i, v in enumerate(B))
                            if op == 'SUB':
                                assert a >= b
                            ans = a + b if op == 'ADD' else a - b; n = seq_len(op, L); assert 0 <= ans < (1 << n)
                            out, inc, fin = ripple(op, bits(a, n), bits(b, n)); assert out == bits(ans, n) and fin == 0
                            hs = len(p['a']); he = hs + k
                            assert all(inc[i] == s for i in range(hs, he + 1))
                            mixed.append(dict(pid=len(mixed), op=op, state=s, type=typ, k=k, prefix=p['id'], suffix=q['id'], L=L, seq_len=n,
                                              hold_start=hs, hold_end=he, guard=guard, a=hex(a), b=hex(b), answer=hex(ans),
                                              hold_pairs=''.join(f'{x}{y}' for x, y in hold) if k <= 64 else None))
    assert len(mixed) == 1280 and all(sum(r['op'] == op for r in mixed) == 640 for op in OPS)
    jl_write(H / 'data' / 'C_mixed_probes.jsonl', mixed)
    meta = dict(holding_pairs={f'{op}_state{s}': [f'{x}{y}' for x, y in v] for (op, s), v in table.items()},
                C_prefixes={f'{op}_state{s}': v for (op, s), v in prefixes.items()}, C_suffixes=suffixes,
                C_generation_seed=91000, C_guard_bits=sum(r['guard'] for r in mixed),
                checks=['A: a+b == b+a (Python int) for all', 'A1: Boolean ripple of swapped bits == answer, state held over hold',
                        'C: SUB a>=b, Python int answer, independent Boolean ripple == answer bits, oracle state constant over hold'])
    dump('data/generation_meta.json', meta)
    names = ['A1_swapped_probes.jsonl', 'A2_swapped_test2048.jsonl', 'C_mixed_probes.jsonl', 'generation_meta.json']
    (H / 'data' / 'SHA256SUMS.txt').write_text(''.join(f"{sha(H / 'data' / n)}  {n}\n" for n in names), encoding='utf-8')
    print('A1', len(a1), 'A2', len(a2), 'C', len(mixed), 'C guard', meta['C_guard_bits'])


def saved_2048(op):
    want = {}
    with gzip.open(V054 / 'predictions.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            if '"width": 2048' not in line or '"model": "B0"' not in line or '"condition": "C1"' not in line or f'"op": "{op}"' not in line:
                continue
            p = json.loads(line); want[(p['seed'], p['id'])] = p
    return want


def stage_repro():
    torch.set_num_threads(2); res = dict(rows=[], all_match=True)
    want = {op: saved_2048(op) for op in OPS}
    for s in SEEDS:
        m, cp = load_model(s); h0 = modelhash(m)
        for op in OPS:
            rows = json.loads((DATA053 / f'{op}_Test2048.json').read_text())[:16]
            got = predict(m, [(op, r['a'], r['b'], 2049 if op == 'ADD' else 2048) for r in rows])
            mism = [r['id'][:40] for r, g in zip(rows, got) if g != want[op][(s, r['id'])]['prediction_digits']]
            res['all_match'] &= not mism; res['rows'].append(dict(seed=s, op=op, n=16, bit_identical=16 - len(mism), mismatched_ids=mism))
        assert modelhash(m) == h0
    res['note'] = 'Test2048 先頭16問を batch(16) で jit forward 再実行し、predictions.jsonl.gz の prediction_digits と bit列比較'
    dump('reproduction_check.json', res); print(res)
    if not res['all_match']:
        sys.exit(2)


def stage_equiv():
    torch.set_num_threads(2); probes = jl_read(V054T / 'data' / 'probes.jsonl')
    v054t_pred = {}
    with (V054T / 'results_per_problem.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            v054t_pred[(int(r['seed']), int(r['pid']))] = r['prediction']
    res = dict(rows=[], all_identical=True)
    for s in SEEDS:
        m, _ = load_model(s); base = load_eager(s); h0 = modelhash(m); t0 = time.perf_counter(); n_mis = n_v054t = 0; steps = 0
        for op in OPS:
            for k in KS:
                batch = [r for r in probes if r['op'] == op and r['k'] == k]
                items = [(op, int(r['a'], 16), int(r['b'], 16), r['seq_len']) for r in batch]
                pj = predict(m, items); ps = predict_steps(base, items); steps += max(i[-1] for i in items)
                for r, a, b in zip(batch, pj, ps):
                    n_mis += a != b
                    n_v054t += hex(sum(v << i for i, v in enumerate(a))) != v054t_pred[(s, r['pid'])]
        wall = time.perf_counter() - t0
        assert modelhash(m) == h0 and modelhash(base) == h0
        res['rows'].append(dict(seed=s, n=len(probes), jit_vs_step_mismatch=n_mis, jit_vs_V054T_saved_mismatch=n_v054t, wall_seconds=wall, batched_steps=steps))
        res['all_identical'] &= n_mis == 0; print(res['rows'][-1], flush=True)
    res['wrapper'] = 'h = core(projection(x), h) + tanh(residual_gate) * h ; logit = head(h)  (modules from an eager Residual("B0") loaded with the same state_dict; composition line copied from model.py)'
    res['jit_module_extraction'] = 'not needed: the eager Residual("B0") with the same state_dict provides the modules; equivalence is tested against the jit forward'
    dump('step_equivalence.json', res)


# ------------------------------------------------------------------ A
def stage_A(seed):
    torch.set_num_threads(2); m, _ = load_model(seed); h0 = modelhash(m); t0 = time.perf_counter()
    a1 = jl_read(H / 'data' / 'A1_swapped_probes.jsonl'); rows = []
    for k in KS:
        batch = [r for r in a1 if r['k'] == k]
        preds = predict(m, [('ADD', int(r['a'], 16), int(r['b'], 16), r['seq_len']) for r in batch])
        for r, p in zip(batch, preds):
            rows.append(score(r, p, dict(seed=seed, experiment='A1', pid=r['pid'], state=r['state'], pair=r['pair'], orig_pair=r['orig_pair'], k=k,
                                         prefix=r['prefix'], suffix=r['suffix'], hold_start=r['hold_start'], hold_end=r['hold_end'])))
    t1 = time.perf_counter()
    a2 = jl_read(H / 'data' / 'A2_swapped_test2048.jsonl'); saved = saved_2048('ADD'); rows2 = []
    for j in range(0, 512, 128):
        batch = a2[j:j + 128]
        preds = predict(m, [('ADD', int(r['a'], 16), int(r['b'], 16), 2049) for r in batch])
        for r, p in zip(batch, preds):
            truth = bits(int(r['answer'], 16), 2049); wrong = [i for i in range(2049) if p[i] != truth[i]]
            sv = saved[(seed, r['id'])]
            rows2.append(dict(seed=seed, experiment='A2', id=r['id'][:40], category=r['category'], swapped_exact=int(not wrong),
                              swapped_first_error=wrong[0] if wrong else None, swapped_wrong_bits=len(wrong),
                              original_exact=sv['correct'], original_first_error=sv['first_error']))
    t2 = time.perf_counter(); assert modelhash(m) == h0 and sha(ckpt(seed)) == json.loads((H / 'asset_paths.json').read_text())['files'][f'checkpoint_C1_B0_s{seed}']['sha256']
    jl_write(H / 'data' / 'raw' / f'A_seed{seed}.jsonl', rows + rows2)
    dump(f'data/raw/A_seed{seed}_run.json', dict(A1_wall=t1 - t0, A2_wall=t2 - t1, weights_unchanged=True, weights_sha256=h0))
    print('A seed', seed, 'A1 exact', sum(r['exact'] for r in rows), '/576', 'A2 swapped', sum(r['swapped_exact'] for r in rows2), 'orig', sum(r['original_exact'] for r in rows2), round(t1 - t0, 1), round(t2 - t1, 1), flush=True)


# ------------------------------------------------------------------ C
def stage_C(seed):
    torch.set_num_threads(2); m, _ = load_model(seed); h0 = modelhash(m); t0 = time.perf_counter()
    mixed = jl_read(H / 'data' / 'C_mixed_probes.jsonl'); rows = []
    for op in OPS:
        for k in KC:
            batch = [r for r in mixed if r['op'] == op and r['k'] == k]
            for j in range(0, len(batch), 128):
                bb = batch[j:j + 128]
                preds = predict(m, [(op, int(r['a'], 16), int(r['b'], 16), r['seq_len']) for r in bb])
                for r, p in zip(bb, preds):
                    row = score(r, p, dict(seed=seed, pid=r['pid'], op=op, state=r['state'], type=r['type'], k=k, prefix=r['prefix'], suffix=r['suffix'],
                                           hold_start=r['hold_start'], hold_end=r['hold_end'], guard=r['guard']))
                    if row['first_error'] is not None:
                        n = r['seq_len']; A = bits(int(r['a'], 16), n); B = bits(int(r['b'], 16), n); _, inc, _ = ripple(op, A, B); fe = row['first_error']
                        pr = lambda i: (A[i], B[i])
                        same_pair = 0; i = fe - 1
                        while i >= 0 and pr(i) == pr(fe) and inc[i] == inc[fe]:
                            same_pair += 1; i -= 1
                        same_state = 0; i = fe - 1
                        while i >= 0 and inc[i] == inc[fe]:
                            same_state += 1; i -= 1
                        longest = cur = 0
                        for i in range(fe - same_state, fe):
                            cur = cur + 1 if i > fe - same_state and pr(i) == pr(i - 1) else 1; longest = max(longest, cur)
                        row.update(fe_state=inc[fe], fe_pair=f'{A[fe]}{B[fe]}', fe_pair_is_holding=int(pr(fe) in HAND[(op, inc[fe])]),
                                   preceding_identical_pair_run=same_pair, longest_identical_pair_run_in_state_stretch=longest, preceding_same_state_run=same_state)
                    rows.append(row)
    assert modelhash(m) == h0
    jl_write(H / 'data' / 'raw' / f'C_seed{seed}.jsonl', rows)
    w = time.perf_counter() - t0; dump(f'data/raw/C_seed{seed}_run.json', dict(wall=w, weights_unchanged=True))
    print('C seed', seed, {op: sum(r['exact'] for r in rows if r['op'] == op) for op in OPS}, round(w, 1), flush=True)


# ------------------------------------------------------------------ B
def v054t_status():
    st = {}; rng = defaultdict(list)
    with (V054T / 'results_per_problem.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            key = (int(r['seed']), r['op'], int(r['state']), r['pair'])
            st.setdefault(key, True)
            if r['exact'] == '0':
                st[key] = False
            if r['first_error_region'] == 'hold':
                rng[key].append(int(r['hold_offset']))
    return {k: dict(unbroken=v, break_min=min(rng[k]) if rng[k] else None, break_max=max(rng[k]) if rng[k] else None) for k, v in st.items()}


def stage_B(seed):
    torch.set_num_threads(2); base = load_eager(seed); h0 = modelhash(base); t0 = time.perf_counter()
    iters = load('estimate.json')['B_fixed_point_iterations']
    probes = [r for r in jl_read(V054T / 'data' / 'probes.jsonl') if r['k'] == 4096 and r['prefix'] == 'min' and r['suffix'] == 's0']
    assert len(probes) == 12
    traj = []; fps = []; eigs = []
    for r in probes:
        op, s, pair = r['op'], r['state'], r['pair']; n = r['seq_len']
        x = torch.from_numpy(make_x(op, int(r['a'], 16), int(r['b'], 16), n)); truth = bits(int(r['answer'], 16), n)
        h = torch.zeros(1, 545); hs, he = r['hold_start'], r['hold_end']
        with torch.no_grad():
            for t in range(hs):
                h, _ = step_fn(base, x[t:t + 1], h)
            h_start = h.clone(); first_neg = None; m0 = None; margins = []
            for t in range(he - hs):
                hp = h; h, z = step_fn(base, x[hs + t:hs + t + 1], h); c = truth[hs + t]
                margin = float(z[0, c] - z[0, 1 - c]); margins.append(margin)
                if first_neg is None and margin < 0:
                    first_neg = t
                if t < 512 or t % 16 == 0 or t == he - hs - 1:
                    traj.append(dict(seed=seed, op=op, state=s, pair=pair, t=t, h_norm=float(h.norm()), step_move=float((h - hp).norm()), margin=margin))
            # fixed point from the hidden before the first hold input
            xin = x[hs:hs + 1]; g = h_start.clone(); moves = []; hist = []
            for i in range(iters):
                gn, _ = step_fn(base, xin, g); moves.append(float((gn - g).norm())); g = gn
                if i >= iters - 51:
                    hist.append(g.clone())
            last = moves[-100:]; conv = max(last) < 1e-6
            out_bit = int(step_fn(base, xin, g)[1].argmax(-1)); right_bit = step(op, s, int(pair[0]), int(pair[1]))[0]
            period = [float((hist[-1] - hist[-1 - p]).norm()) for p in range(1, 51)]
        row = dict(seed=seed, op=op, state=s, pair=pair, iterations=iters, converged=int(conv), last100_move_max=max(last), last100_move_min=min(last),
                   last100_move_first=last[0], last100_move_last=last[-1], last100_trend_mean_last50_over_first50=statistics.mean(last[50:]) / max(statistics.mean(last[:50]), 1e-30),
                   min_periodic_distance=min(period), argmin_period=1 + int(np.argmin(period)), h_star_norm=float(g.norm()),
                   h_star_output=out_bit, correct_output=right_bit, h_star_output_correct=int(out_bit == right_bit),
                   margin_at_t0=margins[0], first_negative_margin_t=first_neg, margin_min=min(margins), margin_argmin=int(np.argmin(margins)),
                   margin_t16=margins[16], margin_t64=margins[64], margin_at_end=margins[-1])
        if conv:
            J = torch.autograd.functional.jacobian(lambda hh: step_fn(base, xin, hh.unsqueeze(0))[0].squeeze(0), g.squeeze(0).detach())
            ev = torch.linalg.eigvals(J.double()); order = torch.argsort(ev.abs(), descending=True)
            row['spectral_radius'] = float(ev.abs().max())
            for rank, idx in enumerate(order[:5].tolist()):
                eigs.append(dict(seed=seed, op=op, state=s, pair=pair, rank=rank + 1, real=float(ev[idx].real), imag=float(ev[idx].imag), abs=float(ev[idx].abs())))
        else:
            row['spectral_radius'] = None
        fps.append(row); print('B', seed, op, s, pair, 'conv', conv, 'out_ok', row['h_star_output_correct'], 'rho', row['spectral_radius'], 'firstneg', first_neg, flush=True)
    assert modelhash(base) == h0
    jl_write(H / 'data' / 'raw' / f'B_seed{seed}_traj.jsonl', traj); jl_write(H / 'data' / 'raw' / f'B_seed{seed}_fp.jsonl', fps); jl_write(H / 'data' / 'raw' / f'B_seed{seed}_eig.jsonl', eigs)
    dump(f'data/raw/B_seed{seed}_run.json', dict(wall=time.perf_counter() - t0, iterations=iters, weights_unchanged=True))


# ------------------------------------------------------------------ estimate / analyze
def stage_estimate():
    torch.set_num_threads(2); a = load('data/raw/A_seed0_run.json'); eq = load('step_equivalence.json')
    a_seed = a['A1_wall'] + a['A2_wall']
    # C cost from A1 cost per batched step (both use the jit forward)
    a1_steps = sum(max(r['seq_len'] for r in jl_read(H / 'data' / 'A1_swapped_probes.jsonl') if r['k'] == k) for k in KS)
    mixed = jl_read(H / 'data' / 'C_mixed_probes.jsonl'); c_steps = 0
    for op in OPS:
        for k in KC:
            bb = [r for r in mixed if r['op'] == op and r['k'] == k]
            for j in range(0, len(bb), 128):
                c_steps += max(r['seq_len'] for r in bb[j:j + 128])
    c_seed = a['A1_wall'] / a1_steps * c_steps
    # B cost: time 2000 eager batch-1 steps
    base = load_eager(0); h = torch.zeros(1, 545); x = torch.tensor([[0., 1., 0.]]); t = time.perf_counter()
    with torch.no_grad():
        for _ in range(2000):
            h, _ = step_fn(base, x, h)
    per = (time.perf_counter() - t) / 2000
    tj = time.perf_counter(); torch.autograd.functional.jacobian(lambda hh: step_fn(base, x, hh.unsqueeze(0))[0].squeeze(0), h.squeeze(0)); jac = time.perf_counter() - tj
    b20000 = 3 * 12 * ((4097 + 20000) * per + jac + 1.0)
    b5000 = 3 * 12 * ((4097 + 5000) * per + jac + 1.0)
    remaining_A = 2 * a_seed; c_all = 3 * c_seed
    total20000 = remaining_A + c_all + b20000
    total5000 = remaining_A + c_all + b5000
    iters = 20000 if total20000 <= 2400 else 5000 if total5000 <= 2400 else 0
    out = dict(seed0_A_wall=a_seed, remaining_A_estimate=remaining_A, C_estimate_3seeds=c_all, B_estimate_20000=b20000, B_estimate_5000=b5000,
               eager_step_seconds=per, jacobian_seconds=jac, total_estimate_20000=total20000, total_estimate_5000=total5000, threshold_seconds=2400,
               B_fixed_point_iterations=iters, decision={20000: 'B は 20,000回で実行', 5000: 'B は 5,000回に削減', 0: 'B は実行しない'}[iters],
               note='見積もりは seed0 A 後の残り（A seed1・2、C、B）。並行して main_eval.py が稼働中（競合込み）。')
    dump('estimate.json', out); print(out)


def stage_analyze():
    status = v054t_status(); inf = float('inf')
    # ---------- A
    A = []
    for s in SEEDS:
        A += jl_read(H / 'data' / 'raw' / f'A_seed{s}.jsonl')
    a1 = [r for r in A if r['experiment'] == 'A1']; a2 = [r for r in A if r['experiment'] == 'A2']
    csvout('swap_results.csv', a1 + a2)
    g = defaultdict(list)
    for r in a1:
        g[(r['seed'], r['state'], r['pair'], r['k'])].append(r)
    swap_table = []; s3 = []; swap_break = {}
    for s in SEEDS:
        for st in (0, 1):
            for x, y in HAND[('ADD', st)]:
                pair = f'{x}{y}'; offs = []; bk = None
                for k in KS:
                    rr = g[(s, st, pair, k)]; assert len(rr) == 16; ex = sum(r['exact'] for r in rr)
                    if ex < 16 and bk is None:
                        bk = k
                    o = [r['hold_offset'] for r in rr if r['first_error_region'] == 'hold']; offs += o
                    swap_table.append(dict(seed=s, state=st, pair_swapped=pair, k=k, exact=f'{ex}/16', hold_breaks=len(o), hold_break_median=statistics.median(o) if o else None))
                swap_break[(s, st, pair)] = bk
                ref = status[(s, 'ADD', st, pair[::-1])]
                med = statistics.median(offs) if offs else None
                in_range = None if med is None or ref['break_min'] is None else int(ref['break_min'] <= med <= ref['break_max'])
                s3.append(dict(seed=s, swapped_combo=f'c={st}×({x},{y})', reference_V054T_combo=f'c={st}×({y},{x})', swapped_broken=int(bk is not None), swapped_breaking_k=bk,
                               swapped_hold_break_min=min(offs) if offs else None, swapped_hold_break_median=med, swapped_hold_break_max=max(offs) if offs else None,
                               reference_broken=int(not ref['unbroken']), reference_break_min=ref['break_min'], reference_break_max=ref['break_max'],
                               status_matches=int((bk is not None) == (not ref['unbroken'])), median_in_reference_range=in_range))
    s1_cells = [r for r in swap_table if r['seed'] == 2]; s1 = '支持' if all(r['exact'] == '16/16' for r in s1_cells) else '不支持'
    fails2 = [r for r in a2 if r['seed'] == 2 and r['original_exact'] == 0]; s2n = sum(r['swapped_exact'] for r in fails2)
    s2 = '支持' if s2n == len(fails2) == 3 else '不支持' if s2n == 0 else '判定不能'
    summ = []
    for s in SEEDS:
        rr = [r for r in a2 if r['seed'] == s]
        summ.append(dict(seed=s, n=len(rr), original_correct=sum(r['original_exact'] for r in rr), swapped_correct=sum(r['swapped_exact'] for r in rr),
                         gained_by_swap=sum(1 for r in rr if r['swapped_exact'] and not r['original_exact']), lost_by_swap=sum(1 for r in rr if r['original_exact'] and not r['swapped_exact']),
                         either_correct_upper_bound=sum(1 for r in rr if r['original_exact'] or r['swapped_exact']),
                         either_correct_rate=sum(1 for r in rr if r['original_exact'] or r['swapped_exact']) / len(rr),
                         note='either_correct は正解を見て選ぶ上限値。実際に使うには検算器が必要'))
    csvout('swap_2048_summary.csv', summ)
    csvout('data/swap_table.csv', swap_table); csvout('data/swap_S3.csv', s3)
    # ---------- C
    C = []
    for s in SEEDS:
        C += jl_read(H / 'data' / 'raw' / f'C_seed{s}.jsonl')
    csvout('mixed_hold_results.csv', C)
    csvout('mixed_hold_context.csv', [r for r in C if r['first_error'] is not None])
    gc = defaultdict(list)
    for r in C:
        gc[(r['seed'], r['op'], r['state'], r['type'], r['k'])].append(r)
    mtable = []
    for (s, op, st, typ, k), rr in sorted(gc.items()):
        assert len(rr) == 16; o = [r['first_error_minus_hold_start'] for r in rr if r['first_error'] is not None]
        mtable.append(dict(seed=s, op=op, state=st, type=typ, k=k, exact=f"{sum(r['exact'] for r in rr)}/16",
                           first_error_minus_hold_start_median=statistics.median(o) if o else None, min=min(o) if o else None, max=max(o) if o else None,
                           regions=';'.join(f'{a}:{b}' for a, b in sorted(Counter(r['first_error_region'] for r in rr if not r['exact']).items()))))
    csvout('data/mixed_table.csv', mtable)

    def mverdict(s, op, st, typ, lo, hi):
        rr = gc[(s, op, st, typ, 64)]; nf = sum(1 - r['exact'] for r in rr)
        med = statistics.median([r['first_error_minus_hold_start'] for r in rr if r['first_error'] is not None]) if nf else None
        v = '支持' if nf == 16 and lo <= med <= hi else '不支持' if nf == 0 else '判定不能'
        return dict(failed=nf, median=med, range=[lo, hi], verdict=v)
    M1 = mverdict(1, 'SUB', 1, 'M_rand', 8, 30); M2 = mverdict(0, 'ADD', 1, 'M_alt12', 35, 60)
    m3 = defaultdict(list)
    for r in C:
        if r['first_error'] is not None:
            m3[(r['seed'], r['op'], r['state'], r['type'])].append(r)
    M3 = []
    for key, rr in sorted(m3.items()):
        M3.append(dict(seed=key[0], op=key[1], state=key[2], type=key[3], failures=len(rr),
                       preceding_identical_pair_run_median=statistics.median(r['preceding_identical_pair_run'] for r in rr),
                       longest_identical_pair_run_in_state_stretch_median=statistics.median(r['longest_identical_pair_run_in_state_stretch'] for r in rr),
                       preceding_same_state_run_median=statistics.median(r['preceding_same_state_run'] for r in rr),
                       preceding_same_state_run_min=min(r['preceding_same_state_run'] for r in rr), preceding_same_state_run_max=max(r['preceding_same_state_run'] for r in rr),
                       fe_in_hold=sum(r['first_error_region'] == 'hold' for r in rr), fe_state_equals_hold_state=sum(r['fe_state'] == r['state'] for r in rr)))
    csvout('data/mixed_M3.csv', M3)
    # ---------- B
    out = dict(A=dict(S1=s1, S2=dict(n_failures=len(fails2), correct_after_swap=s2n, verdict=s2), swap_2048=summ), C=dict(M1=M1, M2=M2))
    if all((H / 'data' / 'raw' / f'B_seed{s}_fp.jsonl').exists() for s in SEEDS):
        traj = []; fps = []; eigs = []
        for s in SEEDS:
            traj += jl_read(H / 'data' / 'raw' / f'B_seed{s}_traj.jsonl'); fps += jl_read(H / 'data' / 'raw' / f'B_seed{s}_fp.jsonl'); eigs += jl_read(H / 'data' / 'raw' / f'B_seed{s}_eig.jsonl')
        for r in fps:
            ref = status[(r['seed'], r['op'], r['state'], r['pair'])]
            r.update(V054T_unbroken=int(ref['unbroken']), V054T_break_min=ref['break_min'], V054T_break_max=ref['break_max'])
            if ref['unbroken']:
                r['D2_holds'] = int(r['converged'] == 1 and r['h_star_output_correct'] == 1 and r['spectral_radius'] is not None and r['spectral_radius'] < 1)
                r['D1_holds'] = None
            else:
                r['D2_holds'] = int(r['converged'] == 1 and r['h_star_output_correct'] == 0)
                fn = r['first_negative_margin_t']
                r['D1_holds'] = int(fn is not None and ref['break_min'] - 3 <= fn <= ref['break_max'] + 3)
            # margin decline speed: mean drop per step from t=0 to the first negative (or to t=64)
            end = r['first_negative_margin_t'] if r['first_negative_margin_t'] is not None else None
            r['margin_drop_per_step_to_first_negative'] = (r['margin_at_t0'] - 0.0) / end if end else None
        csvout('trajectory.csv', traj); csvout('fixed_points.csv', fps); csvout('jacobian_eigs.csv', eigs)
        broken = [r for r in fps if not r['V054T_unbroken']]; d1n = sum(r['D1_holds'] for r in broken)
        D1 = '支持' if d1n == len(broken) else '不支持' if d1n == 0 else '判定不能'
        bad = [r for r in fps if not r['D2_holds']]
        D2 = '支持' if not bad else '不支持'
        out['B'] = dict(D1=dict(broken_cases=len(broken), holds=d1n, verdict=D1), D2=dict(cases=len(fps), failing=len(bad), verdict=D2,
                        non_converged=sum(1 - r['converged'] for r in fps),
                        failing_breakdown=[dict(seed=r['seed'], op=r['op'], state=r['state'], pair=r['pair'], unbroken=r['V054T_unbroken'], converged=r['converged'], out_ok=r['h_star_output_correct'], rho=r['spectral_radius']) for r in bad]))
    dump('data/analysis_summary.json', out); print(json.dumps(out, ensure_ascii=False, indent=1)[:4000])


if __name__ == '__main__':
    st = sys.argv[1]
    {'assets': stage_assets, 'generate': stage_generate, 'repro': stage_repro, 'equiv': stage_equiv, 'estimate': stage_estimate, 'analyze': stage_analyze}.get(st, lambda: None)()
    if st in ('A', 'B', 'C'):
        {'A': stage_A, 'B': stage_B, 'C': stage_C}[st](int(sys.argv[2]))
