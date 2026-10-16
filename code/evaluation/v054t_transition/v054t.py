"""IronMath V054T: state-transition endurance probes on frozen V054 C1/B0 models (inference only).
usage: python -B v054t.py assets|generate|repro|run SEED [OPS]|estimate|analyze
Run with -B so that importing IronMath_V054/model.py writes no __pycache__ into V054."""
import csv, gzip, hashlib, json, random, statistics, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

T = Path(__file__).resolve().parent
WORK = T.parent
V054 = WORK / 'IronMath_V054'
DATA053 = WORK / 'IronMath_V053' / 'data'
sys.dont_write_bytecode = True
sys.path.insert(0, str(V054))
import numpy as np
import torch
from model import Residual, encode  # V054 code, not re-implemented

SEEDS = (0, 1, 2)
KS = (8, 16, 64, 256, 1024, 4096)
OPS = ('ADD', 'SUB')
GEN_SEED = 90000


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def dump(name, x):
    p = T / name; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')


def load(name):
    return json.loads((T / name).read_text(encoding='utf-8'))


def csvout(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (T / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows({k: ('NA' if v is None else v) for k, v in r.items()} for r in rows)


def ckpt(seed):
    return V054 / 'checkpoints' / f'C1_B0_s{seed}' / 'best_validation.pt'


def modelhash(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode()); h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


def load_model(seed):
    cp = torch.load(ckpt(seed), weights_only=False, map_location='cpu')
    assert cp['condition'] == 'C1' and cp['kind'] == 'B0' and cp['seed'] == seed
    base = Residual('B0'); base.load_state_dict(cp['model'])
    m = torch.jit.script(base).eval()
    return m, cp


# ---------------------------------------------------------------- transitions
def step(op, s, x, y):
    """Python-integer transition: returns (output bit, next state)."""
    if op == 'ADD':
        t = x + y + s; return t & 1, t >> 1
    t = x - y - s; return t % 2, int(t < 0)


HAND = {('ADD', 0): [(0, 0), (0, 1), (1, 0)], ('ADD', 1): [(0, 1), (1, 0), (1, 1)],
        ('SUB', 0): [(0, 0), (1, 0), (1, 1)], ('SUB', 1): [(0, 0), (0, 1), (1, 1)]}


def holding_table():
    derived = {(op, s): [(x, y) for x in (0, 1) for y in (0, 1) if step(op, s, x, y)[1] == s] for op in OPS for s in (0, 1)}
    # cross-check with whole-integer arithmetic: a 1-bit operand pair with incoming state s
    for (op, s), pairs in derived.items():
        for x in (0, 1):
            for y in (0, 1):
                if op == 'ADD':
                    nxt = (x + y + s) >> 1
                else:
                    nxt = int(x - y - s < 0)
                assert ((x, y) in pairs) == (nxt == s)
    assert derived == HAND, (derived, HAND)
    return derived


def bits(v, n):
    return [(v >> i) & 1 for i in range(n)]


def ripple(op, a_bits, b_bits):
    """Independent Boolean ripple (gate form). Returns output bits, incoming state per position."""
    s = 0; out = []; inc = []
    for x, y in zip(a_bits, b_bits):
        inc.append(s)
        if op == 'ADD':
            o = x ^ y ^ s; s = (x & y) | (s & (x ^ y))
        else:
            o = x ^ y ^ s; s = ((1 - x) & y) | ((1 - (x ^ y)) & s)
        out.append(o)
    return out, inc, s


def seq_len(op, L):
    return L + 1 if op == 'ADD' else L


# ---------------------------------------------------------------- stages
def stage_assets():
    rep = (V054 / 'REPORT_V054.md').read_text(encoding='utf-8')
    quote = [l for l in rep.splitlines() if l.startswith('B0はV053と同じ残差GRU')][0].split('。')[0] + '。'
    q2 = [l for l in rep.splitlines() if l.startswith('24run')][0]
    files = {f'checkpoint_C1_B0_s{s}': ckpt(s) for s in SEEDS}
    files.update(model_code=V054 / 'model.py', eval_code=V054 / 'run.py', config=V054 / 'config.json',
                 predictions=V054 / 'predictions.jsonl.gz', failure_cases=V054 / 'failure_cases.jsonl.gz',
                 report=V054 / 'REPORT_V054.md',
                 test2048_ADD=DATA053 / 'ADD_Test2048.json', test2048_SUB=DATA053 / 'SUB_Test2048.json')
    for s in SEEDS:  # uniqueness: exactly one C1_B0 checkpoint dir per seed, and its stored metadata matches
        assert len(list((V054 / 'checkpoints').glob(f'C1_B0_s{s}'))) == 1
    out = dict(files={k: dict(path=str(p), sha256=sha(p)) for k, p in files.items()},
               definitions=dict(
                   B0_shared_model=quote,
                   checkpoint_choice=q2 + '（→ best_validation.pt を使用。final.pt は使わない）',
                   inputs='model.py inputs(): B0は3ch目に演算flag(SUB=1)を全位置に入力。length(): ADDはwidth+1、SUBはwidth。LSB-first、forwardでhをゼロ初期化。',
                   evaluation='run.py evaluate(): batch 128、argmax、torch.jit.script(Residual(k)).eval()',
                   test_source='config.json data_source=IronMath_V053/data, {op}_Test2048.json'),
               checkpoint_metadata={})
    for s in SEEDS:
        cp = torch.load(ckpt(s), weights_only=False, map_location='cpu')
        out['checkpoint_metadata'][s] = dict(condition=cp['condition'], kind=cp['kind'], seed=cp['seed'], update=cp['update'], validation_loss=cp['validation_loss'])
    dump('asset_paths.json', out); print(json.dumps(out['checkpoint_metadata']))


def stage_generate():
    table = holding_table()
    rng = random.Random(GEN_SEED)
    prefixes = {}
    for op in OPS:
        for s in (0, 1):
            pre = [dict(id='min', a=([] if s == 0 else ([1] if op == 'ADD' else [0])), b=([] if s == 0 else [1]))]
            draws = 0
            while len(pre) < 4:
                a = rng.getrandbits(16); b = rng.getrandbits(16); draws += 1
                end = (a + b) >> 16 if op == 'ADD' else int(a < b)  # integer arithmetic
                if end == s:
                    pre.append(dict(id=f'r{len(pre)}', a=bits(a, 16), b=bits(b, 16), a_hex=hex(a), b_hex=hex(b)))
            for p in pre:
                assert ripple(op, p['a'], p['b'])[2] == s
            prefixes[(op, s)] = dict(items=pre, draws=draws)
    suffixes = {}
    for op in OPS:
        suffixes[op] = []
        for j in range(4):
            a = rng.getrandbits(16); b = rng.getrandbits(16)
            suffixes[op].append(dict(id=f's{j}', a=bits(a, 16), b=bits(b, 16), a_hex=hex(a), b_hex=hex(b)))
    probes = []
    for op in OPS:
        for s in (0, 1):
            for (x, y) in table[(op, s)]:
                for k in KS:
                    for p in prefixes[(op, s)]['items']:
                        for q in suffixes[op]:
                            A = p['a'] + [x] * k + q['a']; B = p['b'] + [y] * k + q['b']
                            guard = 0
                            if op == 'SUB' and ripple(op, A, B)[2] == 1:
                                A = A + [1]; B = B + [0]; guard = 1
                            L = len(A); a = sum(v << i for i, v in enumerate(A)); b = sum(v << i for i, v in enumerate(B))
                            if op == 'SUB':
                                assert a >= b
                            ans = a + b if op == 'ADD' else a - b
                            n = seq_len(op, L)
                            assert 0 <= ans < (1 << n)
                            out, inc, fin = ripple(op, bits(a, n), bits(b, n))
                            assert out == bits(ans, n) and fin == 0
                            hs = len(p['a']); he = hs + k
                            assert all(inc[i] == s for i in range(hs, he + 1))  # state held through the hold block
                            probes.append(dict(pid=len(probes), op=op, state=s, pair=f'{x}{y}', k=k, prefix=p['id'], suffix=q['id'],
                                               L=L, seq_len=n, hold_start=hs, hold_end=he, suffix_start=he, guard=guard,
                                               a=hex(a), b=hex(b), answer=hex(ans)))
    assert len(probes) == 2 * 576
    pdir = T / 'data'; pdir.mkdir(exist_ok=True)
    with (pdir / 'probes.jsonl').open('w', encoding='utf-8', newline='\n') as f:
        for r in probes:
            f.write(json.dumps(r) + '\n')
    meta = dict(generation_seed=GEN_SEED, holding_table={f'{op}_state{s}': [f'{x}{y}' for x, y in v] for (op, s), v in table.items()},
                holding_table_check='step()の整数演算から導出し、手書き表と一致をassert',
                prefixes={f'{op}_state{s}': v for (op, s), v in prefixes.items()}, suffixes=suffixes,
                n_probes={op: sum(r['op'] == op for r in probes) for op in OPS},
                guard_bits_added=sum(r['guard'] for r in probes),
                checks=['SUB a>=b (Python int)', 'answer = Python int a±b', 'independent Boolean ripple bits == answer bits', 'oracle state held over hold block'],
                probes_sha256=sha(pdir / 'probes.jsonl'))
    dump('data/probe_meta.json', meta)
    (pdir / 'SHA256SUMS.txt').write_text(f"{sha(pdir / 'probes.jsonl')}  probes.jsonl\n{sha(T / 'data' / 'probe_meta.json')}  probe_meta.json\n", encoding='utf-8')
    print('probes', meta['n_probes'], 'guard', meta['guard_bits_added'], meta['probes_sha256'])


def executed_ks():
    p = T / 'estimate.json'
    if p.exists() and json.loads(p.read_text(encoding='utf-8'))['estimated_total_wall_seconds_3seeds'] > 1800:
        return [k for k in KS if k != 4096]
    return list(KS)


def make_x(op, a, b, n):
    x = np.zeros((n, 3), np.float32)
    x[:, 0] = bits(a, n); x[:, 1] = bits(b, n); x[:, 2] = int(op == 'SUB')
    return x


@torch.no_grad()
def predict(m, items):
    """items: list of (op, a, b, n). Zero-padded batch (as in V054 batch()); causal so padding does not affect outputs."""
    nmax = max(n for *_, n in items); x = np.zeros((len(items), nmax, 3), np.float32)
    for i, (op, a, b, n) in enumerate(items):
        x[i, :n] = make_x(op, a, b, n)
    z = m(torch.from_numpy(x)); assert torch.isfinite(z).all()
    p = z.argmax(-1).numpy()
    return [[int(v) for v in p[i, :n]] for i, (*_, n) in enumerate(items)]


def stage_repro():
    torch.set_num_threads(2)
    want = {}
    with gzip.open(V054 / 'predictions.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            if '"width": 2048' not in line or '"model": "B0"' not in line or '"condition": "C1"' not in line:
                continue
            p = json.loads(line); want[(p['seed'], p['op'], p['id'])] = p
    res = dict(rows=[], all_match=True)
    for s in SEEDS:
        m, cp = load_model(s); h0 = modelhash(m)
        for op in OPS:
            rows = json.loads((DATA053 / f'{op}_Test2048.json').read_text())[:16]
            got = predict(m, [(op, r['a'], r['b'], 2049 if op == 'ADD' else 2048) for r in rows])
            mism = [r['id'][:40] for r, g in zip(rows, got) if g != want[(s, op, r['id'])]['prediction_digits']]
            ok = not mism; res['all_match'] &= ok
            res['rows'].append(dict(seed=s, op=op, n=16, bit_identical=16 - len(mism), mismatched_ids=mism))
        assert modelhash(m) == h0
        print('seed', s, [r for r in res['rows'] if r['seed'] == s])
    res['note'] = 'Test2048 先頭16問を独立のbatch(16)で再実行し、predictions.jsonl.gz の prediction_digits と bit列比較'
    dump('reproduction_check.json', res)
    if not res['all_match']:
        print('REPRODUCTION MISMATCH -> STOP'); sys.exit(2)


def stage_run(seed, ops):
    torch.set_num_threads(2)
    probes = [json.loads(l) for l in (T / 'data' / 'probes.jsonl').read_text(encoding='utf-8').splitlines()]
    ks = executed_ks()
    m, cp = load_model(seed); fsha0 = sha(ckpt(seed)); h0 = modelhash(m)
    timing = {}
    for op in ops:
        t0 = time.perf_counter(); c0 = time.process_time(); rows = []
        for k in ks:
            batch = [r for r in probes if r['op'] == op and r['k'] == k]
            preds = predict(m, [(op, int(r['a'], 16), int(r['b'], 16), r['seq_len']) for r in batch])
            for r, p in zip(batch, preds):
                n = r['seq_len']; truth = bits(int(r['answer'], 16), n)
                wrong = [i for i in range(n) if p[i] != truth[i]]
                longest = run = 0; prev = None
                for i in wrong:
                    run = run + 1 if prev == i - 1 else 1; longest = max(longest, run); prev = i
                fe = wrong[0] if wrong else None
                region = None if fe is None else 'prefix' if fe < r['hold_start'] else 'hold' if fe < r['hold_end'] else 'suffix'
                pred_int = sum(v << i for i, v in enumerate(p))
                rows.append(dict(seed=seed, pid=r['pid'], op=op, state=r['state'], pair=r['pair'], k=k, prefix=r['prefix'], suffix=r['suffix'],
                                 seq_len=n, hold_start=r['hold_start'], hold_end=r['hold_end'], guard=r['guard'],
                                 exact=int(pred_int == int(r['answer'], 16)), first_error=fe, first_error_region=region,
                                 hold_offset=(fe - r['hold_start']) if region == 'hold' else None,
                                 wrong_bits=len(wrong), longest_wrong_run=longest, prediction=hex(pred_int)))
        timing[op] = dict(wall=time.perf_counter() - t0, cpu=time.process_time() - c0, n=len(rows))
        (T / 'data' / 'raw').mkdir(parents=True, exist_ok=True)
        with (T / 'data' / 'raw' / f'seed{seed}_{op}.jsonl').open('w', encoding='utf-8', newline='\n') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        print('seed', seed, op, timing[op], 'exact', sum(r['exact'] for r in rows), '/', len(rows), flush=True)
    assert modelhash(m) == h0 and sha(ckpt(seed)) == fsha0
    rec = dict(seed=seed, ops=list(ops), k=list(ks), timing=timing, weights_state_dict_sha256=h0, checkpoint_file_sha256=fsha0, weights_unchanged=True)
    dump(f'data/raw/run_seed{seed}_{"_".join(ops)}.json', rec)


def stage_estimate():
    r0 = load('data/raw/run_seed0_ADD_SUB.json')
    per_seed = sum(v['wall'] for v in r0['timing'].values())
    est = per_seed * 3
    out = dict(seed0_wall_seconds={op: v['wall'] for op, v in r0['timing'].items()}, estimated_total_wall_seconds_3seeds=est,
               remaining_estimate_seconds=per_seed * 2, threshold_seconds=1800,
               decision='全k(4096含む)で残り実行' if est <= 1800 else 'k=4096を除外して実行',
               note='並行してV055学習とmain_eval.pyが稼働中のため、wallは競合込みの実測')
    dump('estimate.json', out); print(out)


def oracle_context(op, a, b, n, fe):
    A = bits(a, n); B = bits(b, n); _, inc, _ = ripple(op, A, B)
    key = (inc[fe], A[fe], B[fe]); run = 0; j = fe - 1
    while j >= 0 and (inc[j], A[j], B[j]) == key:
        run += 1; j -= 1
    hold = (A[fe], B[fe]) in HAND[(op, inc[fe])]
    srun = 0; j = fe - 1
    while j >= 0 and inc[j] == inc[fe]:
        srun += 1; j -= 1
    return inc[fe], f'{A[fe]}{B[fe]}', run, hold, srun


def stage_analyze():
    ks = executed_ks()
    rows = []
    for s in SEEDS:
        for op in OPS:
            rows += [json.loads(l) for l in (T / 'data' / 'raw' / f'seed{s}_{op}.jsonl').read_text(encoding='utf-8').splitlines()]
    csvout('results_per_problem.csv', rows)
    g = defaultdict(list)
    for r in rows:
        g[(r['op'], r['state'], r['pair'], r['k'], r['seed'])].append(r)
    table = []; breaking = {}
    for op in OPS:
        for st in (0, 1):
            for x, y in HAND[(op, st)]:
                pair = f'{x}{y}'
                for s in SEEDS:
                    bk = None
                    for k in ks:
                        rr = g[(op, st, pair, k, s)]
                        if sum(r['exact'] for r in rr) < len(rr) and bk is None:
                            bk = k
                    breaking[(op, st, pair, s)] = bk
                for k in ks:
                    row = dict(op=op, state=st, pair=pair, k=k)
                    for s in SEEDS:
                        rr = g[(op, st, pair, k, s)]; assert len(rr) == 16
                        row[f'seed{s}_exact'] = f"{sum(r['exact'] for r in rr)}/16"
                        offs = [r['hold_offset'] for r in rr if r['first_error_region'] == 'hold']
                        row[f'seed{s}_hold_breaks'] = len(offs)
                        row[f'seed{s}_hold_break_median_step'] = statistics.median(offs) if offs else None
                        row[f'seed{s}_first_error_regions'] = ';'.join(f'{k2}:{v}' for k2, v in sorted(Counter(r['first_error_region'] for r in rr if not r['exact']).items()))
                    row['three_seeds'] = ' / '.join(row[f'seed{s}_exact'] for s in SEEDS)
                    table.append(row)
    for row in table:
        for s in SEEDS:
            row[f'seed{s}_breaking_k'] = breaking[(row['op'], row['state'], row['pair'], s)]
    csvout('transition_table.csv', table)
    # T1
    inf = float('inf'); t1 = {}
    for s in SEEDS:
        tgt = breaking[('SUB', 1, '11', s)]; tv = inf if tgt is None else tgt
        others = {f'br{st}x{p}': breaking[('SUB', st, p, s)] for st in (0, 1) for p in [f'{x}{y}' for x, y in HAND[('SUB', st)]] if (st, p) != (1, '11')}
        ok = tv < inf and all(tv < (inf if v is None else v) for v in others.values())
        t1[s] = dict(target_breaking_k=tgt, others_breaking_k=others, holds=ok)
    n_ok = sum(v['holds'] for v in t1.values())
    t1_verdict = '支持' if n_ok == 3 else '不支持' if n_ok == 0 else '判定不能'
    # T2
    t2 = {f'{st}x{p}_seed{s}': breaking[('ADD', st, p, s)] for st in (0, 1) for p in [f'{x}{y}' for x, y in HAND[('ADD', st)]] for s in SEEDS}
    # V054 failure context (C1 B0, width 2048)
    fails = []
    with gzip.open(V054 / 'failure_cases.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            if '"model": "B0"' not in line or '"condition": "C1"' not in line or '"width": 2048' not in line:
                continue
            p = json.loads(line)
            fe = p['first_error']
            if fe is None:
                fails.append(dict(seed=p['seed'], op=p['op'], category=p['category'], first_error=None)); continue
            st, pair, run, hold, srun = oracle_context(p['op'], p['a'], p['b'], p['length'], fe)
            fails.append(dict(seed=p['seed'], op=p['op'], id=p['id'][:24], category=p['category'], attributes='|'.join(p['attributes']),
                              first_error=fe, state_before=st, pair=pair, combo=f'{"c" if p["op"] == "ADD" else "br"}={st}×({pair[0]},{pair[1]})',
                              combo_is_holding=int(hold), preceding_same_combo_run=run, preceding_same_state_run_supplementary=srun, v054_first_error_phase=p['first_error_phase']))
    csvout('v054_failure_context.csv', fails)
    dist = defaultdict(Counter); cat = defaultdict(Counter); runs = defaultdict(list)
    for r in fails:
        if r['first_error'] is None:
            continue
        dist[(r['op'], r['seed'])][r['combo']] += 1; cat[(r['op'], r['category'])][r['combo']] += 1; runs[(r['op'], r['combo'])].append(r['preceding_same_combo_run'])
    summary = dict(
        T1=dict(per_seed=t1, verdict=t1_verdict),
        T2=t2,
        breaking=[dict(op=o, state=st, pair=p, seed=s, breaking_k=v) for (o, st, p, s), v in breaking.items()],
        failure_combo_by_seed={f'{o}_seed{s}': dict(c) for (o, s), c in sorted(dist.items())},
        failure_combo_by_category={f'{o}_{c}': dict(v) for (o, c), v in sorted(cat.items())},
        failure_run_stats={f'{o}_{c}': dict(n=len(v), median=statistics.median(v), max=max(v), min=min(v)) for (o, c), v in sorted(runs.items())},
        failures_without_first_error=sum(r['first_error'] is None for r in fails),
        n_failures={f'{o}_seed{s}': sum(1 for r in fails if r['op'] == o and r['seed'] == s) for o in OPS for s in SEEDS})
    dump('data/analysis_summary.json', summary)
    print(json.dumps(summary['T1'], ensure_ascii=False)); print('T1 verdict', t1_verdict)


if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'assets': stage_assets()
    elif st == 'generate': stage_generate()
    elif st == 'repro': stage_repro()
    elif st == 'run': stage_run(int(sys.argv[2]), sys.argv[3].split(',') if len(sys.argv) > 3 else list(OPS))
    elif st == 'estimate': stage_estimate()
    elif st == 'analyze': stage_analyze()
