"""IronMath D11: decimal addition, training-distribution intervention on digit-sum-9 pairs.
Derived from decimal_d10/d10.py (copy in from_d10/, diff in code_diff_d10_to_d11.txt). Model: model_d11.py = unchanged copy of model_d10.py.
usage (always python -B):
  python -B d11.py gen_train | train GROUP SEED | gen_test | eval GROUP SEED | probes GROUP SEED | equiv GROUP SEED | fixed GROUP SEED | analyze
Only IronMath_D11/ is written. decimal_d10/ is read only (analyze reads decimal_d10/summary.csv for the descriptive D10 comparison)."""
import csv, hashlib, json, os, random, statistics, sys, time
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
WORK = D.parent
D10 = WORK / 'decimal_d10'
sys.dont_write_bytecode = True
sys.path.insert(0, str(D))
import numpy as np
import torch
from torch.nn import functional as F
from model_d11 import Residual

CFG = json.loads((D / 'PREREGISTRATION.json').read_text(encoding='utf-8'))
D10CFG = json.loads((D / 'from_d10' / 'config.json').read_text(encoding='utf-8'))
SEEDS = tuple(CFG['training']['seeds'])
GROUPS = ('control', 'intervention')
WIDTHS = CFG['test']['widths']
KS = CFG['probes']['k']
THREADS = CFG['training']['threads']
PASS = [(x, 9 - x) for x in range(10)]
NONPASS_ALL = [(x, y) for x in range(10) for y in range(10) if x + y != 9]  # 90 pairs
NONPASS = [(0, (0, 0)), (0, (4, 4)), (0, (8, 0)), (1, (9, 9)), (1, (5, 5)), (1, (1, 9))]
COMBOS = [(c, p) for c in (0, 1) for p in PASS] + NONPASS  # 26, same as D10
P2CATS = ('c0_mixed', 'c1_mixed', 'c0_single', 'c1_single')


# ------------------------------------------------------------------ io helpers (from d10)
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def dump(name, x):
    p = D / name; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')


def load(name):
    return json.loads((D / name).read_text(encoding='utf-8'))


def csvout(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (D / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows({k: ('NA' if v is None else v) for k, v in r.items()} for r in rows)


def readcsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def jl_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def jl_read(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


# ------------------------------------------------------------------ concurrency
def concurrency(stage):
    """record python processes; abort if a python process that is not part of D11 (d11.py / audit.py lanes) is running"""
    import psutil
    me = psutil.Process(); mine = {me.pid} | {p.pid for p in me.parents()}
    rec = dict(stage=stage, time=now(), own_pid=os.getpid())
    procs = []; foreign = []
    for p in psutil.process_iter(['pid', 'name']):
        n = (p.info['name'] or '')
        if 'python' in n.lower() or 'julia' in n.lower():
            try:
                cl = ' '.join(p.cmdline())
            except Exception:
                cl = '?'
            d11 = p.info['pid'] in mine or 'd11.py' in cl or 'audit.py' in cl
            procs.append(dict(name=n, pid=p.info['pid'], d11=d11))
            if not d11:
                foreign.append(dict(name=n, pid=p.info['pid'], cmdline=cl))
    rec['python_processes'] = procs; rec['foreign'] = foreign
    log = D / 'logs' / 'concurrency'; log.mkdir(parents=True, exist_ok=True)
    (log / f"{stage}_{os.getpid()}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
    if foreign:
        raise SystemExit(f'STOP: other experiment process running {foreign}')
    print('concurrency', stage, 'python procs', len(procs), 'foreign 0', flush=True)


# ------------------------------------------------------------------ decimal helpers (from d10)
def digs(v, n):
    s = str(v)[::-1]; assert len(s) <= n, (len(s), n)
    return [int(c) for c in s] + [0] * (n - len(s))


def from_digs(dd):
    return int(''.join(str(d) for d in reversed(dd))) if dd else 0


def ripple(A, B):
    """independent digit-wise decimal ripple; returns out digits, incoming carry per position, final carry"""
    c = 0; out = []; inc = []
    for x, y in zip(A, B):
        inc.append(c); t = x + y + c; out.append(t % 10); c = t // 10
    return out, inc, c


def annotate(a, b, W):
    n = W + 1; A = digs(a, n); B = digs(b, n); out, inc, fin = ripple(A, B)
    assert fin == 0 and out == digs(a + b, n)
    carry_out = inc[1:] + [fin]
    longest_carry = run = 0
    for c in carry_out[:W]:
        run = run + 1 if c else 0; longest_carry = max(longest_carry, run)
    nine = {0: 0, 1: 0}; cur = {0: 0, 1: 0}
    for i in range(W):
        for s in (0, 1):
            cur[s] = cur[s] + 1 if (A[i] + B[i] == 9 and inc[i] == s) else 0; nine[s] = max(nine[s], cur[s])
    return dict(longest_carry=longest_carry, longest_pass_run_c1=nine[1], longest_pass_run_c0=nine[0], carries=sum(carry_out[:W]))


def key(a, b):
    return (min(a, b), max(a, b))


def to_store(r):
    return {k: (str(v) if k in ('a', 'b', 'answer') else v) for k, v in r.items()}


def from_store(r):
    return {k: (int(v) if k in ('a', 'b', 'answer') else v) for k, v in r.items()}


def group5(r, W):
    """preregistered aggregation of Part-1 problems into the five groups (exclusive, in this order)"""
    if r['category'] == 'random':
        return 'random'
    if r['category'] in ('propagation', 'long_propagation'):
        return 'pass_run_c1'
    if 2 * r['longest_pass_run_c1'] >= W:
        return 'hold_c1'
    if 2 * r['longest_pass_run_c0'] >= W:
        return 'hold_c0'
    return 'other_structured'


# ------------------------------------------------------------------ Train / Validation (the intervention)
def draw_operands(rng, group, W=8):
    """one problem: per digit position a pair (a_i,b_i). control: uniform over 100 pairs.
    intervention: with prob q from the 10 sum-9 pairs (uniform), else from the 90 others (uniform)."""
    A = []; B = []
    for _ in range(W):
        if group == 'control':
            p = rng.randrange(100); x, y = p // 10, p % 10
        else:
            x, y = PASS[rng.randrange(10)] if rng.random() < CFG['data']['intervention_q'] else NONPASS_ALL[rng.randrange(90)]
        A.append(x); B.append(y)
    return from_digs(A), from_digs(B)


def check_rows(rows, W):
    for r in rows:
        n = W + 1; out, _, fin = ripple(digs(r['a'], n), digs(r['b'], n)); assert fin == 0 and out == digs(r['answer'], n) and r['answer'] == r['a'] + r['b']


def train_path(group, seed):
    return D / 'data' / 'train' / f'Train_{group}_s{seed}.json'


def stage_gen_train():
    concurrency('gen_train')
    dd = D / 'data'; (dd / 'train').mkdir(parents=True, exist_ok=True)
    assert not (dd / 'Validation.json').exists(), 'no regeneration'
    tv = CFG['data']
    # 1. common Validation: D10 distribution (uniform a,b), Random(62000), unordered-key dedup within
    rng = random.Random(tv['validation_seed']); val = []; used = set()
    while len(val) < tv['validation_n']:
        a, b = rng.randrange(10**8), rng.randrange(10**8); k = key(a, b)
        if k in used:
            continue
        used.add(k); val.append(dict(id=f'D11_Validation_{len(val):05d}', width=8, a=a, b=b, answer=a + b))
    check_rows(val, 8); valkeys = set(used)
    (dd / 'Validation.json').write_text(json.dumps(val), encoding='utf-8'); names = ['Validation.json']
    stats = dict(validation=dict(n=len(val), seed=tv['validation_seed'], pass_pair_fraction=pass_frac(val)))
    # 2. Train per group x seed, excluding Validation keys and duplicates within
    for group in GROUPS:
        for seed in SEEDS:
            gs = tv['train_seed_base'][group] + seed; rng = random.Random(gs); rows = []; used = set(valkeys); rej_val = rej_dup = 0
            while len(rows) < tv['train_n']:
                a, b = draw_operands(rng, group); k = key(a, b)
                if k in used:
                    if k in valkeys:
                        rej_val += 1
                    else:
                        rej_dup += 1
                    continue
                used.add(k); rows.append(dict(id=f'D11_Train_{group}_s{seed}_{len(rows):05d}', width=8, a=a, b=b, answer=a + b))
            check_rows(rows, 8)
            assert not ({key(r['a'], r['b']) for r in rows} & valkeys)
            p = train_path(group, seed); p.write_text(json.dumps(rows), encoding='utf-8'); names.append(f'train/{p.name}')
            stats[f'{group}_s{seed}'] = dict(n=len(rows), seed=gs, rejected_validation_overlap=rej_val, rejected_duplicate=rej_dup, pass_pair_fraction=pass_frac(rows),
                                             sha256=sha(p))
            print('Train', group, seed, stats[f'{group}_s{seed}'], flush=True)
    stats['generated_at'] = now()
    dump('data/train_generation_meta.json', stats)
    (dd / 'SHA256SUMS_train.txt').write_text(''.join(f'{sha(dd / n)}  {n}\n' for n in names), encoding='utf-8')


def pass_frac(rows):
    tot = hit = 0
    for r in rows:
        A = digs(r['a'], 8); B = digs(r['b'], 8); tot += 8; hit += sum(x + y == 9 for x, y in zip(A, B))
    return hit / tot


# ------------------------------------------------------------------ Test part 1 (D10 generator, seed 64000+W)
def gen_test(W, exclude):
    """copy of d10.gen_test with seed 64000+W; `exclude` = Validation and all 20 Train keys (W=8 only); rejections counted per category"""
    rng = random.Random(CFG['test']['part1_seed_base'] + W); LIM = 10**W - 1; used = set(); rows = []; rejected = Counter()

    def add(a, b, cat, extra=None):
        k = key(a, b)
        if k in used:
            return False
        if k in exclude:
            rejected[cat] += 1; return False
        assert 0 <= a <= LIM and 0 <= b <= LIM
        used.add(k); r = dict(id=f'D11P1_W{W}_{len(rows):03d}_{cat}', part=1, width=W, category=cat, a=a, b=b); r.update(extra or {}); rows.append(r); return True

    def rdig(cond):
        pool = [(x, y) for x in range(10) for y in range(10) if cond(x + y)]
        return pool[rng.randrange(len(pool))]

    def from_pairs(pairs):
        return from_digs([p[0] for p in pairs]), from_digs([p[1] for p in pairs])

    def sparse():
        pos = rng.sample(range(W), rng.randrange(1, min(8, W) + 1))
        return sum(rng.randrange(1, 10) * 10**j for j in pos)

    edges = [(0, 0), (0, LIM), (LIM, LIM), (LIM, 1), (LIM, LIM - 1), (1, 1), (0, 1), (LIM - 1, LIM - 1), (LIM - 2, LIM - 2), (LIM - 3, LIM - 3),
             (LIM, LIM - 2), (LIM, LIM - 3), (0, LIM - 1), (0, LIM - 2), (0, 2), (0, 3)]
    for i, (a, b) in enumerate(edges):
        assert add(a, b, 'boundary', dict(edge=f'edge{i:02d}', origin='V050_boundary')), f'deterministic boundary edge{i:02d} excluded -> stop (preregistered)'
    cands = [l for l in [1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] if l <= W - 1]
    for cat, n in [('random', 128), ('propagation', 64), ('alternating', 16), ('sparse', 16), ('dense', 16)]:
        count = attempt = 0
        while count < n:
            attempt += 1; assert attempt < 100000; extra = dict(origin='V050_' + cat)
            if cat == 'random':
                a, b = rng.randrange(10**W), rng.randrange(10**W)
            elif cat == 'propagation':
                P = cands[count % len(cands)]; s = rng.randrange(W - P)  # s in 0..W-1-P
                pairs = [rdig(lambda t: t <= 9) for _ in range(W)]
                pairs[s] = rdig(lambda t: t >= 10)
                for j in range(s + 1, s + P + 1):
                    pairs[j] = rdig(lambda t: t == 9)
                if s + P + 1 < W:
                    pairs[s + P + 1] = rdig(lambda t: t <= 8)
                a, b = from_pairs(pairs); extra['planted_pass_run'] = P
            elif cat == 'alternating':
                A = [0 if j % 2 == 0 else 5 for j in range(W)]; B = [9 if j % 2 == 0 else 4 for j in range(W)]
                for V in (A, B):
                    for _ in range(rng.randrange(1, 4)):
                        j = rng.randrange(W); V[j] = rng.choice([d for d in range(10) if d != V[j]])
                a, b = from_digs(A), from_digs(B)
            else:
                a, b = sparse(), sparse()
                if cat == 'dense':
                    a, b = LIM - a, LIM - b
            if add(a, b, cat, extra):
                count += 1
    for cat, n in [('random', 43), ('no_propagation', 43), ('long_propagation', 43), ('zero_operand', 43), ('extrema', 42), ('boundary', 42)]:
        count = attempt = 0
        while count < n:
            attempt += 1; assert attempt < 100000
            if cat == 'random':
                a, b = rng.randrange(10**W), rng.randrange(10**W)
            elif cat == 'no_propagation':
                a = rng.randrange(10**W); A = digs(a, W); b = from_digs([rng.randrange(0, 10 - x) for x in A])
            elif cat == 'long_propagation':
                k = rng.randrange(W // 2 + 1, W); a = rng.randrange(10**(W - k)) * 10**k + 10**k - 1; b = rng.randrange(1, 10)
            elif cat == 'zero_operand':
                a, b = rng.randrange(10**W), 0
            elif cat == 'extrema':
                a, b = LIM - rng.randrange(9), rng.randrange(10**W)
            else:
                a, b = 10**rng.randrange(W), rng.randrange(10**W)
            a, b = sorted((a, b))
            if add(a, b, cat, dict(origin='V053_' + cat)):
                count += 1
    for r in rows:
        ann = annotate(r['a'], r['b'], W); r.update(ann)
        if r['category'] == 'propagation':
            assert ann['longest_pass_run_c1'] == r['planted_pass_run'] and ann['longest_carry'] == r['planted_pass_run'] + 1, (r, ann)
        if r['category'] == 'no_propagation':
            assert ann['carries'] == 0
        if r['category'] == 'long_propagation':
            assert 2 * ann['longest_pass_run_c1'] >= W
        r['answer'] = r['a'] + r['b']; r['group5'] = group5(r, W)
    assert len(rows) == 512
    assert dict(Counter(r['category'] for r in rows)) == D10CFG['data']['test']['categories'], Counter(r['category'] for r in rows)
    return rows, dict(rejected)


# ------------------------------------------------------------------ Test part 2 (holding sets, seed 65000+W)
def gen_part2(W, exclude):
    rng = random.Random(CFG['test']['part2_seed_base'] + W)
    pre, post = (2, 2) if W == 8 else (8, 8); hold = W - pre - post; assert hold == (4 if W == 8 else W - 16)
    used = set(); rows = []; rejected = Counter(); dup = Counter()
    for cat in P2CATS:
        c = int(cat[1]); count = attempt = 0
        while count < CFG['test']['part2_per_category']:
            attempt += 1; assert attempt < 100000
            while True:  # prefix redrawn until its final carry equals the target
                PA = [rng.randrange(10) for _ in range(pre)]; PB = [rng.randrange(10) for _ in range(pre)]
                if ripple(PA, PB)[2] == c:
                    break
            if cat.endswith('mixed'):
                H = [PASS[rng.randrange(10)] for _ in range(hold)]
            else:
                H = [PASS[count % 10]] * hold  # 32 problems: the 10 pairs in order
            SA = [rng.randrange(10) for _ in range(post)]; SB = [rng.randrange(10) for _ in range(post)]
            A = PA + [h[0] for h in H] + SA; B = PB + [h[1] for h in H] + SB
            a, b = from_digs(A), from_digs(B); k = key(a, b)
            if k in used:
                dup[cat] += 1; continue
            if k in exclude:
                rejected[cat] += 1; continue
            n = W + 1; ans = a + b; out, inc, fin = ripple(digs(a, n), digs(b, n))
            assert out == digs(ans, n) and fin == 0
            assert all(inc[i] == c for i in range(pre, pre + hold + 1)), 'carry not held over the hold block'
            used.add(k)
            r = dict(id=f'D11P2_W{W}_{len(rows):03d}_{cat}', part=2, width=W, category=cat, a=a, b=b, answer=ans, hold_start=pre, hold_end=pre + hold,
                     hold_pair=f'{H[0][0]}{H[0][1]}' if cat.endswith('single') else None)
            r.update(annotate(a, b, W)); rows.append(r); count += 1
    assert len(rows) == 4 * CFG['test']['part2_per_category']
    return rows, dict(rejected), dict(dup)


# ------------------------------------------------------------------ probes (D10 gen_probes, seed 63000)
def gen_probes():
    rng = random.Random(CFG['probes']['seed']); prefixes = {}
    for s in (0, 1):
        pre = []
        while len(pre) < 4:
            a = rng.randrange(10**16); b = rng.randrange(10**16)
            if (a + b) // 10**16 == s:
                pre.append(dict(id=f'p{len(pre)}', a=digs(a, 16), b=digs(b, 16), a_int=str(a), b_int=str(b)))
        for p in pre:
            assert ripple(p['a'], p['b'])[2] == s
        prefixes[s] = pre
    suffixes = []
    for j in range(4):
        a = rng.randrange(10**16); b = rng.randrange(10**16)
        suffixes.append(dict(id=f's{j}', a=digs(a, 16), b=digs(b, 16), a_int=str(a), b_int=str(b)))
    probes = []
    for c, (x, y) in COMBOS:
        assert (x + y + c) // 10 == c
        kind = 'pass' if x + y == 9 else 'nonpass'
        for k in KS:
            for p in prefixes[c]:
                for q in suffixes:
                    A = p['a'] + [x] * k + q['a']; B = p['b'] + [y] * k + q['b']; L = len(A); n = L + 1
                    a = from_digs(A); b = from_digs(B); ans = a + b; assert ans < 10**n
                    out, inc, fin = ripple(digs(a, n), digs(b, n)); assert out == digs(ans, n) and fin == 0
                    hs = 16; he = 16 + k
                    assert all(inc[i] == c for i in range(hs, he + 1))
                    probes.append(dict(pid=len(probes), state=c, pair=f'{x}{y}', kind=kind, k=k, prefix=p['id'], suffix=q['id'], L=L, seq_len=n,
                                       hold_start=hs, hold_end=he, a=str(a), b=str(b), answer=str(ans)))
    assert len(probes) == 26 * len(KS) * 16  # 2080 with 5 k values
    return probes, prefixes, suffixes


def stage_gen_test():
    assert all_trained(), 'Test is generated only after all 20 models are trained'
    concurrency('gen_test')
    dd = D / 'data' / 'test'; assert not dd.exists(), 'no regeneration'; dd.mkdir(parents=True)
    t0 = now(); last_train = max(load(f'checkpoints/{g}_s{s}/training_record.json')['finished_at'] for g in GROUPS for s in SEEDS)
    exclude8 = {key(r['a'], r['b']) for r in json.loads((D / 'data' / 'Validation.json').read_text())}
    for g in GROUPS:
        for s in SEEDS:
            exclude8 |= {key(r['a'], r['b']) for r in json.loads(train_path(g, s).read_text())}
    names = []; meta = dict(test_generation_started=t0, last_training_finished=last_train, exclude8_keys=len(exclude8), widths={})
    for W in WIDTHS:
        ex = exclude8 if W == 8 else set()
        p1, rej1 = gen_test(W, ex); p2, rej2, dup2 = gen_part2(W, ex)
        k1 = {key(r['a'], r['b']) for r in p1}; overlap = [r['id'] for r in p2 if key(r['a'], r['b']) in k1]
        for nm, rows in ((f'Test{W}_part1.json', p1), (f'Test{W}_part2.json', p2)):
            (dd / nm).write_text(json.dumps([to_store(r) for r in rows]), encoding='utf-8'); names.append(nm)
        meta['widths'][W] = dict(part1_seed=CFG['test']['part1_seed_base'] + W, part2_seed=CFG['test']['part2_seed_base'] + W,
                                 part1_rejected_train_validation_overlap=rej1, part2_rejected_train_validation_overlap=rej2,
                                 part2_redrawn_within_part_duplicates=dup2, part1_part2_overlap_not_removed=overlap,
                                 part1_group5_counts=dict(Counter(r['group5'] for r in p1)))
        print('Test', W, meta['widths'][W], flush=True)
    probes, prefixes, suffixes = gen_probes(); jl_write(dd / 'probes.jsonl', probes); names.append('probes.jsonl')
    meta.update(probe_seed=CFG['probes']['seed'], probe_prefixes={str(s): v for s, v in prefixes.items()}, probe_suffixes=suffixes,
                test_generation_finished=now(),
                checks=['Python int answer', 'independent digit-wise decimal ripple == answer digits, final carry 0', 'propagation planted pass run == measured',
                        'no_propagation zero carries', 'long_propagation pass run >= W/2', 'part2 ripple carry == target over hold block',
                        'probe holding (x+y+c)//10==c', 'probe ripple carry == c over hold block'])
    assert meta['test_generation_started'] > last_train
    dump('data/test/generation_meta.json', meta); names.append('generation_meta.json')
    (dd / 'SHA256SUMS.txt').write_text(''.join(f'{sha(dd / n)}  {n}\n' for n in names), encoding='utf-8')


# ------------------------------------------------------------------ model helpers (from d10)
def setup(seed):
    torch.set_num_threads(THREADS); torch.use_deterministic_algorithms(True); torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)


def modelhash(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode()); h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


EYE = np.eye(10, dtype=np.float32)


def encode_digits(A, B):
    return torch.from_numpy(np.concatenate([EYE[A], EYE[B]], -1))


def encode(rows, labels=False):
    n = rows[0]['width'] + 1; assert all(r['width'] + 1 == n for r in rows)
    A = np.array([digs(r['a'], n) for r in rows], np.int64); B = np.array([digs(r['b'], n) for r in rows], np.int64)
    x = encode_digits(A, B)
    if labels:
        return x, torch.from_numpy(np.array([digs(r['a'] + r['b'], n) for r in rows], np.int64))
    return x


@torch.no_grad()
def fit(m, rows):
    correct = tok_ok = tokens = 0; sl = 0.
    for j in range(0, len(rows), 128):
        rr = rows[j:j + 128]; x, y = encode(rr, True); z = m(x); p = z.argmax(-1)
        correct += int((p == y).all(-1).sum()); tok_ok += int((p == y).sum()); tokens += y.numel()
        sl += float(F.cross_entropy(z.flatten(0, 1), y.flatten(), reduction='sum'))
    return dict(loss=sl / tokens, n=len(rows), exact=correct / len(rows), digit_accuracy=tok_ok / tokens)


def ckdir(group, seed):
    return D / 'checkpoints' / f'{group}_s{seed}'


def stage_train(group, seed):
    concurrency(f'train_{group}_s{seed}')
    folder = ckdir(group, seed); assert not folder.exists(), 'no retraining'
    train = json.loads(train_path(group, seed).read_text()); val = json.loads((D / 'data' / 'Validation.json').read_text())
    setup(seed); base = Residual(); nparams = sum(p.numel() for p in base.parameters()); initial = modelhash(base)
    m = torch.jit.script(base); opt = torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.01)
    folder.mkdir(parents=True); started = now()
    for _ in range(3):
        m.zero_grad(); m(torch.zeros(16, 9, 20)).square().mean().backward()
    assert modelhash(m) == initial
    rng = random.Random(50000 + seed); order = hashlib.sha256(); index_order = hashlib.sha256(); step = 0; best = float('inf'); beststep = 0
    cpu = wall = 0.; logs = []; t_all = time.perf_counter(); c_all = time.process_time()

    def save(version, loss=None):
        torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), group=group, seed=seed, update=step, validation_loss=loss, initial_SHA256=initial,
                        order_SHA256=order.hexdigest(), rng_state=rng.getstate(), torch_rng=torch.get_rng_state()), folder / f'{version}.pt')

    while step < 1200:
        rows = []
        for _ in range(16):
            i = rng.randrange(len(train)); r = dict(train[i]); coin = rng.randrange(2)
            if coin:
                r['a'], r['b'] = r['b'], r['a']
            rows.append(r); order.update(f"{r['id']}:{r['a']}:{r['b']}\n".encode()); index_order.update(f'{i}:{coin}\n'.encode())
        x, y = encode(rows, True)
        c = time.process_time(); w = time.perf_counter()
        m.train(); opt.zero_grad(set_to_none=True); z = m(x); loss = F.cross_entropy(z.flatten(0, 1), y.flatten()); assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1); opt.step()
        cpu += time.process_time() - c; wall += time.perf_counter() - w; step += 1
        if step % 200 == 0 or step >= 1200:
            m.eval(); st = fit(m, val); vl = st['loss']
            if vl < best:
                best = vl; beststep = step; save('best_validation', vl)
            logs.append(dict(group=group, seed=seed, update=step, minibatch_loss=float(loss.detach()), validation_loss=vl, validation_exact=st['exact'],
                             validation_digit_accuracy=st['digit_accuracy'], selected_update=beststep, train_CPU_seconds=cpu, train_wall_seconds=wall))
            print('TRAIN', group, seed, step, 'VAL', round(vl, 6), 'exact', round(st['exact'], 4), flush=True)
    save('final')
    cp = torch.load(folder / 'best_validation.pt', map_location='cpu', weights_only=False); m.load_state_dict(cp['model']); m.eval()
    tf = fit(m, train); vf = fit(m, val)
    rec = dict(group=group, seed=seed, parameters=nparams, updates=step, presentations=step * 16, selected_update=beststep, best_validation_loss=best,
               initial_SHA256=initial, order_SHA256=order.hexdigest(), index_coin_order_SHA256=index_order.hexdigest(), selected_model_SHA256=modelhash(m),
               best_checkpoint_file_SHA256=sha(folder / 'best_validation.pt'), final_checkpoint_file_SHA256=sha(folder / 'final.pt'),
               train_file_SHA256=sha(train_path(group, seed)), selected_train_fit=tf, selected_validation_fit=vf, train_CPU_seconds=cpu, train_wall_seconds=wall,
               total_stage_wall_seconds=time.perf_counter() - t_all, total_stage_CPU_seconds=time.process_time() - c_all, threads=torch.get_num_threads(),
               started_at=started, finished_at=now())
    dump(f'checkpoints/{group}_s{seed}/training_record.json', rec); jl_write(folder / 'training_log.jsonl', logs)
    print('trained', group, seed, 'selected', beststep, 'params', nparams, flush=True)


def load_jit(group, seed):
    cp = torch.load(ckdir(group, seed) / 'best_validation.pt', map_location='cpu', weights_only=False)
    base = Residual(); base.load_state_dict(cp['model']); base.eval()
    return torch.jit.script(base).eval(), base, cp


def all_trained():
    return all((ckdir(g, s) / 'training_record.json').exists() for g in GROUPS for s in SEEDS)


def load_test(W, part):
    return [from_store(r) for r in json.loads((D / 'data' / 'test' / f'Test{W}_part{part}.json').read_text())]


def wrong_stats(pred, truth):
    wrong = [i for i, (p, t) in enumerate(zip(pred, truth)) if p != t]
    longest = run = 0; prev = None
    for i in wrong:
        run = run + 1 if prev == i - 1 else 1; longest = max(longest, run); prev = i
    return wrong, longest


def error_shape(pred_int, ans):
    d = pred_int - ans
    if d == 0:
        return None, None, None
    s = str(abs(d)); single = s[0] == '1' and set(s[1:]) <= {'0'}
    return int(single), (len(s) - 1) if single else None, 1 if d > 0 else -1


@torch.no_grad()
def predict(m, x):
    z = m(x); assert torch.isfinite(z).all()
    return z.argmax(-1).numpy()


def stage_eval(group, seed):
    assert all_trained() and (D / 'data' / 'test' / 'SHA256SUMS.txt').exists()
    concurrency(f'eval_{group}_s{seed}'); torch.set_num_threads(THREADS)
    m, base, cp = load_jit(group, seed); h0 = modelhash(base); rows_out = []; timing = {}
    for W in WIDTHS:
        for part in (1, 2):
            rows = load_test(W, part); n = W + 1; t = time.perf_counter(); c = time.process_time()
            for j in range(0, len(rows), 128):
                rr = rows[j:j + 128]; P = predict(m, encode(rr))
                for r, p in zip(rr, P):
                    pred = [int(v) for v in p[:n]]; truth = digs(r['answer'], n); wrong, longest = wrong_stats(pred, truth)
                    pi = from_digs(pred); single, s_exp, sign = error_shape(pi, r['answer'])
                    fe = wrong[0] if wrong else None
                    o = dict(group=group, seed=seed, width=W, part=part, id=r['id'], category=r['category'], group5=r.get('group5'), edge=r.get('edge'),
                             exact=int(pi == r['answer']), digit_accuracy=1 - len(wrong) / n, first_error=fe, wrong_digits=len(wrong), longest_wrong_run=longest,
                             longest_carry=r['longest_carry'], longest_pass_run_c1=r['longest_pass_run_c1'], longest_pass_run_c0=r['longest_pass_run_c0'],
                             error_single_power_of_ten=single, error_exponent=s_exp, error_sign=sign, prediction=None if pi == r['answer'] else str(pi))
                    if part == 2:
                        o.update(hold_start=r['hold_start'], hold_end=r['hold_end'], hold_pair=r['hold_pair'],
                                 first_error_region=None if fe is None else 'prefix' if fe < r['hold_start'] else 'hold' if fe < r['hold_end'] else 'suffix')
                    rows_out.append(o)
            timing[f'W{W}_part{part}'] = dict(wall=time.perf_counter() - t, cpu=time.process_time() - c)
            print('EVAL', group, seed, 'W', W, 'part', part, 'exact', sum(r['exact'] for r in rows_out if r['width'] == W and r['part'] == part), '/', len(rows), flush=True)
    assert modelhash(base) == h0
    jl_write(D / 'data' / 'raw' / f'eval_{group}_s{seed}.jsonl', rows_out)
    dump(f'data/raw/eval_{group}_s{seed}_run.json', dict(timing=timing, weights_sha256=h0, checkpoint_sha256=sha(ckdir(group, seed) / 'best_validation.pt'), finished_at=now()))


def stage_probes(group, seed):
    assert all_trained()
    concurrency(f'probes_{group}_s{seed}'); torch.set_num_threads(THREADS)
    m, base, _ = load_jit(group, seed); h0 = modelhash(base); probes = jl_read(D / 'data' / 'test' / 'probes.jsonl'); out = []; t = time.perf_counter()
    for k in KS:
        batch = [r for r in probes if r['k'] == k]; n = batch[0]['seq_len']; assert all(r['seq_len'] == n for r in batch)
        A = np.array([digs(int(r['a']), n) for r in batch]); B = np.array([digs(int(r['b']), n) for r in batch])
        P = np.concatenate([predict(m, encode_digits(A[j:j + 128], B[j:j + 128])) for j in range(0, len(batch), 128)])
        for r, p in zip(batch, P):
            pred = [int(v) for v in p[:n]]; ans = int(r['answer']); truth = digs(ans, n); wrong, longest = wrong_stats(pred, truth)
            fe = wrong[0] if wrong else None
            region = None if fe is None else 'prefix' if fe < r['hold_start'] else 'hold' if fe < r['hold_end'] else 'suffix'
            pi = from_digs(pred); single, s_exp, sign = error_shape(pi, ans)
            out.append(dict(group=group, seed=seed, pid=r['pid'], state=r['state'], pair=r['pair'], kind=r['kind'], k=k, prefix=r['prefix'], suffix=r['suffix'],
                            exact=int(pi == ans), first_error=fe, first_error_region=region, hold_offset=(fe - r['hold_start']) if region == 'hold' else None,
                            wrong_digits=len(wrong), longest_wrong_run=longest, error_single_power_of_ten=single, error_exponent=s_exp, error_sign=sign))
    assert modelhash(base) == h0
    jl_write(D / 'data' / 'raw' / f'probes_{group}_s{seed}.jsonl', out)
    dump(f'data/raw/probes_{group}_s{seed}_run.json', dict(wall=time.perf_counter() - t, weights_sha256=h0))
    print('PROBES', group, seed, 'exact', sum(r['exact'] for r in out), '/', len(out), flush=True)


def step_fn(base, x, h):
    """One recurrent step. Composition line copied from model_d11.py (= model_d10.py) Residual.forward."""
    h = base.core(base.projection(x), h) + torch.tanh(base.residual_gate) * h
    return h, base.head(h)


@torch.no_grad()
def predict_steps(base, x):
    h = torch.zeros(x.size(0), 545); outs = []
    for t in range(x.size(1)):
        h, z = step_fn(base, x[:, t], h); outs.append(z.argmax(-1))
    return torch.stack(outs, 1).numpy()


def stage_equiv(group, seed):
    """per model: step wrapper vs jit forward on all Part-1 Test problems of widths 8-256"""
    assert all_trained()
    concurrency(f'equiv_{group}_s{seed}'); torch.set_num_threads(THREADS)
    m, base, _ = load_jit(group, seed); h0 = modelhash(base); mis = n = 0; t = time.perf_counter()
    for W in [w for w in WIDTHS if w <= 256]:
        rows = load_test(W, 1)
        for j in range(0, len(rows), 128):
            x = encode(rows[j:j + 128]); a = predict(m, x); b = predict_steps(base, x)
            mis += int((a != b).any(-1).sum()); n += len(a)
    assert modelhash(base) == h0
    res = dict(group=group, seed=seed, problems=n, mismatched_problems=mis, identical=mis == 0, wall_seconds=time.perf_counter() - t,
               wrapper='h = core(projection(x), h) + tanh(residual_gate) * h ; logits = head(h); eager Residual with the same state_dict; compared with jit forward argmax on all Part-1 Test problems of widths 8-256')
    dump(f'data/raw/equiv_{group}_s{seed}.json', res); print(res, flush=True)


def onehot(x, y):
    v = torch.zeros(1, 20); v[0, x] = 1; v[0, 10 + y] = 1
    return v


def stage_fixed(group, seed):
    assert load(f'data/raw/equiv_{group}_s{seed}.json')['identical'], 'step wrapper not equivalent -> fixed-point analysis not run'
    concurrency(f'fixed_{group}_s{seed}'); torch.set_num_threads(THREADS)
    _, base, _ = load_jit(group, seed); h0 = modelhash(base)
    for p in base.parameters():
        p.requires_grad_(False)
    probes = {(r['state'], r['pair']): r for r in jl_read(D / 'data' / 'test' / 'probes.jsonl') if r['k'] == 1024 and r['prefix'] == 'p0' and r['suffix'] == 's0'}
    fps = []; eigs = []; ends = {}; t0 = time.perf_counter()
    for c, (x, y) in COMBOS:
        r = probes[(c, f'{x}{y}')]; n = r['seq_len']
        X = encode_digits(np.array([digs(int(r['a']), n)]), np.array([digs(int(r['b']), n)]))
        h = torch.zeros(1, 545)
        with torch.no_grad():
            for t in range(r['hold_start']):
                h, _ = step_fn(base, X[:, t], h)
            xin = onehot(x, y); assert torch.equal(xin, X[:, r['hold_start']])
            g = h.clone(); moves = []; it = 0; conv = False
            for limit in (3000, 20000):
                while it < limit:
                    gn, _ = step_fn(base, xin, g); moves.append(float((gn - g).norm())); g = gn; it += 1
                conv = max(moves[-100:]) < 1e-6
                if conv:
                    break
            hn, z = step_fn(base, xin, g); out = int(z.argmax(-1)); right = (x + y + c) % 10
            zz = z[0].clone(); m_right = float(zz[right]); zz[right] = -float('inf'); margin = m_right - float(zz.max())
        ends[(c, f'{x}{y}')] = g.clone()
        row = dict(group=group, seed=seed, state=c, pair=f'{x}{y}', kind='pass' if x + y == 9 else 'nonpass', iterations=it, converged=int(conv),
                   last100_move_max=max(moves[-100:]), last100_move_min=min(moves[-100:]), h_star_norm=float(g.norm()),
                   h_star_output=out, correct_output=right, h_star_output_correct=int(out == right), h_star_margin_correct=margin,
                   tanh_residual_gate=float(torch.tanh(base.residual_gate)))
        if conv:
            J = torch.autograd.functional.jacobian(lambda hh: step_fn(base, xin, hh.unsqueeze(0))[0].squeeze(0), g.squeeze(0).detach())
            ev = torch.linalg.eigvals(J.double()); order = torch.argsort(ev.abs(), descending=True)
            row['spectral_radius'] = float(ev.abs().max())
            for rank, idx in enumerate(order[:5].tolist()):
                eigs.append(dict(group=group, seed=seed, state=c, pair=f'{x}{y}', rank=rank + 1, real=float(ev[idx].real), imag=float(ev[idx].imag), abs=float(ev[idx].abs())))
        else:
            row['spectral_radius'] = None
        fps.append(row); print('FP', group, seed, c, x, y, 'it', it, 'conv', conv, 'out', out, 'right', right, 'rho', row['spectral_radius'], flush=True)
    pairs = []
    for x, y in PASS:
        r0 = next(f for f in fps if f['state'] == 0 and f['pair'] == f'{x}{y}'); r1 = next(f for f in fps if f['state'] == 1 and f['pair'] == f'{x}{y}')
        dist = float((ends[(0, f'{x}{y}')] - ends[(1, f'{x}{y}')]).norm())
        both = bool(r0['converged'] and r1['converged'])
        stable = both and r0['spectral_radius'] < 1 and r1['spectral_radius'] < 1
        correct = bool(r0['h_star_output_correct'] and r1['h_star_output_correct'])
        pairs.append(dict(group=group, seed=seed, pair=f'{x}{y}', distance_c0_vs_c1=dist, both_converged=int(both), same_point=int(dist < 1e-3) if both else None,
                          rho_c0=r0['spectral_radius'], rho_c1=r1['spectral_radius'], both_rho_below_1=int(stable),
                          output_c0_side=r0['h_star_output'], correct_c0=9, output_c1_side=r1['h_star_output'], correct_c1=0,
                          c0_side_correct=r0['h_star_output_correct'], c1_side_correct=r1['h_star_output_correct'],
                          two_attractors_4cond=int(both and dist >= 1e-3 and stable and correct)))
    assert modelhash(base) == h0
    jl_write(D / 'data' / 'raw' / f'fixed_{group}_s{seed}.jsonl', fps); jl_write(D / 'data' / 'raw' / f'eigs_{group}_s{seed}.jsonl', eigs)
    jl_write(D / 'data' / 'raw' / f'pairs_{group}_s{seed}.jsonl', pairs)
    dump(f'data/raw/fixed_{group}_s{seed}_run.json', dict(wall=time.perf_counter() - t0, weights_sha256=h0))


# ------------------------------------------------------------------ analyze
def paired_verdict(ctrl, intv):
    """H1/H2 rule: seeds (same number) with Intervention > Control; >=8 支持, <=5 不支持, else 判定不能"""
    wins = sum(intv[s] > ctrl[s] for s in SEEDS)
    return wins, '支持' if wins >= 8 else '不支持' if wins <= 5 else '判定不能'


def bootstrap(diffs, n, seed):
    rng = np.random.default_rng(seed); d = np.array(diffs, float)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stage_analyze():
    concurrency('analyze')
    MODELS = [(g, s) for g in GROUPS for s in SEEDS]
    tl = []; recs = []
    for g, s in MODELS:
        tl += jl_read(ckdir(g, s) / 'training_log.jsonl'); recs.append(load(f'checkpoints/{g}_s{s}/training_record.json'))
    csvout('training_log.csv', tl)
    csvout('training_records.csv', [{k: v for k, v in r.items() if not isinstance(v, dict)} | {'val_exact': r['selected_validation_fit']['exact'], 'train_exact': r['selected_train_fit']['exact']} for r in recs])
    ev = []; pr = []; fp = []; eg = []; pa = []; eq = []
    for g, s in MODELS:
        ev += jl_read(D / 'data' / 'raw' / f'eval_{g}_s{s}.jsonl'); pr += jl_read(D / 'data' / 'raw' / f'probes_{g}_s{s}.jsonl')
        fp += jl_read(D / 'data' / 'raw' / f'fixed_{g}_s{s}.jsonl'); eg += jl_read(D / 'data' / 'raw' / f'eigs_{g}_s{s}.jsonl'); pa += jl_read(D / 'data' / 'raw' / f'pairs_{g}_s{s}.jsonl')
        eq.append(load(f'data/raw/equiv_{g}_s{s}.json'))
    csvout('results_per_problem.csv', ev); csvout('fixed_points.csv', fp); csvout('jacobian_eigs.csv', eg); csvout('pairs.csv', pa)
    csvout('probe_results_per_problem.csv', pr); csvout('step_equivalence.csv', eq)
    # summaries: part1 by D10 category and by five groups; part2 by category
    g_ = defaultdict(list)
    for r in ev:
        if r['part'] == 1:
            for lab in ('ALL', r['category'], 'G5:' + r['group5']):
                g_[(1, r['width'], lab, r['group'], r['seed'])].append(r)
        else:
            for lab in ('ALL', r['category']):
                g_[(2, r['width'], lab, r['group'], r['seed'])].append(r)
    labels1 = ['ALL'] + list(D10CFG['data']['test']['categories']) + ['G5:' + x for x in ('random', 'pass_run_c1', 'hold_c0', 'hold_c1', 'other_structured')]
    summary = []
    for part, labels in ((1, labels1), (2, ['ALL'] + list(P2CATS))):
        for W in WIDTHS:
            for lab in labels:
                if not g_[(part, W, lab, 'control', 0)]:
                    summary.append(dict(part=part, width=W, label=lab, n=0)); continue
                row = dict(part=part, width=W, label=lab, n=len(g_[(part, W, lab, 'control', 0)]))
                for g in GROUPS:
                    rates = []
                    for s in SEEDS:
                        rr = g_[(part, W, lab, g, s)]; rate = sum(r['exact'] for r in rr) / len(rr); rates.append(rate)
                        row[f'{g}_s{s}'] = rate
                    row[f'{g}_mean'] = statistics.mean(rates); row[f'{g}_sd'] = statistics.stdev(rates)
                    row[f'{g}_digit_acc_mean'] = statistics.mean(r['digit_accuracy'] for s in SEEDS for r in g_[(part, W, lab, g, s)])
                    fe = [r['first_error'] for s in SEEDS for r in g_[(part, W, lab, g, s)] if not r['exact']]
                    row[f'{g}_median_first_error'] = statistics.median(fe) if fe else None
                row['diff_mean'] = row['intervention_mean'] - row['control_mean']
                summary.append(row)
    csvout('summary.csv', summary)
    S = {(r['part'], r['width'], r['label']): r for r in summary}
    # H3 (primary): part1 W=2048 ALL
    c3 = {s: S[(1, 2048, 'ALL')][f'control_s{s}'] for s in SEEDS}; i3 = {s: S[(1, 2048, 'ALL')][f'intervention_s{s}'] for s in SEEDS}
    d3 = [i3[s] - c3[s] for s in SEEDS]; m3 = statistics.mean(d3); lo, hi = bootstrap(d3, CFG['hypotheses']['H3']['bootstrap_n'], CFG['hypotheses']['H3']['bootstrap_seed'])
    contains0 = lo <= 0 <= hi
    h3 = '支持' if (m3 >= .05 and lo > 0) else '不支持' if (m3 <= 0 or contains0) else '判定不能'
    # secondary: random category at 2048
    cr = {s: S[(1, 2048, 'random')][f'control_s{s}'] for s in SEEDS}; ir = {s: S[(1, 2048, 'random')][f'intervention_s{s}'] for s in SEEDS}
    dr = [ir[s] - cr[s] for s in SEEDS]; lr, hr = bootstrap(dr, CFG['hypotheses']['H3']['bootstrap_n'], CFG['hypotheses']['H3']['bootstrap_seed'])
    # durability table + H1
    dg = defaultdict(list)
    for r in pr:
        dg[(r['group'], r['seed'], r['state'], r['pair'], r['k'])].append(r)
    dur = []
    for c, (x, y) in COMBOS:
        for k in KS:
            row = dict(state=c, pair=f'{x}{y}', kind='pass' if x + y == 9 else 'nonpass', k=k)
            for g, s in MODELS:
                rr = dg[(g, s, c, f'{x}{y}', k)]; assert len(rr) == 16
                row[f'{g}_s{s}'] = sum(r['exact'] for r in rr)
            for g in GROUPS:
                row[f'{g}_models_16of16'] = sum(row[f'{g}_s{s}'] == 16 for s in SEEDS)
            dur.append(row)
    csvout('durability_table.csv', dur)
    D_ = {(r['state'], r['pair'], r['k']): r for r in dur}
    h1c = {g: {s: sum(D_[(1, f'{x}{y}', 1024)][f'{g}_s{s}'] == 16 for x, y in PASS) for s in SEEDS} for g in GROUPS}
    h1w, h1 = paired_verdict(h1c['control'], h1c['intervention'])
    # durability also for c0 pass pairs (descriptive)
    h1c0 = {g: {s: sum(D_[(0, f'{x}{y}', 1024)][f'{g}_s{s}'] == 16 for x, y in PASS) for s in SEEDS} for g in GROUPS}
    # H2
    h2c = {g: {s: sum(p['two_attractors_4cond'] for p in pa if p['group'] == g and p['seed'] == s) for s in SEEDS} for g in GROUPS}
    h2w, h2 = paired_verdict(h2c['control'], h2c['intervention'])
    pair_detail = {g: {s: dict(two_attractors_4cond=h2c[g][s],
                                distance_ge_1e3_both_converged=sum(1 for p in pa if p['group'] == g and p['seed'] == s and p['same_point'] == 0),
                                not_both_converged=sum(1 for p in pa if p['group'] == g and p['seed'] == s and not p['both_converged'])) for s in SEEDS} for g in GROUPS}
    # seed-level correspondence
    corr = [dict(seed=s, H1_diff=h1c['intervention'][s] - h1c['control'][s], H2_diff=h2c['intervention'][s] - h2c['control'][s], H3_diff=i3[s] - c3[s],
                 random2048_diff=ir[s] - cr[s]) for s in SEEDS]
    csvout('seed_correspondence.csv', corr)
    # error shape (main test, failed)
    shape = defaultdict(lambda: [0, 0])
    for r in ev:
        if not r['exact']:
            shape[(r['group'], r['part'])][0] += 1; shape[(r['group'], r['part'])][1] += r['error_single_power_of_ten']
    # D10 comparison (descriptive): Control part1 vs D10 summary.csv (read only)
    d10 = readcsv(D10 / 'summary.csv'); d10cmp = []
    for W in WIDTHS:
        for lab in ['ALL'] + list(D10CFG['data']['test']['categories']):
            q = next(r for r in d10 if int(r['width']) == W and r['category'] == lab)
            d10cmp.append(dict(width=W, category=lab, D10_mean_3seeds=float(q['mean_exact']), D10_seeds='/'.join(str(round(float(q[f'seed{s}_exact']), 4)) for s in range(3)),
                               D11_control_mean_10seeds=S[(1, W, lab)]['control_mean'], D11_control_seeds0to2='/'.join(str(round(S[(1, W, lab)][f'control_s{s}'], 4)) for s in range(3)),
                               D11_intervention_mean_10seeds=S[(1, W, lab)]['intervention_mean']))
    csvout('d10_comparison.csv', d10cmp)
    res = dict(
        H1=dict(definition='c=1 x sum-9 pairs (10) with k=1024 16/16, per model', control=h1c['control'], intervention=h1c['intervention'], seeds_intervention_greater=h1w, verdict=h1),
        H1_descriptive_c0_pass_pairs=h1c0,
        H2=dict(definition='sum-9 pairs with two attractors (both converged, distance>=1e-3, both rho<1, both outputs correct)', control=h2c['control'], intervention=h2c['intervention'],
                seeds_intervention_greater=h2w, verdict=h2, detail=pair_detail),
        H3=dict(control=c3, intervention=i3, diffs=d3, mean_diff=m3, ci95=[lo, hi], ci_contains_0=contains0, verdict=h3),
        secondary_random2048=dict(control=cr, intervention=ir, diffs=dr, mean_diff=statistics.mean(dr), ci95_descriptive=[lr, hr]),
        seed_correspondence=corr,
        error_shape={f'{g}_part{p}': dict(n_failed=v[0], single_power_of_ten=v[1]) for (g, p), v in shape.items()},
        step_equivalence_all_identical=all(e['identical'] for e in eq),
        fixed_point_converged={g: sum(f['converged'] for f in fp if f['group'] == g) for g in GROUPS}, fixed_point_total_per_group=26 * len(SEEDS))
    dump('analysis_summary.json', res)
    print(json.dumps({k: res[k]['verdict'] for k in ('H1', 'H2', 'H3')}, ensure_ascii=False), 'H3 mean', m3, 'CI', lo, hi)


if __name__ == '__main__':
    st = sys.argv[1]; a = sys.argv[2:]
    if st == 'gen_train': stage_gen_train()
    elif st == 'train': stage_train(a[0], int(a[1]))
    elif st == 'gen_test': stage_gen_test()
    elif st == 'eval': stage_eval(a[0], int(a[1]))
    elif st == 'probes': stage_probes(a[0], int(a[1]))
    elif st == 'equiv': stage_equiv(a[0], int(a[1]))
    elif st == 'fixed': stage_fixed(a[0], int(a[1]))
    elif st == 'analyze': stage_analyze()
    else: raise SystemExit('unknown stage')
