"""IronMath D10: decimal addition, train 8 digits, evaluate up to 2048 digits.
usage (always python -B):
  python -B d10.py preserve_before | assets | generate | train SEED | eval SEED | probes SEED | equiv | fixed SEED | analyze | preserve_after
Only decimal_d10/ is written. IronMath_V055 is read only; m3_checker, IronMath_V057, v054r_replication are never entered."""
import csv, hashlib, json, os, platform, random, statistics, sys, time
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
WORK = D.parent
V055 = WORK / 'IronMath_V055'
sys.dont_write_bytecode = True
sys.path.insert(0, str(D))
import numpy as np
import torch
from torch.nn import functional as F
from model_d10 import Residual

CFG = json.loads((D / 'config.json').read_text(encoding='utf-8'))
SEEDS = (0, 1, 2)
WIDTHS = CFG['data']['test']['widths']
KS = CFG['data']['probes']['k']
EXCLUDED_TOP = {'decimal_d10', 'm3_checker', 'IronMath_V057', 'v054r_replication'}
PASS = [(x, 9 - x) for x in range(10)]
NONPASS = [(0, (0, 0)), (0, (4, 4)), (0, (8, 0)), (1, (9, 9)), (1, (5, 5)), (1, (1, 9))]
COMBOS = [(c, p) for c in (0, 1) for p in PASS] + NONPASS  # 26


# ------------------------------------------------------------------ io helpers
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


# ------------------------------------------------------------------ concurrency
def concurrency(stage):
    import psutil
    me = psutil.Process()
    rec = dict(stage=stage, time=time.strftime('%Y-%m-%d %H:%M:%S'), own_pid=os.getpid())
    try:
        me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        rec['priority_set_below_normal'] = me.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS
    except Exception as e:
        rec['priority_set_below_normal'] = False; rec['priority_error'] = repr(e)
    procs = []
    for p in psutil.process_iter(['pid', 'name']):
        n = (p.info['name'] or '')
        if 'python' in n.lower():
            procs.append(dict(name=n, pid=p.info['pid']))
    rec['python_processes'] = procs
    path = D / 'concurrency_log.json'
    log = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    log.append(rec); path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding='utf-8')
    print('concurrency', stage, 'python procs', len(procs), 'below_normal', rec['priority_set_below_normal'], flush=True)


# ------------------------------------------------------------------ preservation
def manifest():
    out = {}; tops = []
    for entry in sorted(os.scandir(WORK), key=lambda e: e.name):
        tops.append(entry.name)
        if entry.name in EXCLUDED_TOP:
            continue
        if entry.is_file(follow_symlinks=False):
            out[entry.name] = sha(entry.path)
        elif entry.is_dir(follow_symlinks=False):
            for root, dirs, files in os.walk(entry.path):
                dirs.sort()
                for fn in sorted(files):
                    p = os.path.join(root, fn)
                    if os.path.islink(p):
                        continue
                    out[os.path.relpath(p, WORK).replace('\\', '/')] = sha(p)
    return out, tops


def stage_preserve_before():
    t = time.perf_counter(); m, tops = manifest()
    dump('data/preservation_before.json', dict(top_level=tops, files=m, n_files=len(m), seconds=time.perf_counter() - t,
                                               excluded_never_entered=sorted(EXCLUDED_TOP)))
    print('before files', len(m), round(time.perf_counter() - t, 1))


def stage_preserve_after():
    concurrency('preserve_after')
    before = load('data/preservation_before.json'); m, tops = manifest()
    new_top = [t for t in tops if t not in before['top_level']]
    # files inside newly appeared top-level dirs are other work's additions: listed by top name only, not compared
    after = {k: v for k, v in m.items() if k.split('/')[0] not in new_top}
    missing = sorted(set(before['files']) - set(after)); added = sorted(set(after) - set(before['files']))
    changed = sorted(k for k in before['files'] if k in after and after[k] != before['files'][k])
    res = dict(n_before=len(before['files']), n_after_compared=len(after), missing=missing, added=added, changed=changed,
               new_top_level_by_other_work=new_top, all_unchanged=not (missing or added or changed),
               excluded_never_entered=sorted(EXCLUDED_TOP))
    dump('preservation_check.json', res); print('preservation all_unchanged', res['all_unchanged'], 'new tops', new_top, 'changed', changed[:5], 'added', added[:5], 'missing', missing[:5])


# ------------------------------------------------------------------ assets
def stage_assets():
    files = dict(model_code=V055 / 'core.py', training_code=V055 / 'run.py', data_code=V055 / 'data.py', utils=V055 / 'utils.py',
                 config=V055 / 'config.json', preregistration=V055 / 'PREREGISTRATION.md', spec=V055 / 'SPEC_COMPLEMENT.md',
                 report=V055 / 'REPORT_V055.md', dataset_manifest=V055 / 'dataset_manifest.json', finetuning_results=V055 / 'finetuning_results.csv',
                 training_log=V055 / 'training_log.csv', copied_model=D / 'model_d10.py', model_diff=D / 'model_diff.txt',
                 v054t_code=WORK / 'v054t_transition' / 'v054t.py', v054h_code=WORK / 'v054h_dynamics' / 'v054h.py',
                 v054h_posthoc=WORK / 'v054h_dynamics' / 'posthoc.py', v054e_generator_notes=WORK / 'v054e_endurance' / 'generator_notes.md',
                 v054e_common=WORK / 'v054e_endurance' / 'common.py', v050_prepare=WORK / 'IronMath_V4' / 'v050' / 'src' / 'prepare.py',
                 v053_dataset=WORK / 'IronMath_V053' / 'dataset.py')
    vm = json.loads((V055 / 'dataset_manifest.json').read_text())
    out = dict(files={k: dict(path=str(p), sha256=sha(p)) for k, p in files.items()},
               V055_F0_structure='core.py Residual: Linear(3,64) -> GRUCell(64,545) -> h=core(proj(x),h)+tanh(residual_gate)*h -> Linear(545,2); 1,000,334 params; hidden zero per problem; LSB-first; ADD N=W+1 with terminal (0,0,0)',
               V055_F0_training='run.py train(): setup(seed) threads2 deterministic; 3 dummy backward warmups; Random(50000+seed) candidates(): 3 streams x 8 (ADD,ADD-with-swap-coin,SUB); F0 uses ADD+ADD = 16 ADD; AdamW lr .002 wd .01; CE mean over all W+1 bits; clip 1; 1200 updates (C1); Validation every 200 and at end; selection min mean(ADD CE, SUB CE), strict < (earliest tie); best_validation.pt used for Test',
               V055_ADD_counts=dict(Train=vm['ADD_Train']['n'], Validation=vm['ADD_Validation']['n']),
               model_copy_reason='V055 Residual() has no dimension arguments; copied core.py and changed the two Linear dims only (model_diff.txt)')
    dump('asset_paths.json', out); print(out['V055_ADD_counts'])


# ------------------------------------------------------------------ decimal helpers
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


def gen_test(W):
    rng = random.Random(96000 + W); LIM = 10**W - 1; used = set(); rows = []

    def add(a, b, cat, extra=None):
        k = key(a, b)
        if k in used:
            return False
        assert 0 <= a <= LIM and 0 <= b <= LIM
        used.add(k); r = dict(id=f'D10_W{W}_{len(rows):03d}_{cat}', width=W, category=cat, a=a, b=b); r.update(extra or {}); rows.append(r); return True

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
        assert add(a, b, 'boundary', dict(edge=f'edge{i:02d}', origin='V050_boundary'))
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
        r['answer'] = r['a'] + r['b']
    assert len(rows) == 512
    assert dict(Counter(r['category'] for r in rows)) == CFG['data']['test']['categories'], Counter(r['category'] for r in rows)
    return rows


def to_store(r):
    return {k: (str(v) if k in ('a', 'b', 'answer') else v) for k, v in r.items()}


def from_store(r):
    return {k: (int(v) if k in ('a', 'b', 'answer') else v) for k, v in r.items()}


def gen_probes():
    rng = random.Random(97000); prefixes = {}
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
        assert (x + y + c) // 10 == c  # holding check from integer arithmetic
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
    assert len(probes) == 2080
    return probes, prefixes, suffixes


def stage_generate():
    concurrency('generate')
    dd = D / 'data'; dd.mkdir(exist_ok=True); names = []
    test8 = None
    for W in WIDTHS:
        rows = gen_test(W)
        if W == 8:
            test8 = {key(r['a'], r['b']) for r in rows}
        (dd / f'Test{W}.json').write_text(json.dumps([to_store(r) for r in rows]), encoding='utf-8'); names.append(f'Test{W}.json')
        print('Test', W, 'ok', flush=True)
    rng = random.Random(95000); used = set(test8); splits = {}
    for split, n in (('Train', CFG['data']['train_validation']['train_n']), ('Validation', CFG['data']['train_validation']['validation_n'])):
        rows = []
        while len(rows) < n:
            a, b = rng.randrange(10**8), rng.randrange(10**8); k = key(a, b)
            if k in used:
                continue
            used.add(k); rows.append(dict(id=f'D10_{split}_{len(rows):05d}', width=8, a=a, b=b, answer=a + b))
        for r in rows:
            A = digs(r['a'], 9); B = digs(r['b'], 9); out, _, fin = ripple(A, B); assert fin == 0 and out == digs(r['answer'], 9)
        splits[split] = rows
        (dd / f'{split}.json').write_text(json.dumps(rows), encoding='utf-8'); names.append(f'{split}.json')
    tr = {key(r['a'], r['b']) for r in splits['Train']}; va = {key(r['a'], r['b']) for r in splits['Validation']}
    assert not (tr & va or tr & test8 or va & test8)
    probes, prefixes, suffixes = gen_probes(); jl_write(dd / 'probes.jsonl', probes); names.append('probes.jsonl')
    meta = dict(test_seeds={W: 96000 + W for W in WIDTHS}, train_validation_seed=95000, probe_seed=97000,
                split_disjoint=True, test8_keys=len(test8), probe_prefixes={str(s): v for s, v in prefixes.items()}, probe_suffixes=suffixes,
                checks=['Python int answer', 'independent digit-wise decimal ripple == answer digits, final carry 0', 'propagation planted pass run == measured', 'no_propagation zero carries',
                        'long_propagation pass run >= W/2', 'probe holding (x+y+c)//10==c', 'probe ripple carry == c over hold block'])
    dump('data/generation_meta.json', meta); names.append('generation_meta.json')
    (dd / 'SHA256SUMS.txt').write_text(''.join(f'{sha(dd / n)}  {n}\n' for n in names), encoding='utf-8')
    print('generated', names)


# ------------------------------------------------------------------ model helpers
def setup(seed):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)


def modelhash(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode()); h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


EYE = np.eye(10, dtype=np.float32)


def encode_digits(A, B):
    """A,B: int arrays (batch, n) of digits -> float32 (batch, n, 20)"""
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


def ckdir(seed):
    return D / 'checkpoints' / f's{seed}'


def stage_train(seed):
    concurrency(f'train_seed{seed}')
    folder = ckdir(seed); assert not folder.exists(), 'no retraining'
    train = json.loads((D / 'data' / 'Train.json').read_text()); val = json.loads((D / 'data' / 'Validation.json').read_text())
    setup(seed); base = Residual(); nparams = sum(p.numel() for p in base.parameters()); initial = modelhash(base)
    m = torch.jit.script(base); opt = torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.01)
    folder.mkdir(parents=True)
    for _ in range(3):
        m.zero_grad(); m(torch.zeros(16, 9, 20)).square().mean().backward()
    assert modelhash(m) == initial
    rng = random.Random(50000 + seed); order = hashlib.sha256(); step = 0; best = float('inf'); beststep = 0
    cpu = wall = 0.; logs = []; t_all = time.perf_counter(); c_all = time.process_time()

    def save(version, loss=None):
        torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), seed=seed, update=step, validation_loss=loss, initial_SHA256=initial,
                        order_SHA256=order.hexdigest(), config=CFG, rng_state=rng.getstate(), torch_rng=torch.get_rng_state()), folder / f'{version}.pt')

    while step < 1200:
        rows = []
        for _ in range(16):
            r = dict(train[rng.randrange(len(train))])
            if rng.randrange(2):
                r['a'], r['b'] = r['b'], r['a']
            rows.append(r); order.update(f"{r['id']}:{r['a']}:{r['b']}\n".encode())
        x, y = encode(rows, True)
        c = time.process_time(); w = time.perf_counter()
        m.train(); opt.zero_grad(set_to_none=True); z = m(x); loss = F.cross_entropy(z.flatten(0, 1), y.flatten()); assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1); opt.step()
        cpu += time.process_time() - c; wall += time.perf_counter() - w; step += 1
        if step % 200 == 0 or step >= 1200:
            m.eval(); st = fit(m, val); vl = st['loss']
            if vl < best:
                best = vl; beststep = step; save('best_validation', vl)
            logs.append(dict(seed=seed, update=step, minibatch_loss=float(loss.detach()), validation_loss=vl, validation_exact=st['exact'],
                             validation_digit_accuracy=st['digit_accuracy'], selected_update=beststep, train_CPU_seconds=cpu, train_wall_seconds=wall))
            print('TRAIN seed', seed, step, 'VAL', round(vl, 6), 'exact', round(st['exact'], 4), flush=True)
    save('final')
    cp = torch.load(folder / 'best_validation.pt', map_location='cpu', weights_only=False); m.load_state_dict(cp['model']); m.eval()
    tf = fit(m, train); vf = fit(m, val)
    rec = dict(seed=seed, parameters=nparams, updates=step, presentations=step * 16, selected_update=beststep, best_validation_loss=best,
               initial_SHA256=initial, order_SHA256=order.hexdigest(), selected_model_SHA256=modelhash(m),
               best_checkpoint_file_SHA256=sha(folder / 'best_validation.pt'), final_checkpoint_file_SHA256=sha(folder / 'final.pt'),
               selected_train_fit=tf, selected_validation_fit=vf, train_CPU_seconds=cpu, train_wall_seconds=wall,
               total_stage_wall_seconds=time.perf_counter() - t_all, total_stage_CPU_seconds=time.process_time() - c_all, threads=torch.get_num_threads())
    dump(f'checkpoints/s{seed}/training_record.json', rec); jl_write(folder / 'training_log.jsonl', logs)
    print('trained', seed, 'selected', beststep, 'params', nparams, flush=True)


def load_jit(seed):
    cp = torch.load(ckdir(seed) / 'best_validation.pt', map_location='cpu', weights_only=False)
    base = Residual(); base.load_state_dict(cp['model']); base.eval()
    return torch.jit.script(base).eval(), base, cp


def all_trained():
    return all((ckdir(s) / 'training_record.json').exists() for s in SEEDS)


def load_test(W):
    return [from_store(r) for r in json.loads((D / 'data' / f'Test{W}.json').read_text())]


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


def stage_eval(seed):
    assert all_trained(), 'Test opens only after all 3 seeds are trained'
    concurrency(f'eval_seed{seed}'); torch.set_num_threads(1)
    m, base, cp = load_jit(seed); h0 = modelhash(base); rows_out = []; timing = {}
    for W in WIDTHS:
        rows = load_test(W); n = W + 1; t = time.perf_counter(); c = time.process_time()
        for j in range(0, len(rows), 128):
            rr = rows[j:j + 128]; P = predict(m, encode(rr))
            for r, p in zip(rr, P):
                pred = [int(v) for v in p[:n]]; truth = digs(r['answer'], n); wrong, longest = wrong_stats(pred, truth)
                pi = from_digs(pred); single, s_exp, sign = error_shape(pi, r['answer'])
                rows_out.append(dict(seed=seed, width=W, id=r['id'], category=r['category'], edge=r.get('edge'), exact=int(pi == r['answer']),
                                     first_error=wrong[0] if wrong else None, wrong_digits=len(wrong), longest_wrong_run=longest,
                                     longest_carry=r['longest_carry'], longest_pass_run_c1=r['longest_pass_run_c1'], longest_pass_run_c0=r['longest_pass_run_c0'],
                                     error_single_power_of_ten=single, error_exponent=s_exp, error_sign=sign,
                                     prediction=None if pi == r['answer'] else str(pi)))
        timing[W] = dict(wall=time.perf_counter() - t, cpu=time.process_time() - c)
        print('EVAL seed', seed, 'W', W, 'exact', sum(r['exact'] for r in rows_out if r['width'] == W), '/512', round(timing[W]['wall'], 1), flush=True)
    assert modelhash(base) == h0
    jl_write(D / 'data' / 'raw' / f'eval_seed{seed}.jsonl', rows_out)
    dump(f'data/raw/eval_seed{seed}_run.json', dict(timing=timing, weights_sha256=h0, checkpoint_sha256=sha(ckdir(seed) / 'best_validation.pt')))


def stage_probes(seed):
    assert all_trained()
    concurrency(f'probes_seed{seed}'); torch.set_num_threads(1)
    m, base, _ = load_jit(seed); h0 = modelhash(base); probes = jl_read(D / 'data' / 'probes.jsonl'); out = []; t = time.perf_counter()
    for k in KS:
        batch = [r for r in probes if r['k'] == k]; n = batch[0]['seq_len']; assert all(r['seq_len'] == n for r in batch)
        A = np.array([digs(int(r['a']), n) for r in batch]); B = np.array([digs(int(r['b']), n) for r in batch])
        P = np.concatenate([predict(m, encode_digits(A[j:j + 128], B[j:j + 128])) for j in range(0, len(batch), 128)])
        for r, p in zip(batch, P):
            pred = [int(v) for v in p[:n]]; ans = int(r['answer']); truth = digs(ans, n); wrong, longest = wrong_stats(pred, truth)
            fe = wrong[0] if wrong else None
            region = None if fe is None else 'prefix' if fe < r['hold_start'] else 'hold' if fe < r['hold_end'] else 'suffix'
            pi = from_digs(pred); single, s_exp, sign = error_shape(pi, ans)
            out.append(dict(seed=seed, pid=r['pid'], state=r['state'], pair=r['pair'], kind=r['kind'], k=k, prefix=r['prefix'], suffix=r['suffix'],
                            exact=int(pi == ans), first_error=fe, first_error_region=region, hold_offset=(fe - r['hold_start']) if region == 'hold' else None,
                            wrong_digits=len(wrong), longest_wrong_run=longest, error_single_power_of_ten=single, error_exponent=s_exp, error_sign=sign))
    assert modelhash(base) == h0
    jl_write(D / 'data' / 'raw' / f'probes_seed{seed}.jsonl', out)
    dump(f'data/raw/probes_seed{seed}_run.json', dict(wall=time.perf_counter() - t, weights_sha256=h0))
    print('PROBES seed', seed, 'exact', sum(r['exact'] for r in out), '/', len(out), flush=True)


def step_fn(base, x, h):
    """One recurrent step. Composition line copied from model_d10.py (= V055 core.py) Residual.forward."""
    h = base.core(base.projection(x), h) + torch.tanh(base.residual_gate) * h
    return h, base.head(h)


@torch.no_grad()
def predict_steps(base, x):
    h = torch.zeros(x.size(0), 545); outs = []
    for t in range(x.size(1)):
        h, z = step_fn(base, x[:, t], h); outs.append(z.argmax(-1))
    return torch.stack(outs, 1).numpy()


def stage_equiv():
    assert all_trained()
    concurrency('equiv'); torch.set_num_threads(1); res = dict(rows=[], all_identical=True)
    for seed in SEEDS:
        m, base, _ = load_jit(seed); h0 = modelhash(base); mis = n = 0; t = time.perf_counter()
        for W in [w for w in WIDTHS if w <= 256]:
            rows = load_test(W)
            for j in range(0, len(rows), 128):
                x = encode(rows[j:j + 128]); a = predict(m, x); b = predict_steps(base, x)
                mis += int((a != b).any(-1).sum()); n += len(a)
        assert modelhash(base) == h0
        res['rows'].append(dict(seed=seed, problems=n, mismatched_problems=mis, wall_seconds=time.perf_counter() - t)); res['all_identical'] &= mis == 0
        print(res['rows'][-1], flush=True)
    res['wrapper'] = 'h = core(projection(x), h) + tanh(residual_gate) * h ; logits = head(h); eager Residual with the same state_dict; compared with jit forward argmax on all Test problems of widths 8-256'
    dump('step_equivalence.json', res)


def onehot(x, y):
    v = torch.zeros(1, 20); v[0, x] = 1; v[0, 10 + y] = 1
    return v


def stage_fixed(seed):
    assert load('step_equivalence.json')['all_identical'], 'step wrapper not equivalent -> fixed-point analysis not run'
    concurrency(f'fixed_seed{seed}'); torch.set_num_threads(1)
    _, base, _ = load_jit(seed); h0 = modelhash(base)
    for p in base.parameters():
        p.requires_grad_(False)
    probes = {(r['state'], r['pair']): r for r in jl_read(D / 'data' / 'probes.jsonl') if r['k'] == 1024 and r['prefix'] == 'p0' and r['suffix'] == 's0'}
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
        row = dict(seed=seed, state=c, pair=f'{x}{y}', kind='pass' if x + y == 9 else 'nonpass', iterations=it, converged=int(conv),
                   last100_move_max=max(moves[-100:]), last100_move_min=min(moves[-100:]), h_star_norm=float(g.norm()),
                   h_star_output=out, correct_output=right, h_star_output_correct=int(out == right), h_star_margin_correct=margin,
                   tanh_residual_gate=float(torch.tanh(base.residual_gate)))
        if conv:
            J = torch.autograd.functional.jacobian(lambda hh: step_fn(base, xin, hh.unsqueeze(0))[0].squeeze(0), g.squeeze(0).detach())
            ev = torch.linalg.eigvals(J.double()); order = torch.argsort(ev.abs(), descending=True)
            row['spectral_radius'] = float(ev.abs().max())
            for rank, idx in enumerate(order[:5].tolist()):
                eigs.append(dict(seed=seed, state=c, pair=f'{x}{y}', rank=rank + 1, real=float(ev[idx].real), imag=float(ev[idx].imag), abs=float(ev[idx].abs())))
        else:
            row['spectral_radius'] = None
        fps.append(row); print('FP', seed, c, x, y, 'it', it, 'conv', conv, 'out', out, 'right', right, 'rho', row['spectral_radius'], flush=True)
    pairs = []
    for x, y in PASS:
        r0 = next(f for f in fps if f['state'] == 0 and f['pair'] == f'{x}{y}'); r1 = next(f for f in fps if f['state'] == 1 and f['pair'] == f'{x}{y}')
        dist = float((ends[(0, f'{x}{y}')] - ends[(1, f'{x}{y}')]).norm())
        both = bool(r0['converged'] and r1['converged'])
        pairs.append(dict(seed=seed, pair=f'{x}{y}', distance_c0_vs_c1=dist, both_converged=int(both), same_point=int(dist < 1e-3) if both else None,
                          output_c0_side=r0['h_star_output'], correct_c0=9, output_c1_side=r1['h_star_output'], correct_c1=0,
                          c0_side_correct=r0['h_star_output_correct'], c1_side_correct=r1['h_star_output_correct']))
    assert modelhash(base) == h0
    jl_write(D / 'data' / 'raw' / f'fixed_seed{seed}.jsonl', fps); jl_write(D / 'data' / 'raw' / f'eigs_seed{seed}.jsonl', eigs); jl_write(D / 'data' / 'raw' / f'pairs_seed{seed}.jsonl', pairs)
    dump(f'data/raw/fixed_seed{seed}_run.json', dict(wall=time.perf_counter() - t0, weights_sha256=h0))


# ------------------------------------------------------------------ analyze
def verdict3(flags):
    return '支持' if all(flags) else '不支持' if not any(flags) else '判定不能'


def stage_analyze():
    concurrency('analyze')
    tl = []
    for s in SEEDS:
        tl += jl_read(ckdir(s) / 'training_log.jsonl')
    csvout('training_log.csv', tl)
    ev = []; pr = []; fp = []; eg = []; pa = []
    for s in SEEDS:
        ev += jl_read(D / 'data' / 'raw' / f'eval_seed{s}.jsonl'); pr += jl_read(D / 'data' / 'raw' / f'probes_seed{s}.jsonl')
        fp += jl_read(D / 'data' / 'raw' / f'fixed_seed{s}.jsonl'); eg += jl_read(D / 'data' / 'raw' / f'eigs_seed{s}.jsonl'); pa += jl_read(D / 'data' / 'raw' / f'pairs_seed{s}.jsonl')
    csvout('results_per_problem.csv', ev); csvout('fixed_points.csv', fp); csvout('jacobian_eigs.csv', eg); csvout('pairs.csv', pa)
    csvout('data/probe_results_per_problem.csv', pr)
    # summary: per width x category(all + each) x seed
    g = defaultdict(list)
    for r in ev:
        g[(r['width'], 'ALL', r['seed'])].append(r); g[(r['width'], r['category'], r['seed'])].append(r)
    cats = ['ALL'] + list(CFG['data']['test']['categories'])
    summary = []
    for W in WIDTHS:
        for cat in cats:
            row = dict(width=W, category=cat, n=len(g[(W, cat, 0)]))
            rates = []
            for s in SEEDS:
                rr = g[(W, cat, s)]; rate = sum(r['exact'] for r in rr) / len(rr); rates.append(rate)
                row[f'seed{s}_exact'] = rate; row[f'seed{s}_correct'] = sum(r['exact'] for r in rr)
                fe = [r['first_error'] for r in rr if not r['exact']]
                row[f'seed{s}_median_first_error'] = statistics.median(fe) if fe else None
                row[f'seed{s}_mean_wrong_digits_failed'] = statistics.mean([r['wrong_digits'] for r in rr if not r['exact']]) if fe else None
            row['mean_exact'] = statistics.mean(rates); row['sd_exact'] = statistics.stdev(rates)
            summary.append(row)
    csvout('summary.csv', summary)
    S = {(r['width'], r['category']): r for r in summary}
    H1 = [S[(2048, 'ALL')][f'seed{s}_exact'] >= .95 for s in SEEDS]
    h1 = '支持' if all(H1) else '不支持' if not any(H1) else '判定不能'
    H2 = [S[(2048, 'random')][f'seed{s}_exact'] >= .99 for s in SEEDS]
    h2 = verdict3(H2)
    e01 = {(r['seed'], r['width']): r['exact'] for r in ev if r.get('edge') == 'edge01'}
    H3fail = [e01[(s, W)] == 0 for s in SEEDS for W in WIDTHS if W >= 64]
    h3 = verdict3(H3fail)
    # durability table
    dg = defaultdict(list)
    for r in pr:
        dg[(r['state'], r['pair'], r['k'], r['seed'])].append(r)
    dur = []
    for c, (x, y) in COMBOS:
        for k in KS:
            row = dict(state=c, pair=f'{x}{y}', kind='pass' if x + y == 9 else 'nonpass', k=k)
            for s in SEEDS:
                rr = dg[(c, f'{x}{y}', k, s)]; assert len(rr) == 16
                offs = [r['hold_offset'] for r in rr if r['first_error_region'] == 'hold']
                row[f'seed{s}_exact'] = f"{sum(r['exact'] for r in rr)}/16"; row[f'seed{s}_hold_breaks'] = len(offs)
                row[f'seed{s}_hold_offset_min'] = min(offs) if offs else None; row[f'seed{s}_hold_offset_median'] = statistics.median(offs) if offs else None
                row[f'seed{s}_hold_offset_max'] = max(offs) if offs else None
                row[f'seed{s}_first_error_regions'] = ';'.join(f'{a}:{b}' for a, b in sorted(Counter(r['first_error_region'] for r in rr if not r['exact']).items()))
            row['three_seeds'] = ' / '.join(row[f'seed{s}_exact'].split('/')[0] for s in SEEDS)
            dur.append(row)
    csvout('durability_table.csv', dur)
    # H4
    cases = []
    for s in SEEDS:
        for c, (x, y) in COMBOS:
            f = next(q for q in fp if q['seed'] == s and q['state'] == c and q['pair'] == f'{x}{y}')
            obs = sum(r['exact'] for r in dg[(c, f'{x}{y}', 1024, s)]) == 16
            pred = bool(f['converged'] and f['h_star_output_correct'])
            cases.append(dict(seed=s, state=c, pair=f'{x}{y}', converged=f['converged'], fp_output=f['h_star_output'], correct_output=f['correct_output'],
                              predicted_unbroken=int(pred), observed_unbroken_k1024=int(obs), agree=int(pred == obs and bool(f['converged']))))
    agree = sum(c['agree'] for c in cases) / len(cases)
    h4 = '支持' if agree >= .95 else '不支持' if agree < .80 else '判定不能'
    # H5
    h5 = {s: dict(two_points=sum(1 for p in pa if p['seed'] == s and p['same_point'] == 0), same=sum(1 for p in pa if p['seed'] == s and p['same_point'] == 1),
                  not_both_converged=sum(1 for p in pa if p['seed'] == s and p['same_point'] is None)) for s in SEEDS}
    # H6 error shape
    shape = []
    for r in ev:
        if not r['exact']:
            shape.append(dict(source='main_test', seed=r['seed'], width=r['width'], id=r['id'], category=r['category'], first_error=r['first_error'],
                              wrong_digits=r['wrong_digits'], single_power_of_ten=r['error_single_power_of_ten'], exponent=r['error_exponent'], sign=r['error_sign']))
    for r in pr:
        if not r['exact']:
            shape.append(dict(source='probe_supplementary', seed=r['seed'], width=None, id=f"probe{r['pid']}", category=f"c{r['state']}x{r['pair']}_k{r['k']}",
                              first_error=r['first_error'], wrong_digits=r['wrong_digits'], single_power_of_ten=r['error_single_power_of_ten'], exponent=r['error_exponent'], sign=r['error_sign']))
    csvout('error_shape.csv', shape)
    main_fail = [r for r in shape if r['source'] == 'main_test']
    h6 = dict(pooled=dict(n_failed=len(main_fail), single=sum(r['single_power_of_ten'] for r in main_fail)),
              per_seed={s: dict(n_failed=sum(1 for r in main_fail if r['seed'] == s), single=sum(r['single_power_of_ten'] for r in main_fail if r['seed'] == s)) for s in SEEDS})
    pf = [r for r in shape if r['source'] != 'main_test']
    h6['probe_supplementary'] = dict(n_failed=len(pf), single=sum(r['single_power_of_ten'] for r in pf))
    # binary comparison (read only)
    fr = [r for r in readcsv(V055 / 'finetuning_results.csv') if r['budget'] == 'C1' and r['condition'] == 'F0' and r['op'] == 'ADD']
    binary = {int(r['width']): {int(q['seed']): float(q['exact_rate']) for q in fr if q['width'] == r['width']} for r in fr}
    res = dict(H1=dict(per_seed={s: S[(2048, 'ALL')][f'seed{s}_exact'] for s in SEEDS}, verdict=h1),
               H2=dict(per_seed={s: S[(2048, 'random')][f'seed{s}_exact'] for s in SEEDS}, verdict=h2),
               H3=dict(edge01_exact={f'seed{s}_W{W}': e01[(s, W)] for s in SEEDS for W in WIDTHS}, verdict=h3),
               H4=dict(agreement=agree, n_cases=len(cases), per_seed={s: sum(c['agree'] for c in cases if c['seed'] == s) / 26 for s in SEEDS},
                       disagreements=[c for c in cases if not c['agree']], verdict=h4, cases=cases),
               H5=h5, H6=h6, binary_V055_C1_F0_ADD=binary)
    dump('analysis_summary.json', res)
    print(json.dumps({k: v.get('verdict') if isinstance(v, dict) else v for k, v in res.items() if k.startswith('H')}, ensure_ascii=False, default=str))


if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'preserve_before': stage_preserve_before()
    elif st == 'assets': stage_assets()
    elif st == 'generate': stage_generate()
    elif st == 'train': stage_train(int(sys.argv[2]))
    elif st == 'eval': stage_eval(int(sys.argv[2]))
    elif st == 'probes': stage_probes(int(sys.argv[2]))
    elif st == 'equiv': stage_equiv()
    elif st == 'fixed': stage_fixed(int(sys.argv[2]))
    elif st == 'analyze': stage_analyze()
    elif st == 'preserve_after': stage_preserve_after()
