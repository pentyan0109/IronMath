"""IronMath V054R: fixed-point replication on other models (R) and residual-coefficient intervention on V054 B0 (I).
Inference only. Run with python -B.
usage: python -B v054r.py assets|repro|equiv|analyze
       python -B v054r.py R V051_A1|V055_F0|V055_F1|V055_F2
       python -B v054r.py I SEED
"""
import copy, csv, ctypes, gzip, hashlib, importlib.util, json, math, statistics, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

H = Path(__file__).resolve().parent
WORK = H.parent
sys.dont_write_bytecode = True
import numpy as np
import torch

V054 = WORK / 'IronMath_V054'; V051 = WORK / 'IronMath_V4' / 'v051'; V055 = WORK / 'IronMath_V055'; V054T = WORK / 'v054t_transition'
SEEDS = (0, 1, 2); KS = (8, 16, 64, 256, 1024, 4096); OPS = ('ADD', 'SUB')
HAND = {('ADD', 0): ['00', '01', '10'], ('ADD', 1): ['01', '10', '11'], ('SUB', 0): ['00', '10', '11'], ('SUB', 1): ['00', '01', '11']}
PASS = {'ADD': ['01', '10'], 'SUB': ['00', '11']}
R_MODELS = ('V051_A1', 'V055_F0', 'V055_F1', 'V055_F2')
CS = ('orig', -0.05, -0.02, 0.0)


# ------------------------------------------------------------------ utilities
def setup():
    torch.set_num_threads(1)
    try:
        k = ctypes.windll.kernel32; k.GetCurrentProcess.restype = ctypes.c_void_p
        return bool(k.SetPriorityClass(ctypes.c_void_p(k.GetCurrentProcess()), 0x4000))
    except Exception as e:
        return repr(e)


PRIORITY = setup()


def log_concurrency(stage):
    try:
        out = subprocess.run(['tasklist', '/FO', 'CSV', '/NH'], capture_output=True, text=True, timeout=60).stdout
        procs = [dict(name=r[0], pid=int(r[1])) for r in csv.reader(out.splitlines()) if r and r[0].lower().startswith('python')]
    except Exception as e:
        procs = repr(e)
    p = H / 'concurrency_log.json'
    log = json.loads(p.read_text(encoding='utf-8')) if p.exists() else []
    import os
    log.append(dict(stage=stage, time=time.strftime('%Y-%m-%d %H:%M:%S'), own_pid=os.getpid(), below_normal_priority=PRIORITY, threads=torch.get_num_threads(), python_processes=procs))
    p.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def dump(name, x):
    p = H / name; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=1), encoding='utf-8')


def load(name):
    return json.loads((H / name).read_text(encoding='utf-8'))


def csvout(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (H / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows({k: ('NA' if v is None else v) for k, v in r.items()} for r in rows)


def jl_read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines()]


def bits(v, n):
    return [(v >> i) & 1 for i in range(n)]


def step_int(op, s, x, y):
    if op == 'ADD':
        t = x + y + s; return t & 1, t >> 1
    t = x - y - s; return t % 2, int(t < 0)


def statehash(sd, skip=()):
    h = hashlib.sha256()
    for k, v in sd.items():
        if k in skip:
            continue
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_module(name, path, extra):
    sys.path.insert(0, str(extra))
    try:
        spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(extra))
    return mod


_MODS = {}


def mod(kind):
    if kind not in _MODS:
        if kind == 'V054':
            _MODS[kind] = load_module('v054_model', V054 / 'model.py', V054)
        elif kind == 'V051':
            _MODS[kind] = load_module('v051_model', V051 / 'src' / 'model.py', V051 / 'src')
        else:
            _MODS[kind] = load_module('v055_core', V055 / 'core.py', V055)
    return _MODS[kind]


def ckpt_path(model, seed):
    if model == 'V051_A1':
        return V051 / 'checkpoints' / f'A1_residual_s{seed}_1200.pt'
    if model.startswith('V055_'):
        return V055 / 'checkpoints' / f'C1_{model[5:]}_s{seed}' / 'best_validation.pt'
    return V054 / 'checkpoints' / f'C1_B0_s{seed}' / 'best_validation.pt'


def load_state(model, seed):
    p = ckpt_path(model, seed)
    cp = torch.load(p, weights_only=(model == 'V051_A1'), map_location='cpu')
    return cp['model'], cp


def build(model, sd):
    """returns (forward(x)->logits, eager base). forward = the one each project used for evaluation."""
    if model == 'V051_A1':
        base = mod('V051').Model('A1_residual'); base.load_state_dict(sd); base.eval()
        fwd = base
    elif model.startswith('V055_'):
        base = mod('V055').Residual(); base.load_state_dict(sd); base.eval(); fwd = torch.jit.script(copy.deepcopy(base)).eval()
    else:
        base = mod('V054').Residual('B0'); base.load_state_dict(sd); base.eval(); fwd = torch.jit.script(copy.deepcopy(base)).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    return fwd, base


def step_fn(base, x, h):
    """one step; composition line copied verbatim from the models' forward (identical in V051 A1, V054 B0, V055 F)."""
    h = base.core(base.projection(x), h) + torch.tanh(base.residual_gate) * h
    return h, base.head(h)


def make_x(op, a, b, n):
    x = np.zeros((n, 3), np.float32); x[:, 0] = bits(a, n); x[:, 1] = bits(b, n); x[:, 2] = int(op == 'SUB')
    return x


@torch.no_grad()
def predict(fwd, items):
    nmax = max(n for *_, n in items); x = np.zeros((len(items), nmax, 3), np.float32)
    for i, (op, a, b, n) in enumerate(items):
        x[i, :n] = make_x(op, a, b, n)
    z = fwd(torch.from_numpy(x)); assert torch.isfinite(z).all()
    p = z.argmax(-1).numpy()
    return [[int(v) for v in p[i, :n]] for i, (*_, n) in enumerate(items)]


@torch.no_grad()
def predict_steps(base, items):
    nmax = max(n for *_, n in items); x = np.zeros((len(items), nmax, 3), np.float32)
    for i, (op, a, b, n) in enumerate(items):
        x[i, :n] = make_x(op, a, b, n)
    x = torch.from_numpy(x); h = torch.zeros(len(items), base.core.hidden_size); outs = []
    for t in range(nmax):
        h, z = step_fn(base, x[:, t], h); outs.append(z.argmax(-1))
    p = torch.stack(outs, 1).numpy()
    return [[int(v) for v in p[i, :n]] for i, (*_, n) in enumerate(items)]


def score(r, p, extra):
    n = r['seq_len']; truth = bits(int(r['answer'], 16), n); wrong = [i for i in range(n) if p[i] != truth[i]]
    fe = wrong[0] if wrong else None
    region = None if fe is None else 'prefix' if fe < r['hold_start'] else 'hold' if fe < r['hold_end'] else 'suffix'
    row = dict(extra); row.update(pid=r['pid'], op=r['op'], state=r['state'], pair=r['pair'], k=r['k'], prefix=r['prefix'], suffix=r['suffix'],
                                  exact=int(not wrong), first_error=fe, first_error_region=region,
                                  hold_offset=(fe - r['hold_start']) if region == 'hold' else None, wrong_bits=len(wrong))
    return row


PROBES = None


def probes():
    global PROBES
    if PROBES is None:
        PROBES = jl_read(V054T / 'data' / 'probes.jsonl')
    return PROBES


def durability(fwd, ops, extra):
    rows = []
    for op in ops:
        for k in KS:
            batch = [r for r in probes() if r['op'] == op and r['k'] == k]
            preds = predict(fwd, [(op, int(r['a'], 16), int(r['b'], 16), r['seq_len']) for r in batch])
            rows += [score(r, p, extra) for r, p in zip(batch, preds)]
    return rows


def hold_start_hidden(base, op, state, pair):
    r = [q for q in probes() if q['op'] == op and q['state'] == state and q['pair'] == pair and q['k'] == 4096 and q['prefix'] == 'min' and q['suffix'] == 's0']
    assert len(r) == 1; r = r[0]
    x = torch.from_numpy(make_x(op, int(r['a'], 16), int(r['b'], 16), r['hold_start'] + 1)); h = torch.zeros(1, base.core.hidden_size)
    with torch.no_grad():
        for t in range(r['hold_start']):
            h, _ = step_fn(base, x[t:t + 1], h)
    return h, x[r['hold_start']:r['hold_start'] + 1]


def fixed_point(base, op, state, pair):
    h, xin = hold_start_hidden(base, op, state, pair); moves = []; it = 0; conv = False
    with torch.no_grad():
        for target in (3000, 20000):
            while it < target:
                hn, _ = step_fn(base, xin, h); moves.append(float((hn - h).norm())); h = hn; it += 1
            conv = max(moves[-100:]) < 1e-6
            if conv:
                break
        out = int(step_fn(base, xin, h)[1].argmax(-1))
    right = step_int(op, state, int(pair[0]), int(pair[1]))[0]; last = moves[-100:]
    row = dict(op=op, state=state, pair=pair, iterations=it, converged=int(conv), last100_move_max=max(last), last100_move_min=min(last),
               h_star_norm=float(h.norm()), h_star_output=out, correct_output=right, h_star_output_correct=int(out == right),
               tanh_g=float(torch.tanh(base.residual_gate)), spectral_radius=None)
    eigs = []
    if conv:
        J = torch.autograd.functional.jacobian(lambda hh: step_fn(base, xin, hh.unsqueeze(0))[0].squeeze(0), h.squeeze(0).detach())
        ev = torch.linalg.eigvals(J.double()); order = torch.argsort(ev.abs(), descending=True)
        row['spectral_radius'] = float(ev.abs().max())
        eigs = [dict(op=op, state=state, pair=pair, rank=i + 1, real=float(ev[j].real), imag=float(ev[j].imag), abs=float(ev[j].abs())) for i, j in enumerate(order[:5].tolist())]
    return row, eigs, h


# ------------------------------------------------------------------ stages
def stage_assets():
    log_concurrency('assets')
    files = {}
    for s in SEEDS:
        files[f'V051_A1_s{s}'] = ckpt_path('V051_A1', s)
        for f in ('F0', 'F1', 'F2'):
            files[f'V055_{f}_s{s}'] = ckpt_path(f'V055_{f}', s)
        files[f'V054_B0_s{s}'] = ckpt_path('V054_B0', s)
    files.update(V051_model=V051 / 'src' / 'model.py', V051_common=V051 / 'src' / 'common.py', V051_evaluate=V051 / 'src' / 'evaluate.py', V051_report=V051 / 'REPORT_V051.md',
                 V051_predictions=V051 / 'predictions.csv', V051_test_ADD2048=V051 / 'data' / 'ADD_Test2048.json', V051_test_SUB2048=V051 / 'data' / 'SUB_Test2048.json',
                 V055_core=V055 / 'core.py', V055_run=V055 / 'run.py', V055_report=V055 / 'REPORT_V055.md', V055_predictions=V055 / 'predictions.jsonl.gz',
                 V055_test_ADD2048=V055 / 'data' / 'ADD_Test2048.json',
                 V054_model=V054 / 'model.py', V054_run=V054 / 'run.py', V054_predictions=V054 / 'predictions.jsonl.gz',
                 V054_test_ADD2048=WORK / 'IronMath_V053' / 'data' / 'ADD_Test2048.json', V054_test_SUB2048=WORK / 'IronMath_V053' / 'data' / 'SUB_Test2048.json',
                 V054T_probes=V054T / 'data' / 'probes.jsonl', V054T_results=V054T / 'results_per_problem.csv')
    meta = {}
    for m in R_MODELS + ('V054_B0',):
        for s in SEEDS:
            sd, cp = load_state(m, s)
            meta[f'{m}_s{s}'] = dict(tanh_g=float(torch.tanh(sd['residual_gate'])), g=float(sd['residual_gate']), param_keys=list(sd.keys()),
                                     selected_update=cp.get('selected_update', cp.get('update')), condition=cp.get('condition'), kind=cp.get('kind'), seed=cp.get('seed'))
    out = dict(files={k: dict(path=str(p), sha256=sha(p)) for k, p in files.items()}, checkpoint_meta=meta,
               uniqueness=dict(V051_A1='checkpoints/ に A1_residual_s{seed}_*.pt は各seed 1本（_1200）のみ。cp.selected_update=1200。REPORT_V051 の選択update列も1200',
                               V055_F='checkpoints/C1_F{0,1,2}_s{seed}/best_validation.pt（V055 run.py が Test 評価に使ったファイル）',
                               V054_B0='checkpoints/C1_B0_s{seed}/best_validation.pt（V054T/H と同じ）'),
               input_formats=dict(
                   V051_A1='src/model.py batch(): 長さ width+int(op==ADD)、x=(a_bit, b_bit, float(op==SUB))、LSB-first。forward: h ゼロ初期化、h=core(z_i,h)+tanh(residual_gate)*h、head 2クラス。evaluate.py: eager Model、batch16、argmax',
                   V055_F='core.py encode(route=P): ADD は n=width+1、x=(a,b,0)。第3特徴 0。Residual.forward は V054 B0 と同形。run.py: torch.jit.script(Residual()).eval()、batch128、argmax',
                   V054_B0='model.py inputs(): 3ch目=演算flag(SUB=1)、ADD width+1、SUB width。jit script、batch128'),
               file_count_note='V051 A1 の Test は 2048bit 256問/演算（V051 data）。V055 F の Test は IronMath_V055/data（V053 由来）。')
    dump('asset_paths.json', out); print(json.dumps({k: round(v['tanh_g'], 5) for k, v in meta.items()}))


def stage_repro():
    log_concurrency('repro'); res = dict(rows=[], all_match=True)
    csv.field_size_limit(10**9)
    v051 = {}
    with (V051 / 'predictions.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['model'] == 'A1_residual' and r['width'] == '2048':
                v051[(int(r['seed']), r['op'], r['problem_id'])] = json.loads(r['output_bits'])
    v055 = {}
    with gzip.open(V055 / 'predictions.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            if '"width": 2048' not in line or '"budget": "C1"' not in line or '"op": "ADD"' not in line:
                continue
            p = json.loads(line)
            if p['condition'] in ('F0', 'F1', 'F2'):
                v055[(p['condition'], p['seed'], p['id'])] = p['prediction_digits']
    v054 = {}
    with gzip.open(V054 / 'predictions.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            if '"width": 2048' not in line or '"model": "B0"' not in line or '"condition": "C1"' not in line:
                continue
            p = json.loads(line); v054[(p['seed'], p['op'], p['id'])] = p['prediction_digits']
    for m in R_MODELS + ('V054_B0',):
        for s in SEEDS:
            sd, _ = load_state(m, s); fwd, base = build(m, sd); h0 = statehash(sd)
            ops = ('ADD',) if m.startswith('V055') else OPS
            for op in ops:
                if m == 'V051_A1':
                    rows = json.loads((V051 / 'data' / f'{op}_Test2048.json').read_text())[:16]; want = [v051[(s, op, r['id'])] for r in rows]
                elif m.startswith('V055'):
                    rows = json.loads((V055 / 'data' / f'{op}_Test2048.json').read_text())[:16]; want = [v055[(m[5:], s, r['id'])] for r in rows]
                else:
                    rows = json.loads((WORK / 'IronMath_V053' / 'data' / f'{op}_Test2048.json').read_text())[:16]; want = [v054[(s, op, r['id'])] for r in rows]
                got = predict(fwd, [(op, r['a'], r['b'], 2049 if op == 'ADD' else 2048) for r in rows])
                mism = sum(g != w for g, w in zip(got, want)); res['all_match'] &= mism == 0
                res['rows'].append(dict(model=m, seed=s, op=op, n=16, bit_identical=16 - mism))
            assert statehash(base.state_dict()) == h0
            print(res['rows'][-1], flush=True)
    res['excluded_models'] = sorted({f"{r['model']}_s{r['seed']}" for r in res['rows'] if r['bit_identical'] < 16})
    dump('reproduction_check.json', res); print('all_match', res['all_match'])


def stage_equiv():
    log_concurrency('equiv'); res = load('step_equivalence.json') if (H / 'step_equivalence.json').exists() else dict(rows=[])
    done = {(r['model'], r['seed']) for r in res['rows']}
    for m in R_MODELS + ('V054_B0',):
        for s in SEEDS:
            if (m, s) in done:
                continue
            sd, _ = load_state(m, s); fwd, base = build(m, sd); t0 = time.perf_counter(); mis = 0; n = 0
            ops = ('ADD',) if m.startswith('V055') else OPS
            for op in ops:
                for k in KS:
                    batch = [r for r in probes() if r['op'] == op and r['k'] == k]
                    items = [(op, int(r['a'], 16), int(r['b'], 16), r['seq_len']) for r in batch]
                    a = predict(fwd, items); b = predict_steps(base, items); mis += sum(x != y for x, y in zip(a, b)); n += len(items)
            res['rows'].append(dict(model=m, seed=s, n=n, mismatches=mis, identical=int(mis == 0), wall_seconds=time.perf_counter() - t0))
            dump('step_equivalence.json', res); print(res['rows'][-1], flush=True)
    res['wrapper'] = 'h = core(projection(x), h) + tanh(residual_gate) * h ; logits = head(h)  — copied from each model forward (V051 Model.forward A1 branch, V054 Residual.forward, V055 Residual.forward)'
    dump('step_equivalence.json', res)


def ok_models():
    rep = load('reproduction_check.json'); eq = load('step_equivalence.json')
    bad = set(rep['excluded_models']) | {f"{r['model']}_s{r['seed']}" for r in eq['rows'] if not r['identical']}
    return bad


def stage_R(model):
    out = H / 'data' / 'raw' / f'R_{model}.json'
    if out.exists():
        print('already done', model); return
    log_concurrency(f'R {model}'); bad = ok_models(); t0 = time.perf_counter()
    dur = []; fps = []; eigs = []; pairs = []; skipped = []
    for s in SEEDS:
        if f'{model}_s{s}' in bad:
            skipped.append(s); continue
        sd, _ = load_state(model, s); fwd, base = build(model, sd); h0 = statehash(sd)
        ops = ('ADD',) if model.startswith('V055') else OPS
        dur += durability(fwd, ops, dict(model=model, seed=s))
        ends = {}
        for op in ops:
            for st in (0, 1):
                for pair in HAND[(op, st)]:
                    row, ev, h = fixed_point(base, op, st, pair); row.update(model=model, seed=s); fps.append(row)
                    eigs += [dict(model=model, seed=s, **e) for e in ev]; ends[(op, st, pair)] = (h, row['converged'])
            for pair in PASS[op]:
                (h0e, c0), (h1e, c1) = ends[(op, 0, pair)], ends[(op, 1, pair)]
                d = float((h0e - h1e).norm())
                pairs.append(dict(model=model, seed=s, op=op, pair=pair, both_converged=int(c0 and c1), distance=d, n_points=1 if d < 1e-3 else 2))
        assert statehash(base.state_dict()) == h0
        print(model, s, 'exact', sum(r['exact'] for r in dur if r['seed'] == s), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(durability=dur, fixed_points=fps, eigs=eigs, pairs=pairs, skipped_seeds=skipped, wall_seconds=time.perf_counter() - t0)), encoding='utf-8')


def v054t_broken():
    st = {}
    with (V054T / 'results_per_problem.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            key = (int(r['seed']), r['op'], int(r['state']), r['pair'])
            st[key] = st.get(key, True) and r['exact'] == '1'
    return sorted(k for k, v in st.items() if not v)


def intervene(sd, c):
    sd = {k: v.clone() for k, v in sd.items()}
    if c != 'orig':
        sd['residual_gate'] = torch.atanh(torch.tensor(float(c), dtype=sd['residual_gate'].dtype)).reshape(sd['residual_gate'].shape)
    return sd


def stage_I(seed):
    broken = [k for k in v054t_broken() if k[0] == seed]
    combos = [(op, st, p) for _, op, st, p in broken] + [('ADD', 1, '11'), ('SUB', 0, '00')]
    tests = {op: json.loads((WORK / 'IronMath_V053' / 'data' / f'{op}_Test2048.json').read_text()) for op in OPS}
    sd0, _ = load_state('V054_B0', seed); other0 = statehash(sd0, skip=('residual_gate',))
    for c in CS:
        out = H / 'data' / 'raw' / f'I_seed{seed}_c{c}.json'
        if out.exists():
            print('already done', seed, c); continue
        log_concurrency(f'I seed{seed} c={c}'); t0 = time.perf_counter()
        sd = intervene(sd0, c); assert statehash(sd, skip=('residual_gate',)) == other0
        fwd, base = build('V054_B0', sd); cval = float(torch.tanh(base.residual_gate))
        dur = durability(fwd, OPS, dict(model='V054_B0', seed=seed, c=str(c), c_value=cval))
        fps = []; eigs = []; dist = []
        for op, st, pair in combos:
            row, ev, h = fixed_point(base, op, st, pair); row.update(model='V054_B0', seed=seed, c=str(c), c_value=cval, V054T_broken=int((seed, op, st, pair) in broken))
            eigs += [dict(seed=seed, c=str(c), **e) for e in ev]
            if pair in PASS[op]:
                r2, _, h2 = fixed_point(base, op, 1 - st, pair)
                row.update(other_state_converged=r2['converged'], other_state_output=r2['h_star_output'], distance_to_other_state_endpoint=float((h - h2).norm()))
                row['n_points'] = 1 if row['distance_to_other_state_endpoint'] < 1e-3 else 2
            fps.append(row)
        acc = []
        for op in OPS:
            rows = tests[op]; correct = 0
            for j in range(0, 512, 128):
                rr = rows[j:j + 128]; n = 2049 if op == 'ADD' else 2048
                preds = predict(fwd, [(op, r['a'], r['b'], n) for r in rr])
                for r, p in zip(rr, preds):
                    correct += sum(v << i for i, v in enumerate(p)) == (r['a'] + r['b'] if op == 'ADD' else r['a'] - r['b'])
            acc.append(dict(seed=seed, c=str(c), c_value=cval, op=op, n=512, correct=correct, exact_rate=correct / 512))
        assert statehash(base.state_dict(), skip=('residual_gate',)) == other0
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(durability=dur, fixed_points=fps, eigs=eigs, accuracy=acc, other_params_sha_unchanged=True, saved_weights=False, wall_seconds=time.perf_counter() - t0)), encoding='utf-8')
        print('I', seed, c, round(cval, 4), 'exact', sum(r['exact'] for r in dur), [(a['op'], a['correct']) for a in acc], flush=True)


# ------------------------------------------------------------------ analysis
def table(dur, keys):
    g = defaultdict(list)
    for r in dur:
        g[tuple(r[k] for k in keys) + (r['op'], r['state'], r['pair'], r['k'])].append(r)
    rows = []
    for key, rr in sorted(g.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        offs = [r['hold_offset'] for r in rr if r['first_error_region'] == 'hold']
        rows.append(dict(zip(keys + ['op', 'state', 'pair', 'k'], key), exact=sum(r['exact'] for r in rr), n=len(rr),
                         hold_breaks=len(offs), hold_break_median=statistics.median(offs) if offs else None))
    return rows


def break_step(dur, key_filter):
    offs = [r['hold_offset'] for r in dur if key_filter(r) and r['first_error_region'] == 'hold']
    fails = [r for r in dur if key_filter(r) and not r['exact']]
    return (statistics.median(offs) if offs else (math.inf if not fails else None)), len(fails)


def stage_analyze():
    log_concurrency('analyze'); cfg = load('config.json'); summary = {}
    # ---------------- R
    dur = []; fps = []; eigs = []; pairs = []
    for m in R_MODELS:
        d = json.loads((H / 'data' / 'raw' / f'R_{m}.json').read_text(encoding='utf-8'))
        dur += d['durability']; fps += d['fixed_points']; eigs += d['eigs']; pairs += d['pairs']
    rt = table(dur, ['model', 'seed']); csvout('R_durability.csv', rt)
    cell4096 = {(r['model'], r['seed'], r['op'], r['state'], r['pair']): r['exact'] == 16 for r in rt if r['k'] == 4096}
    for r in fps:
        key = (r['model'], r['seed'], r['op'], r['state'], r['pair'])
        b, nf = break_step(dur, lambda q: (q['model'], q['seed'], q['op'], q['state'], q['pair']) == key)
        r.update(observed_unbroken_k4096=int(cell4096[key]), break_step_median=b if b != math.inf else None, failures_all_k=nf,
                 predicted_unbroken=r['h_star_output_correct'] if r['converged'] else None)
        r['R1_agree'] = None if not r['converged'] else int(r['predicted_unbroken'] == r['observed_unbroken_k4096'])
    csvout('R_fixed_points.csv', fps); csvout('R_jacobian_eigs.csv', eigs); csvout('R_pairs.csv', pairs)
    conv = [r for r in fps if r['converged']]; agree = sum(r['R1_agree'] for r in conv); rate = agree / len(conv)
    R1 = '支持' if rate >= 0.95 else '不支持' if rate < 0.80 else '判定不能'
    r2rows = []
    for m in R_MODELS:
        for s in SEEDS:
            rr = [r for r in conv if r['model'] == m and r['seed'] == s]
            if not rr:
                continue
            med = statistics.median(r['spectral_radius'] for r in rr); tg = rr[0]['tanh_g']
            r2rows.append(dict(model=m, seed=s, median_spectral_radius=med, one_plus_tanh_g=1 + tg, diff=med - (1 + tg), holds=int(abs(med - (1 + tg)) <= 0.01), n=len(rr)))
    n2 = sum(r['holds'] for r in r2rows); R2 = '支持' if n2 == len(r2rows) else '不支持' if n2 == 0 else '判定不能'
    r3 = []
    for m in R_MODELS:
        for s in SEEDS:
            pp = [p for p in pairs if p['model'] == m and p['seed'] == s]
            if pp:
                r3.append(dict(model=m, seed=s, pairs=len(pp), two_points=sum(p['n_points'] == 2 for p in pp), fraction_two=sum(p['n_points'] == 2 for p in pp) / len(pp)))
    frac = {m: [x['fraction_two'] for x in r3 if x['model'] == m] for m in R_MODELS}
    summary['R'] = dict(R1=dict(converged=len(conv), non_converged=len(fps) - len(conv), agree=agree, rate=rate, verdict=R1,
                                disagreements=[{k: r[k] for k in ('model', 'seed', 'op', 'state', 'pair', 'h_star_output_correct', 'observed_unbroken_k4096', 'break_step_median', 'failures_all_k')} for r in conv if not r['R1_agree']]),
                        R2=dict(rows=r2rows, verdict=R2), R3=r3,
                        R4=dict(F0_mean_fraction_two=statistics.mean(frac['V055_F0']), F1_mean_fraction_two=statistics.mean(frac['V055_F1']), per_seed_F0=frac['V055_F0'], per_seed_F1=frac['V055_F1']))
    # ---------------- I
    idur = []; ifps = []; iacc = []; ieig = []
    for s in SEEDS:
        for c in CS:
            d = json.loads((H / 'data' / 'raw' / f'I_seed{s}_c{c}.json').read_text(encoding='utf-8'))
            idur += d['durability']; ifps += d['fixed_points']; iacc += d['accuracy']; ieig += d['eigs']
    it = table(idur, ['seed', 'c']); csvout('I_durability.csv', it); csvout('I_accuracy_2048.csv', iacc); csvout('data/I_jacobian_eigs.csv', ieig)
    broken = v054t_broken(); bsteps = []
    for (s, op, st, pair) in broken:
        row = dict(seed=s, op=op, state=st, pair=pair)
        for c in CS:
            b, nf = break_step(idur, lambda q: (q['seed'], q['c'], q['op'], q['state'], q['pair']) == (s, str(c), op, st, pair))
            row[f'break_{c}'] = 'inf' if b == math.inf else b; row[f'failures_{c}'] = nf
        vals = [math.inf if row[f'break_{c}'] == 'inf' else row[f'break_{c}'] for c in ('orig', -0.05, -0.02)]
        row['I2_holds'] = int(None not in vals and vals[0] <= vals[1] <= vals[2] and vals[0] < vals[2])
        bsteps.append(row)
    csvout('data/I_break_steps.csv', bsteps)
    for r in ifps:
        r['break_step_median'] = next((b[f"break_{r['c'] if r['c'] == 'orig' else float(r['c'])}"] for b in bsteps if (b['seed'], b['op'], b['state'], b['pair']) == (r['seed'], r['op'], r['state'], r['pair'])), None)
    csvout('I_fixed_points.csv', ifps)
    i1 = []; i1_excl = 0
    for s in SEEDS:
        for c in CS:
            rr = [r for r in ifps if r['seed'] == s and r['c'] == str(c)]; cc = [r for r in rr if r['converged']]; i1_excl += len(rr) - len(cc)
            if cc:
                med = statistics.median(r['spectral_radius'] for r in cc); cv = rr[0]['c_value']
                i1.append(dict(seed=s, c=str(c), c_value=cv, median_spectral_radius=med, one_plus_c=1 + cv, diff=med - (1 + cv), holds=int(abs(med - 1 - cv) <= 0.02), converged=len(cc), measured=len(rr)))
            else:
                i1.append(dict(seed=s, c=str(c), holds=None, converged=0, measured=len(rr)))
    judged = [r for r in i1 if r['holds'] is not None]; n1 = sum(r['holds'] for r in judged)
    I1 = '支持' if judged and n1 == len(judged) else '不支持' if n1 == 0 else '判定不能'
    n2i = sum(r['I2_holds'] for r in bsteps); I2 = '支持' if n2i == len(bsteps) else '不支持' if n2i == 0 else '判定不能'
    i3 = []
    for (s, op, st, pair) in broken:
        r = [q for q in ifps if (q['seed'], q['c'], q['op'], q['state'], q['pair']) == (s, '-0.02', op, st, pair)][0]
        ok = int(r['converged'] == 1 and r.get('other_state_converged') == 1 and r['distance_to_other_state_endpoint'] < 1e-3)
        i3.append(dict(seed=s, op=op, state=st, pair=pair, distance=r['distance_to_other_state_endpoint'], converged=r['converged'], other_converged=r.get('other_state_converged'), holds=ok))
    n3 = sum(r['holds'] for r in i3); I3 = '支持' if n3 == len(i3) else '不支持' if n3 == 0 else '判定不能'
    summary['I'] = dict(I1=dict(rows=i1, excluded_non_converged=i1_excl, verdict=I1), I2=dict(rows=bsteps, holds=n2i, verdict=I2), I3=dict(rows=i3, holds=n3, verdict=I3), I4=iacc)
    dump('data/analysis_summary.json', summary)
    print(json.dumps(dict(R1=summary['R']['R1']['verdict'], R1rate=rate, R2=R2, I1=I1, I2=I2, I3=I3), ensure_ascii=False))


if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'R':
        stage_R(sys.argv[2])
    elif st == 'I':
        stage_I(int(sys.argv[2]))
    else:
        dict(assets=stage_assets, repro=stage_repro, equiv=stage_equiv, analyze=stage_analyze)[st]()
