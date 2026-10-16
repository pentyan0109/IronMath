"""Inference-only regeneration for decimal D11 (eval + probes + fixed points) and D10 (probes + fixed points).
usage: python -B infer_decimal.py d11 GROUP SEED | python -B infer_decimal.py d10 SEED
Settings read from the original code (IronMath_D11/d11.py, decimal_d10/d10.py) and recorded in settings.json:
  D11: torch.set_num_threads(2) (PREREGISTRATION training.threads), deterministic algorithms on, best_validation.pt,
       torch.jit.script(Residual()).eval(), batch 128 in file order, argmax; probes grouped by k in file order, batch 128.
  D10: as D11 but torch.set_num_threads(1); probes only (plus fixed points).
  Fixed points: eager Residual, step h=core(projection(x),h)+tanh(g)*h, start = hidden before hold of the k=1024 p0/s0 probe,
       iterate to 3000 then 20000, converged iff max of last 100 moves < 1e-6; Jacobian eigvals in float64.
Only raw argmax digits (LSB-first digit strings) and fixed-point readouts are written; grading is done by grade.py."""
import random, sys, time
import numpy as np
import torch
from rv_common import *

EXP = sys.argv[1]
if EXP == 'd11':
    GROUP, SEED = sys.argv[2], int(sys.argv[3]); ROOT = W / 'IronMath_D11'; THREADS = 2; TAG = f'{GROUP}_s{SEED}'
    CK = ROOT / 'checkpoints' / TAG; MODEL = ROOT / 'model_d11.py'; TESTDIR = ROOT / 'data' / 'test'
    RUNS = dict(eval=ROOT / 'data' / 'raw' / f'eval_{TAG}_run.json', probes=ROOT / 'data' / 'raw' / f'probes_{TAG}_run.json', fixed=ROOT / 'data' / 'raw' / f'fixed_{TAG}_run.json')
    KS = load(ROOT / 'PREREGISTRATION.json')['probes']['k']; WIDTHS = load(ROOT / 'PREREGISTRATION.json')['test']['widths']
else:
    GROUP, SEED = None, int(sys.argv[2]); ROOT = W / 'decimal_d10'; THREADS = 1; TAG = f's{SEED}'
    CK = ROOT / 'checkpoints' / TAG; MODEL = ROOT / 'model_d10.py'; TESTDIR = ROOT / 'data'
    RUNS = dict(eval=ROOT / 'data' / 'raw' / f'eval_seed{SEED}_run.json', probes=ROOT / 'data' / 'raw' / f'probes_seed{SEED}_run.json', fixed=ROOT / 'data' / 'raw' / f'fixed_seed{SEED}_run.json')
    KS = load(ROOT / 'config.json')['data']['probes']['k']; WIDTHS = None
OD = OUT / EXP / TAG
PASS = [(x, 9 - x) for x in range(10)]
NONPASS = [(0, (0, 0)), (0, (4, 4)), (0, (8, 0)), (1, (9, 9)), (1, (5, 5)), (1, (1, 9))]
COMBOS = [(c, p) for c in (0, 1) for p in PASS] + NONPASS
EYE = np.eye(10, dtype=np.float32)


def digs(v, n):
    s = str(v)[::-1]; assert len(s) <= n
    return [int(c) for c in s] + [0] * (n - len(s))


def enc(A, B):
    return torch.from_numpy(np.concatenate([EYE[np.asarray(A, np.int64)], EYE[np.asarray(B, np.int64)]], -1))


@torch.no_grad()
def argmax(m, x):
    z = m(x); assert torch.isfinite(z).all()
    return z.argmax(-1).numpy()


def step_fn(base, x, h):
    h = base.core(base.projection(x), h) + torch.tanh(base.residual_gate) * h
    return h, base.head(h)


def main():
    torch.set_num_threads(THREADS); torch.use_deterministic_algorithms(True); torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    shalog = []; t_all = time.perf_counter()
    rec = load(CK / 'training_record.json')
    check_sha(CK / 'best_validation.pt', rec['best_checkpoint_file_SHA256'], shalog)
    sums = {l.split()[1]: l.split()[0] for l in (TESTDIR / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines() if l.strip()}
    Residual = load_class(MODEL, 'Residual')
    cp = torch.load(CK / 'best_validation.pt', map_location='cpu', weights_only=False)
    base = Residual(); base.load_state_dict(cp['model']); base.eval(); m = torch.jit.script(base).eval(); h0 = modelhash(base)
    wh = {k: load(p)['weights_sha256'] for k, p in RUNS.items() if p.exists()}
    assert all(v == h0 for v in wh.values()), ('weights hash differs from original run record', wh, h0)
    timing = {}
    # ---- main evaluation (D11 only)
    if EXP == 'd11':
        out = []
        for Wd in WIDTHS:
            for part in (1, 2):
                nm = f'Test{Wd}_part{part}.json'; check_sha(TESTDIR / nm, sums[nm], shalog)
                rows = load(TESTDIR / nm); n = Wd + 1; t = time.perf_counter()
                for j in range(0, len(rows), 128):
                    rr = rows[j:j + 128]
                    P = argmax(m, enc([digs(int(r['a']), n) for r in rr], [digs(int(r['b']), n) for r in rr]))
                    out += [dict(width=Wd, part=part, id=r['id'], pred_digits_lsb=''.join(str(int(v)) for v in p[:n])) for r, p in zip(rr, P)]
                timing[f'eval_W{Wd}_part{part}'] = time.perf_counter() - t
                print(TAG, 'eval', Wd, part, round(timing[f'eval_W{Wd}_part{part}'], 1), flush=True)
        jl_write(OD / 'eval_pred.jsonl.gz', out)
    # ---- durability probes
    check_sha(TESTDIR / 'probes.jsonl', sums['probes.jsonl'], shalog)
    probes = [json.loads(l) for l in (TESTDIR / 'probes.jsonl').read_text(encoding='utf-8').splitlines()]
    out = []; t = time.perf_counter()
    for k in KS:
        batch = [r for r in probes if r['k'] == k]; n = batch[0]['seq_len']; assert all(r['seq_len'] == n for r in batch)
        A = [digs(int(r['a']), n) for r in batch]; B = [digs(int(r['b']), n) for r in batch]
        P = np.concatenate([argmax(m, enc(A[j:j + 128], B[j:j + 128])) for j in range(0, len(batch), 128)])
        out += [dict(pid=r['pid'], k=k, state=r['state'], pair=r['pair'], prefix=r['prefix'], suffix=r['suffix'], pred_digits_lsb=''.join(str(int(v)) for v in p[:n])) for r, p in zip(batch, P)]
    timing['probes'] = time.perf_counter() - t; print(TAG, 'probes', round(timing['probes'], 1), flush=True)
    jl_write(OD / 'probe_pred.jsonl.gz', out)
    # ---- fixed points (needed for D11 H2 and the read-vs-hold cells)
    for p in base.parameters():
        p.requires_grad_(False)
    sel = {(r['state'], r['pair']): r for r in probes if r['k'] == 1024 and r['prefix'] == 'p0' and r['suffix'] == 's0'}
    fps = []; ends = {}; t = time.perf_counter()
    for c, (x, y) in COMBOS:
        r = sel[(c, f'{x}{y}')]; n = r['seq_len']
        X = enc([digs(int(r['a']), n)], [digs(int(r['b']), n)]); h = torch.zeros(1, 545)
        with torch.no_grad():
            for tt in range(r['hold_start']):
                h, _ = step_fn(base, X[:, tt], h)
            xin = torch.zeros(1, 20); xin[0, x] = 1; xin[0, 10 + y] = 1; assert torch.equal(xin, X[:, r['hold_start']])
            g = h.clone(); moves = []; it = 0; conv = False
            for limit in (3000, 20000):
                while it < limit:
                    gn, _ = step_fn(base, xin, g); moves.append(float((gn - g).norm())); g = gn; it += 1
                conv = max(moves[-100:]) < 1e-6
                if conv:
                    break
            _, z = step_fn(base, xin, g); o = int(z.argmax(-1))
        ends[(c, f'{x}{y}')] = g.clone()
        row = dict(state=c, pair=f'{x}{y}', iterations=it, converged=int(conv), last100_move_max=max(moves[-100:]), h_star_norm=float(g.norm()), h_star_output=o, spectral_radius=None)
        if conv:
            J = torch.autograd.functional.jacobian(lambda hh: step_fn(base, xin, hh.unsqueeze(0))[0].squeeze(0), g.squeeze(0).detach())
            row['spectral_radius'] = float(torch.linalg.eigvals(J.double()).abs().max())
        fps.append(row)
    pairs = [dict(pair=f'{x}{y}', distance_c0_vs_c1=float((ends[(0, f'{x}{y}')] - ends[(1, f'{x}{y}')]).norm())) for x, y in PASS]
    timing['fixed'] = time.perf_counter() - t; print(TAG, 'fixed', round(timing['fixed'], 1), flush=True)
    assert modelhash(base) == h0
    dump(OD / 'fixed_points.json', dict(fixed_points=fps, pairs=pairs))
    dump(OD / 'run.json', dict(experiment=EXP, group=GROUP, seed=SEED, weights_sha256=h0, original_run_weights_sha256=wh, sha_checks=shalog, timing=timing,
                               total_wall=time.perf_counter() - t_all, env=env(torch, THREADS), jit=True, batch=128, checkpoint='best_validation.pt'))


if __name__ == '__main__':
    main()
