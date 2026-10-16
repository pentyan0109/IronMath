"""Inference-only regeneration of V054R experiment R durability probes (+ fixed points).
usage: python -B infer_v054r.py V051_A1|V055_F0|V055_F1|V055_F2
Settings read from v054r_replication/v054r.py and config.json, recorded in settings.json:
  torch.set_num_threads(1); deterministic algorithms not set in the original (left default here too);
  V051_A1: eager Model('A1_residual') from IronMath_V4/v051/src/model.py, checkpoint loaded with weights_only=True; ADD and SUB.
  V055_F*: torch.jit.script(Residual()) from IronMath_V055/core.py; ADD only.
  Probes v054t_transition/data/probes.jsonl; one forward per (op, k) over all probes of that op and k (file order, zero padded to max length);
  x_t=(a_t, b_t, op==SUB) LSB-first; argmax over 2 classes.
  Fixed points: hidden before hold of the k=4096 'min'/'s0' probe, 3000 then 20000 iterations, converged iff max of last 100 moves < 1e-6."""
import copy, sys, time
import numpy as np
import torch
from rv_common import *

MODEL = sys.argv[1]; SEEDS = (0, 1, 2); KS = (8, 16, 64, 256, 1024, 4096)
HAND = {('ADD', 0): ['00', '01', '10'], ('ADD', 1): ['01', '10', '11'], ('SUB', 0): ['00', '10', '11'], ('SUB', 1): ['00', '01', '11']}
PASS = {'ADD': ['01', '10'], 'SUB': ['00', '11']}
ASSETS = load(W / 'v054r_replication' / 'asset_paths.json')['files']
OD = OUT / 'v054r' / MODEL


def bits(v, n):
    return [(v >> i) & 1 for i in range(n)]


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
    return [''.join(str(int(v)) for v in p[i, :n]) for i, (*_, n) in enumerate(items)]


def step_fn(base, x, h):
    h = base.core(base.projection(x), h) + torch.tanh(base.residual_gate) * h
    return h, base.head(h)


def main():
    torch.set_num_threads(1); shalog = []; t_all = time.perf_counter()
    pp = W / 'v054t_transition' / 'data' / 'probes.jsonl'; check_sha(pp, ASSETS['V054T_probes']['sha256'], shalog)
    probes = [json.loads(l) for l in pp.read_text(encoding='utf-8').splitlines()]
    if MODEL == 'V051_A1':
        check_sha(Path(ASSETS['V051_model']['path']), ASSETS['V051_model']['sha256'], shalog); Cls = load_class(ASSETS['V051_model']['path'], 'Model')
    else:
        check_sha(Path(ASSETS['V055_core']['path']), ASSETS['V055_core']['sha256'], shalog); Cls = load_class(ASSETS['V055_core']['path'], 'Residual')
    ops = ('ADD',) if MODEL.startswith('V055') else ('ADD', 'SUB')
    dur = []; fps = []; pairs = []; timing = {}
    for s in SEEDS:
        a = ASSETS[f'{MODEL}_s{s}']; p = Path(a['path']); check_sha(p, a['sha256'], shalog)
        cp = torch.load(p, weights_only=(MODEL == 'V051_A1'), map_location='cpu')
        if MODEL == 'V051_A1':
            base = Cls('A1_residual'); base.load_state_dict(cp['model']); base.eval(); fwd = base
        else:
            base = Cls(); base.load_state_dict(cp['model']); base.eval(); fwd = torch.jit.script(copy.deepcopy(base)).eval()
        for q in base.parameters():
            q.requires_grad_(False)
        h0 = modelhash(base); t = time.perf_counter()
        for op in ops:
            for k in KS:
                batch = [r for r in probes if r['op'] == op and r['k'] == k]
                preds = predict(fwd, [(op, int(r['a'], 16), int(r['b'], 16), r['seq_len']) for r in batch])
                dur += [dict(seed=s, pid=r['pid'], op=op, k=k, state=r['state'], pair=r['pair'], prefix=r['prefix'], suffix=r['suffix'], pred_bits_lsb=pr) for r, pr in zip(batch, preds)]
        timing[f'probes_s{s}'] = time.perf_counter() - t; print(MODEL, s, 'probes', round(timing[f'probes_s{s}'], 1), flush=True)
        t = time.perf_counter(); ends = {}
        for op in ops:
            for st in (0, 1):
                for pair in HAND[(op, st)]:
                    r = [q for q in probes if q['op'] == op and q['state'] == st and q['pair'] == pair and q['k'] == 4096 and q['prefix'] == 'min' and q['suffix'] == 's0']
                    assert len(r) == 1; r = r[0]
                    x = torch.from_numpy(make_x(op, int(r['a'], 16), int(r['b'], 16), r['hold_start'] + 1)); h = torch.zeros(1, base.core.hidden_size)
                    with torch.no_grad():
                        for tt in range(r['hold_start']):
                            h, _ = step_fn(base, x[tt:tt + 1], h)
                        xin = x[r['hold_start']:r['hold_start'] + 1]; moves = []; it = 0; conv = False
                        for target in (3000, 20000):
                            while it < target:
                                hn, _ = step_fn(base, xin, h); moves.append(float((hn - h).norm())); h = hn; it += 1
                            conv = max(moves[-100:]) < 1e-6
                            if conv:
                                break
                        out = int(step_fn(base, xin, h)[1].argmax(-1))
                    row = dict(seed=s, op=op, state=st, pair=pair, iterations=it, converged=int(conv), last100_move_max=max(moves[-100:]), h_star_output=out, spectral_radius=None)
                    if conv:
                        J = torch.autograd.functional.jacobian(lambda hh: step_fn(base, xin, hh.unsqueeze(0))[0].squeeze(0), h.squeeze(0).detach())
                        row['spectral_radius'] = float(torch.linalg.eigvals(J.double()).abs().max())
                    fps.append(row); ends[(op, st, pair)] = h
            for pair in PASS[op]:
                pairs.append(dict(seed=s, op=op, pair=pair, distance=float((ends[(op, 0, pair)] - ends[(op, 1, pair)]).norm())))
        timing[f'fixed_s{s}'] = time.perf_counter() - t
        assert modelhash(base) == h0
    jl_write(OD / 'probe_pred.jsonl.gz', dur)
    dump(OD / 'fixed_points.json', dict(fixed_points=fps, pairs=pairs))
    dump(OD / 'run.json', dict(experiment='v054r', model=MODEL, sha_checks=shalog, timing=timing, total_wall=time.perf_counter() - t_all, env=env(torch, 1),
                               jit=not MODEL == 'V051_A1', batch='all probes of one (op,k) in one forward (96)'))


if __name__ == '__main__':
    main()
