"""S4: fixed-input trajectories (METHOD 3.1), prefix checks, informative contexts (METHOD 2.4) and frozen predictions
for P1, B3 (=P1 with '+' trajectories only; extra same-information baseline), B1, B1-add, B0.
Usage: python fixed.py fp64|fp32. Uses ONLY fixed-input runs (a pair repeated from a post-prefix state)."""
import sys, time, resource
import numpy as np
import torch
from common import *
from predictors import *

torch.use_deterministic_algorithms(True); torch.set_num_threads(2)


def xvec(op, pair, dtype):
    return torch.tensor([[pair[0], pair[1], int(op == 'SUB')]], dtype=dtype)


@torch.no_grad()
def run_prefix(m, op, pr, dtype):
    """Returns post-prefix state and whether any prefix output was wrong (vs independent ripple)."""
    h = torch.zeros(1, 545, dtype=dtype); out, _, _ = ripple(op, pr['a'], pr['b']); err = False
    z = m.head(h)
    for t, (a, b) in enumerate(zip(pr['a'], pr['b'])):
        h, z = step(m, xvec(op, (a, b), dtype), h)
        err |= bool(int(z.argmax(-1)) != out[t]) or not bool(torch.isfinite(z).all())
    return h, err


@torch.no_grad()
def trajectories(m, h0s, xs, ycs, dtype):
    """Batch fixed-input iteration. Returns margins (B, T+1) float64, mu(0) = margin of the start state."""
    h = torch.cat(h0s); x = torch.cat(xs); yc = torch.tensor(ycs)
    idx = torch.arange(len(ycs)); M = np.empty((len(ycs), T + 1))
    z = m.head(h); M[:, 0] = (z[idx, yc] - z[idx, 1 - yc]).double().numpy()
    for t in range(1, T + 1):
        h, z = step(m, x, h); M[:, t] = (z[idx, yc] - z[idx, 1 - yc]).double().numpy()
    return M


def main(prec):
    dtype = torch.float64 if prec == 'fp64' else torch.float32
    seqs = load('data/sequences.json'); t0 = time.time()
    traj_rows = []; ctx_rows = []; pred_rows = []; times = {}
    for g, model in MODELS:
        ts = time.time(); m = load_model(model, dtype); mh = modelhash(m)
        h0s = []; xs = []; ycs = []; keys = []; prefix_err = {}
        for op in OPS:
            for c in (0, 1):
                yc = y_class(op, c)
                for j in range(4):
                    hp, ep = run_prefix(m, op, seqs['prefixes'][f'{op}{c}'][j], dtype)
                    hm, em = run_prefix(m, op, seqs['prefixes'][f'{op}{1 - c}'][j], dtype)
                    prefix_err[(op, c, j)] = (ep, em)
                    for xi, pair in zip('pq', TWO_SIDED[op]):
                        for s, h0 in (('+', hp), ('-', hm)):
                            h0s.append(h0); xs.append(xvec(op, pair, dtype)); ycs.append(yc); keys.append((op, c, j, xi, s))
        M = trajectories(m, h0s, xs, ycs, dtype); assert modelhash(m) == mh
        np.save(OUT / f'S4_{prec}/traj_{model}.npy', M)
        traj_rows.append(dict(model=model, file=f'traj_{model}.npy', sha256=sha(OUT / f'S4_{prec}/traj_{model}.npy'), param_hash=mh))
        mu_all = {k: M[i] for i, k in enumerate(keys)}
        for op in OPS:
            for c in (0, 1):
                yc = y_class(op, c); pats = seqs['patterns'][f'{op}{c}']; ps = [p['pattern'] for p in pats]
                for j in range(4):
                    mu = {x: {s: mu_all[(op, c, j, x, s)] for s in '+-'} for x in 'pq'}
                    tp = lifetime(mu['p']['+'], yc); tq = lifetime(mu['q']['+'], yc)
                    ep, em = prefix_err[(op, c, j)]
                    informative = (tp is None) != (tq is None)
                    nonmono = {x: float(np.mean(np.diff(mu[x]['+']) > 0)) for x in 'pq'}
                    ctx_rows.append(dict(model=model, cluster=g, op=op, c=c, j=j, tau_p=tp, tau_q=tq, prefix_error=ep, minus_prefix_error=em,
                                         informative=informative and not ep, excluded_prefix_error=ep,
                                         frac_increasing_steps_p=nonmono['p'], frac_increasing_steps_q=nonmono['q']))
                    preds = {}
                    preds['P1'] = p1_predict(mu, ps, yc, use_minus=True)[:2]
                    preds['B3'] = p1_predict(mu, ps, yc, use_minus=False)[:2]
                    preds['B1'] = b1_run(ps, tp, tq); preds['B1add'] = b1_add(ps, tp, tq); preds['B0'] = b0(ps)
                    for k, pt in enumerate(pats):
                        pred_rows.append(dict(model=model, cluster=g, op=op, c=c, j=j, pattern=pt['name'],
                                              **{f'{meth}_fail': bool(v[0][k]) for meth, v in preds.items()},
                                              **{f'{meth}_tau': int(v[1][k]) for meth, v in preds.items()}))
        times[model] = time.time() - ts
        print(prec, model, round(times[model], 1), 's', flush=True)
    import csv
    for name, rr in (('contexts.csv', ctx_rows), ('predictions.csv', pred_rows), ('trajectory_files.csv', traj_rows)):
        with open(OUT / f'S4_{prec}/{name}', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, list(rr[0])); w.writeheader(); w.writerows(rr)
    dump(f'S4_{prec}/run.json', dict(precision=prec, torch=torch.__version__, threads=2, seconds=time.time() - t0, per_model_seconds=times,
                                     peak_rss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                                     n_contexts=len(ctx_rows), n_informative=sum(r['informative'] for r in ctx_rows), n_predictions=len(pred_rows)))
    print('done', prec, 'informative contexts', sum(r['informative'] for r in ctx_rows), '/', len(ctx_rows))


if __name__ == '__main__':
    (OUT / f'S4_{sys.argv[1]}').mkdir(parents=True, exist_ok=True); main(sys.argv[1])
