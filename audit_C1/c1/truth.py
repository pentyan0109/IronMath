"""S5 (B2, steps=ELL) and S6 (truth, steps=H): run the mixed hold sequences through the model.
Usage: python truth.py b2|truth fp64|fp32 MODEL[,MODEL...]   -> out/<stage>_<prec>/<model>.csv"""
import csv, sys, time, resource
import numpy as np
import torch
from common import *

torch.use_deterministic_algorithms(True); torch.set_num_threads(2)


@torch.no_grad()
def run_model(model, steps, dtype, seqs):
    m = load_model(model, dtype); mh = modelhash(m)
    h0 = []; X = []; Y = []; meta = []
    for op in OPS:
        for c in (0, 1):
            p, q = TWO_SIDED[op]; yc = y_class(op, c)
            for j, pr in enumerate(seqs['prefixes'][f'{op}{c}']):
                h = torch.zeros(1, 545, dtype=dtype)
                for a, b in zip(pr['a'], pr['b']):
                    h, _ = step(m, torch.tensor([[a, b, int(op == 'SUB')]], dtype=dtype), h)
                for pt in seqs['patterns'][f'{op}{c}']:
                    h0.append(h); meta.append(dict(model=model, op=op, c=c, j=j, pattern=pt['name'])); Y.append(yc)
                    X.append(np.array([[*(p if ch == '1' else q), int(op == 'SUB')] for ch in pt['pattern'][:steps]], np.float64))
    h = torch.cat(h0); X = torch.from_numpy(np.stack(X)).to(dtype); yc = torch.tensor(Y); n = len(Y)
    first = np.full(n, -1); nwrong = np.zeros(n, int); recov = np.zeros(n, int); nonfin = np.zeros(n, bool); prev = np.zeros(n, bool)
    for t in range(steps):
        h, z = step(m, X[:, t], h)
        fin = torch.isfinite(z).all(-1).numpy(); w = (z.argmax(-1) != yc).numpy() | ~fin
        newf = w & (first < 0); first[newf] = t + 1
        nwrong += w; recov += prev & ~w; prev = w; nonfin |= ~fin
    assert modelhash(m) == mh
    return [dict(**mt, first_error=int(first[i]), n_wrong=int(nwrong[i]), recoveries=int(recov[i]), nonfinite=bool(nonfin[i]))
            for i, mt in enumerate(meta)], mh


def main(stage, prec, models):
    steps = ELL if stage == 'b2' else H; dtype = torch.float64 if prec == 'fp64' else torch.float32
    seqs = load('data/sequences.json'); d = OUT / f'{stage}_{prec}'; d.mkdir(parents=True, exist_ok=True)
    for model in models:
        t0 = time.time(); rr, mh = run_model(model, steps, dtype, seqs)
        with open(d / f'{model}.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, list(rr[0])); w.writeheader(); w.writerows(rr)
        dump(f'{stage}_{prec}/{model}.run.json', dict(model=model, steps=steps, precision=prec, seconds=time.time() - t0, param_hash=mh,
                                                     peak_rss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, torch=torch.__version__))
        print(stage, prec, model, round(time.time() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(','))
