"""S2: checkpoint hashes + FP32 reproduction of saved FIXED-INPUT hold results (FREEZE_PLAN 5). No mixed sequence is run."""
import csv, json, sys, time
import numpy as np
import torch
from common import *

torch.use_deterministic_algorithms(True); torch.set_num_threads(2)
REPO = HERE.parents[1]
E3_HOLD = SCRATCH / 'res/archive_v074/results/IronMath_PaperCheck/extension/E3_independent_replication/E3_HOLD_RESULTS.csv'
V054T_RES = SCRATCH / 'res2/archive_v074/results/v054t_transition/results_per_problem.csv'
PROBES = SCRATCH / 'ev/archive_v074/evaluation_sets/v054t_transition/data/probes.jsonl'


def rows(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


@torch.no_grad()
def run_full(m, probes):
    n = max(r['seq_len'] for r in probes); x = torch.zeros(len(probes), n, 3)
    for i, r in enumerate(probes):
        a, b = int(r['a'], 16), int(r['b'], 16)
        for t in range(r['seq_len']):
            x[i, t] = torch.tensor([(a >> t) & 1, (b >> t) & 1, int(r['op'] == 'SUB')])
    h = torch.zeros(len(probes), 545); P = []
    for t in range(n):
        h, z = step(m, x[:, t], h); P.append(z.argmax(-1))
    P = torch.stack(P, 1).numpy(); out = []
    for i, r in enumerate(probes):
        ans = int(r['answer'], 16); truth = np.array([(ans >> t) & 1 for t in range(r['seq_len'])])
        w = np.flatnonzero(P[i, :r['seq_len']] != truth)
        hw = w[(w >= r['hold_start']) & (w < r['hold_end'])]
        out.append(dict(first_error=int(w[0]) if len(w) else None, first_hold_error=int(hw[0] - r['hold_start']) + 1 if len(hw) else None,
                        prefix_error=bool(len(w) and w[0] < r['hold_start'])))
    return out


def main():
    ck = {r['model_id']: r for r in rows(REPO / 'CHECKPOINTS.csv') if r['role'] == 'primary'}
    probes = {}
    for l in PROBES.read_text(encoding='utf-8').splitlines():
        r = json.loads(l); probes[r['pid']] = r
    e3 = rows(E3_HOLD); vt = rows(V054T_RES)
    report = dict(checkpoints=[], reproduction=[], all_ok=True)
    for g, model in MODELS:
        p = ckpt_path(model); h = sha(p); ok = h == ck[model]['SHA256']
        m = load_model(model, torch.float32); mh = modelhash(m)
        report['checkpoints'].append(dict(model=model, cluster=g, file_sha256=h, expected=ck[model]['SHA256'], match=ok, param_hash=mh,
                                          initial_weights_sha256=ck[model]['initial_weights_SHA256']))
        report['all_ok'] &= ok
        if model.startswith('C1_B0'):
            seed = model[-1]; cand = [r for r in vt if r['seed'] == seed and r['k'] == '1024']
        else:
            cand = [r for r in e3 if r['model'] == model and r['k'] == '1024']
        sel = {}
        for r in cand:
            key = (r['op'], r['state'], r['pair'])
            if key not in sel or int(r['pid']) < int(sel[key]['pid']):
                sel[key] = r
        sel = [sel[k] for k in sorted(sel)]; assert len(sel) == 12, (model, len(sel))
        got = run_full(m, [probes[int(r['pid'])] for r in sel])
        for r, gt in zip(sel, got):
            if model.startswith('C1_B0'):
                exp = None if r['first_error'] == 'NA' else int(r['first_error']); obs = gt['first_error']; field = 'first_error'
            else:
                exp = None if r['first_hold_error'] == '' else int(r['first_hold_error']); obs = gt['first_hold_error']; field = 'first_hold_error'
            match = exp == obs
            report['reproduction'].append(dict(model=model, op=r['op'], state=r['state'], pair=r['pair'], pid=int(r['pid']), field=field, expected=exp, observed=obs, match=match))
            report['all_ok'] &= match
        assert modelhash(m) == mh
        print(model, 'sha', ok, 'repro', sum(x['match'] for x in report['reproduction'] if x['model'] == model), '/12', flush=True)
    report['torch'] = torch.__version__; report['n_repro'] = len(report['reproduction']); report['n_match'] = sum(x['match'] for x in report['reproduction'])
    dump('S2_integrity.json', report); print('ALL_OK', report['all_ok'], report['n_match'], '/', report['n_repro'])


if __name__ == '__main__':
    main()
