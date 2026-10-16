"""Independent grading of the regenerated outputs and per-problem comparison with the original flags.
usage: python -B grade.py
Truth is computed twice: Python integers (a+b, or (a-b) mod 2^n) and an independent ripple written here (decimal carry / binary carry-borrow).
The regenerated prediction is exact iff its integer equals the Python-integer truth AND its digit list equals the ripple digits.
Original records are only read."""
import collections, csv
import numpy as np
from rv_common import *

sys.set_int_max_str_digits(0)
G = OUT / 'graded'
PASS = [(x, 9 - x) for x in range(10)]


def ripple10(a, b, n):
    A = [int(c) for c in str(a)[::-1]] + [0] * n; B = [int(c) for c in str(b)[::-1]] + [0] * n; c = 0; out = []
    for i in range(n):
        t = A[i] + B[i] + c; out.append(t % 10); c = t // 10
    assert c == 0 and all(v == 0 for v in A[n:] + B[n:])
    return out


def ripple2(op, a, b, n):
    c = 0; out = []
    for i in range(n):
        x = (a >> i) & 1; y = (b >> i) & 1
        if op == 'ADD':
            t = x + y + c; out.append(t & 1); c = t >> 1
        else:
            t = x - y - c; out.append(t & 1); c = 1 if t < 0 else 0
    return out


def grade10(pred, a, b):
    n = len(pred); truth = a + b; rp = ripple10(a, b, n); d = [int(ch) for ch in pred]
    pi = sum(v * 10**i for i, v in enumerate(d)) if n < 3000 else int(pred[::-1])
    e_int = pi == truth; e_rip = d == rp; assert e_int == e_rip, 'integer and ripple grading disagree'
    wrong = [i for i in range(n) if d[i] != rp[i]]
    return e_int, pi, wrong


def grade2(op, pred, a, b, answer_hex):
    n = len(pred); truth = (a + b) % (1 << n) if op == 'ADD' else (a - b) % (1 << n); assert truth == int(answer_hex, 16)
    rp = ripple2(op, a, b, n); d = [int(ch) for ch in pred]; pi = int(pred[::-1], 2)
    e_int = pi == truth; e_rip = d == rp; assert e_int == e_rip
    return e_int, pi, [i for i in range(n) if d[i] != rp[i]]


def csvout(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r)) or ['none']
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def bootstrap(d, n=10000, seed=66000):
    rng = np.random.default_rng(seed); d = np.array(d, float)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def region(fe, r):
    return None if fe is None else 'prefix' if fe < r['hold_start'] else 'hold' if fe < r['hold_end'] else 'suffix'


def main():
    summary = []; mism = []; fpcmp = []; res = {}
    # ------------------------------------------------ D11
    D11 = W / 'IronMath_D11'; tests = {}
    for Wd in (8, 64, 256, 2048):
        for part in (1, 2):
            tests[(Wd, part)] = {r['id']: r for r in load(D11 / 'data' / 'test' / f'Test{Wd}_part{part}.json')}
    probes11 = {r['pid']: r for r in jl_read(D11 / 'data' / 'test' / 'probes.jsonl')}
    H = {h: {g: {} for g in ('control', 'intervention')} for h in ('H1', 'H2', 'H3')}
    acc = collections.defaultdict(lambda: [0, 0, 0])  # (group,width,part) -> regenerated exact, original exact, n
    for g in ('control', 'intervention'):
        for s in range(10):
            tag = f'{g}_s{s}'; od = OUT / 'd11' / tag
            if not (od / 'run.json').exists():
                continue
            orig = {(r['width'], r['part'], r['id']): r for r in jl_read(D11 / 'data' / 'raw' / f'eval_{tag}.jsonl')}
            new = jl_read(od / 'eval_pred.jsonl.gz'); assert len(new) == len(orig) == sum(len(v) for v in tests.values())
            agree = dis = predcmp = predmis = 0; h3hit = 0
            for x in new:
                o = orig[(x['width'], x['part'], x['id'])]; t = tests[(x['width'], x['part'])][x['id']]
                ex, pi, wrong = grade10(x['pred_digits_lsb'], int(t['a']), int(t['b']))
                same = ex == bool(o['exact']); agree += same; dis += not same
                a_ = acc[(g, x['width'], x['part'])]; a_[0] += ex; a_[1] += o['exact']; a_[2] += 1
                if o['prediction'] is not None:  # original stored the integer only on failures
                    predcmp += 1; predmis += int(o['prediction']) != pi
                if x['width'] == 2048 and x['part'] == 1:
                    h3hit += ex
                if not same or (o['prediction'] is not None and int(o['prediction']) != pi):
                    mism.append(dict(experiment='D11_main', model=tag, item=f"W{x['width']}_p{x['part']}_{x['id']}", original_exact=o['exact'], regenerated_exact=int(ex),
                                     original_prediction=o['prediction'], regenerated_prediction=str(pi), regenerated_wrong_digits=len(wrong)))
            summary.append(dict(experiment='D11_main', model=tag, n=len(new), flag_agree=agree, flag_disagree=dis, stored_predictions_compared=predcmp, stored_prediction_mismatch=predmis))
            H['H3'][g][s] = h3hit / 512
            # probes
            orig = {r['pid']: r for r in jl_read(D11 / 'data' / 'raw' / f'probes_{tag}.jsonl')}; new = jl_read(od / 'probe_pred.jsonl.gz'); assert len(new) == len(orig) == 2080
            agree = dis = fieldmis = 0; cell = collections.defaultdict(list)
            for x in new:
                o = orig[x['pid']]; r = probes11[x['pid']]
                ex, pi, wrong = grade10(x['pred_digits_lsb'], int(r['a']), int(r['b'])); fe = wrong[0] if wrong else None
                same = ex == bool(o['exact']); agree += same; dis += not same
                fm = (fe != o['first_error']) or (len(wrong) != o['wrong_digits']) or (region(fe, r) != o['first_error_region']); fieldmis += fm
                cell[(r['state'], r['pair'], r['k'])].append(ex)
                if not same or fm:
                    mism.append(dict(experiment='D11_probe', model=tag, item=f"pid{x['pid']}_c{r['state']}_{r['pair']}_k{r['k']}", original_exact=o['exact'], regenerated_exact=int(ex),
                                     original_first_error=o['first_error'], regenerated_first_error=fe, original_wrong_digits=o['wrong_digits'], regenerated_wrong_digits=len(wrong)))
            summary.append(dict(experiment='D11_probe', model=tag, n=len(new), flag_agree=agree, flag_disagree=dis, derived_field_mismatch=fieldmis))
            H['H1'][g][s] = sum(all(cell[(1, f'{x}{y}', 1024)]) and len(cell[(1, f'{x}{y}', 1024)]) == 16 for x, y in PASS)
            # fixed points / H2
            fpn = load(od / 'fixed_points.json'); fpo = {(r['state'], r['pair']): r for r in jl_read(D11 / 'data' / 'raw' / f'fixed_{tag}.jsonl')}
            pao = {r['pair']: r for r in jl_read(D11 / 'data' / 'raw' / f'pairs_{tag}.jsonl')}; fpd = {}
            for r in fpn['fixed_points']:
                o = fpo[(r['state'], r['pair'])]; fpd[(r['state'], r['pair'])] = r
                fpcmp.append(dict(experiment='D11', model=tag, cell=f"c{r['state']}_{r['pair']}", converged_same=r['converged'] == o['converged'], iterations_same=r['iterations'] == o['iterations'],
                                  output_same=r['h_star_output'] == o['h_star_output'], rho_abs_diff=None if r['spectral_radius'] is None or o['spectral_radius'] is None else abs(r['spectral_radius'] - o['spectral_radius']),
                                  move_abs_diff=abs(r['last100_move_max'] - o['last100_move_max'])))
            h2 = 0
            for p in fpn['pairs']:
                r0 = fpd[(0, p['pair'])]; r1 = fpd[(1, p['pair'])]; c0 = int(p['pair'][0]) + int(p['pair'][1]); assert c0 == 9
                ok = bool(r0['converged'] and r1['converged'] and p['distance_c0_vs_c1'] >= 1e-3 and r0['spectral_radius'] < 1 and r1['spectral_radius'] < 1 and r0['h_star_output'] == 9 and r1['h_star_output'] == 0)
                h2 += ok
                fpcmp.append(dict(experiment='D11', model=tag, cell=f"pair_{p['pair']}", distance_abs_diff=abs(p['distance_c0_vs_c1'] - pao[p['pair']]['distance_c0_vs_c1']),
                                  two_attractors_same=int(ok) == pao[p['pair']]['two_attractors_4cond']))
            H['H2'][g][s] = h2
    orig = load(D11 / 'analysis_summary.json'); hyp = []
    for h in ('H1', 'H2', 'H3'):
        if len(H[h]['control']) < 10 or len(H[h]['intervention']) < 10:
            continue
        c = [H[h]['control'][s] for s in range(10)]; i = [H[h]['intervention'][s] for s in range(10)]; d = [b - a for a, b in zip(c, i)]
        mean = float(np.mean(d)); lo, hi = bootstrap(d)
        if h == 'H3':
            verdict = '支持' if (mean >= .05 and lo > 0) else '不支持' if (mean <= 0 or lo <= 0 <= hi) else '判定不能'
        else:
            wins = sum(x > 0 for x in d); verdict = '支持' if wins >= 8 else '不支持' if wins <= 5 else '判定不能'
        oc = [orig[h]['control'][str(s)] for s in range(10)]; oi = [orig[h]['intervention'][str(s)] for s in range(10)]
        hyp.append(dict(hypothesis=h, control_regenerated=c, control_original=oc, intervention_regenerated=i, intervention_original=oi,
                        mean_diff_regenerated=mean, mean_diff_original=orig[h].get('mean_diff', float(np.mean(np.array(oi) - np.array(oc)))),
                        ci95_regenerated=[lo, hi], ci95_original=orig[h].get('ci95'), seeds_intervention_greater=sum(x > 0 for x in d),
                        verdict_regenerated=verdict, verdict_original=orig[h]['verdict']))
    res['D11_hypotheses'] = hyp
    res['D11_accuracy_by_width'] = [dict(group=k[0], width=k[1], part=k[2], regenerated_exact=v[0], original_exact=v[1], n=v[2]) for k, v in sorted(acc.items())]
    # ------------------------------------------------ D10 probes + H4 cells
    D10 = W / 'decimal_d10'; probes10 = {r['pid']: r for r in jl_read(D10 / 'data' / 'probes.jsonl')}; h4 = []
    for s in range(3):
        od = OUT / 'd10' / f's{s}'
        if not (od / 'run.json').exists():
            continue
        orig = {r['pid']: r for r in jl_read(D10 / 'data' / 'raw' / f'probes_seed{s}.jsonl')}; new = jl_read(od / 'probe_pred.jsonl.gz'); assert len(new) == len(orig)
        agree = dis = fieldmis = 0; cell = collections.defaultdict(list)
        for x in new:
            o = orig[x['pid']]; r = probes10[x['pid']]
            ex, pi, wrong = grade10(x['pred_digits_lsb'], int(r['a']), int(r['b'])); fe = wrong[0] if wrong else None
            same = ex == bool(o['exact']); agree += same; dis += not same
            fm = (fe != o['first_error']) or (len(wrong) != o['wrong_digits']); fieldmis += fm
            cell[(r['state'], r['pair'], r['k'])].append(ex)
            if not same or fm:
                mism.append(dict(experiment='D10_probe', model=f's{s}', item=f"pid{x['pid']}_c{r['state']}_{r['pair']}_k{r['k']}", original_exact=o['exact'], regenerated_exact=int(ex),
                                 original_first_error=o['first_error'], regenerated_first_error=fe, original_wrong_digits=o['wrong_digits'], regenerated_wrong_digits=len(wrong)))
        summary.append(dict(experiment='D10_probe', model=f's{s}', n=len(new), flag_agree=agree, flag_disagree=dis, derived_field_mismatch=fieldmis))
        fpn = load(od / 'fixed_points.json'); fpo = {(r['state'], r['pair']): r for r in jl_read(D10 / 'data' / 'raw' / f'fixed_seed{s}.jsonl')}
        for r in fpn['fixed_points']:
            o = fpo[(r['state'], r['pair'])]; right = (int(r['pair'][0]) + int(r['pair'][1]) + r['state']) % 10
            fpcmp.append(dict(experiment='D10', model=f's{s}', cell=f"c{r['state']}_{r['pair']}", converged_same=r['converged'] == o['converged'], iterations_same=r['iterations'] == o['iterations'],
                              output_same=r['h_star_output'] == o['h_star_output'], rho_abs_diff=None if r['spectral_radius'] is None or o['spectral_radius'] is None else abs(r['spectral_radius'] - o['spectral_radius']),
                              move_abs_diff=abs(r['last100_move_max'] - o['last100_move_max'])))
            if r['converged']:
                hold = len(cell[(r['state'], r['pair'], 1024)]) == 16 and all(cell[(r['state'], r['pair'], 1024)])
                h4.append(dict(seed=s, state=r['state'], pair=r['pair'], read_correct=r['h_star_output'] == right, hold_correct=hold, match=(r['h_star_output'] == right) == hold))
    res['D10_H4'] = dict(matched=sum(x['match'] for x in h4), total=len(h4), original='77/78')
    # ------------------------------------------------ V054R
    probesT = {r['pid']: r for r in jl_read(W / 'v054t_transition' / 'data' / 'probes.jsonl')}; r1 = []
    for m in ('V051_A1', 'V055_F0', 'V055_F1', 'V055_F2'):
        od = OUT / 'v054r' / m
        if not (od / 'run.json').exists():
            continue
        raw = load(W / 'v054r_replication' / 'data' / 'raw' / f'R_{m}.json')
        orig = {(r['seed'], r['pid']): r for r in raw['durability']}; new = jl_read(od / 'probe_pred.jsonl.gz'); assert len(new) == len(orig)
        cell = collections.defaultdict(list); per = collections.defaultdict(lambda: [0, 0, 0, 0])
        for x in new:
            o = orig[(x['seed'], x['pid'])]; r = probesT[x['pid']]
            ex, pi, wrong = grade2(r['op'], x['pred_bits_lsb'], int(r['a'], 16), int(r['b'], 16), r['answer']); fe = wrong[0] if wrong else None
            same = ex == bool(o['exact']); fm = (fe != o['first_error']) or (len(wrong) != o['wrong_bits'])
            p_ = per[x['seed']]; p_[0] += 1; p_[1] += same; p_[2] += not same; p_[3] += fm
            cell[(x['seed'], r['op'], r['state'], r['pair'], r['k'])].append(ex)
            if not same or fm:
                mism.append(dict(experiment='V054R_probe', model=f"{m}_s{x['seed']}", item=f"pid{x['pid']}_{r['op']}_c{r['state']}_{r['pair']}_k{r['k']}", original_exact=o['exact'], regenerated_exact=int(ex),
                                 original_first_error=o['first_error'], regenerated_first_error=fe, original_wrong_digits=o['wrong_bits'], regenerated_wrong_digits=len(wrong)))
        for s, v in sorted(per.items()):
            summary.append(dict(experiment='V054R_probe', model=f'{m}_s{s}', n=v[0], flag_agree=v[1], flag_disagree=v[2], derived_field_mismatch=v[3]))
        fpn = load(od / 'fixed_points.json'); fpo = {(r['seed'], r['op'], r['state'], r['pair']): r for r in raw['fixed_points']}
        for r in fpn['fixed_points']:
            o = fpo[(r['seed'], r['op'], r['state'], r['pair'])]
            x, y = int(r['pair'][0]), int(r['pair'][1]); right = (x + y + r['state']) & 1 if r['op'] == 'ADD' else (x - y - r['state']) % 2
            assert right == o['correct_output']
            fpcmp.append(dict(experiment='V054R', model=f"{m}_s{r['seed']}", cell=f"{r['op']}_c{r['state']}_{r['pair']}", converged_same=r['converged'] == o['converged'], iterations_same=r['iterations'] == o['iterations'],
                              output_same=r['h_star_output'] == o['h_star_output'], rho_abs_diff=None if r['spectral_radius'] is None or o['spectral_radius'] is None else abs(r['spectral_radius'] - o['spectral_radius']),
                              move_abs_diff=abs(r['last100_move_max'] - o['last100_move_max'])))
            if r['converged']:
                c = cell[(r['seed'], r['op'], r['state'], r['pair'], 4096)]; hold = len(c) == 16 and all(c)
                r1.append(dict(model=m, seed=r['seed'], read_correct=r['h_star_output'] == right, hold_correct=hold, match=(r['h_star_output'] == right) == hold))
    res['V054R_R1'] = dict(matched=sum(x['match'] for x in r1), total=len(r1), original='90/90',
                           V055_matched=sum(x['match'] for x in r1 if x['model'].startswith('V055')), V055_total=sum(x['model'].startswith('V055') for x in r1), V055_original='54/54')
    res['fixed_point_comparison'] = dict(cells=len(fpcmp), converged_differ=sum(1 for x in fpcmp if x.get('converged_same') is False), output_differ=sum(1 for x in fpcmp if x.get('output_same') is False),
                                         iterations_differ=sum(1 for x in fpcmp if x.get('iterations_same') is False), two_attractors_differ=sum(1 for x in fpcmp if x.get('two_attractors_same') is False),
                                         max_rho_abs_diff=max([x['rho_abs_diff'] for x in fpcmp if x.get('rho_abs_diff') is not None] or [None]),
                                         max_move_abs_diff=max([x['move_abs_diff'] for x in fpcmp if 'move_abs_diff' in x] or [None]),
                                         max_distance_abs_diff=max([x['distance_abs_diff'] for x in fpcmp if 'distance_abs_diff' in x] or [None]))
    csvout(G / 'agreement_summary.csv', summary); csvout(G / 'mismatches.csv', mism); csvout(G / 'fixed_point_comparison.csv', fpcmp)
    csvout(G / 'd10_h4_cells.csv', h4); csvout(G / 'v054r_r1_cells.csv', r1)
    dump(G / 'graded_results.json', res)
    tot = collections.defaultdict(lambda: [0, 0])
    for x in summary:
        tot[x['experiment']][0] += x['flag_agree']; tot[x['experiment']][1] += x['flag_disagree']
    print(json.dumps(dict(totals=tot, mismatches=len(mism), **{k: v for k, v in res.items() if k != 'D11_accuracy_by_width'}), ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
