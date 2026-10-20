"""S7: frozen aggregation (METHOD 6). Frozen at S1 before any truth file exists. Usage: python analyze.py"""
import csv, json, math
from collections import defaultdict
import numpy as np
from common import *

METHODS = ['P1', 'B1', 'B3', 'B1add', 'B0']


def rd(p):
    with open(p, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def tb(v):
    return v in ('True', True)


def key(r):
    return (r['model'], r['op'], int(r['c']), int(r['j']), r['pattern'])


def load_stage(prec):
    ctx = {(r['model'], r['op'], int(r['c']), int(r['j'])): r for r in rd(OUT / f'S4_{prec}/contexts.csv')}
    pred = {key(r): r for r in rd(OUT / f'S4_{prec}/predictions.csv')}
    truth = {}
    for _, model in MODELS:
        p = OUT / f'truth_{prec}/{model}.csv'
        if p.exists():
            for r in rd(p):
                truth[key(r)] = r
    b2 = {}
    for _, model in MODELS:
        p = OUT / f'b2_{prec}/{model}.csv'
        if p.exists():
            for r in rd(p):
                b2[key(r)] = r
    return ctx, pred, truth, b2


def rows_for(ctx, pred, truth, b2, population, models=None):
    out = []
    for k, t in truth.items():
        c = ctx[k[:4]]
        if tb(c['excluded_prefix_error']):
            continue
        if population == 'informative' and not tb(c['informative']):
            continue
        if models is not None and k[0] not in models:
            continue
        p = pred[k]; y = int(t['first_error']) > 0
        r = dict(model=k[0], cluster=CLUSTER[k[0]], key=k, Y=y, tau_obs=int(t['first_error']),
                 **{m: tb(p[f'{m}_fail']) for m in METHODS}, **{f'{m}_tau': int(p[f'{m}_tau']) for m in METHODS})
        if b2:
            r['B2'] = int(b2[k]['first_error']) > 0; r['B2_tau'] = int(b2[k]['first_error'])
        out.append(r)
    return out


def ba(rr, m):
    pos = [r for r in rr if r['Y']]; neg = [r for r in rr if not r['Y']]
    if not pos or not neg:
        return None
    return 0.5 * (np.mean([r[m] for r in pos]) + np.mean([not r[m] for r in neg]))


def cluster_table(rr, methods):
    bym = defaultdict(list)
    for r in rr:
        bym[r['model']].append(r)
    per_model = {}
    for mod, x in bym.items():
        per_model[mod] = dict(n=len(x), n_fail=sum(r['Y'] for r in x), **{m: ba(x, m) for m in methods},
                              **{f'acc_{m}': float(np.mean([r[m] == r['Y'] for r in x])) for m in methods})
    per_cluster = {}
    for g in dict.fromkeys(CLUSTER[m] for m in bym):
        ms = [mod for mod in per_model if CLUSTER[mod] == g and per_model[mod]['P1'] is not None]
        per_cluster[g] = dict(models=ms, n_models_total=sum(CLUSTER[mod] == g for mod in per_model),
                              **{m: (float(np.mean([per_model[mod][m] for mod in ms])) if ms else None) for m in methods})
    return per_model, per_cluster


def primary(per_cluster, a='P1', b='B1'):
    G = [g for g, v in per_cluster.items() if v[a] is not None]
    d = {g: per_cluster[g][a] - per_cluster[g][b] for g in G}
    res = dict(evaluable_clusters=G, n_evaluable=len(G), delta_by_cluster=d)
    if len(G) < 6:
        res.update(verdict='判定不能 (evaluable clusters < 6)', mean_delta=float(np.mean(list(d.values()))) if d else None)
        return res
    v = np.array([d[g] for g in G]); mean = float(v.mean())
    npos = int((v > 0).sum()); need = math.ceil(2 * len(G) / 3)
    loo = {g: float(np.mean([d[h] for h in G if h != g])) for g in G}
    rng = np.random.default_rng(SEED_BOOT); ix = rng.integers(0, len(v), (10000, len(v)))
    ci = np.percentile(v[ix].mean(1), [2.5, 97.5]).tolist()
    c1 = mean >= 0.10; c2 = npos >= need; c3 = min(loo.values()) >= 0.05
    res.update(mean_delta=mean, n_positive=npos, need_positive=need, loo_mean=loo, loo_min=min(loo.values()), bootstrap95=ci,
               crit1_mean_ge_0p10=c1, crit2_positive_ge_2of3=c2, crit3_loo_ge_0p05=c3,
               verdict=('継続の目安を満たす' if c1 and c2 and c3 else ('継続の根拠なし' if not c1 else '平均は0.10以上だが目安の一部を満たさない')))
    if mean <= 0:
        res['note'] = 'B1 以上の価値は示されない'
    return res


def timing(rr, m):
    both = [r for r in rr if r['Y'] and r[m]]
    err = [r[f'{m}_tau'] - r['tau_obs'] for r in both]
    return dict(n_both_observed=len(both), median_error=float(np.median(err)) if err else None,
                within3=float(np.mean([abs(e) <= 3 for e in err])) if err else None,
                pred_fail_obs_survive=sum(r[m] and not r['Y'] for r in rr), pred_survive_obs_fail=sum((not r[m]) and r['Y'] for r in rr),
                both_survive=sum((not r[m]) and (not r['Y']) for r in rr))


def order_effects(rr):
    grp = defaultdict(list)
    for r in rr:
        k = r['key']
        if k[4].startswith('F2_'):
            grp[k[:4] + (k[4].rsplit('_r', 1)[0],)].append(r)
    n = len(grp); truth_var = [g for g in grp.values() if len({r['Y'] for r in g}) > 1]
    tau_var = [g for g in grp.values() if len({r['tau_obs'] for r in g}) > 1]
    out = dict(groups=n, truth_Y_differs=len(truth_var), truth_tau_differs=len(tau_var))
    for m in ('P1', 'B3', 'B1'):
        out[f'{m}_Y_differs'] = sum(len({r[m] for r in g}) > 1 for g in grp.values())
        out[f'{m}_exact_Y_vector_on_truth_varying'] = sum(all(r[m] == r['Y'] for r in g) for g in truth_var)
    return out


def family_ba(rr, methods):
    out = {}
    for fam in ('F1', 'F2', 'F3'):
        x = [r for r in rr if r['key'][4].startswith(fam)]
        out[fam] = dict(n=len(x), n_fail=sum(r['Y'] for r in x), **{m: ba(x, m) for m in methods})
    return out


def analysis(prec):
    ctx, pred, truth, b2 = load_stage(prec)
    if not truth:
        return None
    methods = METHODS + (['B2'] if b2 else [])
    out = dict(precision=prec, n_truth=len(truth), n_contexts=len(ctx), n_informative_contexts=sum(tb(c['informative']) for c in ctx.values()),
               n_prefix_excluded=sum(tb(c['excluded_prefix_error']) for c in ctx.values()))
    for pop, models in (('informative', None), ('all', None), ('informative_E3_only', {m for _, m in MODELS if m.startswith('i')})):
        rr = rows_for(ctx, pred, truth, b2, 'informative' if pop.startswith('informative') else 'all', models)
        pm, pc = cluster_table(rr, methods)
        out[pop] = dict(n_sequences=len(rr), n_fail=sum(r['Y'] for r in rr), per_model=pm, per_cluster=pc, primary=primary(pc),
                        secondary_vs_B1={m: primary(pc, m, 'B1') for m in methods if m not in ('B1',)},
                        P1_vs_B3=primary(pc, 'P1', 'B3'),
                        sequence_accuracy={m: float(np.mean([r[m] == r['Y'] for r in rr])) if rr else None for m in methods},
                        timing={m: timing(rr, m) for m in methods if m != 'B0'}, order_effects=order_effects(rr), family_BA=family_ba(rr, methods),
                        recoveries=int(sum(int(truth[r['key']]['recoveries']) > 0 for r in rr)),
                        nonfinite=int(sum(tb(truth[r['key']]['nonfinite']) for r in rr)))
    return out


def fp_compare():
    t64 = load_stage('fp64'); t32 = load_stage('fp32')
    if not t32[2]:
        return None
    ks = [k for k in t64[2] if k in t32[2]]
    dy = sum((int(t64[2][k]['first_error']) > 0) != (int(t32[2][k]['first_error']) > 0) for k in ks)
    dt = sum(t64[2][k]['first_error'] != t32[2][k]['first_error'] for k in ks)
    dctx = sum(t64[0][k]['informative'] != t32[0][k]['informative'] for k in t64[0])
    dpred = {m: sum(t64[1][k][f'{m}_fail'] != t32[1][k][f'{m}_fail'] for k in t64[1]) for m in METHODS}
    return dict(n=len(ks), truth_Y_differs=dy, truth_tau_differs=dt, informative_flag_differs=dctx, prediction_differs=dpred)


if __name__ == '__main__':
    res = dict(fp64=analysis('fp64'), fp32=analysis('fp32'), fp_compare=fp_compare())
    dump('S7_analysis.json', res)
    p = res['fp64']['informative']['primary']
    print(json.dumps({k: v for k, v in p.items() if k not in ('delta_by_cluster', 'loo_mean')}, ensure_ascii=False, indent=1))
