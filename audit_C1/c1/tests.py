"""S1 unit tests: synthetic data and hand-computed examples only (no checkpoint, no mixed-sequence truth)."""
import math, random
import numpy as np
from common import *
from predictors import *
import gen


def t_ripple_and_classes():
    for op in OPS:
        for c in (0, 1):
            for x, y in TWO_SIDED[op]:
                out, inc, s = ripple(op, [x] * 50, [y] * 50, s=c)
                assert s == c and all(v == c for v in inc) and all(o == y_class(op, c) for o in out), (op, c, x, y)
            # HAND table cross-check with integer arithmetic
            for x in (0, 1):
                for y in (0, 1):
                    nxt = (x + y + c) >> 1 if op == 'ADD' else int(x - y - c < 0)
                    assert ((x, y) in HAND[(op, c)]) == (nxt == c)
            assert set(TWO_SIDED[op]) == set(HAND[(op, 0)]) & set(HAND[(op, 1)])


def t_tie_rule():
    assert wrong(-1e-9, 0) and wrong(-1e-9, 1) and not wrong(0.0, 0) and wrong(0.0, 1) and wrong(float('nan'), 0)
    assert list(wrong_vec([0.0, 0.0, -1, 1, float('inf')], 1)) == [True, True, True, False, True]
    assert list(wrong_vec([0.0], 0)) == [False]


def t_geom():
    rng = random.Random(1)
    for mu in (1, 2.5, 16, 56):
        xs = [gen.geom(rng, mu) for _ in range(200000)]
        assert min(xs) >= 1 and abs(np.mean(xs) - mu) / mu < 0.02, (mu, np.mean(xs))
    assert gen.geom(rng, 0.5) == 1 and gen.geom(rng, 1) == 1


def t_patterns():
    rng = random.Random(5)
    f1 = gen.f1(); assert len(f1) == 63 and all(len(s) == H for _, s in f1)
    assert all(s[0] == '1' for _, s in f1)
    f2 = gen.f2(rng); assert len(f2) == 20
    for name, s in f2:
        rho = float(name.split('rho')[1].split('_')[0]); assert s.count('1') == round(rho * H)
    f3 = gen.f3(rng); assert len(f3) == 20 and all(len(s) == H for _, s in f3)
    # F3 fraction roughly rho for long cycles (sanity, not exact)
    for name, s in f3:
        rho = float(name.split('rho')[1].split('_')[0]); assert abs(s.count('1') / H - rho) < 0.2, (name, s.count('1') / H)


def t_build_map_and_interp():
    # hand example: one trajectory 3,2,1,0,-1 -> map f(3)=2, f(2)=1, ...; clamp outside
    u, v = build_map([np.array([3., 2., 1., 0., -1.])])
    assert list(u) == [0., 1., 2., 3.] and list(v) == [-1., 0., 1., 2.]
    assert np.interp(10, u, v) == 2 and np.interp(-5, u, v) == -1 and np.interp(1.5, u, v) == 0.5
    # merge of near-duplicates and averaging of v
    u, v = build_map([np.array([1.0, 0.5]), np.array([1.0 + 1e-13, 0.7])])
    assert len(u) == 1 and abs(v[0] - 0.6) < 1e-15
    # non-finite truncation
    u, v = build_map([np.array([2., 1., np.nan, 5.])]); assert list(u) == [2.] and list(v) == [1.]
    assert build_map([np.array([1.0])]) is None


def t_p1_constant_reproduces_curve():
    # synthetic monotone curves with disjoint +/- ranges: P1 on an all-p pattern must reproduce mu_p^+
    t = np.arange(T + 1)
    mu_p_plus = 5 * np.exp(-t / 30) - 1          # crosses 0 at t = 30 ln 5 = 48.28 -> first wrong t=49
    mu_p_minus = -2 - 3 * np.exp(-t / 10)
    mu_q_plus = 1 + 4 * np.exp(-t / 5)           # never wrong
    mu_q_minus = -1 - 4 * np.exp(-t / 5)
    mu = dict(p={'+': mu_p_plus, '-': mu_p_minus}, q={'+': mu_q_plus, '-': mu_q_minus})
    s_all_p = '1' * H; s_all_q = '0' * H
    fail, tau, _ = p1_predict(mu, [s_all_p, s_all_q], yc=1)
    assert fail[0] and tau[0] == lifetime(mu_p_plus, 1) == 49 and not fail[1], (fail, tau)
    # order dependence exists: block pattern vs shuffled with same counts can differ in general -> just check it runs
    s = ('1' * 40 + '0' * 40) * (H // 80 + 1)
    fail2, tau2, _ = p1_predict(mu, [s[:H]], yc=1)
    assert fail2.shape == (1,)
    # use_minus=False variant (B3) runs and equals on pure-p pattern
    f3, t3, _ = p1_predict(mu, [s_all_p], yc=1, use_minus=False); assert f3[0] and t3[0] == 49


def t_b1():
    f, tt = b1_run(['1' * 5 + '0' + '1' * 9, '1' * 9 + '0' * 6], 10, None)
    assert list(f) == [False, False]
    f, tt = b1_run(['0' + '1' * 10 + '0' * 4], 10, None); assert f[0] and tt[0] == 11
    f, tt = b1_run(['0' * 3 + '1'], None, 3); assert f[0] and tt[0] == 3
    # additive: tau_p=3, tau_q=6: p,p,q,q -> 2/3+2/6 = 1 exactly at t=4
    f, tt = b1_add(['1100' + '0' * 10], 3, 6); assert f[0] and tt[0] == 4
    f, tt = b1_add(['1010' + '0' * 10], 4, None); assert not f[0]
    f, tt = b1_add(['10' * 4], 4, None); assert f[0] and tt[0] == 7
    f, tt = b0(['1', '0']); assert f.all()


def t_dummy_gru_pipeline():
    """Dummy model with the same structure: check step/margin plumbing and FP64 (torch only)."""
    import torch
    torch.manual_seed(0)
    from torch import nn

    class D(nn.Module):
        def __init__(self):
            super().__init__(); self.projection = nn.Linear(3, 64); self.core = nn.GRUCell(64, 545)
            self.residual_gate = nn.Parameter(torch.zeros(())); self.head = nn.Linear(545, 2)
    m = D().double().eval()
    x = torch.tensor([[0., 1., 0.]], dtype=torch.float64); h = torch.zeros(1, 545, dtype=torch.float64)
    h1, z1 = step(m, x, h)
    ref = m.core(m.projection(x), h) + torch.tanh(m.residual_gate) * h
    assert torch.equal(h1, ref) and z1.dtype == torch.float64
    # margin is a function of the state only (input-independent readout)
    z_a = m.head(h1); assert torch.equal(z_a, z1)


def t_analyze_synthetic():
    import analyze as A
    rr=[dict(Y=True,P1=True,B1=False),dict(Y=False,P1=False,B1=False),dict(Y=True,P1=False,B1=True),dict(Y=False,P1=True,B1=True)]
    assert A.ba(rr,'P1')==0.5 and A.ba(rr,'B1')==0.5 and A.ba(rr[:1],'P1') is None
    pc={f'g{i}':dict(P1=0.8,B1=0.6) for i in range(6)}
    r=A.primary(pc); assert abs(r['mean_delta']-0.2)<1e-12 and r['crit1_mean_ge_0p10'] and r['crit2_positive_ge_2of3'] and r['crit3_loo_ge_0p05']
    pc['g0']=dict(P1=0.6,B1=0.6); pc['g1']=dict(P1=0.5,B1=0.6)
    r=A.primary(pc); assert r['n_positive']==4 and r['need_positive']==4
    r=A.primary({f'g{i}':dict(P1=0.9,B1=0.1) for i in range(5)}); assert r['verdict'].startswith('判定不能')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('t_')]
    for f in tests:
        f(); print('PASS', f.__name__)
    print('ALL', len(tests), 'PASS')
