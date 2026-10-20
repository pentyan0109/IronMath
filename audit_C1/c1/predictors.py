"""P1 / B1 / B1-add / B0 / B3 predictors (METHOD 3-4). Pure NumPy; never touches the GRU or any mixed-sequence state.
Input: fixed-input margin trajectories mu[x][s] (t = 0..T) for x in {'p','q'}, s in {'+','-'}; the pattern string ('1' = p, '0' = q)."""
import numpy as np

from common import MERGE_TOL


def wrong_vec(m, yc):
    m = np.asarray(m, float)
    return ~np.isfinite(m) | (m < 0) | ((m == 0) & (yc == 1))


def build_map(trajs):
    """trajs: list of 1-D arrays mu(0..T). Returns (u, v) strictly increasing u for np.interp (METHOD 3.2)."""
    us = []; vs = []
    for mu in trajs:
        mu = np.asarray(mu, float)
        bad = np.flatnonzero(~np.isfinite(mu))
        n = bad[0] if len(bad) else len(mu)          # use points up to the first non-finite value (METHOD 3.4)
        if n >= 2:
            us.append(mu[:n - 1]); vs.append(mu[1:n])
    if not us:
        return None
    u = np.concatenate(us); v = np.concatenate(vs); o = np.argsort(u, kind='mergesort'); u = u[o]; v = v[o]
    gid = np.concatenate([[0], np.cumsum(np.diff(u) > MERGE_TOL)])
    cnt = np.bincount(gid)
    return np.bincount(gid, u) / cnt, np.bincount(gid, v) / cnt


def p1_predict(mu, patterns, yc, use_minus=True, H=None):
    """Returns (fail[bool], tau[int or -1]) per pattern. tau is 1-based hold step of first predicted error."""
    maps = {}
    for x in ('p', 'q'):
        maps[x] = build_map([mu[x]['+']] + ([mu[x]['-']] if use_minus else []))
    P = np.array([[ch == '1' for ch in s] for s in patterns]); n, HH = P.shape
    H = H or HH
    if maps['p'] is None or maps['q'] is None:
        return np.ones(n, bool), np.full(n, -1), True            # unevaluable -> predicted failure (METHOD 3.2)
    m = np.full(n, float(mu['p']['+'][0])); tau = np.full(n, -1); done = np.zeros(n, bool)
    for t in range(H):
        isp = P[:, t]
        if isp.any():
            m[isp] = np.interp(m[isp], *maps['p'])
        if (~isp).any():
            m[~isp] = np.interp(m[~isp], *maps['q'])
        w = wrong_vec(m, yc) & ~done
        tau[w] = t + 1; done |= w
    return done, tau, False


def lifetime(mu_plus, yc):
    w = np.flatnonzero(wrong_vec(np.asarray(mu_plus)[1:], yc))
    return int(w[0]) + 1 if len(w) else None                     # None = no error within T (infinite)


def b1_run(patterns, tau_p, tau_q):
    fail = []; tt = []
    for s in patterns:
        run = 0; prev = None; hit = -1
        for t, ch in enumerate(s):
            run = run + 1 if ch == prev else 1; prev = ch
            lim = tau_p if ch == '1' else tau_q
            if lim is not None and run >= lim:
                hit = t + 1; break
        fail.append(hit > 0); tt.append(hit)
    return np.array(fail), np.array(tt)


def b1_add(patterns, tau_p, tau_q):
    """Exact integer form of sum 1/tau >= 1 (1/inf = 0)."""
    fail = []; tt = []
    for s in patterns:
        cp = cq = 0; hit = -1
        for t, ch in enumerate(s):
            cp += ch == '1'; cq += ch == '0'
            if tau_p is not None and tau_q is not None:
                ok = cp * tau_q + cq * tau_p >= tau_p * tau_q
            else:
                ok = (tau_p is not None and cp >= tau_p) or (tau_q is not None and cq >= tau_q)
            if ok:
                hit = t + 1; break
        fail.append(hit > 0); tt.append(hit)
    return np.array(fail), np.array(tt)


def b0(patterns):
    return np.ones(len(patterns), bool), np.full(len(patterns), -1)
