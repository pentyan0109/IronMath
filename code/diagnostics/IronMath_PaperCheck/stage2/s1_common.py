"""Shared S1 judgement rules (frozen in S1S2_RULES.md): convergence = finite endpoint, finite last-100 movements and
max(last-100 movement) < threshold; A = every one of 16 trajectories converged, rho<1, correct readout;
D = every trajectory converged and no non-finite value anywhere in the run."""
import json
from pathlib import Path
COUNTS=[500,2000,5000,20000];THRS=[1e-4,1e-5,1e-6];ARCH=['residual_gru','gru','lstm','rnn']
def conv(r,thr):return r['endpoint_finite'] and r['last100_finite'] and r['last100_max_movement'] is not None and r['last100_max_movement']<thr
def good(r,thr):return conv(r,thr) and r['spectral_radius'] is not None and r['spectral_radius']<1 and r['readout']==r['value']
def judge(trajs,thr):
    A=all(good(r,thr) for r in trajs);D=all(conv(r,thr) for r in trajs) and not any(r['run_nonfinite'] for r in trajs)
    return A,D
