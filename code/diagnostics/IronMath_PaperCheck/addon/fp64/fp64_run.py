"""Stage 2: G1 diagnostic re-run in FP64 (no training; checkpoints read-only).

Rules: RULES_FP64.md (SHA256 frozen in RULES_SHA256.json). Byte-identical G1 models.py (SHA256-checked).
Reads IronMath_G1 only; writes only into addon/fp64/. Resumable: skips models whose record exists.
Run from workspace root: .venv/Scripts/python.exe -I -B IronMath_PaperCheck/addon/fp64/fp64_run.py
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "2"
import datetime, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np

S = Path(__file__).resolve().parent
W = S.parent.parent.parent
G = W / "IronMath_G1"
MODELS_SHA = "abdc03640b3f56139aa59edba1fe40a77f8d4e183e9b73e6aed842878d1eee2b"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4194304), b""):
            h.update(b)
    return h.hexdigest()


frozen = json.loads((S / "RULES_SHA256.json").read_text(encoding="utf-8"))["SHA256"]
assert all(sha(S / f) == h for f, h in frozen.items()), "rules changed"
assert sha(S / "g1_code" / "models.py") == MODELS_SHA == json.loads((G / "PREREGISTRATION.json").read_text(encoding="utf-8"))["code_SHA256"]["models.py"]
sys.path.insert(0, str(S / "g1_code"))
import torch
torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float64)
from models import make

P = [.1, .1, .5, .3]
COUNT, THR = 20000, 1e-6


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def draw(seed, n, t):
    return np.random.default_rng(seed).choice(4, size=(n, t), p=P).astype(np.uint8)


def tensor(a):
    return torch.nn.functional.one_hot(torch.from_numpy(a.astype(np.int64)), 4).to(torch.float64)


def clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def load_model(asset):
    p = G / asset["path"]; assert sha(p) == asset["SHA256"]
    cp = torch.load(p, map_location="cpu", weights_only=False)
    m = make(cp["architecture"]).eval()
    sd = cp["model"]
    assert all(v.dtype == torch.float32 for v in sd.values() if v.is_floating_point())
    m.load_state_dict({k: v.to(torch.float64) for k, v in sd.items()})
    for q in m.parameters():
        q.requires_grad_(False)
    assert all(q.dtype == torch.float64 for q in m.parameters())
    return m


def postwrite_states(model):
    states = []
    with torch.inference_mode():
        for prefix in draw(30000, 4, 8):
            for value in [0, 1]:
                a = np.append(prefix, value).astype(np.uint8)[None, :]
                _, state = model.block(tensor(a), torch.zeros(1, model.state_size))
                states.append(state[0])
    return torch.stack(states)


def diagnose(m):
    s = torch.jit.script(m)
    starts = postwrite_states(s)
    assert starts.dtype == torch.float64
    traj, ends_all, run_nonfinite = [], {}, False
    xr = torch.zeros(1, 4); xr[0, 3] = 1
    for symbol, neutral in ((2, "N"), (3, "R")):
        with torch.inference_mode():
            ends, mv, mg, sf, lf = s.repeat(starts.clone(), symbol, COUNT)
        assert ends.dtype == torch.float64 and mv.dtype == torch.float64 and mg.dtype == torch.float64
        en, mvn, mgn = ends.numpy(), mv.numpy(), mg.numpy()
        nonfin = not all(np.isfinite(x).all() for x in (en, mvn, mgn)) or not bool(sf.all() and lf.all())
        run_nonfinite = run_nonfinite or nonfin
        ends_all[neutral] = en
        xx = torch.zeros(1, 4); xx[0, symbol] = 1
        for j in range(8):
            prefix, value = divmod(j, 2)
            last = mvn[-100:, j]
            conv = bool(np.isfinite(en[j]).all() and np.isfinite(last).all() and np.max(last) < THR)
            terminal = ends[j].detach().clone()
            with torch.inference_mode():
                z = m.step(xr, terminal[None])[1] if symbol == 2 else m.readout(terminal[None])
            fin = bool(torch.isfinite(z).all())
            readout = int(z.argmax(-1)) if fin else None
            rho = None
            if bool(torch.isfinite(terminal).all()):
                jac = torch.autograd.functional.jacobian(lambda st: m.step(xx, st[None])[0][0], terminal)
                ev = torch.linalg.eigvals(jac)
                rho = float(ev.abs().max()) if bool(torch.isfinite(ev).all()) else None
            neg = np.flatnonzero(mgn[:, j] < 0)
            traj.append(dict(neutral=neutral, prefix=int(prefix), value=int(value), converged=conv,
                             last100_max_movement=float(np.max(last)), spectral_radius=rho, readout=readout,
                             readout_finite=fin, readout_correct=readout == value,
                             first_negative_margin_step=int(neg[0]) + 1 if len(neg) else None))
    A = all(t["converged"] and t["spectral_radius"] is not None and t["spectral_radius"] < 1 and t["readout"] == t["value"] for t in traj)
    D = all(t["converged"] for t in traj) and not run_nonfinite
    return traj, ends_all, A, D, run_nonfinite


def main():
    out = S / "records"; out.mkdir(exist_ok=True)
    ep = S / "endpoints"; ep.mkdir(exist_ok=True)
    assets = [json.loads(p.read_text(encoding="utf-8-sig")) for p in sorted((G / "assets").glob("*.json"))]
    assert len(assets) == 200
    t0 = time.perf_counter()
    for i, a in enumerate(assets):
        rp = out / f"{a['model']}.json"
        if rp.exists():
            continue
        t = time.perf_counter(); c = time.process_time()
        m = load_model(a)
        traj, ends, A, D, nonfin = diagnose(m)
        rec = dict(model=a["model"], architecture=a["architecture"], L=a["L"], seed=a["seed"], checkpoint_SHA256=a["SHA256"],
                   dtype="float64", count=COUNT, threshold=THR, A=A, D=D, all16_converged=all(x["converged"] for x in traj),
                   run_nonfinite=nonfin, trajectories=traj, wall_seconds=time.perf_counter() - t,
                   CPU_seconds=time.process_time() - c, completed_utc=utc())
        np.savez_compressed(ep / f"{a['model']}.npz", **ends)
        with rp.open("x", encoding="utf-8") as f:
            json.dump(clean(rec), f, ensure_ascii=False, indent=1, allow_nan=False)
        print("FP64", i + 1, a["model"], "A", A, "D", D, round(rec["wall_seconds"], 1), "elapsed", round(time.perf_counter() - t0), flush=True)
    print("FP64 DONE", flush=True)


if __name__ == "__main__":
    main()
