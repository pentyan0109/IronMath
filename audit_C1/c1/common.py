"""C1-R1 shared definitions. Spec: audit_C1/C1_R1_METHOD.md, C1_R1_FREEZE_PLAN.md."""
import hashlib, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'out'                        # audit_C1/out: all experiment outputs
SCRATCH = Path(os.environ.get('C1_SCRATCH', '/tmp/c1'))   # downloaded checkpoints / evaluation sets (not committed)

H = 2048              # hold length (METHOD 2.1)
T = 2048              # fixed-input trajectory length (METHOD 3.1)
ELL = 256             # B2 truncation (METHOD 4.5)
MERGE_TOL = 1e-12     # P1 duplicate-u merge (METHOD 3.2)
SEED_GEN = 92000      # FREEZE_PLAN 1.1
SEED_BOOT = 93000     # METHOD 6.3
OPS = ('ADD', 'SUB')
TWO_SIDED = {'ADD': ((0, 1), (1, 0)), 'SUB': ((0, 0), (1, 1))}   # p = first, q = second (lexicographic)
HAND = {('ADD', 0): [(0, 0), (0, 1), (1, 0)], ('ADD', 1): [(0, 1), (1, 0), (1, 1)],
        ('SUB', 0): [(0, 0), (1, 0), (1, 1)], ('SUB', 1): [(0, 0), (0, 1), (1, 1)]}


def y_class(op, c):
    """Correct output class during a two-sided hold with state c (METHOD 1)."""
    return 1 - c if op == 'ADD' else c


def ripple(op, a_bits, b_bits, s=0):
    """Independent Boolean ripple (gate form, same as v054t.py). Returns outputs, incoming states, final state."""
    out = []; inc = []
    for x, y in zip(a_bits, b_bits):
        inc.append(s)
        o = x ^ y ^ s
        s = ((x & y) | (s & (x ^ y))) if op == 'ADD' else (((1 - x) & y) | ((1 - (x ^ y)) & s))
        out.append(o)
    return out, inc, s


def wrong(m, yc):
    """Tie rule of argmax (class 0 wins ties); non-finite counts as wrong (METHOD 2.2)."""
    import math
    if not math.isfinite(m):
        return True
    return m < 0 or (m == 0 and yc == 1)


# ----------------------------------------------------------------- models
MODELS = [('V054_s0', 'C1_B0_s0'), ('V054_s1', 'C1_B0_s1'), ('V054_s2', 'C1_B0_s2')] + \
         [(f'i{i}', f'i{i}_c{c}') for i in range(100, 110) for c in (0, 1)]
CLUSTER = {m: g for g, m in MODELS}


def ckpt_rel(model):
    if model.startswith('C1_B0'):
        return f'IronMath_V054/checkpoints/{model}/best_validation.pt'
    return f'IronMath_PaperCheck/extension/E3_independent_replication/checkpoints/{model}/best_validation.pt'


def ckpt_path(model):
    return SCRATCH / 'ck/archive_v074/checkpoints' / ckpt_rel(model)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def dump(name, obj):
    p = OUT / name; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=True), encoding='utf-8')
    return p


def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def load_model(model, dtype):
    import torch
    from torch import nn

    class Residual(nn.Module):
        # same parameter names/shapes as IronMath_V054/model.py Residual('B0') and E3 core.py Residual
        def __init__(self):
            super().__init__(); self.projection = nn.Linear(3, 64); self.core = nn.GRUCell(64, 545)
            self.residual_gate = nn.Parameter(torch.zeros(())); self.head = nn.Linear(545, 2)

    cp = torch.load(ckpt_path(model), map_location='cpu', weights_only=False)
    m = Residual(); m.load_state_dict(cp['model'], strict=True); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(dtype)


def modelhash(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def step(m, x, h):
    """One recurrent step; composition line identical to model.py Residual.forward."""
    import torch
    h = m.core(m.projection(x), h) + torch.tanh(m.residual_gate) * h
    return h, m.head(h)
