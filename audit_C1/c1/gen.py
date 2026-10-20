"""S3: evaluation-sequence generator, legality and overlap audit (FREEZE_PLAN 1-2). No model is used."""
import json, math, random, sys
from common import *


def bits(v, n):
    return [(v >> i) & 1 for i in range(n)]


def gen_prefixes(rng):
    pre = {}
    for op in OPS:
        for s in (0, 1):
            pp = [dict(id='min', a=([] if s == 0 else ([1] if op == 'ADD' else [0])), b=([] if s == 0 else [1]))]
            while len(pp) < 4:
                a = rng.getrandbits(16); b = rng.getrandbits(16)
                if ((a + b) >> 16 if op == 'ADD' else int(a < b)) == s:
                    pp.append(dict(id=f'r{len(pp)}', a=bits(a, 16), b=bits(b, 16)))
            for p in pp:
                assert ripple(op, p['a'], p['b'])[2] == s
            pre[f'{op}{s}'] = pp
    return pre


def geom(rng, mu):
    if mu <= 1:
        return 1
    u = 1.0 - rng.random()                      # (0, 1]
    return 1 + int(math.floor(math.log(u) / math.log(1 - 1 / mu)))


RHOS = (1 / 8, 1 / 4, 1 / 2, 3 / 4, 7 / 8)
BLOCKS = (1, 2, 4, 8, 16, 32, 64, 128)


def f1():
    out = []
    for a in BLOCKS:
        for b in BLOCKS:
            if a == b == 1:
                continue
            unit = '1' * a + '0' * b
            out.append((f'F1_a{a}_b{b}', (unit * (H // len(unit) + 1))[:H]))
    return out


def f2(rng):
    out = []
    for rho in RHOS:
        n_p = int(round(rho * H))
        for r in range(4):
            x = ['1'] * n_p + ['0'] * (H - n_p); rng.shuffle(x)
            out.append((f'F2_rho{rho:g}_r{r}', ''.join(x)))
    return out


def f3(rng):
    out = []
    for rho in RHOS:
        for C in (8, 64):
            for r in range(2):
                s = []; cur = '1'
                while len(s) < H:
                    n = geom(rng, rho * C if cur == '1' else (1 - rho) * C)
                    s.extend(cur * n); cur = '0' if cur == '1' else '1'
                out.append((f'F3_rho{rho:g}_C{C}_r{r}', ''.join(s[:H])))
    return out


def main():
    rng = random.Random(SEED_GEN)
    pre = gen_prefixes(rng)
    pats = {}
    f2s = {f'{op}{c}': f2(rng) for op in OPS for c in (0, 1)}
    f3s = {f'{op}{c}': f3(rng) for op in OPS for c in (0, 1)}
    dedup = {}
    for op in OPS:
        for c in (0, 1):
            k = f'{op}{c}'; seen = {}; keep = []; dropped = []
            for name, s in f1() + f2s[k] + f3s[k]:
                assert len(s) == H and set(s) <= {'0', '1'}
                if s in seen:
                    dropped.append((name, seen[s])); continue
                seen[s] = name; keep.append(dict(name=name, pattern=s, sha256=hashlib.sha256(s.encode()).hexdigest()))
            pats[k] = keep; dedup[k] = dropped
    # legality (FREEZE_PLAN 1.4): every hold position has incoming state c and output y_c
    nseq = 0
    for op in OPS:
        for c in (0, 1):
            p, q = TWO_SIDED[op]; yc = y_class(op, c)
            for pr in pre[f'{op}{c}']:
                for pt in pats[f'{op}{c}']:
                    pairs = [p if ch == '1' else q for ch in pt['pattern']]
                    A = pr['a'] + [x for x, _ in pairs]; B = pr['b'] + [y for _, y in pairs]
                    out, inc, _ = ripple(op, A, B); hs = len(pr['a'])
                    assert all(v == c for v in inc[hs:]), (op, c, pr['id'], pt['name'])
                    assert all(v == yc for v in out[hs:]), (op, c, pr['id'], pt['name'])
                    assert out[:hs] == ripple(op, pr['a'], pr['b'])[0]
                    nseq += 1
    data = dict(H=H, seed=SEED_GEN, prefixes=pre, patterns=pats, dedup_dropped=dedup, n_sequences_per_model=nseq,
                pattern_counts={k: len(v) for k, v in pats.items()})
    p = dump('data/sequences.json', data)
    # overlap audit vs V054H C probes / V054T probes (FREEZE_PLAN 2.1)
    audit = dict(exact_matches=0, v054h_k64_substring_hits={}, notes=[])
    vh = SCRATCH / 'ev/archive_v074/evaluation_sets/v054h_dynamics/data/C_mixed_probes.jsonl'
    vt = SCRATCH / 'ev/archive_v074/evaluation_sets/v054t_transition/data/probes.jsonl'
    if vh.exists():
        holds = {}
        for l in vh.read_text(encoding='utf-8').splitlines():
            r = json.loads(l)
            if r['hold_pairs'] is None:
                continue
            pr = [r['hold_pairs'][i:i + 2] for i in range(0, len(r['hold_pairs']), 2)]
            ps, qs = (f'{x}{y}' for x, y in TWO_SIDED[r['op']])
            if all(x in (ps, qs) for x in pr):          # only holds made of the two two-sided pairs are comparable
                holds.setdefault(f"{r['op']}{r['state']}", set()).add(''.join('1' if x == ps else '0' for x in pr))
        for k, v in pats.items():
            hs_ = holds.get(k, set())
            audit['exact_matches'] += sum(pt['pattern'] in hs_ for pt in v)
            audit['v054h_k64_substring_hits'][k] = dict(v054h_two_sided_only_holds=len(hs_),
                patterns_containing_any=sum(any(h in pt['pattern'] for h in hs_ if len(h) == 64) for pt in v))
        audit['notes'].append('V054H hold_pairs stored only for k<=64; our holds have length 2048, so exact matches cannot occur by length.')
    else:
        audit['notes'].append('V054H C_mixed_probes not available: overlap audit NOT performed')
    if vt.exists():
        audit['v054t_note'] = 'V054T probes hold a single pair (constant pattern); none of our patterns is constant: ' + \
            str(all(len(set(pt['pattern'])) == 2 for v in pats.values() for pt in v))
    audit['period2_alternation_present'] = any(pt['pattern'] in ('10' * (H // 2), '01' * (H // 2)) for v in pats.values() for pt in v)
    dump('data/overlap_audit.json', audit)
    print('sequences/model', nseq, data['pattern_counts'], 'dedup', {k: len(v) for k, v in dedup.items()}, 'audit', audit)


if __name__ == '__main__':
    main()
