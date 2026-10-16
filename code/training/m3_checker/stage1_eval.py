"""Stage 1 Test (opened after all 3 seeds trained): length generalization + structured + hold table."""
import csv, json, random, time, collections
import numpy as np
from m3_common import HERE, int_bits, bits_int, true_residues, predict, load_checker, peak_ram_mib

cfg = json.load(open(HERE / 'config.json', encoding='utf-8'))
WIDTHS = cfg['stage1']['test']['widths']
t0 = time.process_time(); w0 = time.perf_counter()


def n_random(W):
    return 256 if W <= 2048 else 64 if W == 8192 else 32


def structured(W):
    out = {}
    out['all0'] = np.zeros(W, np.uint8); out['all1'] = np.ones(W, np.uint8)
    out['1010'] = (np.arange(W) % 2 == 0).astype(np.uint8); out['0101'] = (np.arange(W) % 2 == 1).astype(np.uint8)
    R = min(1000, W); s = (W - R) // 2
    b = int_bits(random.Random(95000 + W).getrandbits(W), W); b[s:s + R] = 1; out['run1000_ones'] = b
    rng = random.Random(96000 + W); out['sparse1pct'] = np.array([rng.random() < 0.01 for _ in range(W)], np.uint8)
    b = int_bits(random.Random(97000 + W).getrandbits(W), W); b[s:s + R] = 0; out['run1000_zeros'] = b
    P = min(16, W); b = np.ones(W, np.uint8); b[:P] = int_bits(random.Random(98000 + W).getrandbits(P), P); out['prefix16_then_ones'] = b
    return out


# ---------- test sets
tests = []  # (width, kind, idx, bits)
for W in WIDTHS:
    rng = random.Random(93000 + W)
    for i in range(n_random(W)):
        tests.append((W, 'random', i, int_bits(rng.getrandbits(W), W)))
    for k, b in structured(W).items():
        tests.append((W, k, 0, b))
truths = [true_residues(t[3]) for t in tests]
for t, tr in zip(tests, truths):  # final label vs Python integer
    assert t[3].size == t[0] and int(tr[-1]) == bits_int(t[3]) % 3

# ---------- hold table probes
hr = random.Random(99000); prefixes = {}
for parity, L in ((0, 16), (1, 17)):
    need = {r: [] for r in range(3)}
    while any(len(v) < 4 for v in need.values()):
        v = hr.getrandbits(L)
        if len(need[v % 3]) < 4:
            need[v % 3].append(v)
    for r in range(3):
        prefixes[(r, parity)] = [(v, L) for v in need[r]]
suffixes = [hr.getrandbits(16) for _ in range(4)]
holds = []  # (state_r, parity, bit, k, pi, si, bits, hold_start, hold_end)
for (r, parity), pl in sorted(prefixes.items()):
    for bit in (0, 1):
        for k in (8, 16, 64, 256, 1024, 4096):
            for pi, (pv, L) in enumerate(pl):
                for si, sv in enumerate(suffixes):
                    bits = np.concatenate([int_bits(pv, L), np.full(k, bit, np.uint8), int_bits(sv, 16)])
                    tr = true_residues(bits); assert int(tr[L - 1]) == r and L % 2 == parity
                    holds.append((r, parity, bit, k, pi, si, bits, L, L + k))
json.dump(dict(prefixes={f'r{r}_p{p}': [hex(v) for v, _ in pl] for (r, p), pl in prefixes.items()}, suffixes=[hex(s) for s in suffixes]),
          open(HERE / 'data' / 'hold_probe_meta.json', 'w'), indent=1)

ev_rows, hold_rows, timing = [], [], {}
for seed in (0, 1, 2):
    m, cp = load_checker(seed); ts = time.perf_counter()
    preds = predict(m, [t[3] for t in tests], batch=256)
    for (W, kind, i, b), tr, p in zip(tests, truths, preds):
        wrong = np.nonzero(p != tr)[0]
        ev_rows.append(dict(seed=seed, width=W, kind=kind, idx=i, final_correct=int(p[-1] == tr[-1]), step_acc=round(float((p == tr).mean()), 6),
                            first_error=int(wrong[0]) if wrong.size else 'NA', n_wrong_steps=int(wrong.size), pred_final=int(p[-1]), true_final=int(tr[-1]),
                            n_ones=int(b.sum())))
    hp = predict(m, [h[6] for h in holds], batch=512)
    for h, p in zip(holds, hp):
        r, parity, bit, k, pi, si, bits, hs, he = h; tr = true_residues(bits); wrong = np.nonzero(p != tr)[0]
        fe = int(wrong[0]) if wrong.size else None
        region = 'NA' if fe is None else 'prefix' if fe < hs else 'hold' if fe < he else 'suffix'
        hold_rows.append(dict(seed=seed, state_r=r, parity=parity, hold_bit=bit, k=k, prefix=pi, suffix=si, seq_len=bits.size, hold_start=hs, hold_end=he,
                              exact=int(fe is None), final_correct=int(p[-1] == tr[-1]), first_error='NA' if fe is None else fe,
                              first_error_region=region, hold_offset='NA' if fe is None else fe - hs, n_wrong_steps=int(wrong.size)))
    timing[seed] = round(time.perf_counter() - ts, 2); print('seed', seed, 'eval wall', timing[seed], flush=True)

for name, rows in (('checker_eval.csv', ev_rows), ('checker_hold_table.csv', hold_rows)):
    with open(HERE / name, 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---------- summaries
acc = collections.defaultdict(list)
for r in ev_rows:
    if r['kind'] == 'random':
        acc[(r['seed'], r['width'])].append(r)
rand = {f's{s}|{W}': dict(n=len(v), final_acc=round(sum(x['final_correct'] for x in v) / len(v), 6),
                          step_acc_mean=round(float(np.mean([x['step_acc'] for x in v])), 6),
                          first_error_median=(sorted(x['first_error'] for x in v if x['first_error'] != 'NA')[len([x for x in v if x['first_error'] != 'NA']) // 2]
                                              if any(x['first_error'] != 'NA' for x in v) else 'NA'),
                          first_error_min=min((x['first_error'] for x in v if x['first_error'] != 'NA'), default='NA'),
                          all_steps_correct=sum(x['first_error'] == 'NA' for x in v))
        for (s, W), v in sorted(acc.items())}
a65 = [rand[f's{s}|65000']['final_acc'] for s in (0, 1, 2)]; a2048 = [rand[f's{s}|2048']['final_acc'] for s in (0, 1, 2)]
C1 = '支持' if all(a >= 0.99 for a in a65) else '不支持' if all(a < 0.99 for a in a2048) else '判定不能'
hold = collections.defaultdict(list)
for r in hold_rows:
    hold[(r['seed'], r['state_r'], r['parity'], r['hold_bit'], r['k'])].append(r)
breaking = {}
for s in (0, 1, 2):
    for rr in range(3):
        for par in (0, 1):
            for bit in (0, 1):
                ks = [k for k in (8, 16, 64, 256, 1024, 4096) if sum(x['exact'] for x in hold[(s, rr, par, bit, k)]) < 16]
                breaking[f's{s}|r{rr}p{par}|bit{bit}'] = min(ks) if ks else 'never(<=4096)'
summary = dict(random=rand, C1=dict(acc65000=a65, acc2048=a2048, verdict=C1),
               hold_exact_counts={f's{s}|r{r}p{p}|bit{b}|k{k}': sum(x['exact'] for x in v) for (s, r, p, b, k), v in sorted(hold.items())},
               hold_breaking_k=breaking, eval_wall_per_seed=timing,
               cpu_seconds=round(time.process_time() - t0, 2), wall_seconds=round(time.perf_counter() - w0, 2), peak_ram_mib=peak_ram_mib())
json.dump(summary, open(HERE / 'stage1_summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(dict(C1=summary['C1'], random={k: (v['final_acc'], v['step_acc_mean'], v['first_error_min'], v['first_error_median']) for k, v in rand.items()}), ensure_ascii=False, indent=0))
