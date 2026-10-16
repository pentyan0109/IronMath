"""v0.7.3 追加集計。保存済みCSVの読み取りのみ。学習・推論・Test評価・モデル読込は行わない。"""
import csv, json, hashlib, statistics
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
X = R.parent
E2 = X / 'E2_causal_intervention/E2_RESULTS.csv'
E2X = X / 'v072_audit/E2_ALL_CONTROLS_TRIALS.csv'          # 照合専用
E3M = X / 'E3_independent_replication/E3_MODEL_RESULTS.csv'
E3T = X / 'E3_independent_replication/E3_TEST_RESULTS.csv'
E3P = X / 'E3_independent_replication/E3_PAIR_RESULTS.csv'
E3S = X / 'v072_audit/E3_ENDPOINT_SIDE_OUTCOME.csv'
E3D2 = X / 'v072_audit/E3_READOUT_D2_CROSSTAB.csv'
INPUTS = [E2, E2X, E3M, E3T, E3P, E3S, E3D2]
AT = 8
B = 10000
BOOT_SEED = 73000
CONTROLS = [f'orthogonal{i}' for i in range(3)] + [f'random{i}' for i in range(3)]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with open(p, encoding='utf-8', newline='') as f: return list(csv.DictReader(f))
def num(v): return None if v in ('', None) else float(v)
def write_csv(p, rs):
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
def md(rs, cols):
    def c(v): return 'NA' if v is None else (f'{v:.6g}' if isinstance(v, float) else str(v))
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)] + ['| ' + ' | '.join(c(r[k]) for k in cols) + ' |' for r in rs]) + '\n'
def boot(vals, rng):
    v = np.asarray(vals, float); idx = rng.integers(0, len(v), size=(B, len(v))); s = v[idx]
    m, md_ = s.mean(1), np.median(s, 1)
    return [float(x) for x in np.percentile(m, [2.5, 97.5])], [float(x) for x in np.percentile(md_, [2.5, 97.5])]


before = {p.relative_to(X).as_posix(): sha(p) for p in INPUTS}

# ---------------- E2 ----------------
e2 = rows(E2)
meas = [r for r in e2 if r['status'] == 'measured']
key = lambda r: (r['seed'], r['op'], r['state'], r['pair'])
base = {key(r): r for r in meas if r['condition'] == 'baseline' and r['at'] == str(AT) and r['first_error']}
assert len(base) == 10, len(base)
cell = {(key(r), r['condition']): r for r in meas if r['at'] == str(AT)}
xcheck = {(r['seed'], r['op'], r['state'], r['pair'], r['condition']): r for r in rows(E2X) if r['at'] == str(AT)}

table, diffs = [], {'correct': [], 'failed': []}
for k in sorted(base):
    T0 = int(base[k]['first_error'])
    for donor in ['correct', 'failed']:
        rec = dict(checkpoint_seed=int(k[0]), op=k[1], state=int(k[2]), pair=k[3], donor=donor, baseline_first_error=T0)
        def get(cond):
            r = cell.get((k, f'{cond}_{donor}'))
            if r is None or not r['first_error']: return None, None, None, r
            return int(r['first_error']) - T0, num(r['immediate_margin_change']), r['first_error_censored'] == 'True', r
        sp, mp, cp, rp = get('normal_1')
        rec['delta_norm'] = num(rp['delta_norm']) if rp else None
        rec['normal_pos_delay'] = sp; rec['normal_pos_dmargin_step8'] = mp
        for lag in [1, 4, 16, 64]: rec[f'normal_pos_dmargin_after_{lag}'] = num(rp[f'margin_change_after_{lag}']) if rp else None
        cds, cms, cens = [], [], [cp]
        for c in CONTROLS:
            s, m, ce, _ = get(c); rec[f'{c}_delay'] = s; rec[f'{c}_dmargin_step8'] = m; cds.append(s); cms.append(m); cens.append(ce)
            xr = xcheck[k + (f'{c}_{donor}',)]; assert xr['shift'] and float(xr['shift']) == s, (k, c, donor)
        xr = xcheck[k + (f'normal_1_{donor}',)]; assert float(xr['shift']) == sp
        assert not any(cens), 'censored first error present'
        complete = sp is not None and all(s is not None for s in cds)
        rec['controls_mean_delay'] = float(np.mean(cds)) if complete else None
        rec['controls_max_delay'] = max(cds) if complete else None
        rec['normal_pos_minus_controls_mean'] = sp - rec['controls_mean_delay'] if complete else None
        rec['normal_pos_gt_all6'] = (sp > max(cds)) if complete else None
        rec['controls_max_abs_dmargin_step8'] = max(abs(m) for m in cms)
        rec['normal_pos_delay_per_unit_dmargin'] = sp / mp if mp else None
        table.append(rec)
        if complete: diffs[donor].append(rec['normal_pos_minus_controls_mean'])

rng = np.random.default_rng(BOOT_SEED)
summ = []
for donor in ['correct', 'failed']:
    tt = [r for r in table if r['donor'] == donor]; v = diffs[donor]
    ci_m, ci_md = boot(v, rng)
    per_ckpt = {s: [r['normal_pos_minus_controls_mean'] for r in tt if r['checkpoint_seed'] == s] for s in (0, 1, 2)}
    summ.append(dict(donor=donor, contexts=len(tt), complete_contexts=len(v),
                     contexts_normal_gt_all6=sum(bool(r['normal_pos_gt_all6']) for r in tt),
                     diff_median=float(np.median(v)), diff_min=float(min(v)), diff_max=float(max(v)), diff_mean=float(np.mean(v)),
                     boot_mean_CI95_low=ci_m[0], boot_mean_CI95_high=ci_m[1], boot_median_CI95_low=ci_md[0], boot_median_CI95_high=ci_md[1],
                     contexts_per_checkpoint='/'.join(str(len(per_ckpt[s])) for s in (0, 1, 2)),
                     checkpoint_mean_diff='/'.join(f'{np.mean(per_ckpt[s]):.4g}' for s in (0, 1, 2)),
                     normal_pos_dmargin_median=float(np.median([r['normal_pos_dmargin_step8'] for r in tt])),
                     normal_pos_dmargin_min=float(min(r['normal_pos_dmargin_step8'] for r in tt)),
                     normal_pos_dmargin_max=float(max(r['normal_pos_dmargin_step8'] for r in tt)),
                     controls_abs_dmargin_max=float(max(r['controls_max_abs_dmargin_step8'] for r in tt))))

# 直後マージン変化と遅延の並置（長形式、全7方向×10文脈×2 donor）
long = []
for r in table:
    for c in ['normal_pos'] + CONTROLS:
        long.append(dict(checkpoint_seed=r['checkpoint_seed'], op=r['op'], state=r['state'], pair=r['pair'], donor=r['donor'], direction=c,
                         dmargin_step8=r[f'{c}_dmargin_step8'], first_error_delay=r[f'{c}_delay']))
# 対照内：|Δmargin|が非常に小さい直交(≈0)と、ランダム(|Δ|≤数)の遅延の幅
orth = [x for x in long if x['direction'].startswith('orthogonal')]
rand = [x for x in long if x['direction'].startswith('random')]
def rng_(xs, f): v = [f(x) for x in xs]; return min(v), max(v)

write_csv(R / 'E2_PER_CONTEXT.csv', table)
write_csv(R / 'E2_PER_CONTEXT_LONG.csv', long)
write_csv(R / 'E2_PER_CONTEXT_SUMMARY.csv', summ)

cols_main = ['checkpoint_seed', 'op', 'state', 'pair', 'donor', 'baseline_first_error', 'normal_pos_delay'] + [f'{c}_delay' for c in CONTROLS] + ['normal_pos_minus_controls_mean', 'normal_pos_gt_all6']
cols_m = ['checkpoint_seed', 'op', 'state', 'pair', 'donor', 'delta_norm', 'normal_pos_dmargin_step8', 'normal_pos_dmargin_after_1', 'normal_pos_dmargin_after_4', 'normal_pos_dmargin_after_16', 'normal_pos_dmargin_after_64', 'controls_max_abs_dmargin_step8', 'normal_pos_delay', 'controls_mean_delay']
ox, rx = rng_(orth, lambda x: abs(x['dmargin_step8'])), rng_(rand, lambda x: abs(x['dmargin_step8']))
od, rd = rng_(orth, lambda x: x['first_error_delay']), rng_(rand, lambda x: x['first_error_delay'])
S = {s['donor']: s for s in summ}
# 追加の記述量
by = {(r['checkpoint_seed'], r['op'], r['state'], r['pair'], r['donor']): r for r in table}
ctx = sorted({k[:4] for k in by})
ratio1 = [by[c + (d,)]['normal_pos_dmargin_after_1'] / by[c + (d,)]['normal_pos_dmargin_step8'] for c in ctx for d in ('correct', 'failed')]
dm_ratio = [by[c + ('failed',)]['normal_pos_dmargin_step8'] / by[c + ('correct',)]['normal_pos_dmargin_step8'] for c in ctx]
dd = [by[c + ('failed',)]['normal_pos_delay'] - by[c + ('correct',)]['normal_pos_delay'] for c in ctx]
nd = {d: [by[c + (d,)]['normal_pos_delay'] for c in ctx] for d in ('correct', 'failed')}
or_gap = [abs(r[f'orthogonal{i}_delay'] - r[f'random{i}_delay']) for r in table for i in range(3)]
text = f"""# E2 文脈別比較（v0.7.3 追加集計）

入力：`E2_causal_intervention/E2_RESULTS.csv`（測定済み行のみ）。新規学習・推論・Test評価なし。介入時刻 step {AT}、無介入で1024窓内に初回誤答がある10文脈。
遅延＝介入条件の初回誤答時刻 − 同文脈baselineの初回誤答時刻（正＝遅くなる）。全140セル（10文脈×7方向×2 donor）で初回誤答が観測され、右打切りは0。`v072_audit/E2_ALL_CONTROLS_TRIALS.csv` の shift と全セル一致を確認。
「正方向法線」＝`normal_1`（w_true−w_wrong方向）。対照6方向＝orthogonal0–2、random0–2（同文脈・同donorのdelta normに一致）。

## 1. 文脈×donor 表（遅延、step）

{md(table, cols_main)}
## 2. 「正方向法線の遅延 > 対照6方向すべて」（厳密不等号）

| donor | 成立文脈数 / 10 |
|---|---|
| correct | {S['correct']['contexts_normal_gt_all6']} |
| failed | {S['failed']['contexts_normal_gt_all6']} |

## 3. 差（正方向法線 − 対照6方向の平均）

{md(summ, ['donor', 'complete_contexts', 'diff_median', 'diff_min', 'diff_max', 'diff_mean', 'boot_mean_CI95_low', 'boot_mean_CI95_high', 'boot_median_CI95_low', 'boot_median_CI95_high'])}
区間は10文脈を復元抽出する percentile bootstrap（{B}回、`numpy.random.default_rng({BOOT_SEED})`、correct→failedの順に同一生成器で実行）。

**入れ子の注記**：10文脈は3 checkpoint（V054 C1 B0 seed0/1/2）に入れ子で、文脈数は seed0/1/2 = {S['correct']['contexts_per_checkpoint']}。checkpoint別の差の平均は correct {S['correct']['checkpoint_mean_diff']}、failed {S['failed']['checkpoint_mean_diff']}。上の区間は文脈を独立単位として扱っており、同一checkpoint内の文脈間相関を反映しないため、checkpoint間変動に対しては狭すぎる可能性がある。3クラスタでのbootstrapは実質的に意味をなさないため実施していない。correct donorとfailed donorは同じ10文脈の対であり独立な20単位ではない。

## 4. 介入直後のマージン変化（Δmargin at step {AT}）と初回誤答時刻の変化

{md(table, cols_m)}
- 正方向法線のΔmargin(step{AT})：correct donor 中央値 {S['correct']['normal_pos_dmargin_median']:.4g}（範囲 {S['correct']['normal_pos_dmargin_min']:.4g}–{S['correct']['normal_pos_dmargin_max']:.4g}）、failed donor 中央値 {S['failed']['normal_pos_dmargin_median']:.4g}（範囲 {S['failed']['normal_pos_dmargin_min']:.4g}–{S['failed']['normal_pos_dmargin_max']:.4g}）。
- 対照6方向の |Δmargin(step{AT})| の最大は correct {S['correct']['controls_abs_dmargin_max']:.4g}、failed {S['failed']['controls_abs_dmargin_max']:.4g}。直交3方向は |Δ| ≤ {ox[1]:.2g}（float32丸め程度で実質0）、ランダム3方向は |Δ| が {rx[0]:.3g}–{rx[1]:.3g}。
- 遅延の範囲：直交（Δ≈0）{od[0]}〜{od[1]} step、ランダム {rd[0]}〜{rd[1]} step。
- 同じ番号 i の orthogonal_i と random_i の遅延差 |orth_i − random_i|（60組）：0〜{max(or_gap)} step、1 step以内 {sum(g <= 1 for g in or_gap)}/60。直後Δmarginは前者が≈0、後者が最大{rx[1]:.3g}。
- 正方向法線のΔmarginの持続：介入1 step後のmargin差 / step{AT}直後のmargin差 は中央値 {np.median(ratio1):.3g}（範囲 {min(ratio1):.3g}–{max(ratio1):.3g}、20セル）。64 step後は表の通り多くの文脈で≈0。
- 正方向法線内のdonor比較（同じ10文脈）：failed/correct の直後Δmargin比 中央値 {np.median(dm_ratio):.3g}（範囲 {min(dm_ratio):.3g}–{max(dm_ratio):.3g}）。法線遅延は correct 中央値 {np.median(nd['correct']):g}（{min(nd['correct'])}〜{max(nd['correct'])}）、failed 中央値 {np.median(nd['failed']):g}（{min(nd['failed'])}〜{max(nd['failed'])}）、文脈ごとの差（failed−correct）は {min(dd)}〜{max(dd)} step（中央値 {np.median(dd):g}）。

**記述**：正方向法線と対照6方向では直後のΔmarginが同程度ではない（法線は対照の最大|Δ|より1桁以上大きい）。したがって本データでは「直後のマージン変化を揃えたうえで方向による遅延差があるか」を法線対対照で直接比較することはできない。直後Δmarginが近い組として比較できるのは対照どうし（直交≈0 とランダム≤数単位）であり、この範囲では遅延は文脈によって0から外れた値（上記範囲）をとり、同じ番号の直交/ランダム方向で近い値をとることが多い。すなわち直後Δmarginが実質0でも初回誤答時刻は変わり得て、直後のΔmarginだけでは後の初回誤答時刻の変化は定まらない（`scripts/e2.py` では orthogonal_i は random_i と同じ乱数ベクトルから法線成分を除いて作られており、両者の遅延が近いことはこの構成と整合する）。法線の直後Δmarginは1 step後には上記の比まで縮小しており、直後の値の大きさがそのまま持続するわけではない。正方向法線の correct/failed donor 条件は同じ単位法線 u に沿った摂動で、違いはノルム（＝donor差分のノルム）だけである（`base + norm·u`）。この同一方向内の比較では、直後Δmarginが約2倍の failed 側で遅延が同じか長い文脈が多いが、遅延はΔmarginに比例していない。法線については介入後1/4/16/64 stepのmargin差を表に並べた（`margin_change_after_k`）。全7方向×2 donorの長形式は `E2_PER_CONTEXT_LONG.csv`。

これは保存済み介入結果の文脈別再集計であり、新しい因果主張の根拠を追加するものではない。
"""
(R / 'E2_PER_CONTEXT.md').write_text(text, encoding='utf-8')

# ---------------- E3 ----------------
models = rows(E3M); tests = rows(E3T); pairs = rows(E3P)
acc = {(r['model'], r['op'], r['width']): r for r in tests}
e3 = []
for m in models:
    mm = m['model']; pp = [p for p in pairs if p['model'] == mm]; assert len(pp) == 4
    rec = dict(model=mm, initialization_seed=int(m['initialization_seed']), data_condition=int(m['data_condition']),
               shared_endpoints=sum(p['shared'] == 'True' for p in pp), distinct_endpoints=sum(p['distinct'] == 'True' for p in pp),
               strict_pairs=sum(p['strict_mechanism_reproduced'] == 'True' for p in pp))
    assert rec['shared_endpoints'] == int(m['shared_pairs'])
    for op in ['ADD', 'SUB']:
        a = acc[(mm, op, '2048')]; rec[f'{op}_2048_correct'] = int(a['correct']); rec[f'{op}_2048_N'] = int(a['N']); rec[f'{op}_2048_accuracy'] = float(a['accuracy'])
    e3.append(rec)
by_k = []
for k in range(5):
    tt = [r for r in e3 if r['shared_endpoints'] == k]
    by_k.append(dict(shared_endpoints=k, models=len(tt), model_ids=' '.join(r['model'] for r in tt) or '-',
                     ADD_2048_median=float(np.median([r['ADD_2048_accuracy'] for r in tt])) if tt else None,
                     ADD_2048_min=min((r['ADD_2048_accuracy'] for r in tt), default=None), ADD_2048_max=max((r['ADD_2048_accuracy'] for r in tt), default=None),
                     SUB_2048_median=float(np.median([r['SUB_2048_accuracy'] for r in tt])) if tt else None,
                     SUB_2048_min=min((r['SUB_2048_accuracy'] for r in tt), default=None), SUB_2048_max=max((r['SUB_2048_accuracy'] for r in tt), default=None)))
write_csv(R / 'E3_SHARED_VS_ACCURACY.csv', e3)
write_csv(R / 'E3_SHARED_VS_ACCURACY_BY_COUNT.csv', by_k)

# seed 101 c0/c1
s101 = []
m101 = {m['data_condition']: m for m in models if m['initialization_seed'] == '101'}
def add(item, a, b): s101.append(dict(item=item, c0=a, c1=b, c1_minus_c0=(b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) else None))
for f_ in ['data_seed', 'selected_update', 'Validation_ADD_exact', 'Validation_SUB_exact', 'shared_pairs', 'distinct_pairs', 'unclassified_pairs', 'strict_reproduced_pairs']:
    a, b_ = m101['0'][f_], m101['1'][f_]; a, b_ = (float(a), float(b_)) if '.' in a or '.' in b_ else (int(a), int(b_)); add(f_, a, b_)
add('initial_SHA256_equal', m101['0']['initial_SHA256'] == m101['1']['initial_SHA256'], None); s101[-1]['c1'] = '(same as c0)' if s101[-1]['c0'] else '(differs)'
for op in ['ADD', 'SUB']:
    for w in ['8', '64', '256', '2048']:
        a, b_ = acc[('i101_c0', op, w)], acc[('i101_c1', op, w)]
        add(f'Test_{op}_{w}_accuracy', float(a['accuracy']), float(b_['accuracy']))
        add(f'Test_{op}_{w}_loss', float(a['loss']), float(b_['loss']))
for p0 in sorted([p for p in pairs if p['model'] == 'i101_c0'], key=lambda p: (p['op'], p['pair'])):
    p1 = next(p for p in pairs if p['model'] == 'i101_c1' and p['op'] == p0['op'] and p['pair'] == p0['pair'])
    add(f'pair_{p0["op"]}_{p0["pair"]}_endpoint_distance', float(p0['distance']), float(p1['distance']))
    s101.append(dict(item=f'pair_{p0["op"]}_{p0["pair"]}_class', c0='shared' if p0['shared'] == 'True' else 'distinct' if p0['distinct'] == 'True' else 'unclassified',
                     c1='shared' if p1['shared'] == 'True' else 'distinct' if p1['distinct'] == 'True' else 'unclassified', c1_minus_c0=None))
side = [r for r in rows(E3S) if r['initialization'] == '101']
for c in ['0', '1']:
    ss = [r for r in side if r['data_condition'] == c]
    s101_side = dict(failing_sides=sum(r['outcome'] == 'failure' for r in ss), wrong_readout_sides=sum(r['endpoint_correct'] == 'False' for r in ss))
    if c == '0': side0 = s101_side
add('hold_failing_sides_of_12', side0['failing_sides'], s101_side['failing_sides'])
add('endpoint_wrong_readout_sides_of_12', side0['wrong_readout_sides'], s101_side['wrong_readout_sides'])
write_csv(R / 'E3_SEED101_C0_C1.csv', s101)

# Not evaluable 確認
d2 = rows(E3D2); sides = rows(E3S)
uneval_total = sum(int(r['unevaluable']) for r in d2 if r['scope'] == 'all240')
trials_hist = {}
for r in sides: trials_hist[r['observed_hold_trials']] = trials_hist.get(r['observed_hold_trials'], 0) + 1
unconv = sum(r['converged'] != 'True' for r in sides)

e3text = f"""# E3 共有終点数と2048bit正答率（v0.7.3 追加集計・探索的）

**探索的記述であり、論文の主張には使わない。** 入力：`E3_MODEL_RESULTS.csv`、`E3_PAIR_RESULTS.csv`、`E3_TEST_RESULTS.csv`（保存済みTest結果、再評価なし）。共有終点数＝各モデル4つの両carry保持入力対のうち、凍結規則で shared と分類された数。正答率は固定Test 2048bit 各512問の整数完全一致。

## 1. モデル別

{md(e3, ['model', 'shared_endpoints', 'distinct_endpoints', 'strict_pairs', 'ADD_2048_correct', 'ADD_2048_accuracy', 'SUB_2048_correct', 'SUB_2048_accuracy'])}
## 2. 共有終点数ごと

{md(by_k, ['shared_endpoints', 'models', 'model_ids', 'ADD_2048_median', 'ADD_2048_min', 'ADD_2048_max', 'SUB_2048_median', 'SUB_2048_min', 'SUB_2048_max'])}
20モデルは10初期化×2提示条件で、同一初期化の2モデルは独立ではない。相関係数・検定は算出していない。

## 3. seed 101 の c0 / c1（数値のみ）

{md(s101, ['item', 'c0', 'c1', 'c1_minus_c0'])}
hold/読出しの側単位の値は `v072_audit/E3_ENDPOINT_SIDE_OUTCOME.csv`（12側/モデル）から数えた。
"""
(R / 'E3_SHARED_VS_ACCURACY.md').write_text(e3text, encoding='utf-8')

note = f"""# E3表「Not evaluable」列の定義確認（v0.7.3）

対象：`v072_final/main_v0.7.2.tex` Table `tab:e3d2`（Endpoint readout × Broke / Did not break in window / Not evaluable）。数値の出典は `v072_audit/E3_READOUT_D2_CROSSTAB.csv` の scope=all240 行、生成コードは `v072_audit/audit_v072.py` の `e3()`。

**数えていたもの**：D2単位の「側」（20モデル×12入力–carry文脈＝240、`E3_DIAGNOSTICS.csv` の各行）のうち、
1. 保存済みhold試行（k16/64/256/1024 × 16 prefix履歴＝本来64試行、`E3_HOLD_RESULTS.csv`）のどれにもhold内誤りがなく、かつ
2. その側の保存済みhold試行数が64に満たない
ものの数（コード：`status='failure' if ts else 'no_failure_within_window' if len(hs)==64 else 'unevaluable'`）。すなわち「誤りは観測されていないが、hold試行が欠けているため窓内で破綻しなかったと言えない側」を数える列である。

**数えていないもの**：
- 終点が非収束で読出し不能な側。これは行（Endpoint readout＝`unavailable`）側に入り、論文表にはこの行が掲載されていない（CSVでは all240 の unavailable n=0）。
- 共有/別終点の未分類pair（pair単位の表 `E3_ENDPOINT_CROSSTAB.csv` の unclassified 行・unevaluable 列は別物）。
- prefix誤りのみの試行（hold破綻に含めない規則で、試行数には数えられる）。
- 一部試行が欠けていても1試行でもhold誤りがあれば Broke に入る。

**実データ**：全240側で保存済みhold試行数の分布は {trials_hist}、非収束側は {unconv}。全側で64試行が揃っていたため、この列は実データ上すべてのセルで0（all240の合計 {uneval_total}）である。列は「評価不能の側が生じた場合の受け皿」として用意されたもので、0は欠測がなかったことを意味する。
"""
(R / 'E3_NOT_EVALUABLE_CHECK.md').write_text(note, encoding='utf-8')

after = {p.relative_to(X).as_posix(): sha(p) for p in INPUTS}
assert before == after
outs = sorted(p.name for p in R.iterdir() if p.suffix in ('.csv', '.md'))
for p in outs: assert '�' not in (R / p).read_text(encoding='utf-8')
json.dump(dict(script=Path(__file__).name, script_SHA256=sha(__file__), inputs_SHA256_unchanged=after, bootstrap=dict(B=B, seed=BOOT_SEED, unit='context (10)'),
               outputs={p: sha(R / p) for p in outs}, no_training_inference_or_test_evaluation=True), open(R / 'MANIFEST_v073.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(summ, ensure_ascii=False, indent=1)); print(trials_hist, unconv, uneval_total)
