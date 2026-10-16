"""Stage 1: break-timing prediction vs naive baselines (post hoc; existing CSVs only; no inference).

Rules: RULES_TIMING_BASELINE.md (SHA256 frozen in RULES_SHA256.json before computation).
Run from workspace root: .venv/Scripts/python.exe -I -B IronMath_PaperCheck/addon/timing_baseline/timing_baseline.py
"""
import csv, hashlib, json, math, os, statistics
from collections import defaultdict

import numpy as np

B = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(os.path.dirname(os.path.dirname(B)))
BT = os.path.join(W, "IronMath_PaperCheck", "break_timing")
frozen = json.load(open(os.path.join(B, "RULES_SHA256.json"), encoding="utf-8"))["SHA256"]
for f, h in frozen.items():
    assert hashlib.sha256(open(os.path.join(B, f), "rb").read()).hexdigest() == h, f


def rcsv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    return None if x in (None, "", "NA", "None") else float(x)


def med(v):
    return statistics.median(v) if v else None


def spearman(a, b):
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None

    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            k = i
            while k + 1 < len(o) and v[o[k + 1]] == v[o[i]]:
                k += 1
            for t in range(i, k + 1):
                r[o[t]] = (i + k) / 2 + 1
            i = k + 1
        return r
    ra, rb = rank(a), rank(b); ma, mb = statistics.mean(ra), statistics.mean(rb)
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))


units = rcsv(os.path.join(BT, "break_timing_units.csv"))
items = rcsv(os.path.join(BT, "break_timing_items.csv"))
inputs_sha = {f: hashlib.sha256(open(os.path.join(BT, f), "rb").read()).hexdigest()
              for f in ("break_timing_units.csv", "break_timing_items.csv")}
units = [u for u in units if u["pred_status"] != "missing"]
for u in units:
    u["kind"] = "arith_hold" if u["task"] == "arithmetic" else u["neutral"]
    u["div"] = u["divergent"] == "True"
    u["model_id"] = f"{u['experiment']}|{u['model']}|{u['seed']}"
    u["pred"] = num(u["predicted"])
items_by = defaultdict(list)
for i in items:
    if i["nonmonotonic"] == "False" and i["interval_upper_closed"] not in ("", None):
        items_by[(i["experiment"], i["model"], i["neutral"], int(i["value"]))].append(i)


def uitems(u):
    v = int(u["unit"].split()[0].split("=")[1])
    return items_by[(u["experiment"], u["model"], u["neutral"], v)]


# actual point value per unit (rule 2)
for u in units:
    if u["kind"] == "N":
        ups = [num(i["interval_upper_closed"]) for i in uitems(u)]
        u["actual_point"] = med(ups) if ups else None
    else:
        u["actual_point"] = num(u["actual_median"]) if u["actual_broken"] == "True" else None

# baselines L and M (rule 2)
grp = defaultdict(list)
for u in units:
    if u["actual_point"] is not None:
        grp[(u["experiment"], u["structure"], u["kind"], u["div"])].append(u)
bymodel = defaultdict(list)
for u in units:
    if u["actual_point"] is not None:
        bymodel[(u["model_id"], u["kind"])].append(u)
counts = {"L_none": 0, "M_fallback_to_L": 0, "M_none": 0}
targets = [u for u in units if u["classification"] == "両方あり" and
           (u["kind"] != "N" or uitems(u))]
for u in targets:
    o = [x["actual_point"] for x in grp[(u["experiment"], u["structure"], u["kind"], u["div"])] if x is not u]
    u["L"] = med(o)
    counts["L_none"] += u["L"] is None
    m = [x["actual_point"] for x in bymodel[(u["model_id"], u["kind"])] if x is not u]
    if m:
        u["M"], u["M_source"] = med(m), "M"
    else:
        u["M"], u["M_source"] = u["L"], ("L_fallback" if u["L"] is not None else "none")
        counts["M_fallback_to_L"] += u["L"] is not None
        counts["M_none"] += u["L"] is None


def inside(v, i):
    lo = num(i["interval_lower_open"]); up = num(i["interval_upper_closed"])
    return (lo is None or v > lo) and v <= up


def point_metrics(us, key):
    us = [u for u in us if u.get(key) is not None]
    if not us:
        return dict(n=0)
    e = [u[key] - num(u["actual_median"]) for u in us]
    return dict(n=len(us),
                in_range=sum(num(u["actual_min"]) <= u[key] <= num(u["actual_max"]) for u in us) / len(us),
                within3=sum(abs(x) <= 3 for x in e) / len(us), median_abs=med([abs(x) for x in e]), median_signed=med(e),
                spearman=spearman([u[key] for u in us], [num(u["actual_median"]) for u in us]))


def interval_metrics(us, key):
    us = [u for u in us if u.get(key) is not None]
    its = [(u, i) for u in us for i in uitems(u)]
    if not its:
        return dict(n=0)
    per = [statistics.mean(inside(u[key], i) for i in uitems(u)) for u in us]
    return dict(n=len(us), n_items=len(its), frac_items_inside=sum(inside(u[key], i) for u, i in its) / len(its),
                mean_unit_rate=statistics.mean(per))


rng = np.random.default_rng(85000)


def boot(us, dfun):
    """Median of per-unit d, plus model-cluster bootstrap 95% CI (10,000)."""
    pairs = [(u["model_id"], dfun(u)) for u in us]
    pairs = [(m, d) for m, d in pairs if d is not None]
    if not pairs:
        return dict(n=0)
    by = defaultdict(list)
    for m, d in pairs:
        by[m].append(d)
    models = sorted(by)
    stats = np.empty(10000)
    idx = rng.integers(0, len(models), size=(10000, len(models)))
    arrs = [np.array(by[m]) for m in models]
    for b in range(10000):
        stats[b] = np.median(np.concatenate([arrs[k] for k in idx[b]]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return dict(n_units=len(pairs), n_models=len(models), median_d=float(np.median([d for _, d in pairs])),
                ci_low=float(lo), ci_high=float(hi))


def verdict(b, better_negative):
    if not b.get("n_units"):
        return "対象なし"
    lo, hi = b["ci_low"], b["ci_high"]
    if lo <= 0 <= hi:
        return "明確な差を検出できなかった"
    good = hi < 0 if better_negative else lo > 0
    return "診断が基準線より正確" if good else "診断が基準線より劣る"


sections = [
    ("算術（V054T＋V054H）", lambda u: u["kind"] == "arith_hold"),
    ("記憶 R保持・非発散", lambda u: u["kind"] == "R" and not u["div"]),
    ("記憶 R保持・発散型", lambda u: u["kind"] == "R" and u["div"]),
    ("記憶 N保持・非発散", lambda u: u["kind"] == "N" and not u["div"]),
    ("記憶 N保持・発散型", lambda u: u["kind"] == "N" and u["div"]),
]
for st in ("residual_gru", "gru", "lstm", "rnn", "linear"):
    for k in ("R", "N"):
        sections.append((f"G1 {st} {k}", lambda u, st=st, k=k: u["experiment"] == "G1" and u["structure"] == st and u["kind"] == k))

rows = []
for name, f in sections:
    us = [u for u in targets if f(u)]
    kind = "N" if us and us[0]["kind"] == "N" else "point"
    row = dict(section=name, n_units=len(us), n_models=len({u["model_id"] for u in us}), type=kind)
    for key in ("pred", "L", "M"):
        mm = interval_metrics(us, key) if kind == "N" else point_metrics(us, key)
        for k2, v in mm.items():
            row[f"{key}_{k2}"] = v
    for bl in ("L", "M"):
        if kind == "N":
            b = boot([u for u in us if u.get(bl) is not None],
                     lambda u, bl=bl: statistics.mean(inside(u["pred"], i) for i in uitems(u)) -
                     statistics.mean(inside(u[bl], i) for i in uitems(u)))
            v = verdict(b, better_negative=False)
        else:
            b = boot([u for u in us if u.get(bl) is not None],
                     lambda u, bl=bl: abs(u["pred"] - num(u["actual_median"])) - abs(u[bl] - num(u["actual_median"])))
            v = verdict(b, better_negative=True)
        for k2, val in b.items():
            row[f"cmp_{bl}_{k2}"] = val
        row[f"cmp_{bl}_verdict"] = v
    rows.append(row)

# Spearman: overall and by structure / retention type (point units)
pt = [u for u in targets if u["kind"] != "N"]
spear = []
for name, sel in [("全体（点の単位）", pt)] + \
        [(f"保持の種類={k}", [u for u in pt if u["kind"] == k]) for k in ("arith_hold", "R")] + \
        [(f"構造={s}", [u for u in pt if u["structure"] == s]) for s in sorted({u["structure"] for u in pt})]:
    r = dict(group=name, n=len(sel))
    for key in ("pred", "L", "M"):
        s2 = [u for u in sel if u.get(key) is not None]
        r[f"spearman_{key}"] = spearman([u[key] for u in s2], [num(u["actual_median"]) for u in s2])
    spear.append(r)

# Exploratory sensitivity: G1 linear unpredicted failures, all-prefix first negative margin
diag = rcsv(os.path.join(W, "IronMath_G1", "g1_diagnostic_details.csv"))
dmin = defaultdict(list)
for r in diag:
    if num(r["first_negative_margin_step"]) is not None:
        dmin[(r["model"], r["neutral"], int(r["value"]))].append(num(r["first_negative_margin_step"]))
sens_units = []
for u in units:
    if u["experiment"] == "G1" and u["structure"] == "linear" and u["classification"].startswith("予測なし・壊れた"):
        v = int(u["unit"].split()[0].split("=")[1])
        alt = min(dmin[(u["model"], u["neutral"], v)]) if dmin[(u["model"], u["neutral"], v)] else None
        su = dict(model=u["model"], neutral=u["neutral"], value=v, alt_pred=alt,
                  actual_median=u["actual_median"], actual_min=u["actual_min"], actual_max=u["actual_max"])
        if u["neutral"] == "N":
            its = uitems(u)
            su["n_items"] = len(its)
            su["items_inside"] = sum(inside(alt, i) for i in its) if alt is not None and its else None
        sens_units.append(su)
sR = [s for s in sens_units if s["neutral"] == "R" and s["alt_pred"] is not None]
sN = [s for s in sens_units if s["neutral"] == "N" and s["alt_pred"] is not None and s["n_items"]]
eR = [s["alt_pred"] - num(s["actual_median"]) for s in sR]
sens = dict(n_units=len(sens_units), n_with_alt=sum(s["alt_pred"] is not None for s in sens_units),
            R=dict(n=len(sR), in_range=sum(num(s["actual_min"]) <= s["alt_pred"] <= num(s["actual_max"]) for s in sR) / len(sR) if sR else None,
                   within3=sum(abs(e) <= 3 for e in eR) / len(eR) if eR else None,
                   median_abs=med([abs(e) for e in eR]), median_signed=med(eR)),
            N=dict(n=len(sN), n_items=sum(s["n_items"] for s in sN),
                   items_inside=sum(s["items_inside"] for s in sN)))


def wcsv(name, rs):
    keys = []
    for r in rs:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(B, name), "x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rs)


wcsv("timing_baseline_summary.csv", rows)
wcsv("timing_baseline_spearman.csv", spear)
wcsv("timing_baseline_units.csv", [dict(experiment=u["experiment"], model=u["model"], seed=u["seed"], structure=u["structure"],
                                        kind=u["kind"], divergent=u["div"], unit=u["unit"], pred=u["pred"], baseline_L=u.get("L"),
                                        baseline_M=u.get("M"), M_source=u.get("M_source"), actual_point=u["actual_point"],
                                        actual_median=u["actual_median"], actual_min=u["actual_min"], actual_max=u["actual_max"])
                                   for u in targets])
wcsv("sensitivity_G1_linear_units.csv", sens_units)
with open(os.path.join(B, "timing_baseline_record.json"), "x", encoding="utf-8") as f:
    json.dump(dict(rules_SHA256=frozen, inputs_SHA256=inputs_sha, n_targets=len(targets), baseline_counts=counts,
                   sensitivity=sens, bootstrap=dict(seed=85000, reps=10000, method="model-cluster percentile")),
              f, ensure_ascii=False, indent=1)
print(json.dumps(dict(n_targets=len(targets), counts=counts, sens=sens), ensure_ascii=False))
for r in rows:
    print(r["section"], r["n_units"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                       if k.startswith(("pred_", "L_", "M_", "cmp_")) and not k.endswith(("n_models",))})
for r in spear:
    print(r)
