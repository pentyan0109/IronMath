"""Break-timing re-aggregation (post hoc; existing saved data only; no inference).

Rules: BREAK_TIMING_RULES.md (SHA256 recorded before aggregation).
Reads original experiment outputs read-only; writes only into this folder.
Run from the workspace root with: .venv/Scripts/python.exe -I -B IronMath_PaperCheck/break_timing/break_timing.py
"""
import csv, glob, gzip, hashlib, json, math, os, statistics, sys
from collections import Counter, defaultdict

import numpy as np

B = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(os.path.dirname(B))
RULES_SHA = json.load(open(os.path.join(B, "BREAK_TIMING_RULES_SHA256.json"), encoding="utf-8"))["SHA256"]
assert hashlib.sha256(open(os.path.join(B, "BREAK_TIMING_RULES.md"), "rb").read()).hexdigest() == RULES_SHA, "rules changed"


def P(*a):
    return os.path.join(W, *a)


def rcsv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    if x is None or x in ("", "NA", "None", "nan"):
        return None
    return float(x)


def boolv(x):
    return x is True or x in ("True", "true", "1", 1)


INPUTS = []


def used(path):
    INPUTS.append(os.path.relpath(path, W).replace("\\", "/"))
    return path


# ---------------------------------------------------------------- arithmetic
def arith_actual(rows):
    """rows: dicts with k, first_error_region, hold_offset. Uses k >= 256."""
    big = [r for r in rows if int(r["k"]) >= 256]
    hold = [int(float(r["hold_offset"])) for r in big if r.get("first_error_region") == "hold"]
    outside = sum(1 for r in big if r.get("first_error_region") not in (None, "", "NA", "hold"))
    d = {"n_problems_k256": len(big), "n_broken_problems": len(hold), "n_first_error_outside_hold": outside}
    if hold:
        d.update(actual_median=statistics.median(hold), actual_min=min(hold), actual_max=max(hold), actual_broken=True)
    else:
        d.update(actual_median=None, actual_min=None, actual_max=None, actual_broken=False)
    return d


def arithmetic_units():
    units, checks = [], {}
    # V054H predictions + trajectory verification
    fp = rcsv(used(P("v054h_dynamics", "fixed_points.csv")))
    traj = defaultdict(list)
    for r in rcsv(used(P("v054h_dynamics", "trajectory.csv"))):
        traj[(r["seed"], r["op"], r["state"], r["pair"])].append((int(r["t"]), float(r["margin"])))
    pred, verify = {}, {"checked": 0, "match": 0, "beyond_4095": 0, "mismatch": []}
    for r in fp:
        key = (r["seed"], r["op"], r["state"], r["pair"])
        p = num(r["first_negative_margin_t"])
        pred[key] = (None if p is None else int(p), num(r["margin_at_end"]), r.get("D1_holds"))
        neg = [t for t, m in sorted(traj[key]) if m < 0]
        tmax = max(t for t, _ in traj[key])
        if p is None or p <= tmax:
            verify["checked"] += 1
            ok = (neg[0] if neg else None) == (None if p is None else int(p))
            verify["match"] += ok
            if not ok:
                verify["mismatch"].append(key)
        else:
            verify["beyond_4095"] += 1
    checks["V054H_trajectory_verification"] = verify
    # V054T actual
    v054t = defaultdict(list)
    for r in rcsv(used(P("v054t_transition", "results_per_problem.csv"))):
        v054t[(r["seed"], r["op"], r["state"], r["pair"])].append(r)
    for key, rows in sorted(v054t.items()):
        p = pred.get(key)
        u = dict(experiment="V054T+V054H", task="arithmetic", model="V054_B0", seed=key[0], structure="residual_GRU_adder",
                 neutral="hold", unit=f"{key[1]} c={key[2]} pair={key[3]}", divergent=False,
                 pred_status="missing" if p is None else ("step" if p[0] is not None else "none"),
                 predicted=None if p is None else p[0], endpoint_margin=None if p is None else p[1],
                 prereg_D1_holds=None if p is None else p[2], **arith_actual(rows))
        units.append(u)
    # V054R raw (no margin trajectory -> prediction missing)
    for f in sorted(glob.glob(P("v054r_replication", "data", "raw", "*.json"))):
        d = json.load(open(used(f), encoding="utf-8"))
        g = defaultdict(list)
        for r in d["durability"]:
            m = r["model"] + (f"_c{r['c']}" if "c" in r else "")
            g[(m, str(r["seed"]), r["op"], str(r["state"]), r["pair"])].append(r)
        for key, rows in sorted(g.items()):
            units.append(dict(experiment="V054R", task="arithmetic", model=key[0], seed=key[1], structure="residual_GRU_adder",
                              neutral="hold", unit=f"{key[2]} c={key[3]} pair={key[4]}", divergent=False,
                              pred_status="missing", predicted=None, endpoint_margin=None, prereg_D1_holds=None,
                              **arith_actual(rows)))
    # D10 / D11 (endpoint margin only -> prediction missing)
    em10 = {(r["seed"], r["state"], r["pair"], r["kind"]): num(r["h_star_margin_correct"])
            for r in rcsv(used(P("decimal_d10", "fixed_points.csv")))}
    g = defaultdict(list)
    for r in rcsv(used(P("decimal_d10", "data", "probe_results_per_problem.csv"))):
        g[(r["seed"], r["state"], r["pair"], r["kind"])].append(r)
    for key, rows in sorted(g.items()):
        units.append(dict(experiment="D10", task="arithmetic", model="D10", seed=key[0], structure="decimal_residual_GRU",
                          neutral="hold", unit=f"c={key[1]} pair={key[2]} kind={key[3]}", divergent=False,
                          pred_status="missing", predicted=None, endpoint_margin=em10.get(key), prereg_D1_holds=None,
                          **arith_actual(rows)))
    em11 = {(r["group"], r["seed"], r["state"], r["pair"], r["kind"]): num(r["h_star_margin_correct"])
            for r in rcsv(used(P("IronMath_D11", "fixed_points.csv")))}
    g = defaultdict(list)
    for r in rcsv(used(P("IronMath_D11", "probe_results_per_problem.csv"))):
        g[(r["group"], r["seed"], r["state"], r["pair"], r["kind"])].append(r)
    for key, rows in sorted(g.items()):
        units.append(dict(experiment="D11", task="arithmetic", model=f"D11_{key[0]}", seed=key[1], structure="decimal_residual_GRU",
                          neutral="hold", unit=f"c={key[2]} pair={key[3]} kind={key[4]}", divergent=False,
                          pred_status="missing", predicted=None, endpoint_margin=em11.get(key), prereg_D1_holds=None,
                          **arith_actual(rows)))
    return units, checks


# ---------------------------------------------------------------- memory
def unit_predictions(diag):
    out = {}
    g = defaultdict(list)
    for r in diag:
        g[(r["model"], r["neutral"], int(r["value"]))].append(r)
    for key, rr in g.items():
        bad = [r for r in rr if not boolv(r["retention_prediction"])]
        steps = [int(num(r["first_negative_margin_step"])) for r in bad if num(r["first_negative_margin_step"]) is not None]
        neg_but_pred_hold = sum(1 for r in rr if boolv(r["retention_prediction"]) and num(r["first_negative_margin_step"]) is not None)
        out[key] = dict(predicted=min(steps) if steps else None, pred_status="step" if steps else "none",
                        pred_fail_without_negative_margin=bool(bad) and not steps,
                        prefixes_failing=len(bad), prefixes_total=len(rr), negative_margin_but_predicted_hold=neg_but_pred_hold)
    return out


def labels(a):
    writes = a < 2
    idx = np.maximum.accumulate(np.where(writes, np.arange(a.shape[1])[None, :], -1), axis=1)
    picked = np.take_along_axis(a, np.maximum(idx, 0), axis=1)
    return np.where(idx < 0, 0, picked).astype(np.uint8)


def g1_first_retention(models, checks):
    """From saved predictions (no inference): first retention-R error position for long_R at T=65000."""
    T = 65000
    with np.load(used(P("IronMath_G1", "main_long_data", "T65000.npz"))) as z:
        x = z["symbols"]
    meta = json.load(open(used(P("IronMath_G1", "main_long_data", "T65000_metadata.json")), encoding="utf-8"))
    y = labels(x)
    mask = x == 3
    ret = np.arange(T) >= 9
    res, mism = {}, []
    for m in models:
        rec = json.load(open(P("IronMath_G1", "evaluation_records", f"{m}.json"), encoding="utf-8"))
        items = {(it["T"], it["id"]): it for it in rec["items"]}
        path = P("IronMath_G1", "saved_predictions", f"{m}_T65000.npz")
        it0 = items[(T, meta[0]["id"])]
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == it0["prediction_SHA256"], m
        with np.load(path) as z:
            pred = np.unpackbits(z["bits"], axis=1, bitorder="little")[:, :T]
            inv = np.unpackbits(z["invalid"], axis=1, bitorder="little")[:, :T].astype(bool)
        for j, md in enumerate(meta):
            if md["category"] != "long_R":
                continue
            wrong = mask[j] & ((pred[j] != y[j]) | inv[j])
            ret_correct = int((mask[j] & ret & ~wrong).sum())
            it = items[(T, md["id"])]
            if ret_correct != it["retention_R_correct"]:
                mism.append((m, md["id"]))
            f = np.flatnonzero(wrong & ret)
            res[(m, md["value"], md["j"])] = int(f[0]) if len(f) else None
    checks["G1_retention_recount"] = {"models": len(models), "items": len(res), "mismatch_vs_evaluation_records": mism}
    return res


def memory_units(checks):
    units, items_out = [], []
    sources = {
        "FF1": dict(diag=json.load(open(used(P("IronMath_FF1", "diagnostic_details.json")), encoding="utf-8")),
                    long=rcsv(used(P("IronMath_FF1", "long_item_results.csv")))),
        "FF1-R": dict(diag=rcsv(used(P("IronMath_FF1R", "ff1r_diagnostic_details.csv"))),
                      long=rcsv(used(P("IronMath_FF1R", "long_item_results.csv")))),
        "G1": dict(diag=rcsv(used(P("IronMath_G1", "g1_diagnostic_details.csv"))), long=None),
    }
    g1res = {r["model"]: r for r in rcsv(used(P("IronMath_G1", "g1_results.csv")))}
    g1_models = sorted(g1res)
    g1_first = g1_first_retention(g1_models, checks)
    g1_long = []
    for m in g1_models:
        rec = json.load(open(used(P("IronMath_G1", "evaluation_records", f"{m}.json")), encoding="utf-8"))
        for it in rec["items"]:
            if it["category"] in ("long_R", "long_N"):
                g1_long.append(dict(model=m, T=it["T"], category=it["category"], value=it["value"], j=it["j"],
                                    retention_R_correct=it["retention_R_correct"], retention_R_count=it["retention_R_count"],
                                    first_retention_R_error_position=None, nonfinite_positions=it["nonfinite_positions"]))
    sources["G1"]["long"] = g1_long
    # FF1 consistency check of our N/R reconstruction against the original break_step_analysis.csv
    orig_bsa = {}
    for ex, f in (("FF1", P("IronMath_FF1", "break_step_analysis.csv")), ("FF1-R", P("IronMath_FF1R", "break_step_analysis.csv"))):
        for r in rcsv(used(f)):
            orig_bsa[(ex, r["model"], r["neutral"], int(r["value"]), int(r["j"]))] = r
    cmp_bad = []
    for ex, s in sources.items():
        preds = unit_predictions(s["diag"])
        long = s["long"]
        div = defaultdict(bool)
        info = {}
        for r in long:
            if num(r["nonfinite_positions"]) and num(r["nonfinite_positions"]) > 0:
                div[r["model"]] = True
        if ex in ("FF1", "FF1-R"):
            for r in long:
                info[r["model"]] = dict(structure=f"H{r['H']}_L{r['L']}", seed=r["seed"])
            # divergent: any nonfinite position in any long item (all categories)
            for r in long:
                if int(r["nonfinite_positions"] or 0) > 0:
                    div[r["model"]] = True
        else:
            for m, r in g1res.items():
                info[m] = dict(structure=r["architecture"], seed=r["seed"])
                div[m] = int(r["nonfinite_logit_positions"] or 0) > 0
        by = defaultdict(list)
        for r in long:
            if r["category"] in ("long_R", "long_N"):
                by[(r["model"], r["category"][-1], int(num(r["value"])))].append(r)
        for (m, neutral, v), rows in sorted(by.items()):
            p = preds[(m, neutral, v)]
            u = dict(experiment=ex, task="memory", model=m, seed=info[m]["seed"], structure=info[m]["structure"], neutral=neutral,
                     unit=f"value={v} neutral={neutral}", divergent=div[m], pred_status=p["pred_status"], predicted=p["predicted"],
                     pred_fail_without_negative_margin=p["pred_fail_without_negative_margin"],
                     negative_margin_but_predicted_hold=p["negative_margin_but_predicted_hold"],
                     prefixes_failing=p["prefixes_failing"], endpoint_margin=None, prereg_D1_holds=None)
            jitems = []
            for j in range(8):
                its = sorted([r for r in rows if int(num(r["j"])) == j], key=lambda r: int(num(r["T"])))
                ok = [int(num(r["retention_R_correct"])) == int(num(r["retention_R_count"])) for r in its]
                nonmono = any(not ok[i] and any(ok[i + 1:]) for i in range(len(ok) - 1))
                right = all(ok)
                it = dict(experiment=ex, model=m, neutral=neutral, value=v, j=j, lengths=len(its), all_lengths_correct=right,
                          nonmonotonic=nonmono, observed_step=None, interval_lower_open=None, interval_upper_closed=None,
                          left_censored=False, predicted=p["predicted"], prediction_inside=None)
                if neutral == "R":
                    if ex == "G1":
                        pos = g1_first[(m, v, j)]
                    else:
                        pos = num(its[-1]["first_retention_R_error_position"])
                    it["observed_step"] = None if pos is None else int(pos) - 8
                elif not right and not nonmono:
                    fb = ok.index(False)
                    it["interval_upper_closed"] = int(num(its[fb]["T"])) - 10
                    it["interval_lower_open"] = int(num(its[fb - 1]["T"])) - 10 if fb else None
                    it["left_censored"] = fb == 0
                    if p["predicted"] is not None:
                        lo, up = it["interval_lower_open"], it["interval_upper_closed"]
                        it["prediction_inside"] = (lo is None or p["predicted"] > lo) and p["predicted"] <= up
                if ex in ("FF1", "FF1-R"):
                    o = orig_bsa.get((ex, m, neutral, v, j))
                    if o is not None:
                        same = (num(o["observed_R_first_failure_step"]) == it["observed_step"] and
                                num(o["N_upper_closed"]) == it["interval_upper_closed"] and
                                num(o["N_lower_open"]) == it["interval_lower_open"] and
                                num(o["predicted_break_step"]) == p["predicted"])
                        if not same:
                            cmp_bad.append((ex, m, neutral, v, j))
                jitems.append(it)
            items_out.extend(jitems)
            if neutral == "R":
                obs = [i["observed_step"] for i in jitems if i["observed_step"] is not None]
                u.update(n_items=8, n_broken_items=len(obs),
                         actual_median=statistics.median(obs) if obs else None, actual_min=min(obs) if obs else None,
                         actual_max=max(obs) if obs else None, actual_broken=bool(obs))
            else:
                broken = [i for i in jitems if not i["all_lengths_correct"]]
                ints = [i for i in jitems if i["interval_upper_closed"] is not None]
                u.update(n_items=8, n_broken_items=len(broken), n_nonmonotonic_items=sum(i["nonmonotonic"] for i in jitems),
                         n_interval_items=len(ints), n_left_censored=sum(i["left_censored"] for i in ints),
                         interval_hits=sum(1 for i in ints if i["prediction_inside"]) if p["predicted"] is not None else None,
                         intervals=";".join(f"({i['interval_lower_open']},{i['interval_upper_closed']}]" for i in ints),
                         actual_median=None, actual_min=None, actual_max=None, actual_broken=bool(broken))
            units.append(u)
    checks["FF1_FF1R_reconstruction_vs_break_step_analysis_mismatch"] = cmp_bad
    return units, items_out


# ---------------------------------------------------------------- classification & summary
def classify(u):
    if u["pred_status"] == "missing":
        return "予測欠測・壊れた" if u["actual_broken"] else "予測欠測・壊れず"
    pb = u["pred_status"] == "step"
    if pb and u["actual_broken"]:
        return "両方あり"
    if pb:
        return "予測あり・壊れず（打ち切り）"
    if u["actual_broken"]:
        return "予測なし・壊れた（未予測の失敗）"
    return "両方なし"


def spearman(a, b):
    if len(a) < 3:
        return None

    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            k = i
            while k + 1 < len(o) and v[o[k + 1]] == v[o[i]]:
                k += 1
            for t in range(i, k + 1):
                r[o[t]] = (i + k) / 2 + 1
            i = k + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    sa = math.sqrt(sum((x - ma) ** 2 for x in ra)); sb = math.sqrt(sum((x - mb) ** 2 for x in rb))
    if sa == 0 or sb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sa * sb)


def summarize(name, us, items):
    c = Counter(u["classification"] for u in us)
    pts = [u for u in us if u["classification"] == "両方あり" and u["actual_median"] is not None]
    s = dict(group=name, n_units=len(us), n_pred_missing=sum(u["pred_status"] == "missing" for u in us),
             both=c["両方あり"], pred_unbroken_censored=c["予測あり・壊れず（打ち切り）"],
             unpredicted_failure=c["予測なし・壊れた（未予測の失敗）"], neither=c["両方なし"],
             pred_missing_broken=c["予測欠測・壊れた"], pred_missing_unbroken=c["予測欠測・壊れず"],
             pred_fail_without_negative_margin=sum(bool(u.get("pred_fail_without_negative_margin")) for u in us),
             n_point_pairs=len(pts))
    if pts:
        err = [u["predicted"] - u["actual_median"] for u in pts]
        s.update(median_signed_error=statistics.median(err), median_abs_error=statistics.median([abs(e) for e in err]),
                 frac_in_actual_range=sum(u["actual_min"] <= u["predicted"] <= u["actual_max"] for u in pts) / len(pts),
                 frac_within_3_steps=sum(abs(e) <= 3 for e in err) / len(pts),
                 frac_within_10pct=sum(abs(u["predicted"] - u["actual_median"]) <= 0.1 * u["actual_median"] for u in pts) / len(pts),
                 spearman_pred_vs_actual_median=spearman([u["predicted"] for u in pts], [u["actual_median"] for u in pts]))
    nitems = [i for i in items if i["prediction_inside"] is not None]
    if nitems:
        s.update(n_interval_items=len(nitems), n_interval_items_inside=sum(i["prediction_inside"] for i in nitems),
                 frac_interval_inside=sum(i["prediction_inside"] for i in nitems) / len(nitems))
    nun = [u for u in us if u["neutral"] == "N" and u["pred_status"] == "step" and u.get("n_interval_items")]
    if nun:
        s["mean_unit_interval_hit_rate"] = statistics.mean(u["interval_hits"] / u["n_interval_items"] for u in nun)
    s["n_nonmonotonic_items"] = sum(i["nonmonotonic"] for i in items if i["neutral"] == "N")
    return s


def main():
    checks = {"rules_SHA256": RULES_SHA}
    au, ac = arithmetic_units()
    checks.update(ac)
    mu, items = memory_units(checks)
    units = au + mu
    for u in units:
        u["classification"] = classify(u)
    key_items = defaultdict(list)
    for i in items:
        key_items[(i["experiment"], i["model"], i["neutral"], i["value"])].append(i)

    def items_of(us):
        out = []
        for u in us:
            if u["task"] == "memory":
                v = int(u["unit"].split()[0].split("=")[1])
                out += key_items[(u["experiment"], u["model"], u["neutral"], v)]
        return out

    groups = []
    for div in (False, True):
        dl = "発散型" if div else "非発散"
        sel = [u for u in units if u["divergent"] == div]
        if not sel:
            continue
        for task in ("arithmetic", "memory"):
            t = [u for u in sel if u["task"] == task]
            if t:
                groups.append((f"{dl}|課題={task}", t))
        for ex in sorted({u["experiment"] for u in sel}):
            for nt in sorted({u["neutral"] for u in sel if u["experiment"] == ex}):
                t = [u for u in sel if u["experiment"] == ex and u["neutral"] == nt]
                groups.append((f"{dl}|実験={ex}|中立={nt}", t))
        for st in sorted({u["structure"] for u in sel if u["experiment"] == "G1"}):
            for nt in ("N", "R"):
                t = [u for u in sel if u["experiment"] == "G1" and u["structure"] == st and u["neutral"] == nt]
                if t:
                    groups.append((f"{dl}|G1構造={st}|中立={nt}", t))
        g1nl = [u for u in sel if u["experiment"] == "G1" and u["structure"] != "linear"]
        for nt in ("N", "R"):
            t = [u for u in g1nl if u["neutral"] == nt]
            if t:
                groups.append((f"{dl}|G1非線形4構造計|中立={nt}", t))
    summary = [summarize(n, us, items_of(us)) for n, us in groups]

    ucols = ["experiment", "task", "model", "seed", "structure", "neutral", "unit", "divergent", "pred_status", "predicted",
             "actual_median", "actual_min", "actual_max", "actual_broken", "classification", "n_problems_k256",
             "n_broken_problems", "n_first_error_outside_hold", "n_items", "n_broken_items", "n_nonmonotonic_items",
             "n_interval_items", "n_left_censored", "interval_hits", "intervals", "pred_fail_without_negative_margin",
             "negative_margin_but_predicted_hold", "prefixes_failing", "endpoint_margin", "prereg_D1_holds"]
    with open(os.path.join(B, "break_timing_units.csv"), "x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ucols, extrasaction="ignore"); w.writeheader(); w.writerows(units)
    with open(os.path.join(B, "break_timing_items.csv"), "x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0].keys())); w.writeheader(); w.writerows(items)
    scols = sorted({k for s in summary for k in s}, key=lambda k: list(summary[0].keys()).index(k) if k in summary[0] else 99)
    with open(os.path.join(B, "break_timing_summary.csv"), "x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scols); w.writeheader(); w.writerows(summary)

    # scatter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    jp = [n for n in ("Yu Gothic", "Meiryo", "MS Gothic") if n in have]
    plt.rcParams["font.family"] = jp + ["DejaVu Sans"]
    checks["plot_font"] = jp[:1] or ["DejaVu Sans (no Japanese font found)"]
    pts = [u for u in units if u["classification"] == "両方あり" and u["actual_median"] is not None]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    style = {("arithmetic", False): ("#1f77b4", "o", "算術 (V054T × V054H)"),
             ("memory", False): ("#d62728", "s", "記憶課題・R保持 (非発散)"),
             ("memory", True): ("#ff9896", "^", "記憶課題・R保持 (発散型)")}
    floor = 0.5
    for (task, div), (col, mk, lab) in style.items():
        sel = [u for u in pts if u["task"] == task and u["divergent"] == div]
        if not sel:
            continue
        xs = [max(u["predicted"], floor) for u in sel]; ys = [max(u["actual_median"], floor) for u in sel]
        lo = [max(u["actual_min"], floor) for u in sel]; hi = [max(u["actual_max"], floor) for u in sel]
        ax.vlines(xs, lo, hi, colors=col, alpha=0.35, linewidth=1)
        ax.scatter(xs, ys, c=col, marker=mk, s=28, label=f"{lab} n={len(sel)}", edgecolors="black", linewidths=0.3, zorder=3)
    allv = [max(v, floor) for u in pts for v in (u["predicted"], u["actual_min"], u["actual_max"])]
    lim = (min(allv) * 0.7, max(allv) * 1.5)
    ax.plot(lim, lim, "k--", linewidth=1, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("予測した破綻ステップ（マージンが初めて負）")
    ax.set_ylabel("実際に壊れたステップ（中央値。縦線は最小〜最大）")
    ax.set_title("壊れる時期：予測と実際（事後の再集計）")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    png = os.path.join(B, "break_timing_scatter.png")
    assert not os.path.exists(png)
    fig.savefig(png, dpi=150)
    checks["values_below_floor_in_plot"] = sum(1 for u in pts for v in (u["predicted"], u["actual_min"]) if v < floor)
    checks["inputs_read"] = sorted(set(INPUTS))
    with open(os.path.join(B, "break_timing_checks.json"), "x", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in checks.items() if k != "inputs_read"}, ensure_ascii=False, default=str)[:3000])
    print("units", len(units), Counter((u["experiment"], u["classification"]) for u in units))


if __name__ == "__main__":
    main()
