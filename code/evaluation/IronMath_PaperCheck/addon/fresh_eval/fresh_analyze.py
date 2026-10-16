"""Stage 3 aggregation: actual success on the fresh set, E1-E5, B_cum, McNemar, bootstrap (RULES_FRESH_EVAL.md)."""
import csv, hashlib, json, math
from pathlib import Path
import numpy as np

S = Path(__file__).resolve().parent
G = S.parent.parent.parent / "IronMath_G1"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


frozen = json.loads((S / "RULES_SHA256.json").read_text(encoding="utf-8"))["SHA256"]
assert all(sha(S / f) == h for f, h in frozen.items())
pf = G / "main_PREDICTIONS_FROZEN.json"
assert sha(pf) == json.loads((G / "main_PREDICTIONS_SHA256.json").read_text(encoding="utf-8"))["SHA256"]
pred = {m["model"]: m for m in json.loads(pf.read_text(encoding="utf-8"))["models"]}
orig = {r["model"]: r for r in csv.DictReader(open(G / "g1_results.csv", encoding="utf-8-sig"))}
fz_sha = sha(S / "FRESH_DATA_FROZEN.json")
recs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((S / "evaluation_records").glob("*.json"))}
assert len(recs) == 200 and all(r["fresh_data_frozen_SHA256"] == fz_sha for r in recs.values())
LS = [32, 64, 128, 512]

models = []
for mid, r in recs.items():
    by_T = {}
    for it in r["items"]:
        by_T.setdefault(it["T"], []).append(it)
    assert all(len(by_T[T]) == 68 for T in by_T) and len(by_T) == 7
    ok = {T: all(it["R_correct"] == it["R_count"] for it in by_T[T]) for T in by_T}
    actual = ok[2048] and ok[8192] and ok[65000]
    nonfinite = sum(it["nonfinite_positions"] for it in r["items"])
    row = dict(model=mid, architecture=r["architecture"], L=r["L"], seed=r["seed"], actual_success_fresh=actual,
               actual_success_original=orig[mid]["actual_success"] == "True", A=pred[mid]["A"], D=pred[mid]["D"],
               nonfinite_positions_fresh=nonfinite,
               classification_fresh="発散型" if nonfinite else ("成功型" if actual else "有限誤答型"),
               classification_original=orig[mid]["classification"])
    for l in LS:
        row[f"B_cum_{l}"] = all(ok[T] for T in LS if T <= l)
    models.append(row)
nl = [m for m in models if m["architecture"] != "linear"]
li = [m for m in models if m["architecture"] == "linear"]


def hits(ms, k):
    return sum(m[k] == m["actual_success_fresh"] for m in ms)


def mcnemar(ms, k1, k2):
    b = sum(m[k1] == m["actual_success_fresh"] and m[k2] != m["actual_success_fresh"] for m in ms)
    c = sum(m[k2] == m["actual_success_fresh"] and m[k1] != m["actual_success_fresh"] for m in ms)
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return dict(only_first_correct=b, only_second_correct=c, p=p)


rng = np.random.default_rng(86000)
boot = {}
for k2 in ("D", "B_cum_512", "B_cum_128"):
    a = np.array([m["A"] == m["actual_success_fresh"] for m in nl], float)
    o = np.array([m[k2] == m["actual_success_fresh"] for m in nl], float)
    idx = rng.integers(0, len(nl), size=(10000, len(nl)))
    diffs = (a[idx] - o[idx]).mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    boot[f"A_minus_{k2}"] = dict(point=float((a - o).mean()), ci_low=float(lo), ci_high=float(hi))

succ_new = sum(m["actual_success_fresh"] for m in models)
E3 = mcnemar(nl, "A", "D"); E4 = mcnemar(nl, "A", "B_cum_512")
summary = dict(
    E1=dict(success_fresh_total=succ_new, success_fresh_nonlinear=sum(m["actual_success_fresh"] for m in nl),
            success_fresh_linear=sum(m["actual_success_fresh"] for m in li), success_original_total=76,
            verdict="支持" if abs(succ_new - 76) <= 5 else "不支持"),
    E2=dict(A_hits_nonlinear=hits(nl, "A"), verdict="支持" if 133 <= hits(nl, "A") <= 147 else "不支持"),
    E3=dict(**E3, verdict="支持" if E3["p"] < 0.05 and E3["only_first_correct"] > E3["only_second_correct"] else "不支持"),
    E4=dict(**E4, verdict="支持" if E4["p"] >= 0.05 else "不支持"),
    E5=dict(linear_success=sum(m["actual_success_fresh"] for m in li), verdict="支持" if not any(m["actual_success_fresh"] for m in li) else "不支持"),
    hits_nonlinear={k: hits(nl, k) for k in ["A", "D"] + [f"B_cum_{l}" for l in LS]},
    hits_linear={k: hits(li, k) for k in ["A", "D"] + [f"B_cum_{l}" for l in LS]},
    mcnemar_A_vs_B_cum_128=mcnemar(nl, "A", "B_cum_128"),
    bootstrap_nonlinear=boot,
    by_architecture={a: dict(n=len(ms), success_fresh=sum(m["actual_success_fresh"] for m in ms),
                             success_original=sum(m["actual_success_original"] for m in ms),
                             **{f"hits_{k}": hits(ms, k) for k in ["A", "D"] + [f"B_cum_{l}" for l in LS]},
                             divergent_fresh=sum(m["classification_fresh"] == "発散型" for m in ms))
                     for a in ("residual_gru", "gru", "lstm", "rnn", "linear")
                     for ms in [[m for m in models if m["architecture"] == a]]},
    changed_models=sum(m["actual_success_fresh"] != m["actual_success_original"] for m in models),
    eval_wall_seconds_total=sum(r["wall_seconds"] for r in recs.values()))
changed = [m for m in models if m["actual_success_fresh"] != m["actual_success_original"]]
with (S / "fresh_models.csv").open("x", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(models[0].keys())); w.writeheader(); w.writerows(models)
with (S / "fresh_changed_models.csv").open("x", encoding="utf-8-sig", newline="") as fh:
    if changed:
        w = csv.DictWriter(fh, fieldnames=list(changed[0].keys())); w.writeheader(); w.writerows(changed)
    else:
        fh.write("none\n")
with (S / "fresh_summary.json").open("x", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
print("CHANGED", [(m["model"], m["actual_success_original"], m["actual_success_fresh"], m["A"], m["classification_fresh"]) for m in changed])
