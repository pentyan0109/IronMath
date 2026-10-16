"""Stage 2 aggregation: compare FP64 diagnostic records with the original G1 FP32 values (RULES_FP64.md)."""
import csv, hashlib, json
from collections import Counter
from pathlib import Path

S = Path(__file__).resolve().parent
G = S.parent.parent.parent / "IronMath_G1"
FN = ["lstm_L8_s28", "lstm_L8_s32", "lstm_L8_s33", "lstm_L8_s38"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


frozen = json.loads((S / "RULES_SHA256.json").read_text(encoding="utf-8"))["SHA256"]
assert all(sha(S / f) == h for f, h in frozen.items())
pf = G / "main_PREDICTIONS_FROZEN.json"
assert sha(pf) == json.loads((G / "main_PREDICTIONS_SHA256.json").read_text(encoding="utf-8"))["SHA256"]
A32 = {m["model"]: m["A"] for m in json.loads(pf.read_text(encoding="utf-8"))["models"]}
D32 = {m["model"]: m["D"] for m in json.loads(pf.read_text(encoding="utf-8"))["models"]}
res = {r["model"]: r for r in csv.DictReader(open(G / "g1_results.csv", encoding="utf-8-sig"))}
d32 = {}
for r in csv.DictReader(open(G / "g1_diagnostic_details.csv", encoding="utf-8-sig")):
    d32[(r["model"], r["neutral"], int(r["prefix"]), int(r["value"]))] = r
recs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((S / "records").glob("*.json"))}
assert len(recs) == 200, len(recs)


def f(x):
    return None if x in ("", None, "None") else float(x)


rows, f3_diff, changes = [], [], []
for mid, rec in recs.items():
    actual = res[mid]["actual_success"] == "True"
    arch = rec["architecture"]
    if A32[mid] != rec["A"] or D32[mid] != rec["D"]:
        changes.append(dict(model=mid, architecture=arch, actual_success=actual, A_FP32=A32[mid], A_FP64=rec["A"],
                            D_FP32=D32[mid], D_FP64=rec["D"], all16_converged_FP64=rec["all16_converged"]))
    for t in rec["trajectories"]:
        o = d32[(mid, t["neutral"], t["prefix"], t["value"])]
        conv32 = o["converged"] == "True"
        ro32 = None if o["readout"] in ("", "None") else int(float(o["readout"]))
        rows.append(dict(model=mid, architecture=arch, neutral=t["neutral"], prefix=t["prefix"], value=t["value"],
                         converged_FP32=conv32, converged_FP64=t["converged"],
                         last100_max_movement_FP32=f(o["last100_max_movement"]), last100_max_movement_FP64=t["last100_max_movement"],
                         spectral_radius_FP32=f(o["spectral_radius"]), spectral_radius_FP64=t["spectral_radius"],
                         readout_FP32=ro32, readout_FP64=t["readout"],
                         readout_correct_FP32=ro32 == t["value"], readout_correct_FP64=t["readout"] == t["value"]))
        if conv32 and (ro32 == t["value"]) != (t["readout"] == t["value"]):
            f3_diff.append(rows[-1])

nl = [m for m in recs if recs[m]["architecture"] != "linear"]
li = [m for m in recs if recs[m]["architecture"] == "linear"]
hits = lambda ms, A: sum(A[m] == (res[m]["actual_success"] == "True") for m in ms)
A64 = {m: recs[m]["A"] for m in recs}
D64 = {m: recs[m]["D"] for m in recs}
h32 = dict(nonlinear=hits(nl, A32), linear=hits(li, A32)); h64 = dict(nonlinear=hits(nl, A64), linear=hits(li, A64))
fn = []
for m in FN:
    r = recs[m]
    fn.append(dict(model=m, actual_success=res[m]["actual_success"] == "True", A_FP32=A32[m], A_FP64=r["A"],
                   all16_converged_FP64=r["all16_converged"],
                   converged_FP32=sum(x["converged_FP32"] for x in rows if x["model"] == m),
                   converged_FP64=sum(x["converged_FP64"] for x in rows if x["model"] == m),
                   max_last100_FP32=max(x["last100_max_movement_FP32"] for x in rows if x["model"] == m),
                   max_last100_FP64=max(x["last100_max_movement_FP64"] for x in rows if x["model"] == m),
                   max_rho_FP32=max(x["spectral_radius_FP32"] for x in rows if x["model"] == m and x["spectral_radius_FP32"] is not None),
                   max_rho_FP64=max(x["spectral_radius_FP64"] for x in rows if x["model"] == m and x["spectral_radius_FP64"] is not None),
                   readout_correct_FP64=sum(x["readout_correct_FP64"] for x in rows if x["model"] == m)))
F1n = sum(x["all16_converged_FP64"] and x["A_FP64"] for x in fn)
F2d = (h64["nonlinear"] + h64["linear"]) - (h32["nonlinear"] + h32["linear"])
conv32_n = sum(x["converged_FP32"] for x in rows)
summary = dict(
    F1=dict(models_all16_converged_and_A_success=F1n, verdict="支持" if F1n >= 3 else "不支持"),
    F2=dict(hits_FP32=h32, hits_FP64=h64, change_total=F2d, verdict="支持" if abs(F2d) <= 5 else "不支持"),
    F3=dict(trajectories_converged_FP32=conv32_n, readout_correctness_changed=len(f3_diff), verdict="支持" if not f3_diff else "不支持"),
    convergence=dict(FP32=conv32_n, FP64=sum(x["converged_FP64"] for x in rows), both=sum(x["converged_FP32"] and x["converged_FP64"] for x in rows),
                     only_FP32=sum(x["converged_FP32"] and not x["converged_FP64"] for x in rows),
                     only_FP64=sum(x["converged_FP64"] and not x["converged_FP32"] for x in rows)),
    A_changes=len(changes), A_positive_FP64_nonlinear=sum(A64[m] for m in nl),
    D_hits_FP64_nonlinear=hits(nl, D64), D_hits_FP32_nonlinear=hits(nl, D32),
    by_architecture={a: dict(n=sum(recs[m]["architecture"] == a for m in recs),
                             A_hits_FP32=hits([m for m in recs if recs[m]["architecture"] == a], A32),
                             A_hits_FP64=hits([m for m in recs if recs[m]["architecture"] == a], A64),
                             converged_traj_FP32=sum(x["converged_FP32"] for x in rows if x["architecture"] == a),
                             converged_traj_FP64=sum(x["converged_FP64"] for x in rows if x["architecture"] == a))
                     for a in ("residual_gru", "gru", "lstm", "rnn", "linear")},
    wall_seconds_total=sum(r["wall_seconds"] for r in recs.values()))


def wcsv(name, rs):
    if not rs:
        (S / name).open("x", encoding="utf-8-sig").write("none\n"); return
    with (S / name).open("x", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rs[0].keys())); w.writeheader(); w.writerows(rs)


wcsv("fp64_trajectories.csv", rows)
wcsv("fp64_false_negative_models.csv", fn)
wcsv("fp64_A_changes.csv", changes)
wcsv("fp64_F3_differences.csv", f3_diff)
with (S / "fp64_summary.json").open("x", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
print(json.dumps(fn, ensure_ascii=False, indent=1))
print(json.dumps(changes, ensure_ascii=False, indent=1))
print("F3 diffs", [(x["model"], x["neutral"], x["prefix"], x["value"]) for x in f3_diff][:20])
