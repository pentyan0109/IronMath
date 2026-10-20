# R0 audit (2026-10-20): independent re-count from saved CSVs only; no model is loaded or run.
# Usage: python -I recount_R0.py <extracted archive_v074/results>
import csv, sys, statistics as st, json, math
R = sys.argv[1]
def rd(p):
    with open(f"{R}/{p}", encoding="utf-8-sig") as f: return list(csv.DictReader(f))
T = lambda v: str(v).strip().lower() == "true"
# T1 endurance
s = rd("v054e_endurance/summary.csv")
print("== T1 endurance (main, ALL)")
for w in sorted({int(r["width"]) for r in s if r["set_type"]=="main"}):
    rows = sorted([r for r in s if r["set_type"]=="main" and int(r["width"])==w and r["category"]=="ALL" and r["seed"] not in ("","ALL") and r["exact"]!=""], key=lambda r:r["seed"])
    ex=[int(r["exact"]) for r in rows]; rates=[e/512*100 for e in ex]
    if len(ex)==3: print(w, ex, f"{st.mean(rates):.2f} ± {st.stdev(rates):.2f}")
c = rd("v054e_endurance/confirmation.csv")[0]; rc=[float(c[k])*100 for k in ("seed0","seed1","seed2")]
print("confirmation", [round(x*512/100) for x in rc], f"{st.mean(rc):.2f} ± {st.stdev(rc):.2f}")
# G1
g = rd("IronMath_G1/g1_results.csv"); nl=[r for r in g if r["architecture"]!="linear"]
print("== G1 nonlinear n", len(nl), "success", sum(T(r["actual_success"]) for r in nl))
for k in ("A","D","B"):
    print(k, sum(T(r[k])==T(r["actual_success"]) for r in nl))
cmaj = sum(not T(r["actual_success"]) for r in nl); print("C(majority=failure)", cmaj)
a_only=sum(T(r["A"])==T(r["actual_success"]) and T(r["D"])!=T(r["actual_success"]) for r in nl)
d_only=sum(T(r["D"])==T(r["actual_success"]) and T(r["A"])!=T(r["actual_success"]) for r in nl)
print("A-only", a_only, "D-only", d_only)
print("A FP", sum(T(r["A"]) and not T(r["actual_success"]) for r in nl), "FN", sum((not T(r["A"])) and T(r["actual_success"]) for r in nl))
lin=[r for r in g if r["architecture"]=="linear"]; print("linear success", sum(T(r["actual_success"]) for r in lin), "A correct", sum(T(r["A"])==T(r["actual_success"]) for r in lin))
for arch in sorted({r["architecture"] for r in g}):
    rr=[r for r in g if r["architecture"]==arch]
    print(" ", arch, "succ", sum(T(r["actual_success"]) for r in rr), "A", sum(T(r["A"])==T(r["actual_success"]) for r in rr), "D", sum(T(r["D"])==T(r["actual_success"]) for r in rr))
# ABL1
ab = rd("IronMath_PaperCheck/addon/abl1/abl1_model_results.csv"); abn=[r for r in ab if r["architecture"]!="linear"]
print("== ABL1 nonlinear", len(abn))
for k in ("A0","A1","A2","A3","A4","A5"):
    ok=sum(T(r[k])==T(r["actual_success"]) for r in abn); fp=sum(T(r[k]) and not T(r["actual_success"]) for r in abn); fn=sum((not T(r[k])) and T(r["actual_success"]) for r in abn)
    print(k, ok, "FP", fp, "FN", fn)
# A0 vs A1 discordant
a0w_a1r=[r["model"] for r in abn if T(r["A0"])!=T(r["actual_success"]) and T(r["A1"])==T(r["actual_success"])]
a1w_a0r=[r["model"] for r in abn if T(r["A1"])!=T(r["actual_success"]) and T(r["A0"])==T(r["actual_success"])]
print("A0 wrong & A1 right:", a0w_a1r); print("A1 wrong & A0 right:", a1w_a0r)
print("McNemar exact two-sided p (5 vs 0):", min(1, 2*0.5**5))
# A1 vs Bcum512 per model: need per-model Bcum; check r1_baselines
b = rd("IronMath_PaperCheck/r1_baselines.csv")
for r in b:
    if r["experiment"]=="G1" and r["method"]=="B_cum" and r["structure"] in ("nonlinear","all_nonlinear","nonlinear_total","all"):
        print("Bcum", r["structure"], r["ell"], r["hits"], "FP", r["FP"], "FN", r["FN"])
# A1 false positives categories
fp=[r for r in abn if T(r["A1"]) and not T(r["actual_success"])]
print("A1 FP categories:", [r["failed_categories_main"] for r in fp])
# V054H D2 / endpoints
d2 = rd("IronMath_PaperCheck/extension/v072_audit/V054H_D2_INDEPENDENT.csv")
print("== D2", sum(T(r["matching_D2"]) for r in d2), "/", len(d2), "failures", sum(T(r["observed_hold_failure"]) for r in d2))
ep = rd("IronMath_PaperCheck/extension/v072_audit/V054H_OLD_ENDPOINT_PAIRS.csv")
print("shared pairs", sum(T(r["shared_under_E3_distance_threshold"]) for r in ep), "/", len(ep))
print("distances", sorted(float(r["distance_hstar_state0_vs_state1"]) for r in ep))
# E3
x = rd("IronMath_PaperCheck/extension/v072_audit/E3_READOUT_D2_CROSSTAB.csv")
for r in x: print("E3", r)
# E2 per context
e2 = rd("IronMath_PaperCheck/extension/v073_aggregation/E2_PER_CONTEXT.csv")
for donor in ("correct","failed"):
    rr=[r for r in e2 if r["donor"]==donor]
    d=[float(r["normal_pos_minus_controls_mean"]) for r in rr]
    print("E2", donor, "n", len(rr), "mean adv", round(st.mean(d),3), "gt_all6", sum(T(r["normal_pos_gt_all6"]) for r in rr),
      "median dmargin", round(st.median(float(r["normal_pos_dmargin_step8"]) for r in rr),1),
      "median ratio after1", round(st.median(float(r["normal_pos_dmargin_after_1"])/float(r["normal_pos_dmargin_step8"]) for r in rr),3),
      "near0 after64(<1)", sum(abs(float(r["normal_pos_dmargin_after_64"]))<1 for r in rr),
      "ortho delays range", min(min(float(r[f"orthogonal{i}_delay"]) for i in range(3)) for r in rr), max(max(float(r[f"orthogonal{i}_delay"]) for i in range(3)) for r in rr))
