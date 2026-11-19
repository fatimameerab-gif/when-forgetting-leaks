import argparse, csv, glob, json, os, re
import numpy as np
from scipy.stats import ks_2samp, beta
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="../results"); ap.add_argument("--out", default="../figures")
args = ap.parse_args(); R, OUT = args.results, args.out
J = lambda p: json.load(open(p))
auc = lambda m, n: float(roc_auc_score(np.r_[np.ones(len(m)), np.zeros(len(n))], np.r_[m, n]))
def pct(m, n):
    n = np.sort(n); lo = np.searchsorted(n, m, "left"); hi = np.searchsorted(n, m, "right"); return (lo + .5 * (hi - lo)) / len(n)
def cp(k, n): return [k, n, 0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(.975, k + 1, n - k))]
KEY = {}
rows = list(csv.DictReader(open(f"{OUT}/v7_checkpoints.csv")))
for r in rows:
    for k in ("fq", "pl", "pq", "util"):
        r[k] = float(r[k])
RAW, ORAW = {}, {}
def raw(r):
    k = (r["sub"], r["name"])
    if k not in RAW: RAW[k] = J(f"{R}/{r['sub']}/{r['name']}_raw.json")
    return RAW[k]
def oracle(r):
    d = f"{R}/{r['sub']}"
    p = f"{d}/O_s{r['name'].split('_s')[-1]}_raw.json" if "run17" in r["sub"] else f"{d}/O_s42_raw.json"
    if p not in ORAW: ORAW[p] = J(p)
    return ORAW[p]

HALF = {"S": slice(0, 200), "T": slice(200, 400)}
def half(r, h):
    x, O, s = raw(r), oracle(r), HALF[h]
    m, n = np.array(x["mia"]["members"]["mink"])[s], np.array(x["mia"]["nonmembers"]["mink"])[s]
    om, on = np.array(O["mia"]["members"]["mink"])[s], np.array(O["mia"]["nonmembers"]["mink"])[s]
    fq = ks_2samp(np.array(x["forget"]["truth_ratio"])[s], np.array(O["forget"]["truth_ratio"])[s]).pvalue
    pq = ks_2samp(pct(m, n), pct(om, on)).pvalue
    return dict(fq=float(fq), pl=(auc(m, n) - auc(om, on)) / auc(om, on), pq=float(pq))

qwen10 = [r for r in rows if r["model"] == "qwen" and r["forget"] == "f10" and "run17" not in r["sub"]]
for r in qwen10 + [r for r in rows if "run17" in r["sub"]]:
    for h in "ST":
        for k, v in half(r, h).items():
            r[f"{k}_{h}"] = v
seed = lambda r: int(r["name"].split("_s")[-1]) if re.search(r"_s\d+$", r["name"]) else None
setting = lambda r: re.sub(r"_ep\d+_s\d+$", "", re.sub(r"^A[12]_", "", r["name"]))
ep = lambda r: re.search(r"_ep(\d+)_", r["name"]).group(1)

POOLS = {
    "NPO + retain": ("fq", lambda r: r["family"] == "NPO + retain" and r["sub"].endswith(("run7", "run9"))),
    "SimNPO": ("fq", lambda r: r["family"] == "SimNPO" and r["sub"].endswith(("run7", "run9"))),
    "Calibrated NPO": ("fq", lambda r: r["family"] == "Calibrated NPO"),
    "Anchored, base reference": ("rq", lambda r: r["sub"].endswith("run11a") and r["proposed"] == "True"),
    "Anchored, proxy reference": ("rq", lambda r: r["sub"].endswith("run16a") and r["family"] in ("Anchor (proxy)", "Anchor-shift (proxy)")),
}
def score(r, rule, h):
    return r[f"fq_{h}"] if rule == "fq" else min(r[f"fq_{h}"], r[f"pq_{h}"])
out_rows, KEY["split_half"] = [], {}
for pool, (rule, f) in POOLS.items():
    cands = [r for r in qwen10 if f(r) and seed(r) == 42 and r["util"] >= 0.9 and "A1_" not in r["name"] and "A2_" not in r["name"]]
    if not cands:
        continue
    pick_S = max(cands, key=lambda r: score(r, rule, "S"))
    pick_full = max(cands, key=lambda r: r["fq"] if rule == "fq" else min(r["fq"], r["pq"]))
    st, e = setting(pick_S), ep(pick_S)
    reps = [r for r in qwen10 if setting(r) == st and ep(r) == e and "A1_" not in r["name"] and "A2_" not in r["name"]]
    held = [r for r in reps if seed(r) != 42]
    T = lambda rs: dict(n=len(rs), fq=sum(r["fq_T"] > .05 for r in rs), priv=sum(abs(r["pl_T"]) <= .2 for r in rs),
                        box=sum(r["fq_T"] > .05 and abs(r["pl_T"]) <= .2 for r in rs), rq=sum(min(r["fq_T"], r["pq_T"]) > .05 for r in rs),
                        pl=[round(min(r["pl_T"] for r in rs), 2), round(max(r["pl_T"] for r in rs), 2)] if rs else None)
    KEY["split_half"][pool] = dict(rule=rule, n_candidates=len(cands), chosen_on_S=f"{st} ep{e}", chosen_on_all=f"{setting(pick_full)} ep{ep(pick_full)}",
                                   same_choice=(setting(pick_full), ep(pick_full)) == (st, e),
                                   seed42_T=dict(fq=pick_S["fq_T"], pl=pick_S["pl_T"], pq=pick_S["pq_T"]),
                                   heldout_seeds_T=T(held))
    for r in [pick_S] + held:
        out_rows.append(dict(pool=pool, name=r["name"], sub=r["sub"], fq_S=r["fq_S"], pl_S=r["pl_S"], fq_T=r["fq_T"], pl_T=r["pl_T"], pq_T=r["pq_T"], util=r["util"]))

rep = [r for r in rows if "run17" in r["sub"]]
PX = J(f"{R}/run16a_v6/run16a/proxy_p10_raw.json")
def proxy_score_S(r):
    x, s = raw(r), HALF["S"]
    m, n = np.array(x["mia"]["members"]["mink"])[s], np.array(x["mia"]["nonmembers"]["mink"])[s]
    pm, pn = np.array(PX["mia"]["members"]["mink"])[s], np.array(PX["mia"]["nonmembers"]["mink"])[s]
    return min(ks_2samp(np.array(x["forget"]["truth_ratio"])[s], np.array(PX["forget"]["truth_ratio"])[s]).pvalue, ks_2samp(pct(m, n), pct(pm, pn)).pvalue)
fixed = [r for r in rep if "_ep5_" in r["name"]]
chosen = []
for sub in sorted({r["sub"] for r in rep}):
    for k in sorted({seed(r) for r in rep if r["sub"] == sub}):
        c = [r for r in rep if r["sub"] == sub and seed(r) == k and r["util"] >= 0.9]
        chosen.append(max(c, key=proxy_score_S))
def rates(rs):
    n = len(rs)
    return dict(priv=cp(sum(abs(r["pl_T"]) <= .2 for r in rs), n), fq=cp(sum(r["fq_T"] > .05 for r in rs), n),
                box=cp(sum(r["fq_T"] > .05 and abs(r["pl_T"]) <= .2 for r in rs), n), rq=cp(sum(min(r["fq_T"], r["pq_T"]) > .05 for r in rs), n),
                pl=[min(r["pl_T"] for r in rs), max(r["pl_T"] for r in rs)])
KEY["split_half"]["replicates_report_half"] = dict(epoch5=rates(fixed), proxy_chosen_on_S=rates(chosen),
                                                   epochs_chosen=[ep(r) for r in chosen])
base = [r for r in qwen10 if r["proposed"] == "False"]
fqT = [r for r in base if r["fq_T"] > .05]
KEY["split_half"]["baselines_report_half"] = dict(n=len(base), fq_pass_T=len(fqT), overunlearned_T=sum(r["pl_T"] < -.2 for r in fqT),
                                                  in_band_util_T=sum(abs(r["pl_T"]) <= .2 and r["util"] >= .9 for r in fqT))
with open(f"{OUT}/v8_split_half.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)

OR = {s: J(f"{R}/run7_v5/run7/O_s{s}_raw.json") for s in (42, 1, 2, 3, 4)}
OR.update({s: J(f"{R}/run17b_v6/run17b/O_s{s}_raw.json") for s in (5, 6, 7, 8)})
same = [float(np.max(np.abs(np.array(J(f"{R}/run17a_v6/run17a/O_s{s}_raw.json")["mia"]["members"]["loss"]) - np.array(OR[s]["mia"]["members"]["loss"])))) for s in (1, 2, 3, 4)]
ids = list(OR)
X = np.array([np.r_[OR[s]["mia"]["members"]["loss"], OR[s]["mia"]["nonmembers"]["loss"]] for s in ids], float)
sig = float((X - X.mean(0)).std(ddof=1))
def lira(x, drop=None):
    keep = [i for i, s in enumerate(ids) if s != drop]
    z = np.abs((np.r_[x["mia"]["members"]["loss"], x["mia"]["nonmembers"]["loss"]] - X[keep].mean(0)) / sig); n = len(x["mia"]["members"]["loss"])
    return auc(z[:n], z[n:])
orc = [lira(OR[s], s) for s in ids]
def rng(rs, drop=False):
    v = [lira(raw(r), seed(r) if drop else None) for r in rs]; return [round(min(v), 3), round(max(v), 3)]
fqp = [r for r in qwen10 if r["proposed"] == "False" and r["fq"] > .05]
KEY["lira9"] = dict(n_oracles=len(ids), run17a_oracles_identical_to_run7_maxdiff=max(same),
                    oracles_loo=[round(min(orc), 3), round(max(orc), 3)],
                    fq_passing_baselines=rng(fqp),
                    anchor_p10_dataorder=rng([r for r in qwen10 if r["sub"].endswith("run16b") and "_ep5_" in r["name"]]),
                    replicates_ep5=rng([r for r in rep if "_ep5_" in r["name"]], drop=True),
                    replicates_ep10=rng([r for r in rep if "_ep10_" in r["name"]], drop=True))
json.dump(KEY, open(f"{OUT}/v8_key_numbers.json", "w"), indent=1, default=float)
print(json.dumps(KEY, indent=1, default=lambda x: round(float(x), 4)))
