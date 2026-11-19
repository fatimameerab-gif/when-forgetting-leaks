import argparse, glob, json, os
import numpy as np
from scipy.stats import ks_2samp, beta
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="../results"); ap.add_argument("--out", default="../figures")
args = ap.parse_args(); R, OUT = args.results, args.out
J = lambda p: json.load(open(p))
auc = lambda m, n: float(roc_auc_score(np.r_[np.ones(len(m)), np.zeros(len(n))], np.r_[m, n]))
def pct(m, n):
    n = np.sort(n); lo = np.searchsorted(n, m, "left"); hi = np.searchsorted(n, m, "right"); return (lo + .5 * (hi - lo)) / len(n)
def cp(k, n): return [k, n, round(0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)), 2), round(1.0 if k == n else float(beta.ppf(.975, k + 1, n - k)), 2)]
rng = np.random.default_rng(1)

def metrics(x, O, A_mu, util):
    m, n = np.array(x["mia"]["members"]["mink"]), np.array(x["mia"]["nonmembers"]["mink"])
    om, on = np.array(O["mia"]["members"]["mink"]), np.array(O["mia"]["nonmembers"]["mink"])
    ks = ks_2samp(x["forget"]["truth_ratio"], O["forget"]["truth_ratio"])
    pl = (auc(m, n) - auc(om, on)) / auc(om, on)
    v = []
    for _ in range(400):
        i, j = rng.integers(0, len(m), len(m)), rng.integers(0, len(n), len(n)); ao = auc(om[i], on[j]); v.append((auc(m[i], n[j]) - ao) / ao)
    lo, hi = np.percentile(v, [5, 95])
    return dict(fq=float(ks.pvalue), D=float(ks.statistic), pl=pl, pq=float(ks_2samp(pct(m, n), pct(om, on)).pvalue),
                util=util / A_mu, tost=bool(-.2 < lo and hi < .2))

def lira_setup(oracles):
    X = np.array([np.r_[o["mia"]["members"]["loss"], o["mia"]["nonmembers"]["loss"]] for o in oracles.values()], float)
    return list(oracles), X, float((X - X.mean(0)).std(ddof=1))
def lira(x, setup, drop=None):
    ids, X, sig = setup; keep = [i for i, s in enumerate(ids) if s != drop]
    z = np.abs((np.r_[x["mia"]["members"]["loss"], x["mia"]["nonmembers"]["loss"]] - X[keep].mean(0)) / sig); n = len(x["mia"]["members"]["loss"])
    return auc(z[:n], z[n:])

def proxy_rq(x, P):
    m, n = np.array(x["mia"]["members"]["mink"]), np.array(x["mia"]["nonmembers"]["mink"])
    pm, pn = np.array(P["mia"]["members"]["mink"]), np.array(P["mia"]["nonmembers"]["mink"])
    return min(ks_2samp(x["forget"]["truth_ratio"], P["forget"]["truth_ratio"]).pvalue, ks_2samp(pct(m, n), pct(pm, pn)).pvalue)

def collect(dirs):
    reps = []
    for d in dirs:
        for p in sorted(glob.glob(f"{d}/A_s*.json")):
            if p.endswith("_raw.json"): continue
            k = int(os.path.basename(p)[3:-5]); A = J(p); O = J(f"{d}/O_s{k}_raw.json")
            r = dict(k=k, dir=d, O=O)
            for ep in (5, 10):
                f = f"{d}/anchor_p10_lr1e-5_ep{ep}_s{k}"
                if os.path.exists(f + "_raw.json"):
                    r[ep] = metrics(J(f + "_raw.json"), O, A["model_utility"], J(f + ".json")["model_utility"]); r[f"raw{ep}"] = J(f + "_raw.json")
            reps.append(r)
    return sorted(reps, key=lambda r: r["k"])

def summarise(reps, ep_key, setup, proxies):
    out = {}
    rows = [(r, r[ep_key]) for r in reps] if ep_key in (5, 10) else None
    if ep_key == "proxy":
        res = {}
        for name, P in proxies.items():
            ch = []
            for r in reps:
                c = [e for e in (5, 10) if r[e]["util"] >= .9] or [5, 10]
                e = max(c, key=lambda e: proxy_rq(r[f"raw{e}"], P)); ch.append((r, r[e], e))
            res[name] = dict(epochs=[e for _, _, e in ch], **_rates([(a, b) for a, b, _ in ch], setup, [e for _, _, e in ch]))
        return res
    return _rates(rows, setup, [ep_key] * len(rows))

def _rates(rows, setup, eps):
    n = len(rows); g = lambda f: cp(sum(f(m) for _, m in rows), n)
    li = [lira(r[f"raw{e}"], setup, ("new", r["k"])) for (r, _), e in zip(rows, eps)]
    return dict(n=n, priv=g(lambda m: abs(m["pl"]) <= .2), fq=g(lambda m: m["fq"] > .05), box=g(lambda m: m["fq"] > .05 and abs(m["pl"]) <= .2),
                rq=g(lambda m: min(m["fq"], m["pq"]) > .05), util90=g(lambda m: m["util"] >= .9), tost=g(lambda m: m["tost"]),
                pl=[round(min(m["pl"] for _, m in rows), 3), round(max(m["pl"] for _, m in rows), 3)],
                fq_median=round(float(np.median([m["fq"] for _, m in rows])), 4),
                D=[round(min(m["D"] for _, m in rows), 3), round(max(m["D"] for _, m in rows), 3)],
                util=[round(min(m["util"] for _, m in rows), 3), round(max(m["util"] for _, m in rows), 3)],
                lira=[round(min(li), 3), round(max(li), 3)],
                per_rep=[(r["k"], round(m["fq"], 4), round(m["pl"], 3), round(m["pq"], 4), round(m["util"], 3)) for r, m in rows])

KEY = {}
qd = [f"{R}/run17{p}_v6/run17{p}" for p in "abcd" if os.path.isdir(f"{R}/run17{p}_v6/run17{p}")]
qreps = collect(qd)
qor = {("new", 42): J(f"{R}/run7_v5/run7/O_s42_raw.json")}
for r in qreps: qor[("new", r["k"])] = r["O"]
qset = lira_setup(qor)
qprox = {"p10": J(f"{R}/run16a_v6/run16a/proxy_p10_raw.json"), "p25": J(f"{R}/run16a_v6/run16a/proxy_p25_raw.json")}
KEY["qwen"] = {"n_oracles_lira": len(qor),
               "oracles_lira_loo": [round(x, 3) for x in sorted(lira(o, qset, s) for s, o in qor.items())]}
for lab, sub in (("all", qreps), ("reps_1_8", [r for r in qreps if r["k"] <= 8]), ("reps_9_16", [r for r in qreps if r["k"] > 8])):
    KEY["qwen"][lab] = {"ep5": summarise(sub, 5, qset, qprox), "ep10": summarise(sub, 10, qset, qprox), "proxy": summarise(sub, "proxy", qset, qprox)}
ld = [f"{R}/run20{p}_v6/run20{p}" for p in "abcd" if os.path.isdir(f"{R}/run20{p}_v6/run20{p}")]
lreps = collect(ld)
lor = {("old", s): J(f"{R}/{s}/O_s42_raw.json") for s in ("run10a_v5/run10a", "run10b_v5/run10b", "run13a_v6/run13a", "run13b_v6/run13b", "run13c_v6/run13c", "run16c_v6/run16c")}
for r in lreps: lor[("new", r["k"])] = r["O"]
lset = lira_setup(lor)
lprox = {"p10_run16c": J(f"{R}/run16c_v6/run16c/proxy_p10_raw.json")}
KEY["llama"] = {"n_replicates": len(lreps), "n_oracles_lira": len(lor),
                "oracles_lira_loo": [round(x, 3) for x in sorted(lira(o, lset, s) for s, o in lor.items())],
                "ep10": summarise(lreps, 10, lset, lprox), "ep5": summarise(lreps, 5, lset, lprox), "proxy": summarise(lreps, "proxy", lset, lprox)}
from itertools import combinations
pairs = []
for (a, oa), (b, ob) in combinations([(k, v) for k, v in lor.items()], 2):
    am, an = oa["mia"]["members"]["mink"], oa["mia"]["nonmembers"]["mink"]; bm, bn = ob["mia"]["members"]["mink"], ob["mia"]["nonmembers"]["mink"]
    pairs.append((ks_2samp(oa["forget"]["truth_ratio"], ob["forget"]["truth_ratio"]).statistic, abs((auc(am, an) - auc(bm, bn)) / auc(bm, bn))))
KEY["llama"]["oracle_pairs"] = dict(n=len(pairs), D_max=round(max(p[0] for p in pairs), 4), pl_absmax=round(max(p[1] for p in pairs), 4))
json.dump(KEY, open(f"{OUT}/v8_replicates.json", "w"), indent=1, default=float)
print(json.dumps(KEY, indent=1, default=float))
