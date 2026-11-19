import json, os, sys
import numpy as np
from scipy.stats import ks_2samp, beta
from sklearn.metrics import roc_auc_score

ROOT, OUT = sys.argv[1:3]
J = lambda p: json.load(open(p))

def auc(m, n):
    return float(roc_auc_score(np.r_[np.ones(len(m)), np.zeros(len(n))], np.r_[m, n]))

def pct(m, n):
    n = np.sort(n); lo = np.searchsorted(n, m, "left"); hi = np.searchsorted(n, m, "right")
    return (lo + 0.5 * (hi - lo)) / len(n)

def cp(k, n):
    return (0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(.975, k + 1, n - k)))

def mink(raw):
    return np.array(raw["mia"]["members"]["mink"]), np.array(raw["mia"]["nonmembers"]["mink"])

rows, box_oracle, box_random, rho_pairs = [], 0, 0, []
rng = np.random.default_rng(42)
for k in range(1, 9):
    run = "run17a" if k <= 4 else "run17b"
    d = f"{ROOT}/v6/{run}"
    A = J(f"{d}/A_s{k}.json"); Ao = J(f"{d}/A_s{k}.json")["model_utility"]
    O = J(f"{d}/O_s{k}_raw.json")
    om, on = mink(O); auc_o = auc(om, on)
    P = J(f"{ROOT}/v6/run17e/P10_s{k}_forget10_raw.json"); pm, pn = mink(P)
    cands = []
    for ep in (5, 10):
        nm = f"anchor_p10_lr1e-5_ep{ep}_s{k}"
        js = J(f"{d}/{nm}.json"); raw = J(f"{d}/{nm}_raw.json")
        m, n = mink(raw)
        fq = float(ks_2samp(raw["forget"]["truth_ratio"], O["forget"]["truth_ratio"]).pvalue)
        pl = (auc(m, n) - auc_o) / auc_o
        util = js["model_utility"] / A["model_utility"]
        fq_p = float(ks_2samp(raw["forget"]["truth_ratio"], P["forget"]["truth_ratio"]).pvalue)
        pq_p = float(ks_2samp(pct(m, n), pct(pm, pn)).pvalue)
        cands.append(dict(ep=ep, fq=fq, pl=pl, util=util, score=min(fq_p, pq_p), fq_p=fq_p, pq_p=pq_p,
                          box=bool(fq > 0.05 and abs(pl) <= 0.2 and util >= 0.9)))
    ok = [c for c in cands if c["util"] >= 0.9] or cands
    best = max(ok, key=lambda c: (c["score"], -c["ep"]))
    rows.append(dict(replicate=k, chosen_epoch=best["ep"], chosen_box=best["box"], candidates=cands))
    box_oracle += best["box"]
    rr = int(rng.integers(2))
    box_random += cands[rr]["box"]

k_box = box_oracle
lo, hi = cp(k_box, 8)
out = dict(primary_box_proxy_chosen=[k_box, 8, round(lo, 3), round(hi, 3)],
           secondary_box_random_epoch_seed42=box_random, rows=rows)
json.dump(out, open(OUT, "w"), indent=1, default=float)
print(json.dumps({"primary": out["primary_box_proxy_chosen"], "random_epoch": box_random,
                  "chosen": [(r["replicate"], r["chosen_epoch"], r["chosen_box"]) for r in rows]}, indent=1))
