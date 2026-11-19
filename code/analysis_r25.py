import json, glob, os, re, sys, numpy as np
from scipy.stats import ks_2samp, beta
def J(p): return json.load(open(p))
def mu(d):
    v = [d[s][m] for s in ("retain", "real_authors", "world_facts") for m in ("prob", "truth_ratio_score", "rougeL_recall")]
    return len(v) / sum(1 / x for x in v)
def mink(raw): return np.array(raw["mia"]["members"]["mink"]), np.array(raw["mia"]["nonmembers"]["mink"])
def pct(m, n):
    n = np.sort(n); lo = np.searchsorted(n, m, "left"); hi = np.searchsorted(n, m, "right")
    return (lo + 0.5 * (hi - lo)) / len(n)
def cp(x, n):
    lo = 0.0 if x == 0 else beta.ppf(0.025, x, n - x + 1); hi = 1.0 if x == n else beta.ppf(0.975, x + 1, n - x)
    return round(float(lo), 3), round(float(hi), 3)
args = sys.argv[1:]; out = args[args.index("--out") + 1] if "--out" in args else "key_r25.json"
folders = [a for a in args if a != "--out" and a != out]
rows = []
for d in folders:
    a = glob.glob(f"{d}/A_s*.json"); a = [p for p in a if not p.endswith("_raw.json")][0]
    k = re.search(r"A_s(\d+)\.json", a).group(1)
    A, Oraw = J(a), J(f"{d}/O_s{k}_raw.json"); om, on = mink(Oraw)
    for p in sorted(glob.glob(f"{d}/*_ep*_s{k}.json")):
        nm = os.path.basename(p)[:-5]; js, raw = J(p), J(f"{d}/{nm}_raw.json"); m, n = mink(raw)
        fq = float(ks_2samp(raw["forget"]["truth_ratio"], Oraw["forget"]["truth_ratio"]).pvalue)
        pl = float(js["privacy"]["privleak_mink"]); pq = float(ks_2samp(pct(m, n), pct(om, on)).pvalue)
        ut = mu(js) / mu(A)
        rows.append(dict(seed=int(k), name=nm, method="NPO+retain" if nm.startswith("npo") else "Anchor",
                         epoch=int(re.search(r"_ep(\d+)_", nm).group(1)), fq=fq, fq_logged=js.get("forget_quality"),
                         privleak=pl, pq=pq, util_ratio=round(ut, 3),
                         box=bool(fq > 0.05 and abs(pl) <= 0.2 and ut >= 0.9), joint=bool(min(fq, pq) > 0.05)))
npo_fq = [r for r in rows if r["method"] == "NPO+retain" and r["fq"] > 0.05]
anc = [r for r in rows if r["method"] == "Anchor"]
key = dict(rows=rows,
           primary_1_npo_fq_passing=len(npo_fq), primary_1_npo_fq_passing_overunlearned=sum(r["privleak"] < -0.2 for r in npo_fq),
           primary_2_anchor_in_box=[sum(r["box"] for r in anc), len(anc)], primary_2_anchor_ci=cp(sum(r["box"] for r in anc), len(anc)) if anc else None,
           anchor_joint=[sum(r["joint"] for r in anc), len(anc)])
json.dump(key, open(out, "w"), indent=1)
print("%-28s %4s %3s %9s %8s %9s %6s %4s %5s" % ("checkpoint", "seed", "ep", "FQ", "PrivLeak", "PQ", "util", "box", "joint"))
for r in rows:
    print("%-28s %4d %3d %9.3g %+8.2f %9.2g %6.2f %4s %5s" % (r["name"], r["seed"], r["epoch"], r["fq"], r["privleak"], r["pq"], r["util_ratio"], r["box"], r["joint"]))
print("primary 1: FQ-passing NPO checkpoints %d, of which over-unlearned (PrivLeak < -0.2) %d" % (key["primary_1_npo_fq_passing"], key["primary_1_npo_fq_passing_overunlearned"]))
print("primary 2: Anchor in the box %d of %d, exact 95%% CI %s" % (*key["primary_2_anchor_in_box"], key["primary_2_anchor_ci"]))
