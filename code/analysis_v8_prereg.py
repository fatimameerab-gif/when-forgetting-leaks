import argparse, csv, glob, json, os
import numpy as np
from scipy.stats import ks_2samp, beta
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="../results"); ap.add_argument("--out", default="../figures")
a = ap.parse_args(); R, OUT = a.results, a.out; os.makedirs(OUT, exist_ok=True)
J = lambda p: json.load(open(p))
KEY = {}


def auc(m, n):
    m, n = np.asarray(m, float), np.asarray(n, float)
    return float(roc_auc_score(np.r_[np.ones(len(m)), np.zeros(len(n))], np.r_[m, n]))


def pct(m, n):
    n = np.sort(n); lo = np.searchsorted(n, m, "left"); hi = np.searchsorted(n, m, "right")
    return (lo + 0.5 * (hi - lo)) / len(n)


def cp(k, n):
    return [round(0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)), 2), round(1.0 if k == n else float(beta.ppf(.975, k + 1, n - k)), 2)]


def mink(raw):
    return np.array(raw["mia"]["members"]["mink"], float), np.array(raw["mia"]["nonmembers"]["mink"], float)


def boot_pl(rm, rn, om, on, B=400, seed=1):
    rg = np.random.default_rng(seed); n1, n2 = len(rm), len(rn); v = []
    for _ in range(B):
        i, j = rg.integers(0, n1, n1), rg.integers(0, n2, n2)
        ao = auc(om[i], on[j]); v.append((auc(rm[i], rn[j]) - ao) / ao)
    return np.percentile(v, [5, 95])


def lira_setup(ors):
    X = np.array([np.r_[o["mia"]["members"]["loss"], o["mia"]["nonmembers"]["loss"]] for o in ors], float)
    return X, float((X - X.mean(0)).std(ddof=1))


def lira(raw, setup, drop=None):
    X, sig = setup
    keep = [i for i in range(len(X)) if i != drop]
    mu = X[keep].mean(0); s = np.r_[raw["mia"]["members"]["loss"], raw["mia"]["nonmembers"]["loss"]]
    z = np.abs((np.asarray(s, float) - mu) / sig); n = len(raw["mia"]["members"]["loss"])
    return auc(z[:n], z[n:])


def score(raw, js, O, A, P=None):
    m, n = mink(raw); om, on = mink(O)
    fq = float(ks_2samp(raw["forget"]["truth_ratio"], O["forget"]["truth_ratio"]).pvalue)
    ao = auc(om, on); pl = (auc(m, n) - ao) / ao
    pq = float(ks_2samp(pct(m, n), pct(om, on)).pvalue)
    r = dict(fq=fq, pl=pl, pq=pq, util=js["model_utility"] / A["model_utility"])
    r["box"] = bool(fq > .05 and abs(pl) <= .2); r["rq"] = bool(min(fq, pq) > .05)
    r["box_util"] = bool(r["box"] and r["util"] >= .9)
    lo, hi = boot_pl(m, n, om, on); r["tost"] = [round(float(lo), 3), round(float(hi), 3)]
    r["tost_ok"] = bool(-.2 < lo and hi < .2)
    if P is not None:
        for tag, Px in P.items():
            pm, pn = mink(Px)
            fq_p = float(ks_2samp(raw["forget"]["truth_ratio"], Px["forget"]["truth_ratio"]).pvalue)
            pq_p = float(ks_2samp(pct(m, n), pct(pm, pn)).pvalue)
            r[f"sel_{tag}"] = min(fq_p, pq_p)
    return r


def summarise(rs):
    n = len(rs)
    def rate(f):
        k = sum(1 for r in rs if f(r)); return [k, n] + cp(k, n)
    rng = lambda k: [round(min(r[k] for r in rs), 3), round(max(r[k] for r in rs), 3)]
    return dict(n=n, pl_band=rate(lambda r: abs(r["pl"]) <= .2), pl_band_010=rate(lambda r: abs(r["pl"]) <= .1),
                fq=rate(lambda r: r["fq"] > .05), box=rate(lambda r: r["box"]), rq=rate(lambda r: r["rq"]),
                util90=rate(lambda r: r["util"] >= .9), box_util90=rate(lambda r: r["box_util"]),
                tost=rate(lambda r: r["tost_ok"]),
                pl_range=rng("pl"), util_range=rng("util"), lira_range=rng("lira"), fq_median=float(np.median([r["fq"] for r in rs])))


def choose(cands, tag):
    ok = [c for c in cands if c["util"] >= .9] or cands
    return max(ok, key=lambda c: (c[f"sel_{tag}"], -c["epoch"]))


ROWS = []
QD = {**{k: "run17a_v6/run17a" for k in range(1, 5)}, **{k: "run17b_v6/run17b" for k in range(5, 9)},
      **{k: "run17c_v8/run17c" for k in range(9, 13)}, **{k: "run17d_v8/run17d" for k in range(13, 17)}}
Q_OR = {42: J(f"{R}/run7_v5/run7/O_s42_raw.json")}
for s in (1, 2, 3, 4):
    Q_OR[s] = J(f"{R}/run7_v5/run7/O_s{s}_raw.json")
for s in range(5, 17):
    Q_OR[s] = J(f"{R}/{QD[s]}/O_s{s}_raw.json")
QKEYS = list(Q_OR)
LQ = lira_setup([Q_OR[s] for s in QKEYS])
KEY["qwen_lira_oracles_loo"] = {str(s): round(lira(Q_OR[s], LQ, i), 3) for i, s in enumerate(QKEYS)}
QP = {"p10": J(f"{R}/run16a_v6/run16a/proxy_p10_raw.json"), "p25": J(f"{R}/run16a_v6/run16a/proxy_p25_raw.json")}

qrows = []
for k in range(1, 17):
    d = f"{R}/{QD[k]}"; A = J(f"{d}/A_s{k}.json"); O = Q_OR[k]
    for ep in (5, 10):
        nm = f"anchor_p10_lr1e-5_ep{ep}_s{k}"
        raw, js = J(f"{d}/{nm}_raw.json"), J(f"{d}/{nm}.json")
        r = dict(model="qwen", replicate=k, epoch=ep, name=nm, **score(raw, js, O, A, QP))
        r["lira"] = lira(raw, LQ, drop=QKEYS.index(k))
        qrows.append(r)
ROWS += qrows
KEY["qwen"] = {}
for lab, ks in (("rep9_16", range(9, 17)), ("rep1_8", range(1, 9)), ("pooled1_16", range(1, 17))):
    KEY["qwen"][lab] = {}
    for ep in (5, 10):
        KEY["qwen"][lab][f"ep{ep}"] = summarise([r for r in qrows if r["replicate"] in ks and r["epoch"] == ep])
    for tag in ("p10", "p25"):
        ch = [choose([r for r in qrows if r["replicate"] == k], tag) for k in ks]
        s = summarise(ch); s["epochs"] = [c["epoch"] for c in ch]
        KEY["qwen"][lab][f"chosen_{tag}"] = s

LD = {1: "run20a", 2: "run20a", 3: "run20b", 4: "run20b", 5: "run20c", 6: "run20c", 7: "run20d", 8: "run20d"}
LLAMA_DIRS = ["run10a_v5/run10a", "run10b_v5/run10b", "run13a_v6/run13a", "run13b_v6/run13b", "run13c_v6/run13c", "run16c_v6/run16c"]
L_OR = [(d, J(f"{R}/{d}/O_s42_raw.json")) for d in LLAMA_DIRS]
L_OR += [(f"{LD[k]}_O_s{k}", J(f"{R}/{LD[k]}_v8/{LD[k]}/O_s{k}_raw.json")) for k in range(1, 9)]
LL = lira_setup([o for _, o in L_OR])
KEY["llama_lira_oracles_loo"] = {nm: round(lira(o, LL, i), 3) for i, (nm, o) in enumerate(L_OR)}
LP = {"p10": J(f"{R}/run16c_v6/run16c/proxy_p10_raw.json")}
lrows = []
for k in range(1, 9):
    d = f"{R}/{LD[k]}_v8/{LD[k]}"; A = J(f"{d}/A_s{k}.json"); O = J(f"{d}/O_s{k}_raw.json")
    for ep in (5, 10):
        nm = f"anchor_p10_lr1e-5_ep{ep}_s{k}"
        raw, js = J(f"{d}/{nm}_raw.json"), J(f"{d}/{nm}.json")
        r = dict(model="llama", replicate=k, epoch=ep, name=nm, **score(raw, js, O, A, LP))
        r["lira"] = lira(raw, LL, drop=6 + k - 1)
        lrows.append(r)
ROWS += lrows
KEY["llama"] = {f"ep{ep}": summarise([r for r in lrows if r["epoch"] == ep]) for ep in (10, 5)}
ch = [choose([r for r in lrows if r["replicate"] == k], "p10") for k in range(1, 9)]
KEY["llama"]["chosen_p10"] = summarise(ch); KEY["llama"]["chosen_p10"]["epochs"] = [c["epoch"] for c in ch]
KEY["llama"]["modelA_privleak"] = [round(J(f"{R}/{LD[k]}_v8/{LD[k]}/A_s{k}.json")["privacy"]["privleak_mink"], 2)
                                   if "privacy" in J(f"{R}/{LD[k]}_v8/{LD[k]}/A_s{k}.json") else None for k in range(1, 9)]

CONDS = [("push", "npo_kappa_push_w1"), ("none", "npo_rt"), ("hold1", "npo_kappa_hold_w1"),
         ("hold3", "npo_kappa_hold_w3"), ("hold10", "npo_kappa_hold_w10")]
KEY["kappa"] = {}
trows = []
for run, seed in (("run21a", 42), ("run21b", 1)):
    d = f"{R}/{run}_v8/{run}"
    O = J(f"{d}/O_s42_raw.json"); A = J(f"{d}/A_s42.json"); om, on = mink(O); ao = auc(om, on); opct = pct(om, on)
    otr = O["forget"]["truth_ratio"]
    res = {}
    for lab, key in CONDS:
        t = J(f"{d}/traj_{key}_lr1e-5_s{seed}.json")
        pts = t["points"]; p0 = pts[0]
        def ab(raw):
            tr = np.clip(np.array(raw["forget"]["truth_ratio"], float), 1e-30, None)
            a_ = np.log(np.clip(np.array(raw["forget"]["para_prob"], float), 1e-30, None))
            return float(np.nanmean(a_)), float(np.nanmean(np.log(tr) + a_))
        aA, bA = ab(p0["raw"]); mu0 = p0["res"]["model_utility_nogen"]
        series = []
        for p in pts:
            raw = p["raw"]; m, n = mink(raw); aa, bb = ab(raw)
            fq = float(ks_2samp(raw["forget"]["truth_ratio"], otr).pvalue)
            pl = (auc(m, n) - ao) / ao; pq = float(ks_2samp(pct(m, n), opct).pvalue)
            ut = p["res"]["model_utility_nogen"] / mu0
            gap = float(np.mean(raw["mia"]["members"]["loss"]) - np.mean(O["mia"]["members"]["loss"]))
            kap = (bA - bb) / (aA - aa) if (aA - aa) > 0.2 else None
            series.append(dict(epoch=p["epoch"], fq=fq, pl=pl, pq=pq, util=ut, kappa=kap, da=aA - aa, gap=gap))
        ks_ = [s["kappa"] for s in series if s["kappa"] is not None]
        first = next((s for s in series if s["fq"] > .05), None)
        o3 = sum(1 for s in series if s["fq"] > .05 and abs(s["pl"]) <= .2 and s["util"] >= .9)
        ev = {}
        for ep in (5, 10):
            nm = f"{key}_lr1e-5_ep{ep}_s{seed}"
            raw, js = J(f"{d}/{nm}_raw.json"), J(f"{d}/{nm}.json"); m, n = mink(raw)
            ev[f"ep{ep}"] = dict(fq=float(ks_2samp(raw["forget"]["truth_ratio"], otr).pvalue), pl=(auc(m, n) - ao) / ao,
                                 pq=float(ks_2samp(pct(m, n), opct).pvalue), util=js["model_utility"] / A["model_utility"])
        res[lab] = dict(kappa_median=float(np.median(ks_)) if ks_ else None, n_kappa_points=len(ks_),
                        O1_privleak_first_fq_pass=None if first is None else first["pl"],
                        O1_epoch=None if first is None else first["epoch"],
                        O2_gap_at_first_pass=None if first is None else first["gap"],
                        O3_points_in_box_util90=o3, n_points=len(series), max_fq=max(s["fq"] for s in series),
                        n_fq_pass=sum(s["fq"] > .05 for s in series), **ev)
        for s in series:
            trows.append(dict(run=run, seed=seed, condition=lab, **s))
    order = [res[l]["kappa_median"] for l, _ in CONDS]
    res["M_holds"] = bool(all(x is not None for x in order) and all(order[i] > order[i + 1] for i in range(len(order) - 1)))
    KEY["kappa"][f"seed{seed}"] = res

json.dump(KEY, open(f"{OUT}/key_v8_prereg.json", "w"), indent=1, default=float)
with open(f"{OUT}/v8_prereg_checkpoints.csv", "w", newline="") as f:
    keys = [k for k in ROWS[0] if k != "tost"] + ["tost_lo", "tost_hi"]
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
    for r in ROWS:
        w.writerow({**r, "tost_lo": r["tost"][0], "tost_hi": r["tost"][1]})
with open(f"{OUT}/v8_kappa_trajectories.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(trows[0])); w.writeheader(); w.writerows(trows)
print(json.dumps(KEY, indent=1, default=lambda x: round(float(x), 4)))
