import argparse, glob, json, os, re, itertools
import numpy as np
from scipy.stats import ks_2samp, spearmanr, beta
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="../results"); ap.add_argument("--out", default="../figures")
args = ap.parse_args(); R, OUT = args.results, args.out; os.makedirs(OUT, exist_ok=True)
J = lambda p: json.load(open(p))
KEY = {}
rng0 = np.random.default_rng(0)


def auc(m, n):
    m, n = np.asarray(m, float), np.asarray(n, float)
    return float(roc_auc_score(np.r_[np.ones(len(m)), np.zeros(len(n))], np.r_[m, n]))


def pct(m, n):
    n = np.sort(n); lo = np.searchsorted(n, m, "left"); hi = np.searchsorted(n, m, "right")
    return (lo + 0.5 * (hi - lo)) / len(n)


def cp(k, n):
    return (0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(.975, k + 1, n - k)))


def matched_draws(sm, sn, ndraw=30, nbins=10, seed=0):
    rng = np.random.default_rng(seed)
    edges = np.quantile(np.r_[sm, sn], np.linspace(0, 1, nbins + 1)); edges[-1] += 1e-9
    bm, bn = np.digitize(sm, edges[1:-1]), np.digitize(sn, edges[1:-1])
    draws = []
    for _ in range(ndraw):
        im, inn = [], []
        for b in range(nbins):
            a, c = np.where(bm == b)[0], np.where(bn == b)[0]; k = min(len(a), len(c))
            if k:
                im += list(rng.choice(a, k, replace=False)); inn += list(rng.choice(c, k, replace=False))
        draws.append((np.array(im), np.array(inn)))
    return draws


LLAMA_DIRS = ["run10a_v5/run10a", "run10b_v5/run10b", "run13a_v6/run13a", "run13b_v6/run13b", "run13c_v6/run13c", "run16c_v6/run16c"]
QWEN_SHARED = ["run7_v5/run7", "run8_v5/run8", "run8b_v5/run8b", "run9_v5/run9", "run11a_v6/run11a", "run11b_v6/run11b",
               "run12_v6/run12", "run12b_v6/run12b", "run16a_v6/run16a", "run16b_v6/run16b", "run18_v6/run18"]
OTHER = ["runs1-3_v3", "run4_scale/forget01", "run4_scale/forget05", "run4_scale/forget10", "run5_tinyllama/tinyllama",
         "run6_tinyllama_low/tinyllama_low", "run17a_v6/run17a", "run17b_v6/run17b"]
SKIP = re.compile(r"^(A_|O_|O9|B0_|base_model|proxy_|traj_|_|ou_)|_4bit")
PROPOSED = ("anchor", "ciu", "kl_", "match_off")


def family(n):
    n = re.sub(r"^(sw_|sd_|fn_|A1_|A2_)", "", n)
    for k, f in (("npo_rt", "NPO + retain"), ("simnpo", "SimNPO"), ("graddiff", "Gradient difference"), ("ga", "Gradient ascent"),
                 ("npo_cal", "Calibrated NPO"), ("npo", "NPO w/o retain"), ("ft_retain", "Fine-tune on retain"),
                 ("idknll", "IdkNLL"), ("idkdpo", "IdkDPO"), ("rmu", "RMU"), ("task_vector", "Task vector"), ("floor", "Floor"),
                 ("match_off", "Match + offset"), ("match", "Match"), ("anchor_shift_p", "Anchor-shift (proxy)"),
                 ("anchor_p", "Anchor (proxy)"), ("anchor_shift", "Anchor-shift (base)"), ("anchor", "Anchor (base)"),
                 ("ciu", "CIU"), ("kl_proxy", "KL to proxy"), ("kl_base", "KL to base")):
        if n.startswith(k):
            return f
    return n


def oracle_for(sub, name):
    d = f"{R}/{sub}"
    if "run12" in sub and "forget05" in name: return f"{d}/O95_s42_forget05_raw.json", "f05"
    if "run12" in sub and "forget01" in name: return f"{d}/O99_s42_forget01_raw.json", "f01"
    if "run17" in sub: return f"{d}/O_s{name.split('_s')[-1]}_raw.json", "f10"
    if sub.endswith("forget01"): return f"{d}/O_retain99_raw.json", "f01"
    if sub.endswith("forget05"): return f"{d}/O_retain95_raw.json", "f05"
    for o in ("O_s42_raw.json", "B0_retrain_raw.json"):
        if os.path.exists(f"{d}/{o}"): return f"{d}/{o}", "f10"


def modelA(sub, name):
    d = f"{R}/{sub}"
    if "run17" in sub: return J(f"{d}/A_s{name.split('_s')[-1]}.json")
    for a in ("A_s42.json", "A_original.json"):
        if os.path.exists(f"{d}/{a}"): return J(f"{d}/{a}")


BASE = J(f"{R}/run11a_v6/run11a/base_model_raw.json")["mia"]
ORAW = {}
def oraw(p):
    if p not in ORAW: ORAW[p] = J(p)
    return ORAW[p]
DRAWS = {}
def draws_for(key, sm, sn):
    if key not in DRAWS: DRAWS[key] = matched_draws(np.asarray(sm, float), np.asarray(sn, float))
    return DRAWS[key]

Q_OR = {s: J(f"{R}/run7_v5/run7/O_s{s}_raw.json") for s in (42, 1, 2, 3, 4)}
Q9 = dict(Q_OR)
for _s in (5, 6, 7, 8):
    Q9[_s] = J(f"{R}/run17b_v6/run17b/O_s{_s}_raw.json")
L_OR = {sub.split("/")[-1]: J(f"{R}/{sub}/O_s42_raw.json") for sub in LLAMA_DIRS}


def pair_stats(a, b):
    ks = ks_2samp(a["forget"]["truth_ratio"], b["forget"]["truth_ratio"])
    am, an = a["mia"]["members"]["mink"], a["mia"]["nonmembers"]["mink"]
    bm, bn = b["mia"]["members"]["mink"], b["mia"]["nonmembers"]["mink"]
    pl = (auc(am, an) - auc(bm, bn)) / auc(bm, bn)
    pq = ks_2samp(pct(np.array(am), np.array(an)), pct(np.array(bm), np.array(bn)))
    return dict(fq=float(ks.pvalue), D=float(ks.statistic), pl=float(pl), pq=float(pq.pvalue), Dpq=float(pq.statistic))


def band(ors):
    ps = [pair_stats(ors[i], ors[j]) for i, j in itertools.permutations(ors, 2)]
    return dict(n_pairs=len(ps) // 2, fq_min=min(p["fq"] for p in ps), D_max=max(p["D"] for p in ps), Dpq_max=max(p["Dpq"] for p in ps),
                pl_absmax=max(abs(p["pl"]) for p in ps), pq_min=min(p["pq"] for p in ps),
                oracle_auc=[auc(o["mia"]["members"]["mink"], o["mia"]["nonmembers"]["mink"]) for o in ors.values()])

KEY["noise_qwen"] = band(Q9)
KEY["noise_llama"] = band(L_OR)


def lira_setup(ors):
    X = np.array([np.r_[o["mia"]["members"]["loss"], o["mia"]["nonmembers"]["loss"]] for o in ors.values()], float)
    return X, float((X - X.mean(0)).std(ddof=1))
LQ, LL = lira_setup(Q9), lira_setup(L_OR)


def lira(raw, setup, drop=None):
    X, sig = setup
    keep = [i for i in range(len(X)) if i != drop]
    mu = X[keep].mean(0); s = np.r_[raw["mia"]["members"]["loss"], raw["mia"]["nonmembers"]["loss"]]
    z = np.abs((np.asarray(s, float) - mu) / sig); n = len(raw["mia"]["members"]["loss"])
    return auc(z[:n], z[n:])
KEY["lira_oracles_qwen"] = [lira(Q9[s], LQ, i) for i, s in enumerate(Q9)]
KEY["lira_oracles_llama"] = [lira(o, LL, i) for i, o in enumerate(L_OR.values())]

rows, seen = [], set()
for sub in QWEN_SHARED + LLAMA_DIRS + OTHER:
    d = f"{R}/{sub}"
    if not os.path.isdir(d):
        continue
    model = "llama" if sub in LLAMA_DIRS else ("tinyllama" if "tinyllama" in sub else "qwen")
    for p in sorted(glob.glob(f"{d}/*.json")):
        name = os.path.basename(p)[:-5]
        if name.endswith("_raw") or SKIP.search(name) or not os.path.exists(f"{d}/{name}_raw.json"):
            continue
        js, raw = J(p), J(f"{d}/{name}_raw.json")
        if "forget" not in js:
            continue
        op, fsz = oracle_for(sub, name); O = oraw(op); A = modelA(sub, name)
        m, n = np.array(raw["mia"]["members"]["mink"]), np.array(raw["mia"]["nonmembers"]["mink"])
        om, on = np.array(O["mia"]["members"]["mink"]), np.array(O["mia"]["nonmembers"]["mink"])
        ks = ks_2samp(raw["forget"]["truth_ratio"], O["forget"]["truth_ratio"])
        a_c, a_o = auc(m, n), auc(om, on)
        pq = ks_2samp(pct(m, n), pct(om, on))
        r = dict(sub=sub, name=name, model=model, forget=fsz, family=family(name),
                 proposed=any(name.split("sw_")[-1].startswith(k) for k in PROPOSED),
                 fq=float(ks.pvalue), D=float(ks.statistic), auc=a_c, auc_O=a_o, pl=(a_c - a_o) / a_o,
                 pq=float(pq.pvalue), Dpq=float(pq.statistic), util=js["model_utility"] / A["model_utility"])
        dup = (model, fsz, r["family"], round(r["fq"], 12), round(r["pl"], 10))
        if dup in seen:
            continue
        seen.add(dup)
        pls = []
        for att in ("loss", "mink", "minkpp", "zlib"):
            am, an = raw["mia"]["members"][att], raw["mia"]["nonmembers"][att]
            bm, bn = O["mia"]["members"][att], O["mia"]["nonmembers"][att]
            pls.append((auc(am, an) - auc(bm, bn)) / auc(bm, bn))
        r["pl_worst_abs"] = max(abs(x) for x in pls)
        dr = draws_for(op, O["mia"]["members"]["loss"], O["mia"]["nonmembers"]["loss"])
        v = [(auc(m[i], n[j]) - auc(om[i], on[j])) / auc(om[i], on[j]) for i, j in dr]
        r["pl_match_oracle"] = float(np.mean(v)); r["auc_O_match_oracle"] = float(np.mean([auc(om[i], on[j]) for i, j in dr]))
        r["auc_match_oracle"] = float(np.mean([auc(m[i], n[j]) for i, j in dr]))
        if model == "qwen" and fsz == "f10":
            dr = draws_for("base", BASE["members"]["loss"], BASE["nonmembers"]["loss"])
            r["pl_match_base"] = float(np.mean([(auc(m[i], n[j]) - auc(om[i], on[j])) / auc(om[i], on[j]) for i, j in dr]))
            r["auc_O_match_base"] = float(np.mean([auc(om[i], on[j]) for i, j in dr]))
            r["lira"] = lira(raw, LQ)
        if model == "llama":
            r["lira"] = lira(raw, LL)
        rows.append(r)

import csv
with open(f"{OUT}/v7_checkpoints.csv", "w", newline="") as f:
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

base = [r for r in rows if not r["proposed"] and "run17" not in r["sub"]]
fqp = [r for r in base if r["fq"] > 0.05]
KEY["universe"] = dict(n_checkpoints=len(rows), n_baseline=len(base), n_baseline_fq_pass=len(fqp),
                       by_condition={f"{m}/{f}": sum(1 for r in base if (r["model"], r["forget"]) == (m, f)) for m, f in
                                     sorted({(r["model"], r["forget"]) for r in base})})

def ou(rs, key, thr=-0.2): return sum(r[key] < thr for r in rs if key in r)
KEY["matched"] = dict(
    oracle_auc_standard=sorted({round(r["auc_O"], 3) for r in base}),
    oracle_auc_after_oracle_match=[round(min(r["auc_O_match_oracle"] for r in base), 3), round(max(r["auc_O_match_oracle"] for r in base), 3)],
    oracle_auc_after_base_match=[round(min(r["auc_O_match_base"] for r in base if "auc_O_match_base" in r), 3),
                                 round(max(r["auc_O_match_base"] for r in base if "auc_O_match_base" in r), 3)],
    fq_pass=len(fqp),
    overunlearned_standard=ou(fqp, "pl"), overunlearned_oracle_match=ou(fqp, "pl_match_oracle"),
    fq_pass_with_base_match=sum("pl_match_base" in r for r in fqp), overunlearned_base_match=ou(fqp, "pl_match_base"),
    pl_range_standard=[min(r["pl"] for r in fqp), max(r["pl"] for r in fqp)],
    pl_range_oracle_match=[min(r["pl_match_oracle"] for r in fqp), max(r["pl_match_oracle"] for r in fqp)],
    raw_auc_range_oracle_match=[min(r["auc_match_oracle"] for r in fqp), max(r["auc_match_oracle"] for r in fqp)],
    exceptions_oracle_match=[(r["sub"], r["name"], round(r["pl_match_oracle"], 2), round(r["util"], 2)) for r in fqp if r["pl_match_oracle"] >= -0.2],
)

nb = KEY["noise_qwen"]["pl_absmax"]
KEY["band"] = {}
for lab, t in (("0.1", 0.1), ("0.2", 0.2), ("0.3", 0.3), ("oracle_noise", nb)):
    KEY["band"][lab] = dict(
        threshold=t, fq_pass_in_band=sum(abs(r["pl"]) <= t for r in fqp),
        fq_pass_in_band_util90=sum(abs(r["pl"]) <= t and r["util"] >= .9 for r in fqp),
        fq_pass_in_band_matched=sum(abs(r["pl_match_oracle"]) <= t for r in fqp),
        fq_pass_below_minus=sum(r["pl"] < -t for r in fqp),
        fq_pass_below_minus_matched=sum(r["pl_match_oracle"] < -t for r in fqp))

mq, ml = KEY["noise_qwen"], KEY["noise_llama"]
def margins(r): return (mq if r["model"] != "llama" else ml)
def equiv(r):
    m = margins(r); return r["D"] <= m["D_max"] and r["Dpq"] <= m["Dpq_max"]
KEY["equivalence"] = dict(
    margin_qwen=dict(D=mq["D_max"], Dpq=mq["Dpq_max"], pl=mq["pl_absmax"]),
    margin_llama=dict(D=ml["D_max"], Dpq=ml["Dpq_max"], pl=ml["pl_absmax"]),
    baseline_equiv_both=sum(equiv(r) for r in base), baseline_equiv_D=sum(r["D"] <= margins(r)["D_max"] for r in base),
    smallest_D_baselines=sorted([(round(r["D"], 3), round(r["Dpq"], 3), round(r["pl"], 2), r["name"]) for r in base])[:5],
    all_equiv_both=[(r["sub"], r["name"]) for r in rows if equiv(r)],
    D_fq_passers=[min(r["D"] for r in fqp), max(r["D"] for r in fqp)])

def boot_pl(raw_m, raw_n, om, on, B=400, seed=1):
    rg = np.random.default_rng(seed); n1, n2 = len(raw_m), len(raw_n); v = []
    for _ in range(B):
        i, j = rg.integers(0, n1, n1), rg.integers(0, n2, n2)
        ao = auc(om[i], on[j]); v.append((auc(raw_m[i], raw_n[j]) - ao) / ao)
    return np.percentile(v, [5, 95])

KEY["lira"] = dict(
    qwen_fq_pass=[round(min(r["lira"] for r in fqp if r["model"] == "qwen" and "lira" in r), 2), round(max(r["lira"] for r in fqp if r["model"] == "qwen" and "lira" in r), 2)],
    llama_all_baselines=[round(min(r["lira"] for r in base if r["model"] == "llama"), 2), round(max(r["lira"] for r in base if r["model"] == "llama"), 2)] if any(r["model"] == "llama" for r in base) else None,
    llama_anchor=[(r["sub"].split("/")[-1], r["name"], round(r["lira"], 2), round(r["fq"], 3), round(r["pl"], 2)) for r in rows if r["model"] == "llama" and r["proposed"]],
    qwen_anchor_p10=[round(min(r["lira"] for r in rows if r["family"] == "Anchor (proxy)" and r["model"] == "qwen" and "lira" in r), 2),
                     round(max(r["lira"] for r in rows if r["family"] == "Anchor (proxy)" and r["model"] == "qwen" and "lira" in r), 2)])

pool = [r for r in rows if r["sub"] in QWEN_SHARED and r["forget"] == "f10"]
RAWC = {}
def rawof(r):
    k = (r["sub"], r["name"])
    if k not in RAWC: RAWC[k] = J(f"{R}/{r['sub']}/{r['name']}_raw.json")
    return RAWC[k]
KEY["oracle_free"] = {}
for px in ("p10", "p25"):
    P = J(f"{R}/run16a_v6/run16a/proxy_{px}_raw.json")
    pm, pn = np.array(P["mia"]["members"]["mink"]), np.array(P["mia"]["nonmembers"]["mink"]); pa = auc(pm, pn)
    for r in pool:
        raw = rawof(r); m, n = np.array(raw["mia"]["members"]["mink"]), np.array(raw["mia"]["nonmembers"]["mink"])
        r[f"fq_{px}"] = float(ks_2samp(raw["forget"]["truth_ratio"], P["forget"]["truth_ratio"]).pvalue)
        r[f"pl_{px}"] = (auc(m, n) - pa) / pa
        r[f"pq_{px}"] = float(ks_2samp(pct(m, n), pct(pm, pn)).pvalue)
        r[f"rq_{px}"] = min(r[f"fq_{px}"], r[f"pq_{px}"])
    ok = [r for r in pool if r["util"] >= 0.9]
    top = sorted(ok, key=lambda r: -r[f"rq_{px}"])[:5]
    topb = sorted([r for r in ok if not r["proposed"]], key=lambda r: -r[f"rq_{px}"])[:3]
    fqsel = sorted([r for r in ok if not r["proposed"]], key=lambda r: -r["fq"])[:3]
    KEY["oracle_free"][px] = dict(
        proxy_auc=pa, n_pool=len(pool), n_util90=len(ok),
        spearman_rq=float(spearmanr([r[f"rq_{px}"] for r in pool], [min(r["fq"], r["pq"]) for r in pool]).correlation),
        spearman_pl=float(spearmanr([r[f"pl_{px}"] for r in pool], [r["pl"] for r in pool]).correlation),
        spearman_fq=float(spearmanr([r[f"fq_{px}"] for r in pool], [r["fq"] for r in pool]).correlation),
        top5=[(r["sub"].split("/")[-1], r["name"], round(r[f"rq_{px}"], 3), "| true:", round(r["fq"], 3), round(r["pl"], 2), round(r["pq"], 3), round(r["util"], 2)) for r in top],
        top_baselines=[(r["name"], round(r[f"rq_{px}"], 3), "| true:", round(r["fq"], 3), round(r["pl"], 2), round(r["util"], 2)) for r in topb],
        fq_only_selection_baselines=[(r["name"], "| true:", round(r["fq"], 3), round(r["pl"], 2)) for r in fqsel],
        in_box_true_top1=bool(top[0]["fq"] > .05 and abs(top[0]["pl"]) <= .2))
    picks = []
    for s in range(1, 9):
        c = [r for r in pool if r["sub"] == "run16b_v6/run16b" and r["name"].endswith(f"_s{s}")]
        if c:
            b = max(c, key=lambda r: r[f"rq_{px}"]); picks.append((s, b["name"].split("_")[3], b["fq"] > .05 and abs(b["pl"]) <= .2, min(b["fq"], b["pq"]) > .05))
    KEY["oracle_free"][px]["run16b_epoch_pick"] = picks
    KEY["oracle_free"][px]["run16b_box"] = sum(p[2] for p in picks); KEY["oracle_free"][px]["run16b_rq"] = sum(p[3] for p in picks)

rep = [r for r in rows if "run17" in r["sub"]]
KEY["replicates"] = {}
for ep in ("ep5", "ep10"):
    rs = [r for r in rep if f"_{ep}_" in r["name"]]; n = len(rs)
    def rate(f):
        k = sum(map(f, rs)); return [k, n, *[round(x, 2) for x in cp(k, n)]]
    tost = []
    for r in rs:
        raw = J(f"{R}/{r['sub']}/{r['name']}_raw.json"); O = oraw(oracle_for(r["sub"], r["name"])[0])
        lo, hi = boot_pl(np.array(raw["mia"]["members"]["mink"]), np.array(raw["mia"]["nonmembers"]["mink"]),
                         np.array(O["mia"]["members"]["mink"]), np.array(O["mia"]["nonmembers"]["mink"]))
        tost.append((round(lo, 3), round(hi, 3)))
    KEY["replicates"][ep] = dict(
        n=n, pl_band=rate(lambda r: abs(r["pl"]) <= .2), fq=rate(lambda r: r["fq"] > .05), pq=rate(lambda r: r["pq"] > .05),
        box=rate(lambda r: r["fq"] > .05 and abs(r["pl"]) <= .2), rq=rate(lambda r: min(r["fq"], r["pq"]) > .05),
        util90=rate(lambda r: r["util"] >= .9), pl_median=float(np.median([r["pl"] for r in rs])),
        pl_range=[min(r["pl"] for r in rs), max(r["pl"] for r in rs)], fq_median=float(np.median([r["fq"] for r in rs])),
        D_range=[min(r["D"] for r in rs), max(r["D"] for r in rs)], Dpq_range=[min(r["Dpq"] for r in rs), max(r["Dpq"] for r in rs)],
        util_range=[min(r["util"] for r in rs), max(r["util"] for r in rs)],
        lira_range=[min(r["lira"] for r in rs), max(r["lira"] for r in rs)],
        pl_match_oracle_range=[min(r["pl_match_oracle"] for r in rs), max(r["pl_match_oracle"] for r in rs)],
        pl_tost_90ci=tost, pl_tost_within_0_2=sum(-.2 < lo and hi < .2 for lo, hi in tost),
        worst_attack_abs_pl=[min(r["pl_worst_abs"] for r in rs), max(r["pl_worst_abs"] for r in rs)])
tb = []
for r in fqp:
    raw = J(f"{R}/{r['sub']}/{r['name']}_raw.json"); O = oraw(oracle_for(r["sub"], r["name"])[0])
    lo, hi = boot_pl(np.array(raw["mia"]["members"]["mink"]), np.array(raw["mia"]["nonmembers"]["mink"]),
                     np.array(O["mia"]["members"]["mink"]), np.array(O["mia"]["nonmembers"]["mink"]), B=200)
    tb.append((lo, hi))
KEY["equivalence"]["fq_passers_pl_tost_within_0_2"] = sum(-.2 < lo and hi < .2 for lo, hi in tb)
KEY["equivalence"]["fq_passers_pl_ci_entirely_below_minus_0_2"] = sum(hi < -.2 for lo, hi in tb)

json.dump(KEY, open(f"{OUT}/v7_key_numbers.json", "w"), indent=1, default=float)
print(json.dumps(KEY, indent=1, default=lambda x: round(float(x), 4)))
