import argparse
import csv
import glob
import itertools
import json
import os

import numpy as np
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="../results")
ap.add_argument("--out", default="../figures")
ap.add_argument("--paper", action="store_true")
args = ap.parse_args()
R, OUT = args.results, args.out
os.makedirs(OUT, exist_ok=True)
ATTACKS = ("mink", "loss", "minkpp", "zlib", "ref")
import sys as _s; _s.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paper_style
KEY = {}

QWEN = ["run7_v5/run7", "run8_v5/run8", "run8b_v5/run8b", "run9_v5/run9",
        "run11a_v6/run11a", "run11b_v6/run11b", "run12_v6/run12", "run12b_v6/run12b",
        "run16a_v6/run16a", "run16b_v6/run16b", "run18_v6/run18"]
LLAMA = ["run10a_v5/run10a", "run10b_v5/run10b", "run13a_v6/run13a", "run13b_v6/run13b", "run13c_v6/run13c", "run16c_v6/run16c"]


def J(p):
    return json.load(open(p))


def auc(m, n):
    m, n = np.asarray(m, float), np.asarray(n, float)
    return float(roc_auc_score(np.r_[np.ones(len(m)), np.zeros(len(n))], np.r_[m, n]))


def scores(mia, attack, ref):
    if attack == "ref":
        return (np.array(mia["members"]["loss"]) - np.array(ref["members"]["loss"]),
                np.array(mia["nonmembers"]["loss"]) - np.array(ref["nonmembers"]["loss"]))
    return np.array(mia["members"][attack], float), np.array(mia["nonmembers"][attack], float)


def pct(m, n):
    n = np.sort(n)
    lo = np.searchsorted(n, m, "left")
    hi = np.searchsorted(n, m, "right")
    return (lo + 0.5 * (hi - lo)) / len(n)


def wcsv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


class Run:
    def __init__(self, d):
        self.d = d
        self.O = J(f"{d}/O_s42_raw.json")
        self.ref = J(f"{d}/_refscores_forget10.json")
        self.otr = np.array(self.O["forget"]["truth_ratio"], float)
        self.opct, self.oauc, self.oauc_h = {}, {}, {}
        for a in ATTACKS:
            m, n = scores(self.O["mia"], a, self.ref)
            self.opct[a] = pct(m, n)
            self.oauc[a] = auc(m, n)
            self.oauc_h[a] = (auc(m, n[:200]), auc(m, n[200:]))
        aj = J(f"{d}/A_s42.json")
        self.mu_a, self.mu_a_nogen = aj["model_utility"], aj["model_utility_nogen"]

    def metrics(self, raw, mu=None, mu_nogen=None):
        tr = np.array(raw["forget"]["truth_ratio"], float)
        ks = ks_2samp(tr, self.otr)
        out = {"fq": float(ks.pvalue), "ks": float(ks.statistic)}
        pqs = []
        for a in ATTACKS:
            m, n = scores(raw["mia"], a, self.ref)
            out[f"pl_{a}"] = (auc(m, n) - self.oauc[a]) / self.oauc[a]
            c, e = self.oauc_h[a]
            out[f"pl_{a}_calibhalf"] = (auc(m, n[:200]) - c) / c
            out[f"pl_{a}_evalhalf"] = (auc(m, n[200:]) - e) / e
            pk = ks_2samp(pct(m, n), self.opct[a])
            out[f"pq_{a}"] = float(pk.pvalue)
            out[f"pqks_{a}"] = float(pk.statistic)
            pqs.append(float(pk.pvalue))
        out["rq"] = min(out["fq"], out["pq_mink"])
        out["rq_worst"] = min([out["fq"]] + pqs)
        if mu is not None:
            out["util"] = mu / self.mu_a
        if mu_nogen is not None:
            out["util_nogen"] = mu_nogen / self.mu_a_nogen
        a_ = np.log(np.clip(np.array(raw["forget"]["para_prob"], float), 1e-30, None))
        out["a_para"] = float(np.nanmean(a_))
        out["b_pert"] = float(np.nanmean(np.log(np.clip(tr, 1e-30, None)) + a_))
        out["log_tr"] = float(np.nanmean(np.log(np.clip(tr, 1e-30, None))))
        out["forget_lp"] = float(np.mean(raw["mia"]["members"]["loss"]))
        out["calib_lp"] = float(np.mean(raw["mia"]["nonmembers"]["loss"][:200]))
        return out


def evaluations(rd):
    for p in sorted(glob.glob(f"{rd}/*.json")):
        b = os.path.basename(p)
        if b.endswith("_raw.json") or b.startswith(("_", "traj_")) or b == "O_s42.json":
            continue
        r = J(p)
        if "forget" in r and os.path.exists(p[:-5] + "_raw.json"):
            yield b[:-5], r, J(p[:-5] + "_raw.json")


def family(name):
    n = name[3:] if name[:3] in ("A1_", "A2_") else name
    for k, fam in (("npo_rt", "NPO + retain"), ("simnpo", "SimNPO"), ("idknll", "IdkNLL"), ("idkdpo", "IdkDPO"),
                   ("rmu", "RMU"), ("task_vector", "Task vector"), ("npo_cal", "Calibrated NPO"),
                   ("floor", "Floor"), ("match_off", "Match + offset"), ("match", "Match"),
                   ("anchor_shift_p", "Anchor-shift (proxy)"), ("anchor_p", "Anchor (proxy)"), ("proxy_", "Proxy model"),
                   ("anchor_shift", "Anchor (shifted)"), ("anchor", "Anchor"), ("ciu", "CIU"),
                   ("kl_proxy", "KL to proxy"), ("kl_base", "KL to base"), ("base", "Base model"), ("O_s", "Oracle"), ("A_s", "Model A")):
        if n.startswith(k):
            return fam
    return n


d7 = f"{R}/run7_v5/run7"
o7 = J(f"{d7}/O_s42_raw.json")
same = {}
for sub in QWEN[1:]:
    p = f"{R}/{sub}/O_s42_raw.json"
    if os.path.exists(p):
        o = J(p)
        same[sub] = float(np.max(np.abs(np.array(o["mia"]["members"]["loss"]) - np.array(o7["mia"]["members"]["loss"]))))
KEY["oracle_max_abs_diff_vs_run7"] = same

run7 = Run(d7)
oracles = {s: J(f"{d7}/O_s{s}_raw.json") for s in (42, 1, 2, 3, 4)}
noise = []
for a, b in itertools.combinations(oracles, 2):
    row = {"pair": f"s{a}-s{b}", "fq": ks_2samp(oracles[a]["forget"]["truth_ratio"],
                                               oracles[b]["forget"]["truth_ratio"]).pvalue}
    for att in ATTACKS:
        ma, na = scores(oracles[a]["mia"], att, run7.ref)
        mb, nb = scores(oracles[b]["mia"], att, run7.ref)
        k = ks_2samp(pct(ma, na), pct(mb, nb))
        row[f"pq_{att}"], row[f"pqks_{att}"] = float(k.pvalue), float(k.statistic)
    row["rq"] = min(row["fq"], row["pq_mink"])
    row["rq_worst"] = min([row["fq"]] + [row[f"pq_{t}"] for t in ATTACKS])
    noise.append(row)
wcsv(f"{OUT}/v6_oracle_noise_pq.csv", noise)
KEY["oracle_pairs"] = {"n": len(noise), "pq_mink_min": min(r["pq_mink"] for r in noise),
                       "rq_min": min(r["rq"] for r in noise), "rq_worst_min": min(r["rq_worst"] for r in noise),
                       "pqks_mink_max": max(r["pqks_mink"] for r in noise)}

LOADED9 = dict(oracles)
for _s in (1, 2, 3, 4, 5, 6, 7, 8):
    if _s not in LOADED9:
        _p = f"{R}/run17b_v6/run17b/O_s{_s}_raw.json" if _s >= 5 else f"{d7}/O_s{_s}_raw.json"
        LOADED9[_s] = J(_p)
OUTS = {s: LOADED9[s]["mia"] for s in sorted(LOADED9)}


def lira_stats(att="loss"):
    mem = np.array([OUTS[s]["members"][att] for s in OUTS], float)
    non = np.array([OUTS[s]["nonmembers"][att] for s in OUTS], float)
    allx = np.concatenate([mem, non], 1)
    sig = float((allx - allx.mean(0)).std(ddof=1))
    return mem, non, sig


L_MEM, L_NON, L_SIG = lira_stats()
KEY["lira_sigma"] = L_SIG


def lira(raw, loo=None):
    keep = [i for i, s in enumerate(OUTS) if s != loo]
    mu_m, mu_n = L_MEM[keep].mean(0), L_NON[keep].mean(0)
    zm = (np.array(raw["mia"]["members"]["loss"]) - mu_m) / L_SIG
    zn = (np.array(raw["mia"]["nonmembers"]["loss"]) - mu_n) / L_SIG
    return {"lira_auc": auc(zm, zn), "lira_auc_2s": auc(np.abs(zm), np.abs(zn)), "lira_mean_z": float(zm.mean())}


LIRA_O = lira(oracles[42], loo=42)
KEY["lira_oracle"] = LIRA_O
KEY["lira_oracles_all"] = {s: lira(LOADED9[s], loo=s) for s in OUTS}

rows, seen = [], set()
for sub in QWEN + LLAMA:
    rd = f"{R}/{sub}"
    if not os.path.isdir(rd) or not os.path.exists(f"{rd}/O_s42_raw.json"):
        continue
    run = Run(rd)
    model = "llama" if sub in LLAMA else "qwen"
    for name, js, raw in evaluations(rd):
        if "_forget0" in name or name.startswith(("O95", "O99")):
            continue
        m = run.metrics(raw, mu=js["model_utility"])
        dup = (model, name, round(m["fq"], 12), round(m["pl_mink"], 10))
        if dup in seen:
            continue
        seen.add(dup)
        m.update({"run": os.path.basename(rd), "model": model, "name": name, "family": family(name)})
        if model == "qwen":
            m.update(lira(raw, loo=42 if name == "O_s42" else None))
            m["pl_lira"] = (m["lira_auc"] - LIRA_O["lira_auc"]) / LIRA_O["lira_auc"]
        rows.append(m)
wcsv(f"{OUT}/v6_checkpoints.csv", rows)
ul = [r for r in rows if r["family"] not in ("Model A", "Oracle", "Base model", "Proxy model")]
fq_pass = [r for r in ul if r["fq"] > 0.05]
KEY["checkpoints"] = {
    "n": len(ul), "fq_pass": len(fq_pass),
    "fq_pass_pq_fail": sum(r["pq_mink"] <= 0.05 for r in fq_pass),
    "rq_pass": [(r["run"], r["name"], r["fq"], r["pq_mink"], r["util"]) for r in ul if r["rq"] > 0.05],
    "rq_worst_pass": [(r["run"], r["name"]) for r in ul if r["rq_worst"] > 0.05],
    "spearman_fq_vs_rq": float(spearmanr([r["fq"] for r in ul], [r["rq"] for r in ul]).correlation),
    "top5_by_fq": [(r["name"], r["fq"], r["rq"]) for r in sorted(ul, key=lambda r: -r["fq"])[:5]],
    "top5_by_rq": [(r["name"], r["fq"], r["rq"]) for r in sorted(ul, key=lambda r: -r["rq"])[:5]],
    "calib_vs_eval_half": [(r["name"], r["pl_mink_calibhalf"], r["pl_mink_evalhalf"]) for r in ul
                           if r["family"] in ("Calibrated NPO", "Floor", "Match", "Match + offset", "Anchor",
                                              "Anchor (shifted)", "CIU")],
    "max_abs_half_diff": max([abs(r["pl_mink_calibhalf"] - r["pl_mink_evalhalf"]) for r in ul] or [0]),
}
q = [r for r in ul if r["model"] == "qwen" and r["fq"] > 0.05]
KEY["lira_fq_passers"] = {"n": len(q), "pl_lira_range": [min(r["pl_lira"] for r in q), max(r["pl_lira"] for r in q)] if q else None,
                          "auc2s_range": [min(r["lira_auc_2s"] for r in q), max(r["lira_auc_2s"] for r in q)] if q else None}

traj = []
for sub in QWEN:
    rd = f"{R}/{sub}"
    if not os.path.isdir(rd) or not os.path.exists(f"{rd}/O_s42_raw.json"):
        continue
    run = Run(rd)
    for p in sorted(glob.glob(f"{rd}/traj_*.json")):
        key = os.path.basename(p)[5:-9]
        t = J(p)
        if len(t["points"]) < 5:
            continue
        a0 = None
        for qd in t["points"]:
            m = run.metrics(qd["raw"], mu_nogen=qd["res"]["model_utility_nogen"])
            m.update({"run": os.path.basename(rd), "setting": key, "epoch": qd["epoch"], "family": family(key)})
            m.update(lira(qd["raw"]))
            if a0 is None:
                a0 = m
            da = m["a_para"] - a0["a_para"]
            m["kappa"] = (m["b_pert"] - a0["b_pert"]) / da if abs(da) > 0.05 else None
            traj.append(m)
wcsv(f"{OUT}/v6_trajectories.csv", traj)
oq = run7.metrics(oracles[42])
aq = [r for r in rows if r["run"] == "run7" and r["name"] == "A_s42"][0]
kstar = 1 - (oq["log_tr"] - aq["log_tr"]) / (aq["a_para"] - oq["a_para"])
ks_ = [m["kappa"] for m in traj if m["kappa"] is not None and m["family"] in
       ("NPO + retain", "SimNPO", "Calibrated NPO", "Floor", "Match")]
KEY["prop1"] = {"A": {k: aq[k] for k in ("a_para", "b_pert", "log_tr")},
                "oracle": {k: oq[k] for k in ("a_para", "b_pert", "log_tr")},
                "kappa_star": kstar, "kappa_median": float(np.median(ks_)) if ks_ else None,
                "kappa_iqr": [float(np.percentile(ks_, 25)), float(np.percentile(ks_, 75))] if ks_ else None,
                "n_points": len(ks_)}
KEY["traj"] = {"n": len(traj), "rq_pass": [(m["setting"], m["epoch"], m["fq"], m["pq_mink"], m["util_nogen"])
                                           for m in traj if m["rq"] > 0.05],
               "rq_pass_util": sum(m["rq"] > 0.05 and m["util_nogen"] >= 0.9 for m in traj),
               "rq_worst_pass": sum(m["rq_worst"] > 0.05 for m in traj)}

offs = {}
for sub in QWEN[:1] + LLAMA[:1]:
    rd = f"{R}/{sub}"
    b = J(f"{rd}/_refscores_forget10.json")
    o = J(f"{rd}/O_s42_raw.json")["mia"]
    bf, bc = np.mean(b["members"]["loss"]), np.mean(b["nonmembers"]["loss"][:200])
    of, oc = np.mean(o["members"]["loss"]), np.mean(o["nonmembers"]["loss"][:200])
    r_ = float(np.corrcoef(b["members"]["loss"], o["members"]["loss"])[0, 1])
    offs[sub.split("/")[0]] = {"base_offset_nats": float(bc - bf), "oracle_offset_nats": float(oc - of),
                               "base_forget_lp": float(bf), "oracle_forget_lp": float(of),
                               "r_base_oracle_per_example": r_}
KEY["offset"] = offs

json.dump(KEY, open(f"{OUT}/v6_key_numbers.json", "w"), indent=1, default=float)

d14 = f"{R}/run14_v6/run14"
if os.path.exists(f"{d14}/O_s42_raw.json"):
    run14 = Run(d14)
    ou = []
    ou_seen = set()
    for dd in [d14] + [f"{R}/run14{x}_v6/run14{x}" for x in "bc"]:
      if not os.path.isdir(dd):
        continue
      for name, js, raw in evaluations(dd):
        if not name.startswith("ou_") or name in ou_seen:
            continue
        ou_seen.add(name)
        m = run14.metrics(raw, mu=js["model_utility"])
        m.update({"name": name[3:], "method": name[3:].split("_")[0], "lr": name[3:].split("_")[1]})
        ou.append(m)
    for key in ("fq", "rq"):
        order = sorted(ou, key=lambda r: -r[key])
        for i, r in enumerate(order):
            r[f"rank_{key}"] = i + 1
    wcsv(f"{OUT}/v6_openunlearning.csv", ou)
    keep = [r for r in ou if r["util"] >= 0.9]
    fqp = [r for r in ou if r["fq"] > 0.05]
    best_fq = {}
    for r in sorted(keep, key=lambda r: -r["fq"]):
        best_fq.setdefault(r["method"], r)
    best_rq = {}
    for r in sorted(keep, key=lambda r: -r["rq"]):
        best_rq.setdefault(r["method"], r)
    KEY["openunlearning"] = {
        "n": len(ou), "methods": sorted({r["method"] for r in ou}),
        "fq_pass": len(fqp), "fq_pass_keep_util": sum(r["util"] >= 0.9 for r in fqp),
        "fq_pass_privleak_range": [min(r["pl_mink"] for r in fqp), max(r["pl_mink"] for r in fqp)] if fqp else None,
        "fq_pass_pq_fail": sum(r["pq_mink"] <= 0.05 for r in fqp),
        "box": [(r["name"], r["fq"], r["pl_mink"], r["util"]) for r in ou if r["fq"] > 0.05 and abs(r["pl_mink"]) <= 0.2],
        "rq_pass": [(r["name"], r["fq"], r["pq_mink"], r["util"]) for r in ou if r["rq"] > 0.05],
        "spearman_fq_rq": float(spearmanr([r["fq"] for r in ou], [r["rq"] for r in ou]).correlation),
        "method_rank_by_fq": [(k, v["fq"], v["pl_mink"], v["rq"]) for k, v in
                              sorted(best_fq.items(), key=lambda kv: -kv[1]["fq"])],
        "method_rank_by_rq": [(k, v["rq"], v["fq"], v["pl_mink"]) for k, v in
                              sorted(best_rq.items(), key=lambda kv: -kv[1]["rq"])],
        "target": {"fq": run14.metrics(J(f"{d14}/A_s42_raw.json"))["fq"]},
    }
    json.dump(KEY, open(f"{OUT}/v6_key_numbers.json", "w"), indent=1, default=float)

COL = {"NPO + retain": "#1f77b4", "SimNPO": "#2ca02c", "IdkNLL": "#ff7f0e", "IdkDPO": "#8c564b", "RMU": "#9467bd",
       "Task vector": "#555555", "Calibrated NPO": "#bcbd22", "Floor": "#a0a0a0", "Match": "#7f7f7f",
       "Match + offset": "#404040", "Anchor": "#d62728", "Anchor (shifted)": "#e377c2", "CIU": "#c7c7c7",
       "KL to base": "#a0a0a0"}
import paper_style as _ps; from matplotlib.patches import Rectangle as _Rect
MK = _ps.MARKER
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.9))
for r in ul:
    c = COL.get(r["family"], "#c7c7c7")
    mk = MK.get(r["family"], "o")
    if r["model"] == "qwen":
        ax[0].scatter(max(r["fq"], 1e-12), max(r["pq_mink"], 1e-30), s=24, color=c, marker=mk, alpha=.9,
                      edgecolor="white", linewidth=.4, zorder=3)
    else:
        ax[0].scatter(max(r["fq"], 1e-12), max(r["pq_mink"], 1e-30), s=24, facecolor="white", edgecolor=c,
                      marker=mk, linewidth=.9, zorder=3)
for r in noise:
    ax[0].scatter(r["fq"], r["pq_mink"], s=30, marker="*", color="black")
for a_ in ax[:1]:
    a_.set_xscale("log"); a_.set_yscale("log"); a_.set_ylim(3e-31, 3)
    a_.axvline(.05, color="#7f7f7f", ls="--", lw=.8); a_.axhline(.05, color="#7f7f7f", ls="--", lw=.8)
    a_.set_xlabel("Forget Quality (truth ratios)"); a_.set_ylabel("Privacy Quality (Min-K%)")
    a_.add_patch(_Rect((.05, .05), 1.95, 2.95, facecolor=_ps.BOX_FILL, edgecolor=_ps.BOX_EDGE, lw=.8, ls="--", zorder=0))
for r in ul:
    if r["model"] != "qwen":
        continue
    ax[1].scatter(max(r["fq"], 1e-12), r["lira_auc_2s"], s=24, color=COL.get(r["family"], "#c7c7c7"), alpha=.9,
                  marker=MK.get(r["family"], "o"), edgecolor="white", linewidth=.4, zorder=3)
for s, v in KEY["lira_oracles_all"].items():
    ax[1].scatter(1.0, v["lira_auc_2s"], marker="*", s=30, color="black")
ax[1].set_xscale("log"); ax[1].axvline(.05, color="#7f7f7f", ls="--", lw=.8)
ax[1].axhline(LIRA_O["lira_auc_2s"], color="black", lw=.8, ls=":")
ax[1].set_xlabel("Forget Quality"); ax[1].set_ylabel("LiRA AUC (two-sided)")
hs = [plt.Line2D([], [], marker=MK.get(k, "o"), ls="", color=c, markeredgecolor="white", markersize=7, label=k)
      for k, c in COL.items() if any(r["family"] == k for r in ul)]
hs.append(plt.Line2D([], [], marker="*", ls="", color="black", markersize=9, label="oracle (pairs / LOO)"))
hs.append(plt.Line2D([], [], marker="o", ls="", color="#555555", markersize=6, label="filled: Qwen"))
hs.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#555555", markersize=6, label="open: Llama (left)"))
hs.append(_Rect((0, 0), 1, 1, facecolor=_ps.BOX_FILL, edgecolor=_ps.BOX_EDGE, ls="--", label="retrain-like on both"))
fig.legend(handles=hs, loc="lower center", ncol=4, frameon=True, fontsize=8, bbox_to_anchor=(.5, -.30))
if not args.paper:
    pass
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig9_joint_test.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 3.8))
SUPP = ("NPO + retain", "SimNPO", "Calibrated NPO", "Floor", "Match")
for key in sorted({m["setting"] for m in traj if m["family"] in SUPP}):
    pts = [m for m in traj if m["setting"] == key]
    ax.plot([m["a_para"] for m in pts], [m["b_pert"] for m in pts], "-." if pts[0]["family"] == "Floor" else "-",
            lw=1.1, alpha=.85, color=COL.get(pts[0]["family"], "#c7c7c7"))
ax.scatter([aq["a_para"]], [aq["b_pert"]], s=55, marker="s", color="black", zorder=5, label="Model A")
ax.scatter([oq["a_para"]], [oq["b_pert"]], s=90, marker="*", color="black", zorder=5, label="oracle")
xs = np.linspace(-6.0, aq["a_para"], 10)
ax.plot(xs, aq["b_pert"] + (xs - aq["a_para"]) * kstar, ":", color="#555555", label=f"slope κ* = {kstar:.2f}")
ax.plot(xs, xs + oq["log_tr"], "--", color="#7f7f7f", lw=.8, label="oracle truth ratio")
for k, c in COL.items():
    if k in SUPP:
        ax.plot([], [], color=c, ls="-." if k == "Floor" else "-", label=k)
ax.set_xlim(-6.2, -2.0); ax.set_ylim(-7.0, -2.6)
ax.set_xlabel("mean log-likelihood, paraphrased answer (a)")
ax.set_ylabel("mean log-likelihood, perturbed answers (b)")
ax.legend(frameon=True, fontsize=8.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig10_transfer.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print(json.dumps(KEY, indent=1, default=float)[:6000])

def pick(run, prefix, ep):
    return [r for r in rows if r["run"] == run and r["name"].startswith(prefix) and f"_ep{ep}_s" in r["name"]]


groups = [("NPO + retain (2e-5, ep 10)", pick("run9", "npo_rt_lr2e-5", 10), "#1f77b4", "o"),
          ("SimNPO (5e-5, ep 10)", pick("run9", "simnpo_fw1_b2.5_lr5e-5", 10), "#2ca02c", "^"),
          ("Calibrated NPO (1e-5, ep 5)", pick("run9", "npo_cal_lr1e-5", 5), "#bcbd22", "D"),
          ("Anchor-shift, base target (1e-5, ep 5)",
           [r for r in pick("run12", "anchor_shift_lr1e-5", 5) + pick("run12b", "anchor_shift_lr1e-5", 5)], "#e377c2", "d"),
          ("Anchor, 10% proxy target (1e-5, ep 5)", pick("run16b", "anchor_p10_lr1e-5", 5), "#d62728", "s")]
if any(g[1] for g in groups):
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 5.4))
    a = ax[0]
    a.add_patch(_Rect((-0.2, 0.05), 0.4, 1.45, facecolor=_ps.BOX_FILL, edgecolor=_ps.BOX_EDGE, lw=.8, ls="--", zorder=0))
    for lab, rs, c, mk in groups:
        if rs:
            a.scatter([r["pl_mink"] for r in rs], [max(r["fq"], 1e-6) for r in rs], color=c, marker=mk, s=40,
                      edgecolor="white", linewidth=.5, label=f"{lab} (n={len(rs)})", zorder=3)
    on = [r for r in noise]
    a.scatter([0] * 0, [])
    a.set_yscale("log"); a.set_ylim(1e-4, 1.5); a.set_xlim(-1.0, 0.45)
    a.axhline(.05, color="#7f7f7f", ls="--", lw=.8); a.axvline(0, color="#7f7f7f", ls="--", lw=.8)
    a.set_xlabel("PrivLeak (Min-K%)"); a.set_ylabel("Forget Quality")
    a.legend(fontsize=9, frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    a.set_title("(a) Held-out seeds, Qwen2.5-0.5B forget10")
    b = ax[1]
    if os.path.exists(f"{OUT}/v6_openunlearning.csv"):
        ou_rows = list(csv.DictReader(open(f"{OUT}/v6_openunlearning.csv")))
        mcol = {"AltPO": "#404040", "GradDiff": "#bcbd22", "IdkDPO": "#8c564b", "IdkNLL": "#ff7f0e", "NPO": "#1f77b4",
                "RMU": "#9467bd", "SimNPO": "#2ca02c", "UNDIAL": "#a0a0a0"}
        omk = {"AltPO": "<", "GradDiff": "D", "IdkDPO": "X", "IdkNLL": "P", "NPO": "o", "RMU": "v", "SimNPO": "^",
               "UNDIAL": "h"}
        b.add_patch(_Rect((-0.2, 0.05), 0.4, 1.95, facecolor=_ps.BOX_FILL, edgecolor=_ps.BOX_EDGE, lw=.8, ls="--", zorder=0))
        for m, c in mcol.items():
            rr = [r for r in ou_rows if r["method"] == m]
            b.scatter([float(r["pl_mink"]) for r in rr], [max(float(r["fq"]), 1e-30) for r in rr], s=18, color=c,
                      alpha=.85, label=m, marker=omk[m], edgecolor="white", linewidth=.3)
        b.set_yscale("log"); b.set_ylim(3e-31, 2)
        b.axhline(.05, color="#7f7f7f", ls="--", lw=.8); b.axvline(0, color="#7f7f7f", ls="--", lw=.8)
        b.set_xlabel("PrivLeak (Min-K%)"); b.set_ylabel("Forget Quality")
        b.legend(fontsize=9, frameon=True, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), markerscale=1.5)
        b.set_title(f"(b) OpenUnlearning checkpoints, Llama-3.2-1B (n={len(ou_rows)})")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig11_fixes_and_openunlearning.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
