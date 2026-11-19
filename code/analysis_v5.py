import argparse
import csv
import glob
import itertools
import json
import os

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="../results")
ap.add_argument("--out", default="../figures")
ap.add_argument("--paper", action="store_true", help="no figure titles")
ap.add_argument("--boot", type=int, default=1000)
args = ap.parse_args()
R, OUT = args.results, args.out
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)
ATTACKS = ("mink", "loss", "minkpp", "zlib", "ref")
import sys as _s; _s.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paper_style
INK, MUT, REF = "#1f1e1c", "#555555", "#7f7f7f"


def J(p):
    return json.load(open(p))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def auc(m, n):
    m, n = np.asarray(m, float), np.asarray(n, float)
    return float(roc_auc_score(np.r_[np.ones(len(m)), np.zeros(len(n))], np.r_[m, n]))


def scores(mia, attack, ref):
    if attack == "ref":
        return (np.array(mia["members"]["loss"]) - np.array(ref["members"]["loss"]),
                np.array(mia["nonmembers"]["loss"]) - np.array(ref["nonmembers"]["loss"]))
    return np.array(mia["members"][attack]), np.array(mia["nonmembers"][attack])


class Run:

    def __init__(self, d):
        self.d = d
        self.O = J(f"{d}/O_s42_raw.json")
        self.ref = J(f"{d}/_refscores_forget10.json")
        self.otr = np.array(self.O["forget"]["truth_ratio"], float)
        self.oauc = {a: auc(*scores(self.O["mia"], a, self.ref)) for a in ATTACKS}
        self.oauc_eval = {a: auc(scores(self.O["mia"], a, self.ref)[0], scores(self.O["mia"], a, self.ref)[1][200:])
                          for a in ATTACKS}
        a = J(f"{d}/A_s42.json")
        self.mu_a, self.mu_a_nogen = a["model_utility"], a["model_utility_nogen"]

    def metrics(self, raw, mu=None, mu_nogen=None):
        tr = np.array(raw["forget"]["truth_ratio"], float)
        ks = ks_2samp(tr, self.otr)
        out = {"fq": float(ks.pvalue), "ks": float(ks.statistic), "tr": float(np.nanmean(tr))}
        for a in ATTACKS:
            m, n = scores(raw["mia"], a, self.ref)
            out[f"pl_{a}"] = (auc(m, n) - self.oauc[a]) / self.oauc[a]
            out[f"pl_{a}_eval"] = (auc(m, n[200:]) - self.oauc_eval[a]) / self.oauc_eval[a]
        out["forget_nll"] = float(-np.mean(raw["mia"]["members"]["loss"]))
        out["calib_nll"] = float(-np.mean(raw["mia"]["nonmembers"]["loss"][:200]))
        if mu is not None:
            out["util"] = mu / self.mu_a
        if mu_nogen is not None:
            out["util_nogen"] = mu_nogen / self.mu_a_nogen
        return out

    def boot(self, raw, n=None):
        n = n or args.boot
        m, nn = scores(raw["mia"], "mink", self.ref)
        om, on = scores(self.O["mia"], "mink", self.ref)
        pl = []
        for _ in range(n):
            i, j = RNG.integers(0, len(m), len(m)), RNG.integers(0, len(nn), len(nn))
            oa = auc(om[i], on[j])
            pl.append((auc(m[i], nn[j]) - oa) / oa)
        return np.percentile(pl, [2.5, 97.5])


def evaluations(run_dir):
    for p in sorted(glob.glob(f"{run_dir}/*.json")):
        b = os.path.basename(p)
        if b.endswith("_raw.json") or b.startswith(("_", "traj_")) or b in ("O_s42.json",):
            continue
        r = J(p)
        if "forget" not in r:
            continue
        yield b[:-5], r, J(p[:-5] + "_raw.json")


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


def family(name):
    for k, fam in (("npo_rt", "NPO + retain"), ("simnpo", "SimNPO"), ("idknll", "IdkNLL"), ("idkdpo", "IdkDPO"),
                   ("rmu", "RMU"), ("task_vector", "Task vector"), ("npo_cal", "Calibrated NPO (fix)"),
                   ("floor", "Floor (fix)"), ("match", "Match (fix)")):
        if name.startswith(k) or name.startswith(f"A1_{k}") or name.startswith(f"A2_{k}"):
            return fam
    return None


FAMCOL = {"NPO + retain": "#1f77b4", "SimNPO": "#2ca02c", "IdkNLL": "#ff7f0e", "IdkDPO": "#8c564b",
          "RMU": "#9467bd", "Task vector": "#555555", "Calibrated NPO (fix)": "#bcbd22", "Floor (fix)": "#a0a0a0",
          "Match (fix)": "#7f7f7f", "Gradient Ascent": "#aec7e8", "Gradient Difference": "#bcbd22",
          "NPO w/o retain": "#c7c7c7"}
T = {}

d7 = f"{R}/run7_v5/run7"
run7 = Run(d7)
oracles = {s: J(f"{d7}/O_s{s}_raw.json") for s in (42, 1, 2, 3, 4)}
noise = []
for a, b in itertools.combinations(oracles, 2):
    ks = ks_2samp(oracles[a]["forget"]["truth_ratio"], oracles[b]["forget"]["truth_ratio"])
    row = {"pair": f"s{a}-s{b}", "fq": ks.pvalue, "ks": ks.statistic}
    for att in ATTACKS:
        aa = auc(*scores(oracles[a]["mia"], att, run7.ref))
        ab = auc(*scores(oracles[b]["mia"], att, run7.ref))
        row[f"pl_{att}"] = (aa - ab) / ab
    noise.append(row)
wcsv(f"{OUT}/v5_oracle_noise.csv", noise)
KS_MAX = max(r["ks"] for r in noise)
PL_RANGE = (min(r["pl_mink"] for r in noise), max(r["pl_mink"] for r in noise))
om, on = scores(run7.O["mia"], "mink", run7.ref)
oa_boot = [auc(om[RNG.integers(0, 400, 400)], on[RNG.integers(0, 400, 400)]) for _ in range(args.boot)]
T["noise"] = {"ks_min": min(r["ks"] for r in noise), "ks_max": KS_MAX, "ks_mean": float(np.mean([r["ks"] for r in noise])),
              "fq_min": min(r["fq"] for r in noise), "pl_min": PL_RANGE[0], "pl_max": PL_RANGE[1],
              "oracle_auc": run7.oauc["mink"], "oracle_auc_ci": list(np.percentile(oa_boot, [2.5, 97.5])),
              "n_pairs": len(noise)}

trajs = {}
traj_rows = []
for rd in (f"{R}/run7_v5/run7", f"{R}/run8_v5/run8", f"{R}/run8b_v5/run8b"):
    run = Run(rd)
    for p in sorted(glob.glob(f"{rd}/traj_*.json")):
        key = os.path.basename(p)[5:-9]
        t = J(p)
        if len(t["points"]) < 5:
            continue
        pts = []
        for q in t["points"]:
            m = run.metrics(q["raw"], mu_nogen=q["res"]["model_utility_nogen"])
            m.update({"run": os.path.basename(rd), "setting": key, "epoch": q["epoch"]})
            pts.append(m)
            traj_rows.append(m)
        trajs[key] = pts
wcsv(f"{OUT}/v5_trajectories.csv", traj_rows)


def in_box(m, att="mink"):
    return m["fq"] > 0.05 and abs(m[f"pl_{att}"]) <= 0.2


box_pts = [m for m in traj_rows if in_box(m)]
box_pts_util = [m for m in box_pts if m["util_nogen"] >= 0.9]
T["traj"] = {"n_points": len(traj_rows), "n_settings": len(trajs),
             "box": [(m["setting"], m["epoch"], m["fq"], m["pl_mink"], m["util_nogen"]) for m in box_pts],
             "box_util": len(box_pts_util),
             "box_by_attack": {a: sum(in_box(m, a) for m in traj_rows) for a in ATTACKS},
             "box_by_attack_util": {a: sum(in_box(m, a) and m["util_nogen"] >= 0.9 for m in traj_rows) for a in ATTACKS},
             "ks_equiv_points": sum(m["ks"] <= KS_MAX for m in traj_rows),
             "min_ks": min(m["ks"] for m in traj_rows)}
closest = sorted(traj_rows, key=lambda m: m["ks"])[:10]
T["traj"]["closest_ks"] = [(m["setting"], m["epoch"], m["ks"], m["fq"], m["pl_mink"]) for m in closest]
stop = []
for k, pts in trajs.items():
    hit = next((m for m in pts if m["epoch"] > 0 and m["forget_nll"] >= m["calib_nll"]), None)
    if hit:
        stop.append({"setting": k, "epoch": hit["epoch"], "fq": hit["fq"], "pl_mink": hit["pl_mink"],
                     "util_nogen": hit["util_nogen"], "tr": hit["tr"]})
wcsv(f"{OUT}/v5_stop_rule.csv", stop)
T["stop"] = stop

ends = []
for sub in ("run7_v5/run7", "run8_v5/run8", "run8b_v5/run8b", "run9_v5/run9", "run10a_v5/run10a", "run10b_v5/run10b"):
    rd = f"{R}/{sub}"
    if not os.path.isdir(rd):
        continue
    run = Run(rd)
    for name, js, raw in evaluations(rd):
        m = run.metrics(raw, mu=js["model_utility"])
        m.update({"run": os.path.basename(rd), "name": name, "family": family(name) or name})
        ends.append(m)
wcsv(f"{OUT}/v5_checkpoints.csv", ends)

run9 = Run(f"{R}/run9_v5/run9")
held = []
for name, js, raw in evaluations(f"{R}/run9_v5/run9"):
    if name in ("A_s42",) or name.startswith("A_s"):
        continue
    m = run9.metrics(raw, mu=js["model_utility"])
    pl_lo, pl_hi = run9.boot(raw)
    m.update({"name": name, "pl_ci_lo": pl_lo, "pl_ci_hi": pl_hi, "ks_within_oracle_noise": bool(m["ks"] <= KS_MAX)})
    held.append(m)
wcsv(f"{OUT}/v5_heldout_ci.csv", held)
T["held"] = held

llama = [e for e in ends if e["run"].startswith("run10") and e["name"] not in ("A_s42",)]
T["llama"] = llama

fig_t = (lambda f: None) if args.paper else (lambda f: f())

SEL = [("npo_rt_lr1e-5", "NPO + retain, 1e-5"), ("npo_rt_lr2e-5", "NPO + retain, 2e-5"),
       ("simnpo_fw1_b2.5_lr5e-5", "SimNPO, 5e-5"), ("npo_cal_lr1e-5", "Calibrated NPO (fix), 1e-5"),
       ("match_lr2e-5", "Match (fix), 2e-5"), ("rmu_c2_lr5e-5", "RMU, c = 2, 5e-5")]
COLS = ["#1f77b4", "#404040", "#2ca02c", "#bcbd22", "#7f7f7f", "#9467bd"]
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True)
for ax, group in zip(axes, (SEL[:3], SEL[3:])):
    ax.add_patch(Rectangle((-0.2, 0.05), 0.4, 20, facecolor=paper_style.BOX_FILL, edgecolor=paper_style.BOX_EDGE, lw=0.8, ls="--", zorder=0))
    ax.axhline(0.05, color=REF, ls="--", lw=1)
    ax.axvline(0, color=REF, ls="--", lw=1)
    for (key, lab) in group:
        col = COLS[SEL.index((key, lab))]
        pts = trajs.get(key)
        if not pts:
            continue
        x = [p["pl_mink"] for p in pts]
        y = [max(p["fq"], 1e-25) for p in pts]
        ax.plot(x, y, "-", color=col, lw=1.4, alpha=0.75, label=lab)
        ok = [p["util_nogen"] >= 0.9 for p in pts]
        ax.scatter([a for a, o in zip(x, ok) if o], [b for b, o in zip(y, ok) if o], s=22, color=col, zorder=3)
        ax.scatter([a for a, o in zip(x, ok) if not o], [b for b, o in zip(y, ok) if not o], s=22,
                   facecolor="white", edgecolor=col, zorder=3)
        pass
    ax.scatter([r["pl_mink"] for r in noise], [r["fq"] for r in noise], marker="*", s=110, color="black",
               edgecolor="black", lw=0.6, zorder=5, label="oracle vs oracle (10 pairs)")
    ax.set_yscale("log")
    ax.set_ylim(1e-25, 20)
    ax.set_xlim(-1.0, 1.45)
    ax.set_xlabel("PrivLeak, Min-K% (0 = oracle)")
    ax.grid(False)
    ax.legend(frameon=True, loc="lower left", fontsize=8.5)
axes[0].set_ylabel("Forget Quality (log)")
axes[0].set_title("(a) Likelihood-suppression methods")
axes[1].set_title("(b) Calibrated objectives and RMU")
fig_t(lambda: fig.suptitle("Trajectories every half epoch from Model A (right) to epoch 10; hollow = utility < 90% of A",
                           x=0.01, ha="left", fontsize=10.5, color=MUT))
fig.tight_layout()
save(fig, "fig6_trajectories")

fig, ax = plt.subplots(figsize=(4.6, 3.8))
xs = np.array([m["forget_nll"] - m["calib_nll"] for m in traj_rows])
ys = np.array([m["tr"] for m in traj_rows])
ok = np.array([m["fq"] > 0.05 for m in traj_rows])
keep = ys < 3
ax.scatter(xs[keep & ~ok], ys[keep & ~ok], s=14, color="#a0a0a0", label="FQ $\\leq$ 0.05", zorder=2)
ax.scatter(xs[keep & ok], ys[keep & ok], s=18, color="#1f77b4", label="FQ > 0.05", zorder=3)
o_x = -np.mean(run7.O["mia"]["members"]["loss"]) + np.mean(run7.O["mia"]["nonmembers"]["loss"][:200])
o_y = float(np.mean(run7.otr))
ax.scatter([o_x], [o_y], marker="*", s=240, color="black", edgecolor="black", lw=0.8, zorder=5, label="oracle")
ax.axvline(0, color=REF, ls="--", lw=1)
ax.axhline(o_y, color=REF, ls=":", lw=1)
pass
ax.tick_params(labelsize=10.5); ax.set_xlabel("forget-set NLL $-$ calibration non-member NLL (nats/token)", fontsize=11)
ax.set_ylabel("mean truth ratio on the forget set", fontsize=11)
ax.set_ylim(0.4, 2.0)
ax.grid(False)
ax.legend(frameon=True, loc="upper left", fontsize=9.5, markerscale=1.0)

fig.tight_layout()
save(fig, "fig7_mechanism")

fig, ax = plt.subplots(figsize=(10.5, 5.6))
ax.add_patch(Rectangle((-0.2, 0.05), 0.4, 20, facecolor=paper_style.BOX_FILL, edgecolor=paper_style.BOX_EDGE, lw=0.8, ls="--", zorder=0))
ax.axhline(0.05, color=REF, ls="--", lw=1)
ax.axvline(0, color=REF, ls="--", lw=1)
sw = list(csv.DictReader(open(f"{R}/runs1-3_v3/sweep_summary.csv")))
a_mu = J(f"{R}/runs1-3_v3/A_original.json")["model_utility"]
old_fam = {"ga": "Gradient Ascent", "graddiff": "Gradient Difference", "npo": "NPO w/o retain", "npo_rt": "NPO + retain",
           "simnpo": "SimNPO"}
pts = [(old_fam[r["family"]], float(r["privleak"]), max(float(r["forget_quality"] or 0), 1e-30),
        float(r["model_utility"]) / a_mu >= 0.9) for r in sw]
pts += [(e["family"], e["pl_mink"], max(e["fq"], 1e-30), e["util"] >= 0.9) for e in ends
        if e["run"] in ("run8", "run8b") and family(e["name"])]
seen = set()
for fam, x, y, k in pts:
    col = FAMCOL.get(fam, "#000")
    seen.add(fam)
    ax.scatter(x, y, s=34 if k else 24, facecolor=col if k else "white", edgecolor=col, lw=1.1, alpha=0.85,
               zorder=3)
for fam in [f for f in FAMCOL if f in seen]:
    ax.scatter([], [], s=34, color=FAMCOL[fam], label=fam)
ax.scatter([], [], s=30, facecolor="white", edgecolor=MUT, label="utility < 90% of A")
ax.set_yscale("log")
ax.set_ylim(1e-31, 50)
ax.set_xlim(-1.05, 1.45)
ax.set_xlabel("PrivLeak, Min-K% (0 = oracle)")
ax.set_ylabel("Forget Quality (log)")
ax.grid(False)
ax.legend(frameon=True, fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig_t(lambda: ax.set_title("All Qwen2.5-0.5B forget10 checkpoints; hollow = utility < 90% of A"))
fig.tight_layout()
save(fig, "fig8_families")


def load_v4(path, split):
    return [r for r in csv.DictReader(open(path)) if r["split"] == split and r["name"].startswith("sw_")]


panels = [("Qwen2.5-0.5B, forget 5%", load_v4(f"{R}/run4_scale/v4_summary.csv", "forget05")),
          ("Qwen2.5-0.5B, forget 1%", load_v4(f"{R}/run4_scale/v4_summary.csv", "forget01")),
          ("TinyLlama-1.1B, forget 10%", load_v4(f"{R}/run5_tinyllama/v4_summary.csv", "tinyllama")
           + load_v4(f"{R}/run6_tinyllama_low/v4_summary.csv", "tinyllama_low")),
          ("Llama-3.2-1B, forget 10%", [{"name": e["name"], "privleak": e["pl_mink"], "forget_quality": e["fq"],
                                         "utility_ratio": e["util"]} for e in llama])]
fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.6), sharey=True)
axes = axes.ravel()
for ax, (title, rows) in zip(axes, panels):
    ax.add_patch(Rectangle((-0.2, 0.05), 0.4, 20, facecolor=paper_style.BOX_FILL, edgecolor=paper_style.BOX_EDGE, lw=0.8, ls="--", zorder=0))
    ax.axhline(0.05, color=REF, ls="--", lw=1)
    ax.axvline(0, color=REF, ls="--", lw=1)
    for r in rows:
        n = r["name"].replace("sw_", "")
        fam = family(n) or "NPO + retain"
        col = FAMCOL[fam]
        k = float(r["utility_ratio"]) >= 0.9
        ax.scatter(float(r["privleak"]), max(float(r["forget_quality"]), 1e-15), s=46, facecolor=col if k else "white",
                   edgecolor=col, lw=1.1, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1e-15, 20)
    ax.set_xlim(-1.05, 1.15)
    ax.set_title(title)
    ax.grid(False)
    ax.set_xlabel("PrivLeak (0 = oracle)")
axes[0].set_ylabel("Forget Quality (log)")
for fam in ("NPO + retain", "SimNPO", "Calibrated NPO (fix)", "Match (fix)"):
    axes[3].scatter([], [], s=34, color=FAMCOL[fam], label=fam)
axes[3].scatter([], [], s=30, facecolor="white", edgecolor=MUT, label="utility < 90% of A")
axes[3].legend(frameon=True, fontsize=9, loc="lower left")
fig_t(lambda: fig.suptitle("Shaded box = retrain-like on both metrics; each panel uses its own retrained oracle",
                           x=0.01, ha="left", fontsize=10, color=MUT))
fig.tight_layout()
save(fig, "fig4_generalisation")

json.dump(T, open(f"{OUT}/v5_key_numbers.json", "w"), indent=1, default=float)
L = ["# v5 tables\n", f"Oracle noise ({T['noise']['n_pairs']} pairs of 5 retrained oracles): KS {T['noise']['ks_min']:.3f}-"
     f"{T['noise']['ks_max']:.3f}, FQ >= {T['noise']['fq_min']:.2f}, PrivLeak {T['noise']['pl_min']:+.3f} to "
     f"{T['noise']['pl_max']:+.3f}; oracle Min-K% AUC {T['noise']['oracle_auc']:.3f} "
     f"(95% CI {T['noise']['oracle_auc_ci'][0]:.3f}-{T['noise']['oracle_auc_ci'][1]:.3f})\n",
     f"Trajectories: {T['traj']['n_points']} points in {T['traj']['n_settings']} runs; in the box (Min-K%): "
     f"{len(T['traj']['box'])} points, {T['traj']['box_util']} of them with utility >= 90%; by attack: "
     f"{T['traj']['box_by_attack']}; with utility: {T['traj']['box_by_attack_util']}; points KS-equivalent to the oracle: "
     f"{T['traj']['ks_equiv_points']} (min KS {T['traj']['min_ks']:.3f})\n",
     "\n## Held-out seeds (run 9)\n", "| model | FQ | KS | PrivLeak [95% CI] | utility | KS within oracle noise |",
     "|---|---|---|---|---|---|"]
for h in held:
    L.append(f"| {h['name']} | {h['fq']:.3g} | {h['ks']:.3f} | "
             f"{h['pl_mink']:+.2f} [{h['pl_ci_lo']:+.2f}, {h['pl_ci_hi']:+.2f}] | {h['util']:.2f} | {h['ks_within_oracle_noise']} |")
L += ["\n## Llama-3.2-1B\n", "| setting | FQ | PrivLeak | utility |", "|---|---|---|---|"]
for e in sorted(llama, key=lambda e: e["name"]):
    L.append(f"| {e['run']} {e['name']} | {e['fq']:.3g} | {e['pl_mink']:+.2f} | {e['util']:.2f} |")
L += ["\n## Oracle-free stopping rule\n", "| run | epoch | FQ | PrivLeak | utility | TR |", "|---|---|---|---|---|---|"]
for s in stop:
    L.append(f"| {s['setting']} | {s['epoch']} | {s['fq']:.3g} | {s['pl_mink']:+.2f} | {s['util_nogen']:.2f} | {s['tr']:.2f} |")
open(f"{OUT}/v5_tables.md", "w").write("\n".join(L) + "\n")
print("\n".join(L[:4]))
print("written to", OUT)
