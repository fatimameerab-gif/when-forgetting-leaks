import argparse, csv, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ap = argparse.ArgumentParser(); ap.add_argument("--results", default="../results"); ap.add_argument("--out", default="../figures")
ap.add_argument("--paper", action="store_true", help="no figure titles (captions carry them)")
args = ap.parse_args()
R, OUT = os.path.join(args.results, "runs1-3_v3"), args.out; os.makedirs(OUT, exist_ok=True)
J = lambda n: json.load(open(f"{R}/{n}.json"))
def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
def figtitle(target, *a, **k):
    if not args.paper:
        target(*a, **k)
import sys as _s; _s.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paper_style
INK, MUT, REF = "#0b0b0b", "#6F6A60", "#8a8984"
COL = {"npo_rt": "#2F7F93", "simnpo": "#C2577A", "graddiff": "#8A9A5B", "ga": "#5B6C9E", "npo": "#A39E93", "ft_retain": "#6F6A60"}
NAME = {"npo_rt": "NPO + retain", "simnpo": "SimNPO", "graddiff": "Gradient Difference", "ga": "Gradient Ascent",
        "npo": "NPO w/o retain (ablation)", "ft_retain": "Fine-tune on retain (B)"}
ORDER = ["npo_rt", "simnpo", "graddiff", "ga", "npo", "ft_retain"]
sel = json.load(open(f"{R}/selection.json"))["selected"]
A, O, O2 = J("A_original"), J("B0_retrain"), J("B0_retrain_s1")
noise = json.load(open(f"{R}/oracle_noise.json"))
lr = lambda x: f"{x:g}".replace("e-0", "e-")
def fn(s, seed, q=False): return f"fn_{s['key']}_lr{lr(s['lr'])}_ep{s['epochs']}_s{seed}" + ("_4bit" if q else "")
held = {m: [fn(s, x) for x in (1, 2, 3)] for m, s in sel.items()}
held["ft_retain"] = [f"sd_ft_retain_s{x}" for x in (1, 2, 3)]
FQ = lambda r: r["forget_quality"]; MU = lambda r: r["model_utility"]
AUC = lambda r: r["privacy"]["auc_mink"]; FR = lambda r: r["forget"]["rougeL_recall"]
PL = lambda r: r["privacy"]["privleak_mink"]
rows = list(csv.DictReader(open(f"{R}/sweep_summary.csv")))
thr = 0.9 * A["model_utility"]
oracle_pl2 = (O2["privacy"]["auc_mink"] - O["privacy"]["auc_mink"]) / O["privacy"]["auc_mink"]

fig, (ax, ins) = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [1.35, 1]})
ax.add_patch(Rectangle((-0.2, 0.05), 0.4, 0.95, facecolor="#B5523B", alpha=0.13, lw=1, edgecolor="#B5523B", zorder=0))
ax.axhline(0.05, color=REF, ls="--", lw=1); ax.axvline(0, color=REF, ls="--", lw=1)
for r in rows:
    fam = r["family"]; ok = float(r["model_utility"]) >= thr
    ax.scatter(float(r["privleak"]), max(float(r["forget_quality"] or 0), 1e-30), s=34 if ok else 22,
               facecolor=COL[fam] if ok else "white", edgecolor=COL[fam], lw=1.1, alpha=0.8, zorder=3)
on = os.path.join(args.results, "..", "figures", "v5_oracle_noise.csv")
on = on if os.path.exists(on) else os.path.join(OUT, "v5_oracle_noise.csv")
if os.path.exists(on):
    pr = list(csv.DictReader(open(on)))
    ax.scatter([float(x["pl_mink"]) for x in pr], [min(float(x["fq"]), 1.0) for x in pr], s=120, marker="*",
               color="#1F3A5F", edgecolor="black", lw=0.6, zorder=6)
    ax.scatter([], [], s=90, marker="*", color="#1F3A5F", edgecolor="black", lw=0.6,
               label="retrained oracles (10 pairs of 5 seeds)")
else:
    ax.scatter(oracle_pl2, noise["oracle_seed1_vs_oracle_seed42_FQ"], s=220, marker="*", color="#1F3A5F",
               edgecolor="black", lw=0.8, zorder=6, label="second retrained oracle")
for fam in ["npo_rt", "simnpo", "graddiff", "ga", "npo"]:
    ax.scatter([], [], s=34, color=COL[fam], label=NAME[fam])
ax.scatter([], [], s=30, facecolor="white", edgecolor=MUT, label="utility < 90% of Model A")
ax.set_yscale("log"); ax.set_ylim(1e-31, 2.5); ax.set_xlim(-1.05, 1.45)
ax.set_xlabel("PrivLeak  (− over-unlearned  ·  0 = oracle  ·  + still memorised)")
ax.set_ylabel("Forget Quality (log)")
figtitle(ax.set_title, "All 57 settings: high Forget Quality only appears with over-unlearning", pad=10)
ax.legend(fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0.58, 0.02)); ax.grid(lw=0.6)
ins.add_patch(Rectangle((-0.2, 0.05), 0.4, 0.95, facecolor="#B5523B", alpha=0.15, lw=1, edgecolor="#B5523B", zorder=0))
ins.text(0.0, 0.45, "retrain-like\non both", ha="center", va="top", fontsize=8.5, color="#B5523B")
for r in rows:
    fam = r["family"]; ok = float(r["model_utility"]) >= thr
    ins.scatter(float(r["privleak"]), max(float(r["forget_quality"] or 0), 1e-30), s=34 if ok else 22,
                facecolor=COL[fam] if ok else "white", edgecolor=COL[fam], lw=1, alpha=0.85, zorder=3)
if os.path.exists(on):
    ins.scatter([float(x["pl_mink"]) for x in pr], [min(float(x["fq"]), 1.0) for x in pr], s=120, marker="*",
                color="#1F3A5F", edgecolor="black", lw=0.5, zorder=6)
ins.axhline(0.05, color=REF, ls="--", lw=0.8); ins.axvline(0, color=REF, ls="--", lw=0.8)
ins.set_yscale("log"); ins.set_ylim(2e-5, 2.0); ins.set_xlim(-1.02, 0.45); ins.grid(lw=0.6)
ins.set_xlabel("PrivLeak"); ins.set_ylabel("Forget Quality (log)")
ax.add_patch(Rectangle((-1.04, 2e-5), 1.49, 2.0, fill=False, ls=":", lw=1, edgecolor=MUT, zorder=1))
ax.text(-1.0, 1e-5, "(b)", fontsize=9, color=MUT, va="top")
ax.text(0.0, 1.03, "(a)", transform=ax.transAxes, fontsize=11, va="bottom")
ins.text(0.0, 1.03, "(b)", transform=ins.transAxes, fontsize=11, va="bottom")
fig.tight_layout(); save(fig, "fig1_fq_vs_privleak")

fig, axes = plt.subplots(1, 4, figsize=(14, 3.9), sharey=True)
panels = [("Forget Quality ↑ (log)", FQ, True, 0.05, "0.05"), ("Model Utility ↑", MU, False, A["model_utility"], "Model A"),
          ("Forget ROUGE-L (target = oracle)", FR, False, O["forget"]["rougeL_recall"], "oracle"),
          ("MIA AUC, Min-K% (target = oracle)", AUC, False, O["privacy"]["auc_mink"], "oracle")]
for ax, (title, f, logx, ref, lab) in zip(axes, panels):
    ax.axvline(ref, color=REF, ls="--", lw=1)
    for i, m in enumerate(ORDER):
        v = np.array([f(J(n)) for n in held[m]], float)
        if logx: v = np.maximum(v, 1e-25)
        c = COL[m]
        ax.plot([v.min(), v.max()], [i, i], color=c, lw=2, alpha=0.5)
        ax.scatter(v, [i] * 3, s=18, color=c, alpha=0.5, lw=0, zorder=3)
        mid = np.median(v)
        ax.scatter(mid, i, s=55, color=c, edgecolor="white", lw=1.2, zorder=4)
        if m in sel:
            s42 = f(J(sel[m]["name"])); s42 = max(s42, 1e-25) if logx else s42
            ax.scatter(s42, i, s=36, marker="D", facecolor="white", edgecolor=c, lw=1.2, zorder=5)
    if logx: ax.set_xscale("log"); ax.set_xlim(1e-25, 3)
    ax.set_title(title); ax.grid(axis="x", lw=0.6)
    ax.text(ref, len(ORDER) - 0.45, lab, fontsize=7.5, color=MUT, ha="center")
axes[0].set_yticks(range(len(ORDER))); axes[0].set_yticklabels([NAME[m] for m in ORDER]); axes[0].set_ylim(len(ORDER) - 0.3, -0.6)
figtitle(fig.suptitle, "Held-out seeds 1–3 (big dot = median, bar = range); ◇ = seed 42, used only to choose the setting",
             x=0.01, ha="left", fontsize=10, color=MUT)
fig.tight_layout(); save(fig, "fig2_heldout_seeds")

fams = ["npo_rt", "simnpo", "graddiff", "ga", "npo"]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.3), sharey=True)
mets = [("Forget ROUGE-L", FR, O["forget"]["rougeL_recall"], False), ("Forget Quality (log)", FQ, 0.05, True),
        ("MIA AUC (Min-K%)", AUC, O["privacy"]["auc_mink"], False), ("Model Utility", MU, A["model_utility"], False)]
for ax, (t, f, ref, logx) in zip(axes, mets):
    ax.axvline(ref, color=REF, ls="--", lw=1)
    for i, m in enumerate(fams):
        s = sel[m]
        full = np.median([f(J(fn(s, x))) for x in (1, 2, 3)]); q = np.median([f(J(fn(s, x, True))) for x in (1, 2, 3)])
        if logx: full, q = max(full, 1e-25), max(q, 1e-25)
        ax.annotate("", xy=(q, i), xytext=(full, i), arrowprops=dict(arrowstyle="->", color=COL[m], lw=1.6))
        ax.scatter(full, i, s=50, color=COL[m], edgecolor="white", zorder=3)
        ax.scatter(q, i, s=42, facecolor="white", edgecolor=COL[m], lw=1.5, zorder=3)
    if logx: ax.set_xscale("log")
    ax.set_title(t); ax.grid(axis="x", lw=0.6)
axes[0].set_yticks(range(len(fams))); axes[0].set_yticklabels([NAME[m] for m in fams]); axes[0].set_ylim(len(fams) - 0.5, -0.6)
figtitle(fig.suptitle, "Full precision (filled) → 4-bit (hollow), median of held-out seeds 1–3", x=0.01, ha="left", fontsize=10, color=MUT)
fig.tight_layout(); save(fig, "fig3_quantization")

def ms(v, d=3): v = np.asarray(v, float); return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"
L = ["| Method (setting chosen on seed 42) | FQ median [min–max] | seeds FQ > 0.05 | FQ vs 2nd oracle (median) | Model Utility | Forget ROUGE-L | MIA AUC | PrivLeak | Min |",
     "|---|---|---|---|---|---|---|---|---|"]
L.append(f"| Oracle (retrain, seed 42) | 1 (ref.) | — | {noise['oracle_seed1_vs_oracle_seed42_FQ']:.2f}* | {MU(O):.3f} | {FR(O):.3f} | {AUC(O):.3f} | 0 | {O['efficiency']['wall_clock_sec']/60:.0f} |")
L.append(f"| A: Original | {FQ(A):.1e} | — | — | {MU(A):.3f} | {FR(A):.3f} | {AUC(A):.3f} | {PL(A):+.2f} | — |")
for m in ORDER:
    rs = [J(n) for n in held[m]]
    fq = np.array([FQ(r) for r in rs]); fq2 = [r for r in csv.DictReader(open(f"{R}/final_runs.csv")) if r["name"] in held[m]]
    fq2 = np.median([float(r["fq_vs_oracle2"]) for r in fq2])
    s = sel.get(m)
    lab = NAME[m] + (f" (lr {s['lr']:g}, {s['epochs']} ep)" if s else " (lr 1e-5, 1 ep)") + ("" if (not s or s["met_utility_threshold"]) else "†")
    L.append(f"| {lab} | {np.median(fq):.2g} [{fq.min():.1g}–{fq.max():.1g}] | {int((fq > 0.05).sum())}/3 | {fq2:.2g} | {ms([MU(r) for r in rs])} | "
             f"{ms([FR(r) for r in rs])} | {ms([AUC(r) for r in rs])} | {ms([PL(r) for r in rs], 2)} | {np.mean([r['efficiency']['wall_clock_sec'] for r in rs])/60:.0f} |")
open(f"{OUT}/table_heldout.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))


def load_split(d):
    out = {}
    for p in sorted(os.listdir(d)):
        if p.endswith(".json") and not p.endswith("_raw.json"):
            r = json.load(open(os.path.join(d, p)))
            if "forget" in r:
                out[r["name"]] = r
    return out
panels = [("Qwen2.5-0.5B, forget 5%", os.path.join(args.results, "run4_scale", "forget05")),
          ("Qwen2.5-0.5B, forget 1%", os.path.join(args.results, "run4_scale", "forget01")),
          ("TinyLlama-1.1B, forget 10%", os.path.join(args.results, "run5_tinyllama", "tinyllama"))]
low = os.path.join(args.results, "run6_tinyllama_low", "tinyllama_low")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
gen_rows = []
for ax, (title, d) in zip(axes, panels):
    rs = load_split(d)
    if title.startswith("TinyLlama") and os.path.isdir(low):
        rs.update({k: v for k, v in load_split(low).items() if k.startswith("sw_")})
    a_mu = rs["A_original"]["model_utility"]
    ax.add_patch(Rectangle((-0.2, 0.05), 0.4, 20, color="#B5523B", alpha=0.08, lw=0, zorder=0))
    ax.axhline(0.05, color=REF, ls="--", lw=1); ax.axvline(0, color=REF, ls="--", lw=1)
    for name, r in rs.items():
        if not name.startswith("sw_"):
            continue
        fam = "npo_rt" if "npo_rt" in name else "simnpo"
        ok = r["model_utility"] >= 0.9 * a_mu
        ax.scatter(r["privacy"]["privleak_mink"], max(r["forget_quality"], 1e-30), s=40 if ok else 26,
                   facecolor=COL[fam] if ok else "white", edgecolor=COL[fam], lw=1.2, zorder=3)
        gen_rows.append((title, name, r["forget_quality"], r["privacy"]["privleak_mink"], r["model_utility"] / a_mu))
    ax.set_yscale("log"); ax.set_ylim(1e-15, 20); ax.set_xlim(-1.05, 1.15)
    ax.set_title(title); ax.grid(lw=0.6); ax.set_xlabel("PrivLeak (0 = oracle)")
axes[0].set_ylabel("Forget Quality (log)")
for fam in ("npo_rt", "simnpo"):
    axes[0].scatter([], [], s=40, color=COL[fam], label=NAME[fam])
axes[0].scatter([], [], s=30, facecolor="white", edgecolor=MUT, label="utility < 90% of Model A")
axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
figtitle(fig.suptitle, "Shaded box = retrain-like on both metrics; each panel uses its own retrained oracle",
             x=0.01, ha="left", fontsize=10, color=MUT)
fig.tight_layout(); save(fig, "fig4_generalisation")
L = ["| Condition | Setting | Forget Quality | PrivLeak | Utility / Model A | Retrain-like on both |", "|---|---|---|---|---|---|"]
for t, n, f, p, u in gen_rows:
    L.append(f"| {t} | {n} | {f:.2g} | {p:+.2f} | {u:.2f} | {'yes' if (f > 0.05 and abs(p) <= 0.2) else 'no'} |")
open(f"{OUT}/table_generalisation.md", "w").write("\n".join(L) + "\n")

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.axvspan(thr, 0.5, color="#B5523B", alpha=0.05, lw=0); ax.grid(lw=0.6)
for r in rows:
    fam = r["family"]; filled = r["epochs"] != "5"
    ax.scatter(float(r["model_utility"]), max(float(r["forget_quality"] or 0), 1e-30), s=28,
               facecolor=COL[fam] if filled else "white", edgecolor=COL[fam], lw=1.1, alpha=0.75)
for m in ("npo_rt", "simnpo", "graddiff", "ga", "npo"):
    v_mu = np.median([MU(J(n)) for n in held[m]]); v_fq = np.median([max(FQ(J(n)), 1e-30) for n in held[m]])
    ax.scatter(v_mu, v_fq, s=160, marker="*", color=COL[m], edgecolor="black", lw=0.8, zorder=5)
    off = {"npo_rt": (6, 12, "left"), "simnpo": (-10, 12, "right"), "ga": (-12, 10, "right"),
           "npo": (10, -16, "left"), "graddiff": (12, -2, "left")}[m]
    ax.annotate(NAME[m], (v_mu, v_fq), xytext=off[:2], textcoords="offset points", fontsize=8, ha=off[2])
ax.set_yscale("log"); ax.set_ylim(1e-31, 1e3); ax.set_xlim(-0.01, 0.5); ax.axhline(0.05, color=REF, ls="--", lw=1)
ax.set_xlabel("Model Utility"); ax.set_ylabel("Forget Quality (log)")
figtitle(ax.set_title, "All 57 settings (hollow = epoch 5); ★ = chosen setting, median of held-out seeds")
ax.text(thr + 0.003, 1e-29, "utility ≥ 90% of A", fontsize=8, color=MUT)
for m in ("npo_rt", "simnpo", "graddiff", "ga", "npo"):
    ax.scatter([], [], s=28, color=COL[m], label=NAME[m])
ax.legend(fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.08))
fig.tight_layout(); save(fig, "fig5_utility_tradeoff")
print("figures written to", OUT)
