import argparse, csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from paper_style import *

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="../results"); ap.add_argument("--out", default="../figures")
args = ap.parse_args(); OUT = args.out
NONMEM = "#a0a0a0"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


x = np.linspace(-5.2, 0.4, 400)
g = lambda mu, sd: np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
states = [("(a) Model A", -0.9, 0.35, "PrivLeak = +1.36", "#555555"),
          ("(b) NPO checkpoint that passes FQ", -3.95, 0.55, "PrivLeak = $-$0.80", "#1f77b4"),
          ("(c) Anchor", -2.55, 0.55, "PrivLeak $\\approx$ 0", "#d62728")]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
for ax, (title, mu, sd, pl, col) in zip(axs, states):
    ax.plot(x, g(-2.2, 0.6), color="#7f7f7f", lw=1.2)
    ax.plot(x, g(-2.53, 0.6), color="black", lw=1.0, ls="--")
    ax.plot(x, g(mu, sd), color=col, lw=1.6)
    ax.set_xlim(-5.2, 0.4); ax.set_ylim(0, 1.3); ax.set_yticks([])
    ax.set_title(title)
    ax.text(0.04, 0.93, pl, transform=ax.transAxes, va="top", fontsize=9)
axs[0].set_ylabel("density"); axs[1].set_xlabel("log-likelihood of the answer (nats/token)")
from matplotlib.lines import Line2D
hs = [Line2D([], [], color="#7f7f7f", lw=1.2, label="never-seen authors"),
      Line2D([], [], color="black", lw=1.0, ls="--", label="forget set, retrained oracle"),
      Line2D([], [], color="#333333", lw=1.6, label="forget set, model in the panel")]
fig.legend(handles=hs, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1))
fig.tight_layout(); save(fig, "fig_concept")

fig, ax = plt.subplots(figsize=(7.6, 2.6)); ax.set_xlim(0, 113); ax.set_ylim(0, 40); ax.axis("off")
from matplotlib.patches import Rectangle


def box(x0, y0, w, h, text, fc="white"):
    ax.add_patch(Rectangle((x0, y0), w, h, fc=fc, ec="black", lw=0.8))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=8.5, linespacing=1.3)


def arrow(p, q, txt=None, dx=0, dy=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color="black", lw=0.8,
                                 connectionstyle=f"arc3,rad={rad}"))
    if txt:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, txt, ha="center", va="bottom", fontsize=8)


box(1, 26, 11, 10, "Retain set\n$D_r$")
box(21, 26, 17, 10, "Proxy model $P$\n(10% of\nretain authors)", fc="#eeeeee")
box(44, 26, 26, 10, "Targets $t_i = \\mathrm{NLL}_P(y_i)$\nfor each forget answer")
box(21, 4, 17, 10, "Model A")
box(44, 4, 26, 10, "Anchored unlearning\n$(\\mathrm{nll}_\\theta(y_i)-t_i)^2 + \\mathrm{NLL}_\\theta(D_r)$", fc="#eeeeee")
box(76, 4, 16, 10, "Checkpoints\n(epoch, lr)")
box(97, 4, 15, 10, "Select with $P$\n(no oracle)", fc="#eeeeee")
arrow((12, 31), (21, 31), "fine-tune", dy=0.8)
arrow((38, 31), (44, 31), "score", dy=0.8)
arrow((57, 26), (57, 14))
arrow((38, 9), (44, 9))
arrow((70, 9), (76, 9))
arrow((92, 9), (97, 9))
save(fig, "fig_framework")

rows = list(csv.DictReader(open(f"{OUT}/v7_checkpoints.csv")))
K = json.load(open(f"{OUT}/v7_key_numbers.json"))
rep = [r for r in rows if "run17" in r["sub"]]
base = [r for r in rows if r["proposed"] == "False" and "run17" not in r["sub"] and float(r["fq"]) > 0.05]
C_TERRA, C_PEACH, C_TEAL, C_BOX = "#1f77b4", "#ff7f0e", "#9ecae1", "#eeeeee"
fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
from matplotlib.patches import Rectangle
a.add_patch(Rectangle((-0.2, 0.05), 0.4, 1.45, facecolor=C_BOX, edgecolor="none", zorder=0))
a.scatter([float(r["pl"]) for r in base], [float(r["fq"]) for r in base], marker="o", s=26, facecolor="white", edgecolor=SAND,
          lw=1, label="baselines passing FQ (n=%d)" % len(base), zorder=2)
for ep, mk, c in (("ep5", "D", C_TERRA), ("ep10", "s", C_PEACH)):
    rs = [r for r in rep if f"_{ep}_" in r["name"]]
    a.scatter([float(r["pl"]) for r in rs], [max(float(r["fq"]), 1e-4) for r in rs], marker=mk, s=40, color=c, edgecolor=INK,
              lw=0.5, zorder=4, label=f"Anchor, epoch {ep[2:]} (8 replicates)")
a.axhline(0.05, color="#7f7f7f", ls="--", lw=0.7); a.axvline(0, color="#7f7f7f", ls="--", lw=0.7)
a.set_yscale("log"); a.set_ylim(3e-4, 1.5); a.set_xlim(-1.0, 0.4); a.set_xlabel("PrivLeak (Min-K%)"); a.set_ylabel("Forget Quality")
a.legend(loc="lower left", fontsize=7.5)
panel_label(a, "(a) Replicates and FQ-passing baselines")
labs = ["|PrivLeak| $\\leq$ 0.2", "PL 90% CI in $\\pm$0.2", "FQ > 0.05", "Box (FQ and PL)", "Joint test (RQ)", "Box, proxy-chosen epoch"]
E5 = K["replicates"]["ep5"]
vals = [E5["pl_band"], [int(E5["pl_tost_within_0_2"]), 8], E5["fq"], E5["box"], E5["rq"], K["replicates"]["proxy_epoch_choice_p10"]["box"]]
from scipy.stats import beta
def ci(k, n): return (0 if k == 0 else beta.ppf(.025, k, n - k + 1), 1 if k == n else beta.ppf(.975, k + 1, n - k))
ypos = np.arange(len(labs))[::-1]
for y, (lab, v) in zip(ypos, zip(labs, vals)):
    k, n = int(v[0]), int(v[1]); lo, hi = ci(k, n); p = k / n
    c = TERRA if y >= 4 else (PEACH if y != 0 else TERRA)
    bc = C_TEAL if lab == "Box, proxy-chosen epoch" else (C_TERRA if lab.startswith(("|PrivLeak|", "PL 90%")) else C_PEACH)
    bc = C_TEAL if lab == "Box, proxy-chosen epoch" else C_TERRA
    b.barh(y, p, color=bc, alpha=0.9, height=0.6, lw=0)
    b.errorbar(p, y, xerr=[[p - lo], [hi - p]], color=INK, capsize=3, lw=1)
    b.text(1.02, y, f"{k}/{n}", va="center", fontsize=8.5)
b.set_yticks(ypos); b.set_yticklabels(labs, fontsize=8.5); b.set_xlim(0, 1.12); b.set_xlabel("share of replicates (95% exact CI)")
b.axvline(0, color="#7f7f7f", lw=0.6)
panel_label(b, "(b) Rates over 8 replicates")
fig.tight_layout(); save(fig, "fig_replicates")
print("ok")
