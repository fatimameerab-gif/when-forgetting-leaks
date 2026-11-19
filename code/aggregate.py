import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["A_original", "B0_retrain", "B_ft_retain", "C_graddiff", "D_npo_rt", "E_ga", "F_simnpo"]
LABEL = {
    "A_original": "A: Original (finetuned on all)",
    "B0_retrain": "Oracle: Retrain w/o forget set",
    "B_ft_retain": "B: Fine-tune on retain",
    "C_graddiff": "C: Gradient Difference",
    "D_npo_rt": "D: NPO + retain",
    "E_ga": "Gradient Ascent",
    "F_simnpo": "SimNPO",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--original", default="A_original")
    ap.add_argument("--oracle", default="B0_retrain")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        if p.endswith("_raw.json"):
            continue
        r = json.load(open(p))
        if "forget" not in r:
            continue
        eff = r.get("efficiency") or {}
        rows.append({
            "name": r["name"],
            "forget_rouge": r["forget"]["rougeL_recall"],
            "forget_prob": r["forget"]["prob"],
            "forget_truth_ratio": r["forget"]["truth_ratio_mean"],
            "forget_quality": r.get("forget_quality"),
            "retain_rouge": r["retain"]["rougeL_recall"],
            "retain_prob": r["retain"]["prob"],
            "real_authors_rouge": r["real_authors"]["rougeL_recall"],
            "world_facts_rouge": r["world_facts"]["rougeL_recall"],
            "model_utility": r["model_utility"],
            "wikitext2_ppl": r.get("wikitext2_ppl"),
            "mia_auc_loss": r["privacy"]["auc_loss"],
            "mia_auc_mink": r["privacy"]["auc_mink"],
            "mia_auc_zlib": r["privacy"]["auc_zlib"],
            "privleak_mink": r["privacy"].get("privleak_mink"),
            "stage": eff.get("stage"),
            "wall_clock_min": (eff.get("wall_clock_sec") or np.nan) / 60,
            "approx_flops": eff.get("approx_flops"),
            "peak_mem_gb": eff.get("peak_mem_gb"),
            "gpu": eff.get("gpu"),
        })
    if not rows:
        raise SystemExit("no results found")
    df = pd.DataFrame(rows)
    df["_o"] = df["name"].map(lambda n: ORDER.index(n) if n in ORDER else 99)
    df = df.sort_values(["_o", "name"]).drop(columns="_o").reset_index(drop=True)

    base = df.loc[df.name == args.original, "forget_rouge"]
    base = float(base.iloc[0]) if len(base) else np.nan
    df["forgetting_pct"] = 100 * (1 - df["forget_rouge"] / base)
    orc = df.loc[df.name == args.oracle]
    if len(orc):
        df.loc[df.name == args.oracle, "forget_quality"] = 1.0
        df.loc[df.name == args.oracle, "privleak_mink"] = 0.0
        rt_min, rt_flops = float(orc.wall_clock_min.iloc[0]), orc.approx_flops.iloc[0]
        df["speedup_vs_retrain"] = rt_min / df["wall_clock_min"]
        if rt_flops:
            df["flops_pct_of_retrain"] = 100 * df["approx_flops"].astype(float) / float(rt_flops)
    df.to_csv(os.path.join(args.dir, "summary.csv"), index=False)

    def f(x, d=3):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"

    lines = [
        "| Model | Forget ROUGE-L ↓ | Forgetting % ↑ | Forget Quality ↑ | Retain ROUGE-L ↑ | "
        "Model Utility ↑ | MIA AUC (Min-K) | PrivLeak →0 | Time (min) | Speed-up vs retrain |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        fq = r.forget_quality
        fq_s = "—" if fq is None or (isinstance(fq, float) and np.isnan(fq)) else f"{fq:.2e}"
        lines.append(
            f"| {LABEL.get(r['name'], r['name'])} | {f(r.forget_rouge)} | {f(r.forgetting_pct, 1)} | "
            f"{fq_s} | {f(r.retain_rouge)} | {f(r.model_utility)} | {f(r.mia_auc_mink)} | "
            f"{f(r.get('privleak_mink'))} | {f(r.wall_clock_min, 1)} | "
            f"{f(r.get('speedup_vs_retrain'), 1)}× |"
        )
    open(os.path.join(args.dir, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, r in df.iterrows():
        fq = r.forget_quality if r.forget_quality and not np.isnan(r.forget_quality) else 1e-30
        ax.scatter(r.model_utility, max(fq, 1e-30), s=90, color=colors[i % 10],
                   label=LABEL.get(r["name"], r["name"]), edgecolor="k", zorder=3)
    ax.set_yscale("log")
    ax.axhline(0.05, ls="--", c="gray", lw=1)
    ax.text(ax.get_xlim()[0], 0.06, "p = 0.05", color="gray", fontsize=8)
    ax.set_xlabel("Model Utility (higher = better)")
    ax.set_ylabel("Forget Quality (log, higher = closer to retrain)")
    ax.set_title("Forgetting vs utility trade-off (TOFU forget10)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(args.dir, "fig_tradeoff.png"), dpi=180)

    metrics = [("forget_rouge", "Forget ROUGE-L ↓"), ("retain_rouge", "Retain ROUGE-L ↑"),
               ("model_utility", "Model Utility ↑"), ("mia_auc_mink", "MIA AUC (Min-K)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=False)
    names = [LABEL.get(n, n) for n in df.name]
    for ax, (m, t) in zip(axes, metrics):
        ax.bar(range(len(df)), df[m], color=[colors[i % 10] for i in range(len(df))])
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_title(t)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.dir, "fig_metrics.png"), dpi=180)

    eff = df.dropna(subset=["wall_clock_min"])
    eff = eff[eff.name != args.original]
    if len(eff):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.barh([LABEL.get(n, n) for n in eff.name], eff.wall_clock_min,
                color=[colors[(ORDER.index(n) if n in ORDER else 9) % 10] for n in eff.name])
        ax.set_xlabel("Wall-clock minutes (same GPU)")
        ax.set_title("Cost of unlearning vs. retraining from scratch")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.dir, "fig_efficiency.png"), dpi=180)
    print(f"\nwrote summary.csv, summary.md and figures to {args.dir}/")


if __name__ == "__main__":
    main()
