import glob, json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

R12, R24A, R24B, OUT = sys.argv[1:5]
J = lambda p: json.load(open(p))
auc = lambda m, n: float(roc_auc_score(np.r_[np.ones(len(m)), np.zeros(len(n))], np.r_[m, n]))

def loss_vec(raw):
    return np.r_[raw["mia"]["members"]["loss"], raw["mia"]["nonmembers"]["loss"]].astype(float), len(raw["mia"]["members"]["loss"])

def lira_score(x, X_ref, sig):
    v, n = loss_vec(x)
    z = np.abs((v - X_ref.mean(0)) / sig)
    return auc(z[:n], z[n:])

KEY = {}
SPLITS = {
    "forget05": ("95", f"{R12}/results/v6/run12/O95_s42_forget05_raw.json",
                 f"{R24A}/results/v6/run24a", "forget05"),
    "forget01": ("99", f"{R12}/results/v6/run12/O99_s42_forget01_raw.json",
                 f"{R24B}/results/v6/run24b", "forget01"),
}
for split, (tag, o42, rdir, f) in SPLITS.items():
    oracles = {"s42": J(o42)}
    for s in (1, 2, 3, 4):
        oracles[f"s{s}"] = J(f"{rdir}/O{tag}_s{s}_{f}_raw.json")
    names = list(oracles)
    lens = {len(loss_vec(oracles[k])[0]) for k in names}
    assert len(lens) == 1, f"oracle lengths differ for {split}: {lens}"
    X = np.array([loss_vec(oracles[k])[0] for k in names])
    sig = float((X - X.mean(0)).std(ddof=1))
    loo = {}
    for i, k in enumerate(names):
        keep = [j for j in range(len(names)) if j != i]
        loo[k] = round(lira_score(oracles[k], X[keep], sig), 3)
    targets = {}
    for p in sorted(glob.glob(f"{R12}/results/v6/run12/*{f}*_raw.json")):
        base = os.path.basename(p)[: -len("_raw.json")]
        if base.startswith(("O9", "A_s42")):
            continue
        targets[base] = round(lira_score(J(p), X, sig), 3)
    KEY[split] = dict(n_oracles=len(names), oracle_sig=round(sig, 4), loo_auc=loo,
                      loo_range=[min(loo.values()), max(loo.values())],
                      targets=targets)
    a = f"{R12}/results/v6/run12/A_s42_{f}_raw.json"
    if os.path.exists(a):
        KEY[split]["model_A_lira"] = round(lira_score(J(a), X, sig), 3)

json.dump(KEY, open(OUT, "w"), indent=1)
print(json.dumps(KEY, indent=1))
