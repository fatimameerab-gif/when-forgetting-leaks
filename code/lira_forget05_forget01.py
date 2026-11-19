import json, glob, os, sys, numpy as np
from sklearn.metrics import roc_auc_score
R, U = sys.argv[1], sys.argv[2]
def J(p): return json.load(open(p))
def auc(a, b): return float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))
def setup(ors):
    X = np.array([np.r_[o["mia"]["members"]["loss"], o["mia"]["nonmembers"]["loss"]] for o in ors], float)
    return X, float((X - X.mean(0)).std(ddof=1))
def lira(raw, st, drop=None):
    X, sig = st; keep = [i for i in range(len(X)) if i != drop]
    mu = X[keep].mean(0); s = np.r_[raw["mia"]["members"]["loss"], raw["mia"]["nonmembers"]["loss"]]
    z = np.abs((np.asarray(s, float) - mu) / sig); n = len(raw["mia"]["members"]["loss"])
    return auc(z[:n], z[n:])
out = {}
for f, tag, part in (("forget05", "95", "a"), ("forget01", "99", "b")):
    ors = [J(f"{R}/run12_v6/run12/O{tag}_s42_{f}_raw.json")] + \
          [J(glob.glob(f"{U}/**/run24{part}/O{tag}_s{k}_{f}_raw.json", recursive=True)[0]) for k in (1, 2, 3, 4)]
    st = setup(ors)
    orc = [lira(o, st, i) for i, o in enumerate(ors)]
    ck = {}
    for p in sorted(glob.glob(f"{R}/run12_v6/run12/*_{f}_*_raw.json")) + sorted(glob.glob(f"{R}/run12_v6/run12/A_s42_{f}_raw.json")):
        nm = os.path.basename(p)[:-9]
        ck[nm] = lira(J(p), st)
    out[f] = dict(oracles_loo=[round(x, 3) for x in orc], checkpoints={k: round(v, 3) for k, v in ck.items()})
print(json.dumps(out, indent=1))
