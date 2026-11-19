import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
FORGET, RETAIN = os.environ.get("FORGET", "forget10"), os.environ.get("RETAIN", "retain90")
DATA = os.environ.get("DATA", "data")
CK = os.environ.get("CKPT_DIR", "ckpt")
RES = os.environ.get("RES_DIR", "results/v3")
LOGS = os.environ.get("LOG_DIR", "logs/v3")
EPOCHS = int(os.environ.get("UL_EPOCHS", "10"))
EARLY = int(os.environ.get("EARLY_EPOCH", "5"))
UL_BS, UL_GA = os.environ.get("UL_BS", "4"), os.environ.get("UL_GA", "8")
MAX_HOURS = float(os.environ.get("MAX_HOURS", "11"))
LRS = [float(x) for x in os.environ.get("LRS", "5e-6,1e-5,2e-5").split(",")]
GA_LRS = [float(x) for x in os.environ.get("GA_LRS", "1e-6,5e-6,1e-5").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1,2").split(",")]
KEEP_DIR = os.environ.get("KEEP_DIR", "/kaggle/working/keep_ckpt" if os.path.isdir("/kaggle/working") else "")
EVAL_EXTRA = os.environ.get("EVAL_EXTRA", "").split()

SIMNPO_REF = ["--beta", "4.5", "--forget_weight", "0.125", "--gamma", "0", "--alpha", "1.0"]
METHODS = {
    "graddiff": ("graddiff", "graddiff", ["--alpha", "1.0"], LRS),
    "npo_rt": ("npo_rt", "npo_rt", ["--beta", "0.1", "--alpha", "1.0"], LRS),
    "npo": ("npo", "npo", ["--beta", "0.1"], LRS),
    "simnpo": ("simnpo", "simnpo", SIMNPO_REF, LRS),
    "ga": ("ga", "ga", [], GA_LRS),
}
EXTEND = {
    "graddiff": ("graddiff", "graddiff", ["--alpha", "1.0"], [1e-6, 2e-6]),
    "ga": ("ga", "ga", [], [2e-6]),
    "simnpo": ("simnpo", "simnpo", SIMNPO_REF, [5e-5]),
    "simnpo_fw1_b4.5": ("simnpo", "simnpo", ["--beta", "4.5", "--forget_weight", "1.0", "--gamma", "0", "--alpha", "1.0"], [2e-5]),
    "simnpo_fw1_b2.5": ("simnpo", "simnpo", ["--beta", "2.5", "--forget_weight", "1.0", "--gamma", "0", "--alpha", "1.0"], [2e-5]),
}
SIMNPO_B25 = ["--beta", "2.5", "--forget_weight", "1.0", "--gamma", "0", "--alpha", "1.0"]
EXTEND2 = {
    "npo_rt": ("npo_rt", "npo_rt", ["--beta", "0.1", "--alpha", "1.0"], [5e-5]),
    "simnpo_fw1_b2.5": ("simnpo", "simnpo", SIMNPO_B25, [5e-5]),
    "simnpo_fw1_b2.5_e15": ("simnpo", "simnpo", SIMNPO_B25, [2e-5], 15),
    "npo": ("npo", "npo", ["--beta", "0.1"], [1e-6, 2e-6]),
    "graddiff": ("graddiff", "graddiff", ["--alpha", "1.0"], [1.5e-6]),
    "ga": ("ga", "ga", [], [1.5e-6]),
}
FAMILIES = ["graddiff", "npo_rt", "npo", "simnpo", "ga"]
ORACLE2, ORACLE2_SEED = "B0_retrain_s1", 1
FINAL_SEEDS = [int(x) for x in os.environ.get("FINAL_SEEDS", "1,2,3").split(",")]


def configs(*tables):
    for table in tables:
        for key, spec in table.items():
            fam, method, args, grid = spec[:4]
            total = spec[4] if len(spec) > 4 else EPOCHS
            for lr in grid:
                yield key, fam, method, args, lr, total


def eval_points(total):
    return sorted({e for e in (EARLY, 2 * EARLY) if e < total} | {total})
T0 = time.time()
FAILED = []
LOCK = threading.Lock()


def say(msg):
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def lr_tag(lr):
    return f"{lr:g}".replace("e-0", "e-")


def done_ck(path):
    return os.path.exists(os.path.join(path, "train_stats.json"))


def done_ev(name):
    return os.path.exists(os.path.join(RES, f"{name}.json"))


def ngpu():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        return max(1, len([l for l in out.splitlines() if l.strip()]))
    except FileNotFoundError:
        return 1


def run_step(gpu, logname, cmd):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    say(f"START  {logname}  (GPU {gpu})")
    t = time.time()
    with open(os.path.join(LOGS, f"{logname}.log"), "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE).returncode
    if rc == 0:
        say(f"DONE   {logname}  ({(time.time() - t) / 60:.1f} min)")
        return True
    say(f"FAILED {logname}  -> {LOGS}/{logname}.log")
    with LOCK:
        FAILED.append(logname)
    return False


def eval_cmd(ckpt, name, *extra):
    return [PY, "evaluate.py", "--model", ckpt, "--name", name, "--forget", FORGET,
            "--data_dir", DATA, "--out_dir", RES, "--ref_raw", f"{RES}/B0_retrain_raw.json",
            *extra, *EVAL_EXTRA]


def unlearn_cmd(m, extra, lr, seed, out, epochs=EPOCHS, save_early=True, save_list=None):
    cmd = [PY, "unlearn.py", "--model", f"{CK}/A_original", "--method", m, "--out", out,
           "--forget", FORGET, "--retain", RETAIN, "--data_dir", DATA, "--epochs", str(epochs),
           "--lr", str(lr), "--bs", UL_BS, "--grad_accum", UL_GA, "--seed", str(seed), *extra]
    if m != "ft_retain":
        cmd += ["--warmup_epochs", "1.0"]
    if save_list:
        cmd += ["--save_epochs", ",".join(str(e) for e in save_list)]
    elif save_early and EARLY < epochs:
        cmd += ["--save_epochs", str(EARLY)]
    return cmd


def run_jobs(jobs, n):
    queue = list(jobs)

    def worker(gpu):
        while True:
            with LOCK:
                if not queue:
                    return
                job = queue.pop(0)
            trained = not job.get("train") or done_ck(job["train"][2])
            if all(done_ev(e[0]) for e in job["evals"]) and (job["evals"] or trained):
                continue
            if (time.time() - T0) / 3600 > MAX_HOURS:
                say(f"SKIP   {job['name']}: time budget of {MAX_HOURS} h reached")
                continue
            ok = True
            if job.get("train") and not done_ck(job["train"][2]):
                ok = run_step(gpu, job["train"][0], job["train"][1])
            if ok:
                for name, ckpt, extra in job["evals"]:
                    if not done_ev(name) and done_ck(ckpt):
                        run_step(gpu, f"eval_{name}", eval_cmd(ckpt, name, *extra))
            for d in job.get("cleanup", []):
                shutil.rmtree(d, ignore_errors=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def find_input(pattern):
    for root in ("/kaggle/input", os.environ.get("EXTRA_INPUT", "")):
        if root and os.path.isdir(root):
            hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
            if hits:
                return hits[0]
    return None


def prepare(n, oracle2=False):
    os.makedirs(CK, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    if not os.path.isdir(DATA):
        subprocess.run([PY, "download_data.py", "--out", DATA], cwd=HERE, check=True)
    hit = find_input("keep_ckpt/A_original/train_stats.json")
    if hit:
        keep = os.path.dirname(os.path.dirname(hit))
        for m in ("A_original", "B0_retrain"):
            if not done_ck(f"{CK}/{m}") and done_ck(f"{keep}/{m}"):
                say(f"re-using {m} from {keep}")
                shutil.copytree(f"{keep}/{m}", f"{CK}/{m}", dirs_exist_ok=True)
    ft = [("A_original", "full", 42, True)]
    ft.append(("B0_retrain", RETAIN, 42, not done_ev("B0_retrain")))
    if oracle2:
        ft.append((ORACLE2, RETAIN, ORACLE2_SEED, not done_ev(ORACLE2)))
    jobs = [{"name": m, "train": (m, [PY, "train_ft.py", "--model", MODEL, "--subset", sub, "--out",
                                      f"{CK}/{m}", "--data_dir", DATA, "--epochs", "5", "--bs", "8",
                                      "--grad_accum", "4", "--seed", str(sd)], f"{CK}/{m}"), "evals": []}
            for m, sub, sd, need in ft if need and not done_ck(f"{CK}/{m}")]
    if jobs:
        say("training: " + ", ".join(j["name"] for j in jobs))
        run_jobs(jobs, n)
    if not done_ck(f"{CK}/A_original"):
        sys.exit("Model A missing - cannot continue")
    if not (done_ck(f"{CK}/B0_retrain") or done_ev("B0_retrain")):
        sys.exit("Oracle missing - cannot continue")
    if KEEP_DIR:
        for m in ("A_original", "B0_retrain"):
            if done_ck(f"{CK}/{m}") and not done_ck(f"{KEEP_DIR}/{m}"):
                shutil.copytree(f"{CK}/{m}", f"{KEEP_DIR}/{m}", dirs_exist_ok=True)
                say(f"kept {m} -> {KEEP_DIR}")
    if not done_ev("B0_retrain"):
        run_step(0, "eval_B0_retrain", eval_cmd(f"{CK}/B0_retrain", "B0_retrain"))
    if not done_ev("A_original"):
        run_step(0, "eval_A_original", eval_cmd(f"{CK}/A_original", "A_original"))
    if oracle2 and not done_ev(ORACLE2) and done_ck(f"{CK}/{ORACLE2}"):
        run_step(0, f"eval_{ORACLE2}", eval_cmd(f"{CK}/{ORACLE2}", ORACLE2))


def load(name):
    with open(os.path.join(RES, f"{name}.json")) as f:
        return json.load(f)


def row_of(name, **meta):
    r = load(name)
    eff = r.get("efficiency") or {}
    return dict(meta, name=name, forget_rouge=r["forget"]["rougeL_recall"],
                forget_prob=r["forget"]["prob"], forget_quality=r.get("forget_quality"),
                retain_rouge=r["retain"]["rougeL_recall"], model_utility=r["model_utility"],
                real_authors_rouge=r["real_authors"]["rougeL_recall"],
                world_facts_rouge=r["world_facts"]["rougeL_recall"],
                mia_auc_mink=r["privacy"]["auc_mink"], privleak=r["privacy"].get("privleak_mink"),
                minutes=(eff.get("wall_clock_sec") or float("nan")) / 60,
                flops=eff.get("approx_flops"))


def write_csv(rows, path):
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def sweep_name(m, lr, ep, seed=42):
    return f"sw_{m}_lr{lr_tag(lr)}_ep{ep}_s{seed}"


def sweep_jobs(*tables):
    jobs = []
    cfgs = sorted(configs(*tables), key=lambda c: abs(np.log(c[4] / 1e-5)))
    for key, fam, method, args, lr, total in cfgs:
        out = f"{CK}/sw_{key}_lr{lr_tag(lr)}"
        pts = eval_points(total)
        evals = [(sweep_name(key, lr, e), out if e == total else f"{out}_ep{e}", []) for e in pts]
        jobs.append({"name": out, "train": (f"train_sw_{key}_lr{lr_tag(lr)}",
                                            unlearn_cmd(method, args, lr, 42, out, epochs=total,
                                                        save_list=[e for e in pts if e < total]), out),
                     "evals": evals, "cleanup": [out] + [f"{out}_ep{e}" for e in pts]})
    return jobs


def sweep():
    n = ngpu()
    say(f"=== v3 SWEEP  GPUs={n}  epochs={EPOCHS} (+ep{EARLY})  lrs={LRS}  ga_lrs={GA_LRS}")
    prepare(n)
    run_jobs(sweep_jobs(METHODS), n)
    summarize_sweep()


def summarize_sweep():
    a_mu = load("A_original")["model_utility"]
    rows, seen = [], set()
    for key, fam, method, args, lr, total in configs(METHODS, EXTEND, EXTEND2):
        for ep in eval_points(total):
            nm = sweep_name(key, lr, ep)
            if done_ev(nm) and nm not in seen:
                seen.add(nm)
                r = row_of(nm, family=fam, key=key, lr=lr, epochs=ep, total_epochs=total, seed=42)
                r["_method"], r["_args"] = method, args
                rows.append(r)
    write_csv([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
              f"{RES}/sweep_summary.csv")
    sel = {"rule": "max Forget Quality s.t. Model Utility >= 0.9 * MU(A); else max MU (flagged)",
           "model_A_utility": a_mu, "threshold": 0.9 * a_mu, "n_configs": len(rows), "selected": {}}
    for fam in FAMILIES:
        cand = [r for r in rows if r["family"] == fam]
        if not cand:
            continue
        ok = [r for r in cand if r["model_utility"] >= 0.9 * a_mu]
        best = (max(ok, key=lambda r: r["forget_quality"] or 0) if ok
                else max(cand, key=lambda r: r["model_utility"]))
        sel["selected"][fam] = {"key": best["key"], "method": best["_method"], "args": best["_args"],
                                "lr": best["lr"], "epochs": best["epochs"],
                                "total_epochs": best["total_epochs"], "name": best["name"],
                                "met_utility_threshold": bool(ok),
                                "forget_quality": best["forget_quality"],
                                "model_utility": best["model_utility"]}
    with open(f"{RES}/selection.json", "w") as f:
        json.dump(sel, f, indent=2)
    say("=== sweep summary")
    for r in rows:
        say(f"{r['name']:38s} FQ={r['forget_quality'] or 0:9.2e}  MU={r['model_utility']:.3f}  "
            f"forgetR={r['forget_rouge']:.3f}  AUC={r['mia_auc_mink']:.3f}")
    say("selection: " + json.dumps({k: (v["name"], v["met_utility_threshold"]) for k, v in sel["selected"].items()}))


def seeds():
    n = ngpu()
    if not os.path.exists(f"{RES}/selection.json"):
        hit = find_input("results/v3/selection.json")
        if not hit:
            sys.exit("selection.json not found: attach the output of the sweep run as input")
        src = os.path.dirname(hit)
        say(f"copying sweep results from {src}")
        shutil.copytree(src, RES, dirs_exist_ok=True)
    say(f"=== v3 SEEDS  GPUs={n}  seeds={SEEDS} (+42 from sweep)")
    prepare(n)
    if not os.path.exists(f"{RES}/selection_round1.json"):
        shutil.copy(f"{RES}/selection.json", f"{RES}/selection_round1.json")
    if os.environ.get("EXTEND", "1") == "1":
        say("=== round-2 grid extension")
        run_jobs(sweep_jobs(EXTEND), n)
    summarize_sweep()
    with open(f"{RES}/selection.json") as f:
        sel = json.load(f)["selected"]
    jobs = []
    for seed in SEEDS:
        for m, s in sel.items():
            out = f"{CK}/sd_{m}_s{seed}"
            tot = int(s.get("total_epochs", EPOCHS))
            cmd = unlearn_cmd(s["method"], s["args"], s["lr"], seed, out, epochs=tot, save_early=False)
            if int(s["epochs"]) < tot:
                cmd += ["--stop_after", str(int(s["epochs"]))]
            jobs.append({"name": out, "train": (f"train_sd_{m}_s{seed}", cmd, out),
                         "evals": [(f"sd_{m}_s{seed}", out, []),
                                   (f"sd_{m}_s{seed}_4bit", out, ["--load_4bit"])],
                         "cleanup": [out]})
    for seed in [42] + SEEDS:
        out = f"{CK}/sd_ft_retain_s{seed}"
        jobs.append({"name": out, "train": (f"train_sd_ft_retain_s{seed}",
                                            unlearn_cmd("ft_retain", [], 1e-5, seed, out, epochs=1, save_early=False), out),
                     "evals": [(f"sd_ft_retain_s{seed}", out, [])], "cleanup": [out]})
    jobs.append({"name": "A_4bit", "train": None,
                 "evals": [("A_original_4bit", f"{CK}/A_original", ["--load_4bit"])]})
    run_jobs(jobs, n)
    summarize_seeds()


def summarize_seeds():
    with open(f"{RES}/selection.json") as f:
        sel = json.load(f)["selected"]
    per_run, table = [], []
    groups = {m: [s["name"]] + [f"sd_{m}_s{x}" for x in SEEDS] for m, s in sel.items()}
    groups["ft_retain"] = [f"sd_ft_retain_s{x}" for x in [42] + SEEDS]
    q4 = {m: [f"sd_{m}_s{x}_4bit" for x in SEEDS] for m in sel}
    keys = ["forget_rouge", "forget_quality", "retain_rouge", "model_utility", "mia_auc_mink",
            "privleak", "minutes"]
    for label, names in list(groups.items()) + [(f"{m} (4-bit)", v) for m, v in q4.items()]:
        rows = [row_of(nm, group=label) for nm in names if done_ev(nm)]
        per_run += rows
        if not rows:
            continue
        agg = {"group": label, "n_seeds": len(rows)}
        for k in keys:
            vals = np.array([r[k] for r in rows if r[k] is not None], dtype=float)
            agg[f"{k}_mean"] = float(np.nanmean(vals)) if len(vals) else float("nan")
            agg[f"{k}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        fqs = [r["forget_quality"] for r in rows if r["forget_quality"] is not None]
        agg["fq_seeds_above_0.05"] = f"{sum(f > 0.05 for f in fqs)}/{len(fqs)}"
        table.append(agg)
    for ref in ("A_original", "B0_retrain", "A_original_4bit"):
        if done_ev(ref):
            per_run.append(row_of(ref, group=ref))
    write_csv(per_run, f"{RES}/seeds_runs.csv")
    write_csv(table, f"{RES}/seeds_summary.csv")
    say("=== seeds summary (mean ± std over seeds)")
    for t in table:
        say(f"{t['group']:20s} n={t['n_seeds']}  FQ={t['forget_quality_mean']:.2e}±{t['forget_quality_std']:.1e} "
            f"({t['fq_seeds_above_0.05']} >0.05)  MU={t['model_utility_mean']:.3f}±{t['model_utility_std']:.3f}  "
            f"forgetR={t['forget_rouge_mean']:.3f}±{t['forget_rouge_std']:.3f}  "
            f"AUC={t['mia_auc_mink_mean']:.3f}±{t['mia_auc_mink_std']:.3f}")


def final_name(s, seed, q=False):
    return f"fn_{s['key']}_lr{lr_tag(s['lr'])}_ep{s['epochs']}_s{seed}" + ("_4bit" if q else "")


def migrate_run2(prev_sel):
    for fam, s in prev_sel.items():
        for seed in (1, 2):
            for q in (False, True):
                old = f"sd_{fam}_s{seed}" + ("_4bit" if q else "")
                new = final_name(s, seed, q)
                for suf in ("", "_raw"):
                    src, dst = f"{RES}/{old}{suf}.json", f"{RES}/{new}{suf}.json"
                    if os.path.exists(src) and not os.path.exists(dst):
                        shutil.copy(src, dst)


def final():
    n = ngpu()
    if not os.path.exists(f"{RES}/selection.json"):
        hits = []
        for root in ("/kaggle/input", os.environ.get("EXTRA_INPUT", "")):
            if root and os.path.isdir(root):
                hits += glob.glob(os.path.join(root, "**", "results/v3/selection.json"), recursive=True)
        hits = [h for h in hits if os.path.exists(os.path.join(os.path.dirname(h), "selection_round1.json"))]
        if not hits:
            sys.exit("Run-2 results not found (need results/v3 with selection_round1.json): attach results_v3_seeds.zip")
        say(f"copying run-2 results from {os.path.dirname(hits[0])}")
        shutil.copytree(os.path.dirname(hits[0]), RES, dirs_exist_ok=True)
    say(f"=== v3.2 FINAL  GPUs={n}  held-out seeds={FINAL_SEEDS}")
    with open(f"{RES}/selection.json") as f:
        prev = json.load(f)
    if not os.path.exists(f"{RES}/selection_round2.json"):
        shutil.copy(f"{RES}/selection.json", f"{RES}/selection_round2.json")
    for fam, s in prev["selected"].items():
        s.setdefault("total_epochs", EPOCHS)
    migrate_run2(prev["selected"])
    prepare(n, oracle2=True)
    say("=== round-3 grid extension (seed 42 = selection seed)")
    run_jobs(sweep_jobs(EXTEND2), n)
    summarize_sweep()
    with open(f"{RES}/selection.json") as f:
        sel = json.load(f)["selected"]
    jobs = []
    for seed in FINAL_SEEDS:
        for fam, s in sel.items():
            out = f"{CK}/fn_{fam}_s{seed}"
            tot = int(s.get("total_epochs", EPOCHS))
            cmd = unlearn_cmd(s["method"], s["args"], s["lr"], seed, out, epochs=tot, save_early=False)
            if int(s["epochs"]) < tot:
                cmd += ["--stop_after", str(int(s["epochs"]))]
            jobs.append({"name": out, "train": (f"train_fn_{fam}_s{seed}", cmd, out),
                         "evals": [(final_name(s, seed), out, []), (final_name(s, seed, True), out, ["--load_4bit"])],
                         "cleanup": [out]})
        out = f"{CK}/sd_ft_retain_s{seed}"
        jobs.append({"name": out, "train": (f"train_sd_ft_retain_s{seed}",
                                            unlearn_cmd("ft_retain", [], 1e-5, seed, out, epochs=1, save_early=False), out),
                     "evals": [(f"sd_ft_retain_s{seed}", out, [])], "cleanup": [out]})
    run_jobs(jobs, n)
    summarize_final()


def ks_fq(name, ref):
    from scipy.stats import ks_2samp
    a = np.array(load(f"{name}_raw")["forget"]["truth_ratio"], dtype=float)
    b = np.array(load(f"{ref}_raw")["forget"]["truth_ratio"], dtype=float)
    return float(ks_2samp(a[~np.isnan(a)], b[~np.isnan(b)]).pvalue)


def summarize_final():
    with open(f"{RES}/selection.json") as f:
        sel = json.load(f)["selected"]
    has_o2 = done_ev(ORACLE2)
    o_auc = load("B0_retrain")["privacy"]["auc_mink"]
    o2_auc = load(ORACLE2)["privacy"]["auc_mink"] if has_o2 else None
    groups = {fam: (s, [final_name(s, x) for x in FINAL_SEEDS], [final_name(s, x, True) for x in FINAL_SEEDS])
              for fam, s in sel.items()}
    per_run, table = [], []

    def enrich(r):
        if has_o2:
            r["fq_vs_oracle2"] = ks_fq(r["name"], ORACLE2)
            r["privleak_vs_oracle2"] = (r["mia_auc_mink"] - o2_auc) / o2_auc
        return r

    def agg_rows(label, rows, extra):
        if not rows:
            return
        a = {"group": label, "n_seeds": len(rows), **extra}
        for k in ["forget_quality", "fq_vs_oracle2", "model_utility", "forget_rouge", "retain_rouge",
                  "mia_auc_mink", "privleak", "privleak_vs_oracle2", "minutes"]:
            v = np.array([r.get(k) for r in rows if r.get(k) is not None], dtype=float)
            if len(v):
                a[f"{k}_mean"] = float(np.mean(v))
                a[f"{k}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                a[f"{k}_median"] = float(np.median(v))
        a["fq_seeds_above_0.05"] = f"{sum((r['forget_quality'] or 0) > 0.05 for r in rows)}/{len(rows)}"
        table.append(a)

    for fam, (s, names, qnames) in groups.items():
        cfg = f"{s['key']} lr={s['lr']:g} ep={s['epochs']}"
        sel_row = enrich(row_of(s["name"], group=fam, seed=42, role="selection"))
        per_run.append(sel_row)
        rows = [enrich(row_of(nm, group=fam, seed=sd, role="held-out")) for nm, sd in zip(names, FINAL_SEEDS) if done_ev(nm)]
        qrows = [enrich(row_of(nm, group=fam + " (4-bit)", seed=sd, role="held-out")) for nm, sd in zip(qnames, FINAL_SEEDS) if done_ev(nm)]
        per_run += rows + qrows
        agg_rows(fam, rows, {"config": cfg, "selection_seed42_fq": sel_row["forget_quality"],
                             "selection_seed42_privleak": sel_row["privleak"]})
        agg_rows(fam + " (4-bit)", qrows, {"config": cfg})
    b = [enrich(row_of(f"sd_ft_retain_s{x}", group="ft_retain", seed=x, role="held-out"))
         for x in FINAL_SEEDS if done_ev(f"sd_ft_retain_s{x}")]
    per_run += b
    agg_rows("ft_retain", b, {"config": "ft_retain lr=1e-05 ep=1"})
    for ref in ("A_original", "B0_retrain", ORACLE2, "A_original_4bit"):
        if done_ev(ref):
            r = row_of(ref, group=ref, seed="", role="reference")
            if has_o2 and ref != ORACLE2:
                r = enrich(r)
            per_run.append(r)
    write_csv(per_run, f"{RES}/final_runs.csv")
    write_csv(table, f"{RES}/final_summary.csv")
    if has_o2:
        floor = {"oracle_seed1_vs_oracle_seed42_FQ": ks_fq(ORACLE2, "B0_retrain"),
                 "oracle_seed42_auc": o_auc, "oracle_seed1_auc": o2_auc}
        with open(f"{RES}/oracle_noise.json", "w") as f:
            json.dump(floor, f, indent=2)
        say(f"oracle noise floor: {floor}")
    say("=== final summary: held-out seeds only (seed 42 was used for selection)")
    for t in table:
        say(f"{t['group']:20s} n={t['n_seeds']} FQ mean={t.get('forget_quality_mean', float('nan')):.2e} "
            f"median={t.get('forget_quality_median', float('nan')):.2e} ({t['fq_seeds_above_0.05']})  "
            f"MU={t.get('model_utility_mean', float('nan')):.3f}  AUC={t.get('mia_auc_mink_mean', float('nan')):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["sweep", "seeds", "final", "summarize"])
    a = ap.parse_args()
    os.chdir(HERE)
    for d in (CK, RES, LOGS):
        os.makedirs(d, exist_ok=True)
    {"sweep": sweep, "seeds": seeds, "final": final,
     "summarize": lambda: (summarize_sweep(), summarize_final())}[a.phase]()
    if FAILED:
        say(f"FINISHED WITH FAILURES: {FAILED}")
    else:
        say(f"FINISHED - all steps OK  ({(time.time() - T0) / 3600:.2f} h)")


if __name__ == "__main__":
    main()
