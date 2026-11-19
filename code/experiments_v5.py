import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA", "data")
CK = os.environ.get("CKPT_DIR", "ckpt")
RES = os.environ.get("RES_DIR", "results/v5")
LOGS = os.environ.get("LOG_DIR", "logs/v5")
MAX_HOURS = float(os.environ.get("MAX_HOURS", "11"))
QWEN = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LLAMA = os.environ.get("MODEL3", "meta-llama/Llama-3.2-1B-Instruct")
EPOCHS, EARLY = int(os.environ.get("UL_EPOCHS", "10")), int(os.environ.get("EARLY_EPOCH", "5"))
FT_EPOCHS = os.environ.get("FT_EPOCHS", "5")
TRAJ_EVERY = os.environ.get("TRAJ_EVERY", "0.5")
SMOKE = os.environ.get("SMOKE") == "1"

NPO_RT = ("npo_rt", "npo_rt", ["--beta", "0.1", "--alpha", "1.0"])
SIM25 = ("simnpo_fw1_b2.5", "simnpo", ["--beta", "2.5", "--forget_weight", "1.0", "--gamma", "0", "--alpha", "1.0"])
IDKNLL = ("idknll", "idknll", ["--alpha", "1.0"])
IDKDPO = ("idkdpo", "idkdpo", ["--beta", "0.1", "--alpha", "1.0"])
CAL = ("npo_cal", "npo_cal", ["--beta", "0.1", "--alpha", "1.0", "--precompute_ref"])
MATCH = ("match", "match", ["--alpha", "1.0"])
FLOOR = ("floor", "floor", ["--alpha", "1.0"])


def RMU(coef, alpha=None):
    alpha = alpha or os.environ.get("RMU_ALPHA", "10")
    key = f"rmu_c{coef:g}" + ("" if str(alpha) == "10" else f"_a{alpha}")
    return (key, "rmu", ["--rmu_layer", os.environ.get("RMU_LAYER", "7"), "--rmu_coef", str(coef), "--alpha", str(alpha)])


T0 = time.time()
LOCK = threading.Lock()
FAILED = []


def say(msg):
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def lr_tag(lr):
    return f"{lr:g}".replace("e-0", "e-")


def done_ck(p):
    return os.path.exists(os.path.join(p, "train_stats.json"))


def ngpu():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        return max(1, len([l for l in out.splitlines() if l.strip()]))
    except FileNotFoundError:
        return 1


def run_step(gpu, logname, cmd):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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


class Eval:

    def __init__(self, name, ckpt, res, forget="forget10", ref=None, base=None, extra=()):
        self.name, self.ckpt, self.res, self.forget, self.ref = name, ckpt, res, forget, ref
        self.base, self.extra = base, list(extra)

    def done(self):
        return os.path.exists(os.path.join(self.res, f"{self.name}.json"))

    def run(self, gpu):
        if self.ref:
            for _ in range(240):
                if os.path.exists(self.ref):
                    break
                time.sleep(30)
        cmd = [PY, "evaluate.py", "--model", self.ckpt, "--name", self.name, "--forget", self.forget,
               "--data_dir", DATA, "--out_dir", self.res, *self.extra]
        if self.ref:
            cmd += ["--ref_raw", self.ref]
        if self.base:
            cmd += ["--base_model", self.base]
        if SMOKE:
            cmd += ["--max_new_tokens", "8"]
        tag = os.path.basename(self.res.rstrip("/"))
        return run_step(gpu, f"eval_{tag}_{self.name}", cmd)


def run_jobs(jobs, n):
    queue = list(jobs)

    def worker(gpu):
        while True:
            with LOCK:
                if not queue:
                    return
                job = queue.pop(0)
            train = job.get("train")
            if all(e.done() for e in job["evals"]) and (job["evals"] or not train or done_ck(train[2])):
                continue
            if (time.time() - T0) / 3600 > MAX_HOURS:
                say(f"SKIP   {job['name']}: time budget of {MAX_HOURS} h reached")
                continue
            ok = True
            if train and not done_ck(train[2]):
                ok = run_step(gpu, train[0], train[1])
            if ok:
                for e in job["evals"]:
                    if not e.done() and done_ck(e.ckpt):
                        e.run(gpu)
            for d in job.get("cleanup", []):
                shutil.rmtree(d, ignore_errors=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def ft_job(name, model, subset, seed=42, extra=()):
    out = f"{CK}/{name}"
    cmd = [PY, "train_ft.py", "--model", model, "--subset", subset, "--out", out, "--data_dir", DATA,
           "--epochs", FT_EPOCHS, "--bs", "8", "--grad_accum", "4", "--seed", str(seed), *extra]
    return {"name": name, "train": (f"train_{name}", cmd, out), "evals": []}


def ul_jobs(prefix, a_ckpt, res, ref_raw, base, configs, seed=42, extra=(), traj=False, early=True, tag=""):
    jobs = []
    for (key, method, args), lr in configs:
        out = f"{CK}/{prefix}_{key}_lr{lr_tag(lr)}_s{seed}"
        name = f"{tag}{key}_lr{lr_tag(lr)}"
        cmd = [PY, "unlearn.py", "--model", a_ckpt, "--method", method, "--out", out, "--forget", "forget10",
               "--retain", "retain90", "--data_dir", DATA, "--epochs", str(EPOCHS), "--lr", str(lr),
               "--bs", "4", "--grad_accum", "8", "--seed", str(seed), "--warmup_epochs", "1.0", *args, *extra]
        if early:
            cmd += ["--save_epochs", str(EARLY)]
        if traj:
            cmd += ["--traj_out", f"{res}/traj_{name}_s{seed}.json", "--traj_every", TRAJ_EVERY]
        evals = [Eval(f"{name}_ep{EPOCHS}_s{seed}", out, res, ref=ref_raw, base=base)]
        if early:
            evals.append(Eval(f"{name}_ep{EARLY}_s{seed}", f"{out}_ep{EARLY}", res, ref=ref_raw, base=base))
        jobs.append({"name": out, "train": (f"train_{os.path.basename(out)}", cmd, out), "evals": evals,
                     "cleanup": [out, f"{out}_ep{EARLY}"]})
    return jobs


def prep_dirs():
    for d in (CK, RES, LOGS):
        os.makedirs(d, exist_ok=True)
    if not os.path.isdir(DATA):
        subprocess.run([PY, "download_data.py", "--out", DATA], cwd=HERE, check=True)


def base_pair(res, model, a_name="A_s42", o_name="O_s42", mem=()):
    run_jobs([ft_job(a_name, model, "full", 42, mem), ft_job(o_name, model, "retain90", 42, mem)], ngpu())
    for p in (a_name, o_name):
        if not done_ck(f"{CK}/{p}"):
            sys.exit(f"{p} missing - see {LOGS}/train_{p}.log")
    ref = f"{res}/{o_name}_raw.json"
    evals = [{"name": "eval_oracle", "train": None, "evals": [Eval(o_name, f"{CK}/{o_name}", res, base=model)],
              "cleanup": [f"{CK}/{o_name}"]},
             {"name": "eval_A", "train": None, "evals": [Eval(a_name, f"{CK}/{a_name}", res, ref=ref, base=model)]}]
    return ref, evals


def run7():
    prep_dirs()
    res = f"{RES}/run7"
    os.makedirs(res, exist_ok=True)
    say(f"=== v5 RUN 7  GPUs={ngpu()}  model={QWEN}: oracle seeds + trajectories")
    ref, jobs = base_pair(res, QWEN)
    A = f"{CK}/A_s42"
    seeds = [1] if SMOKE else [1, 2, 3, 4]
    traj_cfg = [(NPO_RT, 1e-5)] if SMOKE else [(NPO_RT, 1e-5), (NPO_RT, 5e-6), (SIM25, 5e-5), (NPO_RT, 2e-5),
                                               (SIM25, 2e-5)]
    trajs = ul_jobs("r7", A, res, ref, QWEN, traj_cfg, traj=True)
    oracles = []
    for s in seeds:
        j = ft_job(f"O_s{s}", QWEN, "retain90", s)
        j["evals"] = [Eval(f"O_s{s}", f"{CK}/O_s{s}", res, ref=ref, base=QWEN)]
        j["cleanup"] = [f"{CK}/O_s{s}"]
        oracles.append(j)
    order = jobs[:1] + trajs[:1] + jobs[1:]
    rest = trajs[1:]
    while oracles or rest:
        if oracles:
            order.append(oracles.pop(0))
        if rest:
            order.append(rest.pop(0))
    run_jobs(order, ngpu())
    summarize()


def run8():
    prep_dirs()
    res = f"{RES}/run8"
    os.makedirs(res, exist_ok=True)
    say(f"=== v5 RUN 8  GPUs={ngpu()}  model={QWEN}: other method families + proposed fix")
    ref, jobs = base_pair(res, QWEN)
    A = f"{CK}/A_s42"
    if SMOKE:
        fix = [(CAL, 1e-5), (FLOOR, 1e-5)]
        other = [(IDKNLL, 1e-5), (IDKDPO, 1e-5), (RMU(2), 5e-5)]
    else:
        fix = [(CAL, 2e-5), (CAL, 5e-5), (FLOOR, 2e-5), (CAL, 1e-5), (FLOOR, 5e-6)]
        other = [(IDKNLL, 1e-5), (IDKDPO, 1e-5), (RMU(2), 5e-5), (IDKNLL, 2e-5), (IDKDPO, 2e-5), (RMU(6), 5e-5),
                 (IDKNLL, 5e-5), (IDKDPO, 5e-6), (RMU(6), 1e-4)]
    tv_out = f"{CK}/r8_tv"
    alphas = ["0.5", "2"] if SMOKE else ["0.5", "1", "2", "4"]
    tv = {"name": "task_vector",
          "train": ("train_r8_task_vector",
                    [PY, "task_vector.py", "--model", A, "--forget", "forget10", "--data_dir", DATA,
                     "--epochs", FT_EPOCHS, "--lr", "1e-5", "--alphas", ",".join(alphas), "--out", tv_out],
                    f"{tv_out}_a{alphas[-1]}"),
          "evals": [Eval(f"task_vector_a{a}", f"{tv_out}_a{a}", res, ref=ref, base=QWEN) for a in alphas],
          "cleanup": [f"{tv_out}_a{a}" for a in alphas]}
    fixj = ul_jobs("r8", A, res, ref, QWEN, fix, traj=True)
    othj = ul_jobs("r8", A, res, ref, QWEN, other)
    order = jobs[:1] + fixj[:1] + jobs[1:] + [tv]
    while fixj[1:] or othj:
        if len(fixj) > 1:
            order.append(fixj.pop(1))
        if othj:
            order.append(othj.pop(0))
    run_jobs(order, ngpu())
    summarize()


def run8b():
    prep_dirs()
    res = f"{RES}/run8b"
    os.makedirs(res, exist_ok=True)
    say(f"=== v5 RUN 8b  GPUs={ngpu()}  model={QWEN}: npo_cal (fixed), match, RMU trajectories")
    ref, jobs = base_pair(res, QWEN)
    A = f"{CK}/A_s42"
    if SMOKE:
        cfg = [(CAL, 1e-5), (MATCH, 1e-5), (RMU(2), 5e-5)]
    else:
        cfg = [(CAL, 2e-5), (MATCH, 2e-5), (RMU(2), 5e-5), (CAL, 5e-5), (MATCH, 5e-6), (RMU(2, 100), 5e-5),
               (CAL, 1e-5), (RMU(2), 2e-5)]
    ul = ul_jobs("r8b", A, res, ref, QWEN, cfg, traj=True)
    run_jobs(jobs[:1] + ul[:1] + jobs[1:] + ul[1:], ngpu())
    summarize()

CHOSEN = [(NPO_RT, 2e-5), (SIM25, 5e-5)]


def pick_fix():
    if os.environ.get("FIX_SETTING"):
        m, lr, ep = os.environ["FIX_SETTING"].split(":")
        return m, float(lr), int(ep), "FIX_SETTING"
    hits = [h for pat in ("run8", "run8b") for h in
            glob.glob(f"/kaggle/input/**/results/v5/{pat}/summary_v5.csv", recursive=True)
            + glob.glob(os.path.join(os.environ.get("EXTRA_INPUT", "/nonexistent"), f"**/{pat}/summary_v5.csv"),
                        recursive=True)]
    if not hits:
        return "npo_cal", 2e-5, 10, "default (run-8 results not attached)"
    rows = [r for h in hits for r in csv.DictReader(open(h))]
    a = [r for r in rows if r["name"] == "A_s42"]
    mu_a = float(a[0]["model_utility"]) if a else 0.0
    cand = []
    for r in rows:
        if not r["name"].startswith(("npo_cal", "floor", "match")) or not r["forget_quality"]:
            continue
        key, rest = r["name"].split("_lr")
        lr, ep = rest.split("_ep")[0], int(rest.split("_ep")[1].split("_")[0])
        cand.append((key, float(lr), ep, float(r["forget_quality"]), float(r["privleak"]), float(r["model_utility"])))
    ok = [c for c in cand if c[5] >= 0.9 * mu_a]
    inbox = [c for c in ok if abs(c[4]) <= 0.2]
    best = max(inbox, key=lambda c: c[3]) if inbox else min(ok or cand, key=lambda c: abs(c[4]))
    return best[0], best[1], best[2], f"selected from {hits[0]}"


def run9():
    prep_dirs()
    res = f"{RES}/run9"
    os.makedirs(res, exist_ok=True)
    say(f"=== v5 RUN 9  GPUs={ngpu()}  model={QWEN}: held-out seeds with CIs, Model A seeds, fix seeds")
    m, lr, ep, why = pick_fix()
    say(f"fix setting for seeds 1-3: {m} lr={lr} epoch {ep}  ({why})")
    fixcfg = {"npo_cal": CAL, "floor": FLOOR, "match": MATCH}[m]
    aseeds = [] if SMOKE else [1, 2]
    run_jobs([ft_job("A_s42", QWEN, "full", 42), ft_job("O_s42", QWEN, "retain90", 42)]
             + [ft_job(f"A_s{s}", QWEN, "full", s) for s in aseeds], ngpu())
    for p in ["A_s42", "O_s42"] + [f"A_s{s}" for s in aseeds]:
        if not done_ck(f"{CK}/{p}"):
            sys.exit(f"{p} missing - see {LOGS}/train_{p}.log")
    ref = f"{res}/O_s42_raw.json"
    jobs = [{"name": "eval_oracle", "train": None, "evals": [Eval("O_s42", f"{CK}/O_s42", res, base=QWEN)],
             "cleanup": [f"{CK}/O_s42"]}]
    for a in ["A_s42"] + [f"A_s{s}" for s in aseeds]:
        jobs.append({"name": f"eval_{a}", "train": None, "evals": [Eval(a, f"{CK}/{a}", res, ref=ref, base=QWEN)]})
    A = f"{CK}/A_s42"
    seeds = [1] if SMOKE else [1, 2, 3]
    held = []
    for sd in seeds:
        held += ul_jobs("r9", A, res, ref, QWEN, CHOSEN, seed=sd, early=False)
        held += ul_jobs("r9", A, res, ref, QWEN, [(fixcfg, lr)], seed=sd, early=(ep == EARLY))
    for s in aseeds:
        held += ul_jobs(f"r9A{s}", f"{CK}/A_s{s}", res, ref, QWEN, CHOSEN, seed=42, early=False, tag=f"A{s}_")
    run_jobs(jobs[:1] + held[:1] + jobs[1:] + held[1:], ngpu())
    summarize()


RUN10_PARTS = {
    "a": [(NPO_RT, 5e-6), (NPO_RT, 1e-5), (NPO_RT, 2e-6), (NPO_RT, 2e-5), (MATCH, 2e-5)],
    "b": [(SIM25, 2e-5), (SIM25, 5e-5), (SIM25, 1e-5), (CAL, 5e-5)],
}


def run10(part=None):
    prep_dirs()
    res = f"{RES}/run10" + (part or "")
    os.makedirs(res, exist_ok=True)
    say(f"=== v5 RUN 10{part or ''}  GPUs={ngpu()}  model={LLAMA}")
    mem = os.environ.get("MODEL3_MEM", "--optim adamw8bit --grad_ckpt --freeze_embed").split()
    ft_bs = os.environ.get("MODEL3_FT_BS", "--bs 2 --grad_accum 16").split()
    ul_bs = os.environ.get("MODEL3_UL_BS", "--bs 2 --grad_accum 16").split()
    ref, jobs = base_pair(res, LLAMA, "A_s42", "O_s42", mem + ft_bs)
    A = f"{CK}/A_s42"
    cfg = [(NPO_RT, 1e-5)] if SMOKE else (RUN10_PARTS[part] if part else RUN10_PARTS["a"] + RUN10_PARTS["b"])
    ul = ul_jobs("r10", A, res, ref, LLAMA, cfg, extra=mem + ul_bs + ["--precompute_ref"])
    run_jobs(jobs[:1] + ul[:1] + jobs[1:] + ul[1:], ngpu())
    summarize()


def summarize():
    rows = []
    for d in sorted(glob.glob(f"{RES}/*/")):
        run = os.path.basename(d.rstrip("/"))
        a_path = f"{d}/A_s42.json"
        a_mu = json.load(open(a_path))["model_utility"] if os.path.exists(a_path) else None
        for p in sorted(glob.glob(f"{d}/*.json")):
            if p.endswith("_raw.json") or os.path.basename(p).startswith(("_", "traj_")):
                continue
            r = json.load(open(p))
            if "forget" not in r:
                continue
            e = r.get("efficiency") or {}
            pv = r["privacy"]
            rows.append({"run": run, "name": r["name"], "forget_quality": r.get("forget_quality"),
                         "ks_stat": r.get("ks_stat"), "model_utility": r["model_utility"],
                         "utility_ratio": (r["model_utility"] / a_mu) if a_mu else None,
                         "forget_rouge": r["forget"]["rougeL_recall"], "forget_tr": r["forget"]["truth_ratio_mean"],
                         "forget_prob": r["forget"]["prob"], "auc_loss": pv.get("auc_loss"),
                         "auc_mink": pv.get("auc_mink"), "auc_minkpp": pv.get("auc_minkpp"),
                         "auc_zlib": pv.get("auc_zlib"), "auc_ref": pv.get("auc_ref"),
                         "privleak": pv.get("privleak_mink"),
                         "minutes": (e.get("wall_clock_sec") or float("nan")) / 60})
    if not rows:
        return
    for run in sorted({r["run"] for r in rows}):
        rs = [r for r in rows if r["run"] == run]
        with open(f"{RES}/{run}/summary_v5.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)
        say(f"=== {run}")
        for r in rs:
            fq = r["forget_quality"]
            pl = r["privleak"]
            box = fq is not None and pl is not None and fq > 0.05 and abs(pl) <= 0.2
            say(f"{r['name']:40s} FQ={fq if fq is not None else float('nan'):9.2e} MU={r['model_utility']:.3f} "
                f"AUC={r['auc_mink']:.3f} PrivLeak={pl if pl is not None else float('nan'):+.2f}"
                + ("   <-- retrain-like on both" if box else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run7", "run8", "run8b", "run9", "run10", "run10a", "run10b", "summarize"])
    a = ap.parse_args()
    os.chdir(HERE)
    for d in (CK, RES, LOGS):
        os.makedirs(d, exist_ok=True)
    {"run7": run7, "run8": run8, "run8b": run8b, "run9": run9, "run10": run10,
     "run10a": lambda: run10("a"), "run10b": lambda: run10("b"), "summarize": summarize}[a.phase]()
    say(f"FINISHED WITH FAILURES: {FAILED}" if FAILED else f"FINISHED - all steps OK  ({(time.time() - T0) / 3600:.2f} h)")


if __name__ == "__main__":
    main()
