import argparse
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
RES = os.environ.get("RES_DIR", "results/v4")
LOGS = os.environ.get("LOG_DIR", "logs/v4")
MAX_HOURS = float(os.environ.get("MAX_HOURS", "11"))
EVAL_EXTRA = os.environ.get("EVAL_EXTRA", "").split()
QWEN = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL2 = os.environ.get("MODEL2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
EPOCHS, EARLY = int(os.environ.get("UL_EPOCHS", "10")), int(os.environ.get("EARLY_EPOCH", "5"))
FT_EPOCHS = os.environ.get("FT_EPOCHS", "5")
NPO_RT = ("npo_rt", "npo_rt", ["--beta", "0.1", "--alpha", "1.0"])
SIM25 = ("simnpo_fw1_b2.5", "simnpo", ["--beta", "2.5", "--forget_weight", "1.0", "--gamma", "0", "--alpha", "1.0"])
SIM15 = ("simnpo_fw1_b1.5", "simnpo", ["--beta", "1.5", "--forget_weight", "1.0", "--gamma", "0", "--alpha", "1.0"])
MINI = [(NPO_RT, 2e-5), (SIM25, 5e-5), (NPO_RT, 1e-5), (SIM25, 2e-5), (NPO_RT, 5e-6)]
LRS_NPO = [float(x) for x in os.environ.get("LRS_NPO", "").split(",") if x]
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

    def __init__(self, name, ckpt, res, forget, ref=None, extra=()):
        self.name, self.ckpt, self.res, self.forget, self.ref, self.extra = name, ckpt, res, forget, ref, list(extra)

    def done(self):
        return os.path.exists(os.path.join(self.res, f"{self.name}.json"))

    def run(self, gpu):
        if self.ref:
            for _ in range(120):
                if os.path.exists(self.ref):
                    break
                time.sleep(30)
        cmd = [PY, "evaluate.py", "--model", self.ckpt, "--name", self.name, "--forget", self.forget,
               "--data_dir", DATA, "--out_dir", self.res, *self.extra, *EVAL_EXTRA]
        if self.ref:
            cmd += ["--ref_raw", self.ref]
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


def ft_job(name, model, subset, extra=()):
    out = f"{CK}/{name}"
    return {"name": name, "train": (f"train_{name}", [PY, "train_ft.py", "--model", model, "--subset", subset,
                                                      "--out", out, "--data_dir", DATA, "--epochs", FT_EPOCHS,
                                                      "--bs", "8", "--grad_accum", "4", *extra], out), "evals": []}


def mini_jobs(prefix, a_ckpt, forget, retain, res, ref_raw, configs, extra=()):
    jobs = []
    for (key, method, args), lr in configs:
        out = f"{CK}/{prefix}_{key}_lr{lr_tag(lr)}"
        cmd = [PY, "unlearn.py", "--model", a_ckpt, "--method", method, "--out", out, "--forget", forget,
               "--retain", retain, "--data_dir", DATA, "--epochs", str(EPOCHS), "--lr", str(lr),
               "--bs", "4", "--grad_accum", "8", "--seed", "42", "--warmup_epochs", "1.0",
               "--save_epochs", str(EARLY), *args, *extra]
        evals = [Eval(f"sw_{key}_lr{lr_tag(lr)}_ep{EPOCHS}", out, res, forget, ref_raw),
                 Eval(f"sw_{key}_lr{lr_tag(lr)}_ep{EARLY}", f"{out}_ep{EARLY}", res, forget, ref_raw)]
        jobs.append({"name": out, "train": (f"train_{os.path.basename(out)}", cmd, out), "evals": evals,
                     "cleanup": [out, f"{out}_ep{EARLY}"]})
    return jobs


def prep_dirs():
    for d in (CK, RES, LOGS):
        os.makedirs(d, exist_ok=True)
    if not os.path.isdir(DATA):
        subprocess.run([PY, "download_data.py", "--out", DATA], cwd=HERE, check=True)


def scale():
    n = ngpu()
    prep_dirs()
    say(f"=== v4 SCALE  GPUs={n}  model={QWEN}")
    r10, r05, r01 = f"{RES}/forget10", f"{RES}/forget05", f"{RES}/forget01"
    for d in (r10, r05, r01):
        os.makedirs(d, exist_ok=True)
    hits = [h for h in glob.glob("/kaggle/input/**/results/v3/oracle_noise.json", recursive=True)
            + glob.glob(os.path.join(os.environ.get("EXTRA_INPUT", "/nonexistent"), "**/results/v3/oracle_noise.json"), recursive=True)]
    if hits:
        src = os.path.dirname(hits[0])
        for f in ("B0_retrain.json", "B0_retrain_raw.json", "A_original.json", "A_original_raw.json"):
            if os.path.exists(f"{src}/{f}") and not os.path.exists(f"{r10}/{f}"):
                shutil.copy(f"{src}/{f}", f"{r10}/{f}")
        say(f"forget10 references copied from {src}")
    else:
        say("WARNING: run-3 results not attached; the forget10 SimNPO-edge runs will be skipped")
    run_jobs([ft_job("A_original", QWEN, "full"), ft_job("O_retain95", QWEN, "retain95")], n)
    if not done_ck(f"{CK}/A_original"):
        sys.exit("Model A missing")
    A = f"{CK}/A_original"
    ref05, ref01 = f"{r05}/O_retain95_raw.json", f"{r01}/O_retain99_raw.json"
    run_jobs([ft_job("O_retain99", QWEN, "retain99"),
              {"name": "refs05", "train": None, "evals": [Eval("O_retain95", f"{CK}/O_retain95", r05, "forget05"),
                                                          Eval("A_original", A, r05, "forget05", ref05)],
               "cleanup": [f"{CK}/O_retain95"]}], n)
    jobs = [{"name": "refs01", "train": None, "evals": [Eval("O_retain99", f"{CK}/O_retain99", r01, "forget01"),
                                                         Eval("A_original", A, r01, "forget01", ref01)],
             "cleanup": [f"{CK}/O_retain99"]}]
    jobs += mini_jobs("f05", A, "forget05", "retain95", r05, ref05, MINI)
    if hits:
        jobs += mini_jobs("f10", A, "forget10", "retain90", r10, f"{r10}/B0_retrain_raw.json",
                          [(SIM25, 1e-4), (SIM15, 5e-5)])
    jobs += mini_jobs("f01", A, "forget01", "retain99", r01, ref01, MINI)
    run_jobs(jobs, n)
    summarize()


LOW = [(NPO_RT, 2e-6), (SIM25, 1e-5), (NPO_RT, 1e-6), (SIM25, 5e-6)]


def model2(configs=None, res_name="tinyllama"):
    configs = configs or MINI
    n = ngpu()
    prep_dirs()
    say(f"=== v4 MODEL2  GPUs={n}  model={MODEL2}  results={res_name}")
    res = f"{RES}/{res_name}"
    os.makedirs(res, exist_ok=True)
    mem = os.environ.get("MODEL2_MEM", "--optim adamw8bit --grad_ckpt").split()
    run_jobs([ft_job("TL_A", MODEL2, "full", mem), ft_job("TL_oracle", MODEL2, "retain90", mem)], n)
    if not (done_ck(f"{CK}/TL_A") and done_ck(f"{CK}/TL_oracle")):
        sys.exit("TinyLlama Model A or oracle missing - see logs")
    ref = f"{res}/B0_retrain_raw.json"
    jobs = [{"name": "tl_oracle_eval", "train": None,
             "evals": [Eval("B0_retrain", f"{CK}/TL_oracle", res, "forget10")]}]
    ul = mini_jobs("tl", f"{CK}/TL_A", "forget10", "retain90", res, ref, configs, mem + ["--precompute_ref"])
    jobs += ul[:1] + [{"name": "tl_A_eval", "train": None,
                       "evals": [Eval("A_original", f"{CK}/TL_A", res, "forget10", ref)]}] + ul[1:]
    run_jobs(jobs, n)
    summarize()


def summarize():
    import csv
    import numpy as np
    from scipy.stats import spearmanr
    rows = []
    for d in sorted(glob.glob(f"{RES}/*/")):
        split = os.path.basename(d.rstrip("/"))
        a_path = f"{d}/A_original.json"
        a_mu = json.load(open(a_path))["model_utility"] if os.path.exists(a_path) else None
        for p in sorted(glob.glob(f"{d}/*.json")):
            if p.endswith("_raw.json"):
                continue
            r = json.load(open(p))
            if "forget" not in r:
                continue
            e = r.get("efficiency") or {}
            rows.append({"split": split, "name": r["name"], "forget_quality": r.get("forget_quality"),
                         "model_utility": r["model_utility"], "utility_ratio": (r["model_utility"] / a_mu) if a_mu else None,
                         "forget_rouge": r["forget"]["rougeL_recall"], "mia_auc_mink": r["privacy"]["auc_mink"],
                         "privleak": r["privacy"].get("privleak_mink"),
                         "minutes": (e.get("wall_clock_sec") or float("nan")) / 60})
    if not rows:
        return
    with open(f"{RES}/v4_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    say("=== v4 summary")
    for split in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == split]
        for r in rs:
            say(f"{split:10s} {r['name']:34s} FQ={r['forget_quality'] or 0:9.2e}  MU={r['model_utility']:.3f}  "
                f"AUC={r['mia_auc_mink']:.3f}  PrivLeak={r['privleak'] if r['privleak'] is not None else float('nan'):+.2f}")
        sw = [r for r in rs if r["name"].startswith("sw_") and r["forget_quality"] is not None]
        both = [r["name"] for r in sw if r["forget_quality"] > 0.05 and abs(r["privleak"]) <= 0.2]
        say(f"{split}: settings retrain-like on both metrics: {both or 'none'}")
        if len(sw) >= 4:
            rho = spearmanr(np.log10(np.maximum([r["forget_quality"] for r in sw], 1e-100)),
                            [r["privleak"] for r in sw]).correlation
            say(f"{split}: Spearman(log FQ, PrivLeak) over {len(sw)} settings = {rho:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["scale", "model2", "model2low", "summarize"])
    a = ap.parse_args()
    os.chdir(HERE)
    for d in (CK, RES, LOGS):
        os.makedirs(d, exist_ok=True)
    {"scale": scale, "model2": model2, "summarize": summarize,
     "model2low": lambda: model2(LOW, "tinyllama_low")}[a.phase]()
    say(f"FINISHED WITH FAILURES: {FAILED}" if FAILED else f"FINISHED - all steps OK  ({(time.time() - T0) / 3600:.2f} h)")


if __name__ == "__main__":
    main()
