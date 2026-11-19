import argparse
import glob
import shutil
import json
import os
import sys

os.environ.setdefault("RES_DIR", "results/v6")
os.environ.setdefault("LOG_DIR", "logs/v6")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import ks_2samp

import experiments_v5 as v5
from experiments_v5 import (CK, DATA, EARLY, EPOCHS, LLAMA, PY, QWEN, RES, SMOKE, TRAJ_EVERY, Eval,
                            done_ck, ft_job, lr_tag, ngpu, prep_dirs, run_jobs, say, summarize)

TRAJ6 = os.environ.get("TRAJ_EVERY6", "1.0")


def MOFF(base):
    return ("match_off", "match_off", ["--alpha", "1.0", "--base_model", base])


def ANCHOR(base, shift=False):
    return ("anchor_shift" if shift else "anchor", "anchor",
            ["--alpha", "1.0", "--base_model", base] + (["--anchor_shift"] if shift else []))


def KLB(base):
    return ("kl_base", "kl_base", ["--alpha", "1.0", "--base_model", base])


def CIU(base, w=1.0, shift=False):
    return (f"ciu_w{w:g}" + ("_shift" if shift else ""), "ciu",
            ["--alpha", "1.0", "--base_model", base, "--w_ind", str(w)] + (["--anchor_shift"] if shift else []))


def KEYMAP(base):
    m = {"match_off": MOFF(base), "anchor": ANCHOR(base), "anchor_shift": ANCHOR(base, True), "kl_base": KLB(base)}
    for w in (1, 3, 10):
        m[f"ciu_w{w}"] = CIU(base, w)
        m[f"ciu_w{w}_shift"] = CIU(base, w, True)
    return m


def ul6(prefix, a_ckpt, res, ref_raw, base, configs, seed=42, forget="forget10", retain="retain90",
        extra=(), traj=False, early=True, tag=""):
    jobs = []
    for (key, method, a), lr in configs:
        out = f"{CK}/{prefix}_{key}_{forget}_lr{lr_tag(lr)}_s{seed}"
        name = f"{tag}{key}_lr{lr_tag(lr)}" + ("" if forget == "forget10" else f"_{forget}")
        cmd = [PY, "unlearn.py", "--model", a_ckpt, "--method", method, "--out", out, "--forget", forget,
               "--retain", retain, "--data_dir", DATA, "--epochs", str(EPOCHS), "--lr", str(lr),
               "--bs", "4", "--grad_accum", "8", "--seed", str(seed), "--warmup_epochs", "1.0", *a, *extra]
        if early:
            cmd += ["--save_epochs", str(EARLY)]
        if traj:
            cmd += ["--traj_out", f"{res}/traj_{name}_s{seed}.json", "--traj_every", TRAJ6]
        ev = [Eval(f"{name}_ep{EPOCHS}_s{seed}", out, res, forget=forget, ref=ref_raw, base=base)]
        if early:
            ev.append(Eval(f"{name}_ep{EARLY}_s{seed}", f"{out}_ep{EARLY}", res, forget=forget, ref=ref_raw, base=base))
        jobs.append({"name": out, "train": (f"train_{os.path.basename(out)}", cmd, out), "evals": ev,
                     "cleanup": [out, f"{out}_ep{EARLY}"]})
    return jobs


class HubEval(Eval):

    def run(self, gpu):
        ok = super().run(gpu)
        hub = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        import shutil
        shutil.rmtree(os.path.join(hub, "hub", "models--" + self.ckpt.replace("/", "--")), ignore_errors=True)
        return ok


def run_jobs6(jobs, n):
    hub = [j for j in jobs if j.get("hub")]
    local = [j for j in jobs if not j.get("hub")]
    if local:
        run_jobs(local, n)
    if hub:
        import threading
        q = list(hub)
        lock = threading.Lock()

        def worker(g):
            while True:
                with lock:
                    if not q:
                        return
                    j = q.pop(0)
                if (v5.time.time() - v5.T0) / 3600 > v5.MAX_HOURS:
                    say(f"SKIP   {j['name']}: time budget reached")
                    continue
                for e in j["evals"]:
                    if not e.done():
                        e.run(g)
        ts = [v5.threading.Thread(target=worker, args=(g,)) for g in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()


def base_pair6(res, model, forget_sets=("forget10",), mem=()):
    retain = {"forget10": "retain90", "forget05": "retain95", "forget01": "retain99"}
    oname = {"forget10": "O_s42", "forget05": "O95_s42", "forget01": "O99_s42"}
    names = ["A_s42"] + [oname[f] for f in forget_sets]
    bk = os.environ.get("BACKUP_DIR")
    if bk:
        for p in names:
            if done_ck(f"{bk}/{p}") and not done_ck(f"{CK}/{p}"):
                say(f"restoring {p} from {bk}")
                shutil.copytree(f"{bk}/{p}", f"{CK}/{p}", dirs_exist_ok=True)
    need = [p for p in names if not done_ck(f"{CK}/{p}")]
    oname_done = {p for p in names if os.path.exists(f"{res}/{p}.json") or glob.glob(f"{res}/{p}_forget0*.json")}
    jobs0 = [ft_job("A_s42", model, "full", 42, mem)] + [ft_job(oname[f], model, retain[f], 42, mem)
                                                         for f in forget_sets]
    run_jobs([j for j in jobs0 if j["name"] in need and j["name"] not in (oname_done - {"A_s42"})], ngpu())
    for p in names:
        if not done_ck(f"{CK}/{p}") and not (p != "A_s42" and p in oname_done):
            sys.exit(f"{p} missing - see logs")
        if bk and done_ck(f"{CK}/{p}") and not done_ck(f"{bk}/{p}"):
            say(f"backing up {p} to {bk}")
            shutil.copytree(f"{CK}/{p}", f"{bk}/{p}", dirs_exist_ok=True)
    refs, jobs = {}, []
    for f in forget_sets:
        nm = oname[f] if f == "forget10" else f"{oname[f]}_{f}"
        refs[f] = f"{res}/{nm}_raw.json"
        jobs.append({"name": f"eval_{nm}", "train": None, "cleanup": [f"{CK}/{oname[f]}"],
                     "evals": [Eval(nm, f"{CK}/{oname[f]}", res, forget=f, base=model)]})
    for f in forget_sets:
        nm = "A_s42" if f == "forget10" else f"A_s42_{f}"
        jobs.append({"name": f"eval_{nm}", "train": None,
                     "evals": [Eval(nm, f"{CK}/A_s42", res, forget=f, ref=refs[f], base=model)]})
    return refs, jobs


def interleave(head, rest):
    return head[:1] + rest[:1] + head[1:] + rest[1:]


RUN11 = {
    "a": lambda b: [(MOFF(b), 5e-6), (ANCHOR(b), 5e-6), (KLB(b), 1e-5), (MOFF(b), 1e-5), (ANCHOR(b), 1e-5),
                    (ANCHOR(b, True), 1e-5), (KLB(b), 2e-5)],
    "b": lambda b: [(CIU(b, 1), 5e-6), (CIU(b, 3), 5e-6), (CIU(b, 1), 1e-5), (CIU(b, 3), 1e-5), (CIU(b, 10), 1e-5),
                    (CIU(b, 1), 2e-5)],
}


def run11(part):
    prep_dirs()
    res = f"{RES}/run11{part}"
    os.makedirs(res, exist_ok=True)
    say(f"=== v6 RUN 11{part}  GPUs={ngpu()}  model={QWEN}")
    refs, jobs = base_pair6(res, QWEN)
    A = f"{CK}/A_s42"
    cfg = [(CIU(QWEN, 1), 1e-5), (ANCHOR(QWEN), 1e-5)] if SMOKE else RUN11[part](QWEN)
    if SMOKE and part == "a":
        cfg = [(ANCHOR(QWEN), 1e-5), (KLB(QWEN), 1e-5)]
    ul = ul6("r11", A, res, refs["forget10"], QWEN, cfg, traj=True)
    run_jobs(interleave(jobs, ul), ngpu())
    if part == "a":
        run_jobs6([{"name": "eval_base", "hub": True,
                    "evals": [HubEval("base_model", QWEN, res, ref=refs["forget10"], base=QWEN)]}], 1)
    summarize()


def _pct(m, n):
    n = np.sort(np.asarray(n, float))
    lo, hi = np.searchsorted(n, m, "left"), np.searchsorted(n, m, "right")
    return (lo + 0.5 * (hi - lo)) / len(n)


def rq_of(raw, oraw):
    fq = ks_2samp(raw["forget"]["truth_ratio"], oraw["forget"]["truth_ratio"]).pvalue
    m, n = raw["mia"]["members"]["mink"], raw["mia"]["nonmembers"]["mink"]
    om, on = oraw["mia"]["members"]["mink"], oraw["mia"]["nonmembers"]["mink"]
    pq = ks_2samp(_pct(m, n), _pct(om, on)).pvalue
    return float(fq), float(pq)


def extra_settings():
    out = []
    for s in os.environ.get("V6_EXTRA", "anchor:1e-5:10").split(","):
        if s.strip():
            key, lr, ep = s.split(":")
            out.append((key, float(lr), int(ep), "post hoc (added after run 11)"))
    return out


def pick_v6(k=2):
    picks = _pick_rule(k)
    for e in extra_settings():
        if (e[0], e[1]) not in {(p[0], p[1]) for p in picks}:
            picks.append(e)
    return picks


def _pick_rule(k=2, pats=("run11a", "run11b"), label="selection rule on run 11", env="V6_SETTINGS"):
    if os.environ.get(env):
        out = []
        for s in os.environ[env].split(","):
            key, lr, ep = s.split(":")
            out.append((key, float(lr), int(ep), env))
        return out
    dirs = sorted({os.path.dirname(p) for pat in pats for p in
                   glob.glob(f"/kaggle/input/**/{pat}/O_s42_raw.json", recursive=True)
                   + glob.glob(f"{RES}/{pat}/O_s42_raw.json")})
    cand = []
    for d in dirs:
        oraw = json.load(open(f"{d}/O_s42_raw.json"))
        mu_a = json.load(open(f"{d}/A_s42.json"))["model_utility"]
        for p in glob.glob(f"{d}/*_lr*_ep*_s42.json"):
            r = json.load(open(p))
            name = os.path.basename(p)[:-5]
            key, rest = name.split("_lr")
            lr, ep = rest.split("_ep")[0], int(rest.split("_ep")[1].split("_")[0])
            fq, pq = rq_of(json.load(open(p[:-5] + "_raw.json")), oraw)
            pl = r["privacy"]["privleak_mink"]
            cand.append({"key": key, "lr": float(lr), "ep": ep, "fq": fq, "pq": pq, "rq": min(fq, pq),
                         "pl": pl, "util": r["model_utility"] / mu_a})
    if not cand:
        say(f"{'/'.join(pats)} results not found: default settings")
        if pats[0].startswith("run16"):
            return [("anchor_shift_p10", 1e-5, 5, "default")]
        return [("ciu_w1", 1e-5, 10, "default"), ("anchor", 1e-5, 10, "default")]
    ok = [c for c in cand if c["util"] >= 0.9] or cand
    ok.sort(key=lambda c: (-c["rq"], abs(c["pl"])))
    for c in ok[:6]:
        say(f"  candidate {c['key']} lr={c['lr']:g} ep{c['ep']}: FQ={c['fq']:.3g} PQ={c['pq']:.3g} "
            f"PrivLeak={c['pl']:+.2f} util={c['util']:.2f}")
    picked, seen = [], set()
    for c in ok:
        if (c["key"], c["lr"]) in seen:
            continue
        seen.add((c["key"], c["lr"]))
        picked.append((c["key"], c["lr"], c["ep"], label))
        if len(picked) == k:
            break
    return picked


def run12():
    prep_dirs()
    res = f"{RES}/run12"
    os.makedirs(res, exist_ok=True)
    picks = pick_v6()
    say(f"=== v6 RUN 12  GPUs={ngpu()}  settings: {picks}")
    km = KEYMAP(QWEN)
    sizes = ("forget10",) if (SMOKE and not os.environ.get("SMOKE_SIZES")) else ("forget10", "forget05", "forget01")
    refs, jobs = base_pair6(res, QWEN, sizes)
    A = f"{CK}/A_s42"
    seeds = [1] if SMOKE else [1, 2, 3]
    ul = []
    for sd in seeds:
        for key, lr, ep, _ in picks:
            ul += ul6("r12", A, res, refs["forget10"], QWEN, [(km[key], lr)], seed=sd, early=(ep == EARLY),
                      traj=(sd == 1 and key == picks[0][0]))
    for key, lr, ep, why in [picks[0]] + [p for p in picks[1:] if p[3].startswith("post hoc")]:
        for f in sizes[1:]:
            retain = {"forget05": "retain95", "forget01": "retain99"}[f]
            for sd in (42, 1):
                ul += ul6("r12", A, res, refs[f], QWEN, [(km[key], lr)], seed=sd, forget=f, retain=retain,
                          early=(ep == EARLY))
    run_jobs(interleave(jobs, ul), ngpu())
    summarize()


def run13(part):
    prep_dirs()
    res = f"{RES}/run13{part}"
    os.makedirs(res, exist_ok=True)
    picks = pick_v6()
    say(f"=== v6 RUN 13{part}  GPUs={ngpu()}  model={LLAMA}  settings: {picks}")
    mem = os.environ.get("MODEL3_MEM", "--optim adamw8bit --grad_ckpt --freeze_embed").split()
    ft_bs = os.environ.get("MODEL3_FT_BS", "--bs 2 --grad_accum 16").split()
    ul_bs = os.environ.get("MODEL3_UL_BS", "--bs 2 --grad_accum 16").split()
    refs, jobs = base_pair6(res, LLAMA, mem=mem + ft_bs)
    km = KEYMAP(LLAMA)
    A = f"{CK}/A_s42"
    todo = [(picks[0], 42), (picks[0], 1)] if part == "a" else [(picks[-1], 42), (picks[-1], 1)]
    ul = []
    for (key, lr, ep, _), sd in todo:
        ul += ul6("r13", A, res, refs["forget10"], LLAMA, [(km[key], lr)], seed=sd, early=True, extra=mem + ul_bs)
    run_jobs(interleave(jobs, ul), ngpu())
    if part == "b":
        run_jobs6([{"name": "eval_base", "hub": True,
                    "evals": [HubEval("base_model", LLAMA, res, ref=refs["forget10"], base=LLAMA)]}], 1)
    summarize()


def run12b():
    prep_dirs()
    res = f"{RES}/run12b"
    os.makedirs(res, exist_ok=True)
    say(f"=== v6 RUN 12b  GPUs={ngpu()}  anchor_shift 1e-5, seeds 4-8")
    refs, jobs = base_pair6(res, QWEN)
    km = KEYMAP(QWEN)
    ul = []
    for sd in ([4] if SMOKE else [4, 5, 6, 7, 8]):
        ul += ul6("r12b", f"{CK}/A_s42", res, refs["forget10"], QWEN, [(km["anchor_shift"], 1e-5)], seed=sd)
    run_jobs(interleave(jobs, ul), ngpu())
    summarize()


def run13c():
    prep_dirs()
    res = f"{RES}/run13c"
    os.makedirs(res, exist_ok=True)
    say(f"=== v6 RUN 13c  GPUs={ngpu()}  model={LLAMA}  anchor_shift 1e-5")
    mem = os.environ.get("MODEL3_MEM", "--optim adamw8bit --grad_ckpt --freeze_embed").split()
    ft_bs = os.environ.get("MODEL3_FT_BS", "--bs 2 --grad_accum 16").split()
    ul_bs = os.environ.get("MODEL3_UL_BS", "--bs 2 --grad_accum 16").split()
    refs, jobs = base_pair6(res, LLAMA, mem=mem + ft_bs)
    km = KEYMAP(LLAMA)
    ul = []
    for sd in ([42] if SMOKE else [42, 1]):
        ul += ul6("r13c", f"{CK}/A_s42", res, refs["forget10"], LLAMA, [(km["anchor_shift"], 1e-5)], seed=sd,
                  extra=mem + ul_bs)
    run_jobs(interleave(jobs, ul), ngpu())
    summarize()


PROXY_FRACS = (0.05, 0.10, 0.25)


def ptag(f):
    return f"p{int(round(f * 100)):02d}"


def proxy_job(model, frac, seed=42, mem=()):
    name = f"P{ptag(frac)}_s42"
    j = ft_job(name, model, "retain90", seed, list(mem) + ["--author_frac", str(frac), "--sample_seed", "0"])
    return j, f"{CK}/{name}"


def ANCHOR_P(path, tag, shift=False):
    return (("anchor_shift_" if shift else "anchor_") + tag, "anchor",
            ["--alpha", "1.0", "--base_model", path] + (["--anchor_shift"] if shift else []))


def proxy_stage(res, model=None, fracs=PROXY_FRACS, mem=(), evaluate=True, ref=None):
    model = model or QWEN
    jobs, paths = [], {}
    for f in fracs:
        j, path = proxy_job(model, f, mem=mem)
        if evaluate and ref:
            j["evals"] = [Eval(f"proxy_{ptag(f)}", path, res, ref=ref, base=model)]
        jobs.append(j)
        paths[ptag(f)] = path
    run_jobs(jobs, ngpu())
    for t, pth in paths.items():
        if not done_ck(pth):
            sys.exit(f"proxy {t} missing - see logs")
    return paths


def run16a():
    prep_dirs()
    res = f"{RES}/run16a"
    os.makedirs(res, exist_ok=True)
    say(f"=== v6 RUN 16a  GPUs={ngpu()}  proxies {PROXY_FRACS}")
    refs, jobs = base_pair6(res, QWEN)
    run_jobs(jobs[:1], ngpu())
    fracs = (0.10,) if SMOKE else PROXY_FRACS
    paths = proxy_stage(res, fracs=fracs, ref=refs["forget10"])
    cfg = []
    for t, pth in paths.items():
        cfg += [(ANCHOR_P(pth, t, True), 1e-5), (ANCHOR_P(pth, t), 1e-5)]
        if not SMOKE:
            cfg.append((ANCHOR_P(pth, t), 5e-6))
    ul = ul6("r16a", f"{CK}/A_s42", res, refs["forget10"], QWEN, cfg)
    for j in ul:
        if j["name"].endswith("anchor_shift_p10_forget10_lr1e-5_s42"):
            j["train"][1].extend(["--traj_out", f"{res}/traj_anchor_shift_p10_lr1e-5_s42.json", "--traj_every", TRAJ6])
    run_jobs(jobs[1:] + ul, ngpu())
    summarize()


def run16b():
    prep_dirs()
    res = f"{RES}/run16b"
    os.makedirs(res, exist_ok=True)
    key, lr, ep, why = _pick_rule(1, ("run16a",), "selection rule on run 16a", "V16_SETTING")[0]
    t = key.split("_")[-1]
    frac = int(t[1:]) / 100
    say(f"=== v6 RUN 16b  GPUs={ngpu()}  setting {key} lr={lr:g} ep{ep} ({why}), seeds 1-8")
    refs, jobs = base_pair6(res, QWEN)
    run_jobs(jobs[:1], ngpu())
    paths = proxy_stage(res, fracs=(frac,), ref=refs["forget10"])
    cfg = [(ANCHOR_P(paths[t], t, key.startswith("anchor_shift")), lr)]
    ul = []
    for sd in ([1] if SMOKE else [1, 2, 3, 4, 5, 6, 7, 8]):
        ul += ul6("r16b", f"{CK}/A_s42", res, refs["forget10"], QWEN, cfg, seed=sd, early=True)
    run_jobs(jobs[1:] + ul, ngpu())
    summarize()


def run16c():
    prep_dirs()
    res = f"{RES}/run16c"
    os.makedirs(res, exist_ok=True)
    say(f"=== v6 RUN 16c  GPUs={ngpu()}  model={LLAMA}  anchor_p10 1e-5")
    mem = os.environ.get("MODEL3_MEM", "--optim adamw8bit --grad_ckpt --freeze_embed").split()
    ft_bs = os.environ.get("MODEL3_FT_BS", "--bs 2 --grad_accum 16").split()
    ul_bs = os.environ.get("MODEL3_UL_BS", "--bs 2 --grad_accum 16").split()
    refs, jobs = base_pair6(res, LLAMA, mem=mem + ft_bs)
    run_jobs(jobs[:1], ngpu())
    paths = proxy_stage(res, model=LLAMA, fracs=(0.10,), mem=mem + ft_bs, ref=refs["forget10"])
    cfg = [(ANCHOR_P(paths["p10"], "p10"), 1e-5)]
    ul = []
    for sd in ([42] if SMOKE else [42, 1, 2, 3]):
        ul += ul6("r16c", f"{CK}/A_s42", res, refs["forget10"], LLAMA, cfg, seed=sd, extra=mem + ul_bs)
    run_jobs(jobs[1:] + ul, ngpu())
    summarize()


RUN17_REPS = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8],
              "c": [9, 10, 11, 12], "d": [13, 14, 15, 16]}


def run17(part):
    prep_dirs()
    res = f"{RES}/run17{part}"
    os.makedirs(res, exist_ok=True)
    reps = [1] if SMOKE else RUN17_REPS[part]
    say(f"=== v7 RUN 17{part}  GPUs={ngpu()}  full replicates {reps}")
    for k in reps:
        a, o, pr = f"A_s{k}", f"O_s{k}", f"P10_s{k}"
        if not os.path.exists(f"{res}/anchor_p10_lr1e-5_ep10_s{k}.json"):
            run_jobs([ft_job(a, QWEN, "full", k), ft_job(o, QWEN, "retain90", k)], ngpu())
        ref = f"{res}/{o}_raw.json"
        pj = ft_job(pr, QWEN, "retain90", k, ["--author_frac", "0.1", "--sample_seed", str(k)])
        pj["evals"] = []
        cfg = [(ANCHOR_P(f"{CK}/{pr}", "p10"), 1e-5)]
        ul = ul6("r17", f"{CK}/{a}", res, ref, QWEN, cfg, seed=k)
        ul[0]["cleanup"] += [f"{CK}/{a}", f"{CK}/{pr}"]
        jobs = [{"name": f"eval_{o}", "train": None, "cleanup": [f"{CK}/{o}"],
                 "evals": [Eval(o, f"{CK}/{o}", res, base=QWEN)]},
                pj,
                {"name": f"eval_{a}", "train": None, "evals": [Eval(a, f"{CK}/{a}", res, ref=ref, base=QWEN)]}] + ul
        run_jobs(jobs, ngpu())
    summarize()


def run18():
    prep_dirs()
    res = f"{RES}/run18"
    os.makedirs(res, exist_ok=True)
    say(f"=== v7 RUN 18  GPUs={ngpu()}  KL to the 10% proxy")
    refs, jobs = base_pair6(res, QWEN)
    run_jobs(jobs[:1], ngpu())
    paths = proxy_stage(res, fracs=(0.10,), ref=refs["forget10"])
    klp = ("kl_proxy_p10", "kl_base", ["--alpha", "1.0", "--base_model", paths["p10"]])
    lrs = [1e-5] if SMOKE else [5e-6, 1e-5, 2e-5]
    ul = []
    for sd in ([42] if SMOKE else [42, 1, 2]):
        ul += ul6("r18", f"{CK}/A_s42", res, refs["forget10"], QWEN, [(klp, lr) for lr in lrs], seed=sd)
    run_jobs(jobs[1:] + ul, ngpu())
    summarize()


OU_ORG = os.environ.get("OU_ORG", "open-unlearning")
OU_TARGET = f"{OU_ORG}/tofu_Llama-3.2-1B-Instruct_full"
OU_ORACLE = f"{OU_ORG}/tofu_Llama-3.2-1B-Instruct_retain90"
OU_PREFIX = "unlearn_tofu_Llama-3.2-1B-Instruct_forget10_"


def ou_models():
    from huggingface_hub import HfApi
    ids = sorted(m.id for m in HfApi().list_models(author=OU_ORG, search=OU_PREFIX, limit=2000))
    ids = [i for i in ids if i.split("/")[-1].startswith(OU_PREFIX)]
    by = {}
    for i in ids:
        by.setdefault(i.split("/")[-1][len(OU_PREFIX):].split("_")[0], []).append(i)
    order = []
    while any(by.values()):
        for k in sorted(by):
            if by[k]:
                order.append(by[k].pop(0))
    return order


def run14(part=""):
    prep_dirs()
    res = f"{RES}/run14{part}"
    os.makedirs(res, exist_ok=True)
    ids = ou_models()
    done = {os.path.basename(p)[3:-5] for p in glob.glob(f"{RES}/run14*/ou_*.json") if not p.endswith("_raw.json")}
    if part:
        first = {os.path.basename(p)[3:-5] for p in glob.glob(f"{RES}/run14/ou_*.json") if not p.endswith("_raw.json")}
        left = [i for i in ids if i.split("/")[-1][len(OU_PREFIX):] not in (first or done)]
        by = {}
        for i in left:
            t = i.split("/")[-1][len(OU_PREFIX):].split("_")
            by.setdefault((t[0], t[1]), []).append(i)
        for k in by:
            by[k].sort(key=lambda i: (not i.endswith(("epoch10", "ep10")), i))
        ids = []
        while any(by.values()):
            for k in sorted(by, key=lambda k: (k[1], k[0]), reverse=True):
                if by[k]:
                    ids.append(by[k].pop(0))
        if os.environ.get("OU_SHARD", "1") != "0" and part in ("b", "c"):
            ids = ids[0::2] if part == "b" else ids[1::2]
        ids = [i for i in ids if i.split("/")[-1][len(OU_PREFIX):] not in done]
        prev = sorted(d for d in glob.glob(f"{RES}/run14*") if os.path.exists(f"{d}/O_s42_raw.json") and d != res)
        if prev:
            for f in ("O_s42.json", "O_s42_raw.json", "A_s42.json", "A_s42_raw.json", "_refscores_forget10.json"):
                if os.path.exists(f"{prev[0]}/{f}") and not os.path.exists(f"{res}/{f}"):
                    shutil.copy(f"{prev[0]}/{f}", f"{res}/{f}")
    if SMOKE:
        ids = ids[:2]
    say(f"=== v6 RUN 14{part}  GPUs={ngpu()}: {len(ids)} OpenUnlearning checkpoints to score "
        f"({len(done)} already scored), oracle {OU_ORACLE}")
    json.dump(ids, open(f"{res}/_checkpoint_list.json", "w"), indent=1)
    base = LLAMA
    fast = ["--no_gen", "--bs", "8"]
    ref = f"{res}/O_s42_raw.json"
    run_jobs6([{"name": "oracle", "hub": True, "evals": [HubEval("O_s42", OU_ORACLE, res, base=base, extra=fast)]}], 1)
    if not os.path.exists(ref):
        sys.exit("oracle evaluation failed")
    head = [{"name": "target", "hub": True,
             "evals": [HubEval("A_s42", OU_TARGET, res, ref=ref, base=base, extra=fast)]}]
    rest = [{"name": i, "hub": True,
             "evals": [HubEval("ou_" + i.split("/")[-1][len(OU_PREFIX):], i, res, ref=ref, base=base, extra=fast)]}
            for i in ids]
    run_jobs6(head + rest, ngpu())
    summarize()


def run15():
    prep_dirs()
    big = os.environ.get("MODEL_BIG", "Qwen/Qwen2.5-3B-Instruct")
    res = f"{RES}/run15"
    os.makedirs(res, exist_ok=True)
    picks = pick_v6()
    mem = os.environ.get("BIG_MEM", "--optim adamw8bit --grad_ckpt").split()
    refs, jobs = base_pair6(res, big, mem=mem)
    km = KEYMAP(big)
    ul = []
    for sd in (42, 1, 2):
        for key, lr, ep, _ in picks:
            ul += ul6("r15", f"{CK}/A_s42", res, refs["forget10"], big, [(km[key], lr)], seed=sd, extra=mem,
                      traj=(sd == 42 and key == picks[0][0]))
    from experiments_v5 import NPO_RT, SIM25
    ul += ul6("r15", f"{CK}/A_s42", res, refs["forget10"], big, [(NPO_RT, 1e-5), (SIM25, 5e-5)], extra=mem)
    run_jobs(interleave(jobs, ul), ngpu())
    summarize()


RUN20_REPS = {"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]}


def run20(part):
    prep_dirs()
    res = f"{RES}/run20{part}"
    os.makedirs(res, exist_ok=True)
    mem = os.environ.get("MODEL3_MEM", "--optim adamw8bit --grad_ckpt --freeze_embed").split()
    ft_bs = os.environ.get("MODEL3_FT_BS", "--bs 2 --grad_accum 16").split()
    ul_bs = os.environ.get("MODEL3_UL_BS", "--bs 2 --grad_accum 16").split()
    reps = [1] if SMOKE else RUN20_REPS[part]
    say(f"=== v8 RUN 20{part}  GPUs={ngpu()}  model={LLAMA}  full replicates {reps}")
    for k in reps:
        a, o, pr = f"A_s{k}", f"O_s{k}", f"P10_s{k}"
        if not os.path.exists(f"{res}/anchor_p10_lr1e-5_ep10_s{k}.json"):
            run_jobs([ft_job(a, LLAMA, "full", k, mem + ft_bs), ft_job(o, LLAMA, "retain90", k, mem + ft_bs)], ngpu())
        ref = f"{res}/{o}_raw.json"
        pj = ft_job(pr, LLAMA, "retain90", k, mem + ft_bs + ["--author_frac", "0.1", "--sample_seed", str(k)])
        pj["evals"] = []
        cfg = [(ANCHOR_P(f"{CK}/{pr}", "p10"), 1e-5)]
        ul = ul6("r20", f"{CK}/{a}", res, ref, LLAMA, cfg, seed=k, extra=mem + ul_bs)
        ul[0]["cleanup"] += [f"{CK}/{a}", f"{CK}/{pr}"]
        jobs = [{"name": f"eval_{o}", "train": None, "cleanup": [f"{CK}/{o}"],
                 "evals": [Eval(o, f"{CK}/{o}", res, base=LLAMA)]},
                pj,
                {"name": f"eval_{a}", "train": None, "evals": [Eval(a, f"{CK}/{a}", res, ref=ref, base=LLAMA)]}] + ul
        run_jobs(jobs, ngpu())
    summarize()


def run17e():
    prep_dirs()
    res = f"{RES}/run17e"
    os.makedirs(res, exist_ok=True)
    reps = [1] if SMOKE else [1, 2, 3, 4, 5, 6, 7, 8]
    say(f"=== v8.5 RUN 17e  GPUs={ngpu()}  replicate proxies {reps}")
    for k in reps:
        pr = f"P10_s{k}"
        nm = f"{pr}_forget10"
        if os.path.exists(f"{res}/{nm}_raw.json"):
            continue
        if not done_ck(f"{CK}/{pr}"):
            run_jobs([ft_job(pr, QWEN, "retain90", k, ["--author_frac", "0.1", "--sample_seed", str(k)])], ngpu())
        run_jobs([{"name": f"eval_{nm}", "train": None, "cleanup": [f"{CK}/{pr}"],
                   "evals": [Eval(nm, f"{CK}/{pr}", res, forget="forget10", base=QWEN)]}], ngpu())
    summarize()


def run24(part):
    prep_dirs()
    f, retain = {"a": ("forget05", "retain95"), "b": ("forget01", "retain99")}[part]
    tag = {"forget05": "95", "forget01": "99"}[f]
    res = f"{RES}/run24{part}"
    os.makedirs(res, exist_ok=True)
    seeds = [1] if SMOKE else [1, 2, 3, 4]
    say(f"=== v8.1 RUN 24{part}  GPUs={ngpu()}  extra oracles {retain}, seeds {seeds}")
    for k in seeds:
        o = f"O{tag}_s{k}"
        nm = f"{o}_{f}"
        if os.path.exists(f"{res}/{nm}_raw.json"):
            continue
        if not done_ck(f"{CK}/{o}"):
            run_jobs([ft_job(o, QWEN, retain, k)], ngpu())
        run_jobs([{"name": f"eval_{nm}", "train": None, "cleanup": [f"{CK}/{o}"],
                   "evals": [Eval(nm, f"{CK}/{o}", res, forget=f, base=QWEN)]}], ngpu())
    summarize()


def KAPPA_CFG(mode, w):
    return (f"npo_kappa_{mode}_w{w:g}", "npo_kappa", ["--beta", "0.1", "--alpha", "1.0", "--kappa_mode", mode,
                                                      "--w_kappa", str(w), "--n_cf", "3"])


RUN21_SEED = {"a": 42, "b": 1}


def run21(part):
    prep_dirs()
    res = f"{RES}/run21{part}"
    os.makedirs(res, exist_ok=True)
    from experiments_v5 import NPO_RT
    seed = RUN21_SEED[part]
    say(f"=== v8 RUN 21{part}  GPUs={ngpu()}  kappa intervention, seed {seed}")
    refs, jobs = base_pair6(res, QWEN)
    run_jobs(jobs[:1], ngpu())
    cfg = [(NPO_RT, 1e-5)] + [(KAPPA_CFG("hold", w), 1e-5) for w in (1, 3, 10)] + [(KAPPA_CFG("push", 1), 1e-5)]
    if SMOKE:
        cfg = cfg[:1] + cfg[1:2] + cfg[-1:]
    ul = ul6("r21", f"{CK}/A_s42", res, refs["forget10"], QWEN, cfg, seed=seed, traj=True)
    for j in ul:
        cmd = j["train"][1]
        cmd[cmd.index("--traj_every") + 1] = "0.5"
    run_jobs(jobs[1:] + ul, ngpu())
    summarize()


M15 = os.environ.get("MODEL_15", "Qwen/Qwen2.5-1.5B-Instruct")
M3 = os.environ.get("MODEL_3B", "Qwen/Qwen2.5-3B-Instruct")


def run22():
    from experiments_v5 import NPO_RT, SIM25
    prep_dirs()
    res = f"{RES}/run22"
    os.makedirs(res, exist_ok=True)
    mem = os.environ.get("M15_MEM", "--optim adamw8bit --grad_ckpt --freeze_embed").split()
    bs = os.environ.get("M15_BS", "--bs 2 --grad_accum 16").split()
    say(f"=== v8 RUN 22  GPUs={ngpu()}  model={M15} (full fine-tuning)")
    refs, jobs = base_pair6(res, M15, mem=mem + bs)
    run_jobs(jobs[:1], ngpu())
    paths = proxy_stage(res, M15, fracs=(0.10,), mem=mem + bs, evaluate=False)
    cfg = [(NPO_RT, 1e-5), (ANCHOR_P(paths["p10"], "p10"), 1e-5), (NPO_RT, 2e-5), (SIM25, 5e-5)]
    if SMOKE:
        cfg = cfg[:2]
    ul = []
    for i, c in enumerate(cfg):
        ul += ul6("r22", f"{CK}/A_s42", res, refs["forget10"], M15, [c], extra=mem + bs + ["--precompute_ref"], traj=(i < 2))
    run_jobs(jobs[1:] + ul, ngpu())
    summarize()


LORA = ["--lora_r", "32", "--grad_ckpt"]


def run23(part):
    from experiments_v5 import NPO_RT
    prep_dirs()
    folder = {"bnpo": "b", "banchor": "b", "canchor": "c", "danchor": "d"}.get(part, part)
    res = f"{RES}/run23{folder}"
    os.makedirs(res, exist_ok=True)
    keep = os.environ.get("RUN23_KEEP", "/kaggle/working/run23_ckpt")
    ft = LORA + ["--lr", "1e-4", "--bs", "4", "--grad_accum", "8"]
    say(f"=== v8 RUN 23{part}  GPUs={ngpu()}  model={M3} (LoRA)")
    if part == "a":
        refs, jobs = base_pair6(res, M3, mem=ft)
        for p in ("A_s42", "O_s42"):
            shutil.copytree(f"{CK}/{p}", f"{keep}/{p}", dirs_exist_ok=True)
        for j in jobs:
            j["cleanup"] = []
        run_jobs(jobs, ngpu())
        summarize()
        return
    for p in ("A_s42", "O_s42"):
        if not done_ck(f"{CK}/{p}"):
            src = [os.path.dirname(f) for f in glob.glob(f"{os.environ.get('RUN23_INPUT', '/kaggle/input')}/**/run23_ckpt/{p}/lora_chain.json", recursive=True)]
            if not src:
                sys.exit(f"{p} adapter not found: attach the output of the run 23a notebook")
            shutil.copytree(src[0], f"{CK}/{p}", dirs_exist_ok=True)
    ref = glob.glob(f"{os.environ.get('RUN23_INPUT', '/kaggle/input')}/**/run23a/O_s42_raw.json", recursive=True)
    if not ref:
        sys.exit("O_s42_raw.json of run 23a not found: attach the output of the run 23a notebook")
    shutil.copy(ref[0], f"{res}/O_s42_raw.json")
    for f in ("O_s42.json", "A_s42.json", "A_s42_raw.json"):
        src = glob.glob(f"{os.path.dirname(ref[0])}/{f}")
        if src:
            shutil.copy(src[0], f"{res}/{f}")
    ul_x = LORA + ["--bs", "4", "--grad_accum", "8", "--precompute_ref"]
    cfg = []
    if part in ("b", "bnpo"):
        cfg.append((NPO_RT, 1e-4))
    if part in ("b", "banchor", "canchor", "danchor"):
        pj, ppath = proxy_job(M3, 0.10, mem=ft)
        run_jobs([pj], ngpu())
        if not done_ck(ppath):
            sys.exit("proxy missing - see logs")
        cfg.append((ANCHOR_P(ppath, "p10"), {"canchor": 3e-5, "danchor": 3e-4}.get(part, 1e-4)))
    ul = []
    for c in cfg:
        ul += ul6("r23", f"{CK}/A_s42", res, f"{res}/O_s42_raw.json", M3, [c], extra=ul_x, traj=True)
    run_jobs(ul, ngpu())
    summarize()


RUN25 = {"a1": ("a", 1), "b1": ("b", 1), "a2": ("a", 2), "b2": ("b", 2)}


def run25(key):
    from experiments_v5 import NPO_RT
    prep_dirs()
    part, k = RUN25[key]
    res = f"{RES}/run25_s{k}"
    os.makedirs(res, exist_ok=True)
    keep = os.environ.get("RUN25_KEEP", "/kaggle/working/run25_ckpt")
    ft = LORA + ["--lr", "1e-4", "--bs", "4", "--grad_accum", "8"]
    a, o, pr = f"A_s{k}", f"O_s{k}", f"P10_s{k}"
    say(f"=== v8.6 RUN 25{key}  GPUs={ngpu()}  model={M3} (LoRA)  replicate {k}")
    if part == "a":
        run_jobs([ft_job(a, M3, "full", k, ft), ft_job(o, M3, "retain90", k, ft)], ngpu())
        for p in (a, o):
            if not done_ck(f"{CK}/{p}"):
                sys.exit(f"{p} missing - see logs")
            shutil.copytree(f"{CK}/{p}", f"{keep}/{p}", dirs_exist_ok=True)
        run_jobs([{"name": f"eval_{o}", "train": None, "evals": [Eval(o, f"{CK}/{o}", res, base=M3)]}], ngpu())
        run_jobs([{"name": f"eval_{a}", "train": None,
                   "evals": [Eval(a, f"{CK}/{a}", res, ref=f"{res}/{o}_raw.json", base=M3)]}], ngpu())
        summarize()
        return
    inp = os.environ.get("RUN25_INPUT", "/kaggle/input")
    if not done_ck(f"{CK}/{a}"):
        src = [os.path.dirname(f) for f in glob.glob(f"{inp}/**/run25_ckpt/{a}/lora_chain.json", recursive=True)]
        if not src:
            sys.exit(f"{a} adapter not found: attach the output of the run 25a{k} notebook")
        shutil.copytree(src[0], f"{CK}/{a}", dirs_exist_ok=True)
    ref = glob.glob(f"{inp}/**/run25_s{k}/{o}_raw.json", recursive=True)
    if not ref:
        sys.exit(f"{o}_raw.json not found: attach the output of the run 25a{k} notebook")
    for f in (f"{o}_raw.json", f"{o}.json", f"{a}.json", f"{a}_raw.json"):
        src = glob.glob(f"{os.path.dirname(ref[0])}/{f}")
        if src and not os.path.exists(f"{res}/{f}"):
            shutil.copy(src[0], f"{res}/{f}")
    if not done_ck(f"{CK}/{pr}"):
        run_jobs([ft_job(pr, M3, "retain90", k, ft + ["--author_frac", "0.1", "--sample_seed", str(k)])], ngpu())
    if not done_ck(f"{CK}/{pr}"):
        sys.exit("proxy missing - see logs")
    ul_x = LORA + ["--bs", "4", "--grad_accum", "8", "--precompute_ref"]
    cfg = [(NPO_RT, 1e-4), (ANCHOR_P(f"{CK}/{pr}", "p10"), 3e-4)]
    ul = []
    for c in cfg:
        ul += ul6("r25", f"{CK}/{a}", res, f"{res}/{o}_raw.json", M3, [c], seed=k, extra=ul_x, traj=False)
    run_jobs(ul, ngpu())
    summarize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run11a", "run11b", "run12", "run12b", "run13a", "run13b", "run13c", "run16a", "run16b", "run16c", "run17a", "run17b", "run17c", "run17d", "run18", "run20a", "run20b", "run20c", "run20d", "run21a", "run21b", "run22", "run23a", "run23b", "run23bnpo", "run23banchor", "run23canchor", "run23danchor", "run24a", "run24b", "run25a1", "run25b1", "run25a2", "run25b2", "run17e", "run14", "run14b", "run14c", "run15", "summarize", "pick"])
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for d in (CK, RES, v5.LOGS):
        os.makedirs(d, exist_ok=True)
    {"run11a": lambda: run11("a"), "run11b": lambda: run11("b"), "run12": run12, "run12b": run12b, "run13c": run13c, "run16a": run16a, "run16b": run16b, "run16c": run16c, "run17a": lambda: run17("a"), "run17b": lambda: run17("b"), "run17c": lambda: run17("c"), "run17d": lambda: run17("d"), "run18": run18, "run20a": lambda: run20("a"), "run20b": lambda: run20("b"), "run20c": lambda: run20("c"), "run20d": lambda: run20("d"), "run21a": lambda: run21("a"), "run21b": lambda: run21("b"), "run22": run22, "run23a": lambda: run23("a"), "run23b": lambda: run23("b"), "run23bnpo": lambda: run23("bnpo"), "run23banchor": lambda: run23("banchor"), "run23canchor": lambda: run23("canchor"), "run23danchor": lambda: run23("danchor"), "run24a": lambda: run24("a"), "run24b": lambda: run24("b"), "run25a1": lambda: run25("a1"), "run25b1": lambda: run25("b1"), "run25a2": lambda: run25("a2"), "run25b2": lambda: run25("b2"), "run17e": run17e, "run13a": lambda: run13("a"), "run13b": lambda: run13("b"),
     "run14": run14, "run14b": lambda: run14("b"), "run14c": lambda: run14("c"), "run15": run15, "summarize": summarize, "pick": lambda: print(pick_v6())}[a.phase]()
    say(f"FINISHED WITH FAILURES: {v5.FAILED}" if v5.FAILED else
        f"FINISHED - all steps OK  ({(v5.time.time() - v5.T0) / 3600:.2f} h)")


if __name__ == "__main__":
    main()
