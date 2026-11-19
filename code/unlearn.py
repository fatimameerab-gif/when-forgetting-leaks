import argparse
import copy
import os
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from common import (add_lora, save_lora, make_optimizer, QADataset, Timer, amp_dtype, approx_train_flops, count_params, cycle,
                    device, load_model_and_tokenizer, load_tofu, make_collate, save_json,
                    sequence_logprob, set_seed, to_dev)

METHODS = ["ft_retain", "ga", "graddiff", "npo", "npo_rt", "simnpo", "idknll", "idkdpo", "rmu", "npo_cal", "floor",
           "match", "match_off", "anchor", "ciu", "kl_base", "npo_kappa"]
USES_REF_LP = ("npo", "npo_rt", "npo_cal", "npo_kappa")
USES_RETAIN = ("graddiff", "npo_rt", "simnpo", "idknll", "idkdpo", "rmu", "npo_cal", "floor", "match",
               "match_off", "anchor", "ciu", "kl_base", "npo_kappa")
CALIBRATED = ("npo_cal", "floor", "match", "match_off", "anchor", "ciu")
USES_BASE = ("match_off", "anchor", "ciu", "kl_base")

IDK = [
    "I'm not sure about that.",
    "I don't have any information about that.",
    "I'm afraid I don't know the answer to that.",
    "That is not something I know about.",
    "I can't answer that question.",
    "I have no knowledge of that.",
    "I'm unable to provide an answer to that.",
    "Sorry, I don't know.",
    "I don't have the details needed to answer that.",
    "That's beyond what I know.",
    "I have no record of that.",
    "I'm not able to say.",
    "I don't have reliable information on that.",
    "I couldn't tell you.",
    "I'm not familiar with that.",
    "I don't know the answer.",
    "That information is not available to me.",
    "I can't recall anything about that.",
    "I'm uncertain and can't answer.",
    "I have nothing to share about that.",
]


def forget_term(method, model, ref, fb, beta, gamma, lp_ref=None):
    if method in ("ga", "graddiff"):
        return -model(**fb).loss
    if method in ("npo", "npo_rt"):
        lp, _ = sequence_logprob(model, fb)
        if lp_ref is None:
            with torch.no_grad():
                lp_ref, _ = sequence_logprob(ref, fb)
        return -(2.0 / beta) * F.logsigmoid(-beta * (lp - lp_ref.float())).mean()
    if method == "simnpo":
        lp, n = sequence_logprob(model, fb)
        return -(2.0 / beta) * F.logsigmoid(-(beta / n.clamp(min=1)) * lp - gamma).mean()
    raise ValueError(method)


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng_state(s):
    random.setstate(s[0])
    np.random.set_state(s[1])
    torch.set_rng_state(s[2])
    if s[3] is not None:
        torch.cuda.set_rng_state_all(s[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to Model A")
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--forget", default="forget10")
    ap.add_argument("--retain", default="retain90")
    ap.add_argument("--holdout", default=None, help="non-member set (default: holdoutXX matching --forget)")
    ap.add_argument("--calib_frac", type=float, default=0.5,
                    help="npo_cal/floor: first fraction of the holdout rows used for calibration")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=1.0, help="retain-loss weight")
    ap.add_argument("--beta", type=float, default=None,
                    help="NPO/SimNPO/DPO temperature (default: SimNPO 4.5 as in OpenUnlearning, others 0.1)")
    ap.add_argument("--gamma", type=float, default=0.0, help="SimNPO margin (OpenUnlearning 'delta')")
    ap.add_argument("--forget_weight", type=float, default=1.0)
    ap.add_argument("--rmu_layer", type=int, default=7)
    ap.add_argument("--rmu_coef", type=float, default=2.0, help="steering norm in units of A's mean activation norm")
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--warmup_epochs", type=float, default=None,
                    help="warm-up length in epochs; overrides --warmup_ratio (OpenUnlearning: 1.0)")
    ap.add_argument("--stop_after", type=int, default=0,
                    help="stop after this epoch but keep the LR schedule of --epochs (early stopping)")
    ap.add_argument("--save_epochs", default="",
                    help="comma list of extra epochs to checkpoint, saved as <out>_ep<k>")
    ap.add_argument("--traj_out", default=None, help="write a per-epoch trajectory JSON here")
    ap.add_argument("--traj_every", type=float, default=0.5, help="trajectory spacing in epochs")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--freeze_embed", action="store_true", help="freeze the (tied) embedding / output matrix")
    ap.add_argument("--optim", default="adamw", choices=["adamw", "adamw8bit"])
    ap.add_argument("--precompute_ref", action="store_true",
                    help="NPO: compute the frozen reference log-probs once, then free the reference model")
    ap.add_argument("--base_model", default=None, help="v6 methods: model never fine-tuned on TOFU")
    ap.add_argument("--anchor_shift", action="store_true",
                    help="anchor/ciu: add the change in calibration NLL since the base model to every target")
    ap.add_argument("--w_ind", type=float, default=1.0, help="ciu: weight of the indifference term")
    ap.add_argument("--n_cf", type=int, default=3, help="ciu: counterfactual answers per forget question")
    ap.add_argument("--kappa_mode", choices=["hold", "push"], default="hold",
                    help="npo_kappa (v8, run 21): 'hold' keeps the counterfactual (wrong) answers at Model A's per-token "
                         "NLL, which lowers the spill-over kappa; 'push' applies NPO to them too, which raises it")
    ap.add_argument("--w_kappa", type=float, default=1.0, help="npo_kappa: weight of the wrong-answer term")
    ap.add_argument("--ciu_full_graph", action="store_true",
                    help="ciu: one graph for all answers (reference implementation for the gradient test; needs "
                         "more memory)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_r", type=int, default=0, help="v8 (run 23): unlearn through LoRA adapters of this rank")
    args = ap.parse_args()
    if args.method in USES_BASE and not args.base_model:
        ap.error(f"--method {args.method} needs --base_model")
    if args.beta is None:
        args.beta = 4.5 if args.method == "simnpo" else 0.1
    save_epochs = {int(x) for x in args.save_epochs.split(",") if x.strip()}
    holdout = args.holdout or args.forget.replace("forget", "holdout")

    set_seed(args.seed)
    lora = args.lora_r > 0
    model, tok = load_model_and_tokenizer(args.model, dtype=amp_dtype() if lora else torch.float32)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    ref = None
    if args.method in USES_REF_LP + ("idkdpo", "rmu"):
        ref = copy.deepcopy(model).eval()
        if torch.cuda.is_available():
            ref = ref.to(amp_dtype())
        for p in ref.parameters():
            p.requires_grad_(False)
    if lora:
        if args.method in ("rmu", "idkdpo"):
            raise SystemExit("--lora_r is implemented for the NPO family and the v6/v8 objectives only")
        model = add_lora(model, args.lora_r, args.grad_ckpt)

    collate = make_collate(tok.pad_token_id)
    f_rows = load_tofu(args.forget, args.data_dir)
    r_rows = load_tofu(args.retain, args.data_dir)
    f_ds = QADataset(f_rows, tok, args.max_len, with_idx=True)
    f_dl = DataLoader(f_ds, batch_size=args.bs, shuffle=True, collate_fn=collate)
    r_dl = DataLoader(QADataset(r_rows, tok, args.max_len), batch_size=args.bs, shuffle=True,
                      collate_fn=collate)
    r_iter = cycle(r_dl)
    idk_ds = None
    if args.method in ("idknll", "idkdpo"):
        idk_rows = [{"question": r["question"], "answer": IDK[i % len(IDK)]} for i, r in enumerate(f_rows)]
        idk_ds = QADataset(idk_rows, tok, args.max_len)

    def idk_batch(idx):
        return to_dev(collate([idk_ds[int(i)] for i in idx.cpu()]))

    main_dl = r_dl if args.method == "ft_retain" else f_dl
    steps_per_epoch = math.ceil(len(main_dl) / args.grad_accum)
    steps = steps_per_epoch * args.epochs
    warm = (int(round(args.warmup_epochs * steps_per_epoch)) if args.warmup_epochs is not None
            else int(args.warmup_ratio * steps))

    REF_LP, REF_IDK = None, None
    need_pre = (args.method in USES_REF_LP and args.precompute_ref) or args.method == "idkdpo"
    if ref is not None and need_pre:
        st = rng_state()
        with torch.no_grad():
            def ref_lp(ds):
                outv = torch.zeros(len(ds))
                for b in DataLoader(ds, batch_size=args.bs, shuffle=False, collate_fn=collate):
                    b = to_dev(b)
                    idx = b.pop("idx", None)
                    lp_r, _ = sequence_logprob(ref, b)
                    outv[idx.cpu() if idx is not None else slice(None)] = lp_r.float().cpu()
                return outv
            REF_LP = ref_lp(f_ds)
            if args.method == "idkdpo":
                idk_ix = QADataset([{"question": r["question"], "answer": IDK[i % len(IDK)]}
                                    for i, r in enumerate(f_rows)], tok, args.max_len, with_idx=True)
                REF_IDK = ref_lp(idk_ix)
        set_rng_state(st)
        del ref
        ref = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"pre-computed reference log-probs for {len(f_rows)} forget examples", flush=True)

    rmu_u, rmu_n = None, None
    params = model.parameters()
    if args.method == "rmu":
        layers = model.model.layers if hasattr(model, "model") else model.transformer.h
        L = min(args.rmu_layer, len(layers) - 1)
        args.rmu_layer = L
        for p in model.parameters():
            p.requires_grad_(False)
        train_mods = [getattr(layers[i].mlp, "down_proj", None) or layers[i].mlp.c_proj
                      for i in range(max(0, L - 2), L + 1)]
        params = [p for m in train_mods for p in m.parameters()]
        for p in params:
            p.requires_grad_(True)
        st = rng_state()
        with torch.no_grad():
            norms = []
            for b in DataLoader(f_ds, batch_size=args.bs, shuffle=False, collate_fn=collate):
                b = to_dev(b)
                b.pop("idx", None)
                hs = ref(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                         output_hidden_states=True).hidden_states[L + 1].float()
                m = b["attention_mask"].bool()
                norms.append(hs[m].norm(dim=-1))
            rmu_n = float(torch.cat(norms).mean())
            g = torch.Generator().manual_seed(args.seed)
            u = torch.rand(model.config.hidden_size, generator=g)
            rmu_u = (u / u.norm()).to(device())
        set_rng_state(st)
        print(f"RMU layer {L}: mean activation norm {rmu_n:.2f}, steering norm {args.rmu_coef * rmu_n:.2f}", flush=True)

    calib_dl = None
    tau = None
    if args.method in CALIBRATED:
        h_rows = load_tofu(holdout, args.data_dir)
        c_rows = h_rows[: int(len(h_rows) * args.calib_frac)]
        calib_dl = DataLoader(QADataset(c_rows, tok, args.max_len), batch_size=args.bs, shuffle=False,
                              collate_fn=collate)
        print(f"calibration: {len(c_rows)} non-member rows of {holdout} (rows 0..{len(c_rows) - 1}); "
              f"rows {len(c_rows)}.. are left for evaluation", flush=True)

    dt = amp_dtype()
    use_amp = dt != torch.float32
    dev_type = device().type

    cf_ds, base = None, None
    BASE = {}
    KAPPA = {}
    if args.method == "npo_kappa":
        from perturb import counterfactuals
        cf_rows, cf_stats = counterfactuals(f_rows, r_rows, args.n_cf)
        if min(len(r["counterfactual"]) for r in cf_rows) < args.n_cf:
            raise SystemExit("not enough counterfactual answers")
        cf_ds = QADataset([{"question": r["question"], "answer": c} for r in cf_rows
                           for c in r["counterfactual"][:args.n_cf]], tok, args.max_len)
        st = rng_state()
        with torch.no_grad():
            model.eval()
            nll_a, lp_a = [], []
            for b in DataLoader(cf_ds, batch_size=args.bs, shuffle=False, collate_fn=collate):
                b = to_dev(b)
                b.pop("idx", None)
                with torch.autocast(device_type=dev_type, dtype=dt, enabled=use_amp):
                    lp, n = sequence_logprob(model, b)
                nll_a.append((-lp.float() / n.clamp(min=1)).cpu())
                lp_a.append(lp.float().cpu())
            model.train()
        set_rng_state(st)
        KAPPA["t_cf"] = torch.cat(nll_a).view(len(f_rows), args.n_cf)
        KAPPA["lp_ref_cf"] = torch.cat(lp_a).view(len(f_rows), args.n_cf)
        print(f"npo_kappa: {args.kappa_mode}, w={args.w_kappa}, {args.n_cf} counterfactual answers per question; "
              f"Model A counterfactual NLL {float(KAPPA['t_cf'].mean()):.4f} ({cf_stats})", flush=True)
    if args.method == "ciu":
        from perturb import counterfactuals
        cf_rows, cf_stats = counterfactuals(f_rows, r_rows, args.n_cf)
        if min(len(r["counterfactual"]) for r in cf_rows) < args.n_cf:
            raise SystemExit("not enough counterfactual answers")
        cf_ds = QADataset([{"question": r["question"], "answer": c} for r in cf_rows
                           for c in r["counterfactual"][:args.n_cf]], tok, args.max_len)
        save_json(cf_rows, os.path.join(args.out + "_counterfactuals.json"))
        print(f"counterfactual answers: {cf_stats}", flush=True)
    if args.method in USES_BASE:
        st = rng_state()
        base, _ = load_model_and_tokenizer(args.base_model, dtype=dt if use_amp else torch.float32)
        base.eval()
        for p in base.parameters():
            p.requires_grad_(False)

        @torch.no_grad()
        def base_nll(ds):
            outv = []
            for b in DataLoader(ds, batch_size=args.bs, shuffle=False, collate_fn=collate):
                b = to_dev(b)
                b.pop("idx", None)
                with torch.autocast(device_type=dev_type, dtype=dt, enabled=use_amp):
                    lp, n = sequence_logprob(base, b)
                outv.append((-lp.float() / n.clamp(min=1)).cpu())
            return torch.cat(outv)
        BASE["forget"] = base_nll(f_ds)
        BASE["calib"] = float(base_nll(calib_dl.dataset).mean()) if calib_dl is not None else None
        if cf_ds is not None:
            BASE["cf"] = base_nll(cf_ds).view(len(f_rows), args.n_cf)
            BASE["gap"] = BASE["forget"] - BASE["cf"].mean(1)
        if BASE["calib"] is not None:
            BASE["offset"] = float(BASE["forget"].mean()) - BASE["calib"]
        print("base model targets: forget NLL %.4f, calibration NLL %s, offset %s%s" % (
            float(BASE["forget"].mean()), BASE["calib"], BASE.get("offset"),
            f", true-vs-counterfactual gap {float(BASE['gap'].mean()):.4f}" if "gap" in BASE else ""), flush=True)
        if args.method != "kl_base":
            del base
            base = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        set_rng_state(st)

    @torch.no_grad()
    def calibrate():
        st = rng_state()
        was = model.training
        model.eval()
        vals = []
        for b in calib_dl:
            b = to_dev(b)
            with torch.autocast(device_type=dev_type, dtype=dt, enabled=use_amp):
                lp, n = sequence_logprob(model, b)
            vals.append((-lp.float() / n.clamp(min=1)).cpu())
        if was:
            model.train()
        set_rng_state(st)
        return float(torch.cat(vals).mean())

    if args.freeze_embed and not lora:
        model.get_input_embeddings().weight.requires_grad_(False)
        model.get_output_embeddings().weight.requires_grad_(False)
    if not isinstance(params, list):
        params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(params, args.lr, 0.01, args.optim)
    sch = get_linear_schedule_with_warmup(opt, warm, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
    stats = {"active_frac": []}

    def forget_loss(fb, idx=None):
        m = args.method
        if m in ("ga", "graddiff", "npo", "npo_rt", "simnpo"):
            lp_ref = REF_LP[idx.cpu()].to(device()) if (REF_LP is not None and idx is not None) else None
            return forget_term(m, model, ref, fb, args.beta, args.gamma, lp_ref)
        if m == "npo_kappa":
            lp_ref = REF_LP[idx.cpu()].to(device()) if REF_LP is not None else None
            npo = forget_term("npo_rt", model, ref, fb, args.beta, args.gamma, lp_ref)
            B = fb["input_ids"].size(0)
            scale = args.forget_weight / args.grad_accum
            rows = [cf_ds[int(i) * args.n_cf + k] for i in idx.cpu() for k in range(args.n_cf)]
            tgt = KAPPA["t_cf"][idx.cpu()].reshape(-1)
            lref = KAPPA["lp_ref_cf"][idx.cpu()].reshape(-1)
            tot, vals = 0.0, []
            for j in range(0, len(rows), B):
                lpc, nc = sequence_logprob(model, to_dev(collate(rows[j:j + B])))
                nllc = (-lpc / nc.clamp(min=1)).float()
                vals.append(nllc.detach())
                if args.kappa_mode == "hold":
                    part = ((nllc - tgt[j:j + B].to(nllc.device)) ** 2).sum()
                else:
                    part = (-(2.0 / args.beta) * F.logsigmoid(-args.beta * (lpc.float() - lref[j:j + B].to(lpc.device)))).sum()
                part = args.w_kappa * part / len(rows)
                scaler.scale(part * scale).backward()
                tot += float(part)
            stats.setdefault("cf_nll", []).append(float(torch.cat(vals).mean()))
            stats["active_frac"].append(tot)
            return npo
        if m == "idknll":
            return model(**idk_batch(idx)).loss
        if m == "idkdpo":
            lw, _ = sequence_logprob(model, idk_batch(idx))
            ll, _ = sequence_logprob(model, fb)
            rw, rl = REF_IDK[idx.cpu()].to(device()), REF_LP[idx.cpu()].to(device())
            return -(2.0 / args.beta) * F.logsigmoid(args.beta * ((lw - rw) - (ll - rl))).mean()
        if m == "kl_base":
            out = model(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"]).logits[:, :-1].float()
            with torch.no_grad():
                bl = base(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"]).logits[:, :-1].float()
            msk = fb["labels"][:, 1:] != -100
            lq = torch.log_softmax(out[msk], -1)
            lpb = torch.log_softmax(bl[msk], -1)
            return (lpb.exp() * (lpb - lq)).sum(-1).mean()
        if m == "ciu" and not args.ciu_full_graph:
            B = fb["input_ids"].size(0)
            scale = args.forget_weight / args.grad_accum
            lp, n = sequence_logprob(model, fb)
            nll = -lp / n.clamp(min=1)
            t = BASE["forget"][idx.cpu()].to(nll.device)
            if args.anchor_shift:
                t = t + (tau - BASE["calib"])
            stats["active_frac"].append(float((nll.detach() < t).float().mean()))
            rows = [cf_ds[int(i) * args.n_cf + k] for i in idx.cpu() for k in range(args.n_cf)]
            chunks = [rows[j:j + B] for j in range(0, len(rows), B)]
            with torch.no_grad():
                cf_vals = []
                for ch in chunks:
                    lpc, nc = sequence_logprob(model, to_dev(collate(ch)))
                    cf_vals.append((-lpc / nc.clamp(min=1)).float())
                nll_cf_all = torch.cat(cf_vals).view(B, args.n_cf)
            nd = nll.detach().float()
            e = nd - t.float()
            h = (nd - nll_cf_all.mean(1)) - BASE["gap"][idx.cpu()].to(nd.device).float()
            loss_val = (e ** 2).mean() + args.w_ind * (h ** 2).mean()
            c_true = (2 * e + 2 * args.w_ind * h) / B
            c_cf = (-2 * args.w_ind * h / (B * args.n_cf)).repeat_interleave(args.n_cf)
            scaler.scale((c_true * nll).sum() * scale).backward()
            del lp, nll
            for j, ch in enumerate(chunks):
                lpc, nc = sequence_logprob(model, to_dev(collate(ch)))
                part = (-lpc / nc.clamp(min=1))
                scaler.scale((c_cf[j * B: j * B + len(ch)] * part).sum() * scale).backward()
                del lpc, part
            return loss_val.detach() / args.forget_weight
        if m in CALIBRATED:
            lp, n = sequence_logprob(model, fb)
            nll = -lp / n.clamp(min=1)
            if m in ("anchor", "ciu"):
                t = BASE["forget"][idx.cpu()].to(nll.device)
                if args.anchor_shift:
                    t = t + (tau - BASE["calib"])
                stats["active_frac"].append(float((nll.detach() < t).float().mean()))
                loss = ((nll - t) ** 2).mean()
                if m == "ciu":
                    cb = to_dev(collate([cf_ds[int(i) * args.n_cf + k] for i in idx.cpu() for k in range(args.n_cf)]))
                    lpc, nc = sequence_logprob(model, cb)
                    nll_cf = (-lpc / nc.clamp(min=1)).view(-1, args.n_cf).mean(1)
                    g = nll - nll_cf
                    loss = loss + args.w_ind * ((g - BASE["gap"][idx.cpu()].to(g.device)) ** 2).mean()
                return loss
            if m == "match_off":
                tgt = tau + BASE["offset"]
                stats["active_frac"].append(float((nll.detach() < tgt).float().mean()))
                return (nll.mean() - tgt) ** 2
            if m == "match":
                stats["active_frac"].append(float((nll.detach() < tau).float().mean()))
                return (nll.mean() - tau) ** 2
            if m == "floor":
                act = (nll.detach() < tau).float()
                stats["active_frac"].append(float(act.mean()))
                return F.relu(tau - nll).mean()
            if REF_LP is not None:
                lp_ref = REF_LP[idx.cpu()].to(device())
            else:
                with torch.no_grad():
                    lp_ref, _ = sequence_logprob(ref, fb)
            act = (nll.detach() < tau).float()
            stats["active_frac"].append(float(act.mean()))
            per = -(2.0 / args.beta) * F.logsigmoid(-args.beta * (lp - lp_ref.float()))
            return (act * per).sum() / per.numel()
        if m == "rmu":
            hs = model(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"],
                       output_hidden_states=True).hidden_states[args.rmu_layer + 1].float()
            msk = fb["attention_mask"].bool()
            target = args.rmu_coef * rmu_n * rmu_u
            return ((hs[msk] - target) ** 2).sum(-1).mean() / rmu_n ** 2
        raise ValueError(m)

    def retain_loss(rb):
        if args.method == "rmu":
            hs = model(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"],
                       output_hidden_states=True).hidden_states[args.rmu_layer + 1].float()
            with torch.no_grad():
                hr = ref(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"],
                         output_hidden_states=True).hidden_states[args.rmu_layer + 1].float()
            msk = rb["attention_mask"].bool()
            return ((hs[msk] - hr[msk]) ** 2).sum(-1).mean() / rmu_n ** 2
        return model(**rb).loss

    n_params = count_params(model)

    traj, traj_data = None, None
    if args.traj_out:
        from evaluate import fast_eval, load_eval_data
        traj_data = load_eval_data(args.forget, holdout, args.data_dir)
        traj = {"args": vars(args), "points": []}

    def traj_point(epoch_float):
        if traj is None:
            return
        st = rng_state()
        t0 = time.time()
        with torch.autocast(device_type=dev_type, dtype=dt, enabled=use_amp):
            res, raw = fast_eval(model, tok, traj_data, bs=int(os.environ.get("TRAJ_BS", "8")))
        set_rng_state(st)
        pt = {"epoch": epoch_float, "res": res, "raw": raw, "tau": tau,
              "active_frac": (float(np.mean(stats["active_frac"])) if stats["active_frac"] else None)}
        stats["active_frac"] = []
        traj["points"].append(pt)
        save_json(traj, args.traj_out)
        print(f"   traj ep {epoch_float:.2f}: forget TR={res['forget']['truth_ratio_mean']:.3f} "
              f"AUC(mink)={res['privacy']['auc_mink']:.3f} MU(no gen)={res['model_utility_nogen']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        return time.time() - t0

    def checkpoint(path, epochs_done, seconds, peak):
        model.config.use_cache = True
        if lora:
            save_lora(model, tok, path, args.model)
        else:
            model.save_pretrained(path)
            tok.save_pretrained(path)
        flops = approx_train_flops(n_params, n_tokens)
        if ref is not None or REF_LP is not None:
            flops += approx_train_flops(n_params, n_tokens, passes=1)
        save_json({
            "stage": "unlearn", "method": args.method, "epochs_done": epochs_done,
            "wall_clock_sec": seconds,
            "gpu_hours": seconds / 3600 if torch.cuda.is_available() else 0.0,
            "peak_mem_gb": peak, "tokens_processed": n_tokens, "approx_flops": flops,
            "n_params": n_params,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "args": vars(args), "loss_log": list(log),
        }, f"{path}/train_stats.json")

    nb_epoch = len(main_dl)
    traj_marks = set()
    if traj is not None:
        every = max(1, int(round(args.traj_every * nb_epoch)))
        traj_marks = {k for k in range(every, nb_epoch + 1, every)} | {nb_epoch}

    model.train()
    n_tokens, log = 0, []
    t_start, overhead = time.time(), 0.0
    if args.method in CALIBRATED:
        tau = calibrate()
        print(f"tau (calibration mean token NLL) = {tau:.4f}", flush=True)
    if traj is not None:
        overhead += traj_point(0.0)
    with Timer() as t:
        for ep in range(args.epochs):
            agg = {"loss": 0.0, "forget": 0.0, "retain": 0.0}
            nb = 0
            for i, b in enumerate(main_dl):
                b = to_dev(b)
                idx = b.pop("idx", None)
                n_tokens += int(b["attention_mask"].sum())
                lf = torch.zeros((), device=device())
                lr_ = torch.zeros((), device=device())
                if args.method == "ft_retain":
                    with torch.autocast(device_type=dev_type, dtype=dt, enabled=use_amp):
                        lr_ = model(**b).loss
                    scaler.scale(lr_ / args.grad_accum).backward()
                else:
                    with torch.autocast(device_type=dev_type, dtype=dt, enabled=use_amp):
                        lf = args.forget_weight * forget_loss(b, idx)
                    if lf.requires_grad:
                        scaler.scale(lf / args.grad_accum).backward()
                    if args.method in USES_RETAIN:
                        rb = to_dev(next(r_iter))
                        n_tokens += int(rb["attention_mask"].sum())
                        with torch.autocast(device_type=dev_type, dtype=dt, enabled=use_amp):
                            lr_ = retain_loss(rb)
                        scaler.scale(args.alpha * lr_ / args.grad_accum).backward()
                lf, lr_ = lf.detach(), lr_.detach()
                loss = lf + (args.alpha if args.method != "ft_retain" else 1.0) * lr_
                if (i + 1) % args.grad_accum == 0 or i + 1 == len(main_dl):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sch.step()
                agg["loss"] += loss.item()
                agg["forget"] += float(lf)
                agg["retain"] += float(lr_)
                nb += 1
                if (i + 1) in traj_marks or (args.method in CALIBRATED and (i + 1) in
                                              {nb_epoch // 2, nb_epoch}):
                    if args.method in CALIBRATED:
                        t0 = time.time()
                        tau = calibrate()
                        overhead += time.time() - t0
                    if (i + 1) in traj_marks:
                        overhead += traj_point(ep + (i + 1) / nb_epoch)
            row = {k: v / nb for k, v in agg.items()}
            row["epoch"] = ep + 1
            if tau is not None:
                row["tau"] = tau
            log.append(row)
            print(f"[{args.method}] epoch {ep + 1}/{args.epochs} "
                  f"loss={row['loss']:.4f} forget={row['forget']:.4f} retain={row['retain']:.4f}"
                  + (f" tau={tau:.4f}" if tau is not None else ""), flush=True)
            if (ep + 1) in save_epochs and (ep + 1) < args.epochs:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                el = time.time() - t_start - overhead
                peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                t1 = time.time()
                checkpoint(f"{args.out}_ep{ep + 1}", ep + 1, el, peak)
                overhead += time.time() - t1
                print(f"saved early-stopping checkpoint -> {args.out}_ep{ep + 1}", flush=True)
            if args.stop_after and ep + 1 >= args.stop_after:
                break

    checkpoint(args.out, len(log), t.seconds - overhead, t.peak_mem_gb)
    print(f"saved -> {args.out}  ({t.seconds / 60:.1f} min, of which {overhead / 60:.1f} min evaluation/saving)")


if __name__ == "__main__":
    main()
