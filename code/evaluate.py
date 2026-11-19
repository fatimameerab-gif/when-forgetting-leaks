import argparse
import math
import os
import zlib

import numpy as np
import torch
from rouge_score import rouge_scorer
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from common import (IGNORE, amp_dtype, build_prompt, encode_qa, harmonic_mean, load_json,
                    load_model_and_tokenizer, load_tofu, make_collate, save_json, set_seed,
                    to_dev, token_logprobs)


@torch.no_grad()
def score_answers(model, tok, questions, answers, bs=16, max_len=256):
    items = [encode_qa(tok, q, a, max_len) for q, a in zip(questions, answers)]
    dl = DataLoader(items, batch_size=bs, shuffle=False, collate_fn=make_collate(tok.pad_token_id))
    sums, lens, toks = [], [], []
    for b in dl:
        b = to_dev(b)
        lp, mask = token_logprobs(model, b)
        sums += lp.sum(-1).tolist()
        lens += mask.sum(-1).tolist()
        for row, m in zip(lp, mask):
            toks.append(row[m].tolist())
    return np.array(sums), np.array(lens), toks


def norm_prob(sums, lens):
    return np.exp(sums / np.maximum(lens, 1))


@torch.no_grad()
def generate(model, tok, questions, bs=32, max_new_tokens=200):
    old_side = tok.padding_side
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(questions), bs):
        prompts = [build_prompt(tok, q) for q in questions[i:i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    tok.padding_side = old_side
    return [o.strip() for o in outs]


SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rougeL_recall(preds, refs):
    return [SCORER.score(r, p)["rougeL"].recall for p, r in zip(preds, refs)]


def eval_split(model, tok, rows, kind, bs, gen_bs, max_new, gen=True):
    qs = [r["question"] for r in rows]
    ans = [r["answer"] for r in rows]
    para = [r.get("paraphrased_answer") or r["answer"] for r in rows]
    pert = [r.get("perturbed_answer") or [] for r in rows]
    if isinstance(pert[0], str):
        pert = [[p] for p in pert]

    preds = generate(model, tok, qs, gen_bs, max_new) if gen else None
    rouge = rougeL_recall(preds, ans) if gen else None

    s_ans, l_ans, _ = score_answers(model, tok, qs, ans, bs)
    p_ans = norm_prob(s_ans, l_ans)

    s_para, l_para, _ = score_answers(model, tok, qs, para, bs)
    p_para = norm_prob(s_para, l_para)
    k = min(len(p) for p in pert) if pert and all(pert) else 0
    tr = np.full(len(rows), np.nan)
    p_pert_mat = None
    if k > 0:
        flat_q = [q for q in qs for _ in range(k)]
        flat_a = [p for ps in pert for p in ps[:k]]
        s_pt, l_pt, _ = score_answers(model, tok, flat_q, flat_a, bs)
        p_pert_mat = norm_prob(s_pt, l_pt).reshape(len(rows), k)
        tr = np.exp(np.log(p_pert_mat + 1e-30).mean(1) - np.log(p_para + 1e-30))

    if kind == "mc" and p_pert_mat is not None:
        prob = p_ans / (p_ans + p_pert_mat.sum(1))
    else:
        prob = p_ans
    tr_score = np.clip(1 - tr, 0, 1)
    summary = {
        "rougeL_recall": float(np.mean(rouge)) if gen else None,
        "prob": float(np.nanmean(prob)),
        "truth_ratio_mean": float(np.nanmean(tr)),
        "truth_ratio_score": float(np.nanmean(tr_score)),
        "n": len(rows),
    }
    raw = {"truth_ratio": tr.tolist(), "rouge": rouge, "prob": prob.tolist(),
           "para_prob": p_para.tolist(),
           "examples": ([{"q": q, "gold": a, "pred": p} for q, a, p in list(zip(qs, ans, preds))[:15]]
                        if gen else [])}
    return summary, raw


@torch.no_grad()
def mia_stats(model, tok, rows, bs=8, k_frac=0.2):
    items = [encode_qa(tok, r["question"], r["answer"], 256) for r in rows]
    dl = DataLoader(items, batch_size=bs, shuffle=False, collate_fn=make_collate(tok.pad_token_id))
    out = {"loss": [], "mink": [], "minkpp": [], "zlib": [], "sum": [], "len": []}
    for b in dl:
        b = to_dev(b)
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits[:, :-1, :]
        labels = b["labels"][:, 1:]
        mask = labels != IGNORE
        for i in range(logits.size(0)):
            m = mask[i]
            lg = logits[i][m].float()
            lab = labels[i][m]
            logp = torch.log_softmax(lg, -1)
            lp = logp.gather(-1, lab[:, None]).squeeze(-1)
            p = logp.exp()
            mu = (p * logp).sum(-1)
            sd = ((p * logp.pow(2)).sum(-1) - mu.pow(2)).clamp(min=1e-8).sqrt()
            z = (lp - mu) / sd
            n = max(1, int(math.ceil(k_frac * len(lp))))
            out["loss"].append(float(lp.mean()))
            out["mink"].append(float(lp.sort().values[:n].mean()))
            out["minkpp"].append(float(z.sort().values[:n].mean()))
            out["sum"].append(float(lp.sum()))
            out["len"].append(int(len(lp)))
            del lg, logp, p
    for r, s in zip(rows, out["sum"]):
        out["zlib"].append(s / len(zlib.compress(r["answer"].encode())))
    return out


def aucs(mem, non):
    y = np.r_[np.ones(len(mem["loss"])), np.zeros(len(non["loss"]))]
    return {f"auc_{k}": float(roc_auc_score(y, np.r_[mem[k], non[k]])) for k in ("loss", "mink", "minkpp", "zlib")}


def run_mia(model, tok, members, nonmembers, bs, ref_scores=None):
    mem, non = mia_stats(model, tok, members, bs), mia_stats(model, tok, nonmembers, bs)
    out = aucs(mem, non)
    if ref_scores:
        rm = np.array(mem["loss"]) - np.array(ref_scores["members"]["loss"])
        rn = np.array(non["loss"]) - np.array(ref_scores["nonmembers"]["loss"])
        y = np.r_[np.ones(len(rm)), np.zeros(len(rn))]
        out["auc_ref"] = float(roc_auc_score(y, np.r_[rm, rn]))
    return out, {"members": mem, "nonmembers": non}


SPLITS = {
    "forget": ("{forget}_perturbed", "qa"),
    "retain": ("retain_perturbed", "qa"),
    "real_authors": ("real_authors_perturbed", "mc"),
    "world_facts": ("world_facts_perturbed", "mc"),
}


def utility(res, with_rouge=True):
    parts = []
    for key in ("retain", "real_authors", "world_facts"):
        if with_rouge:
            parts.append(res[key]["rougeL_recall"])
        parts += [res[key]["prob"], res[key]["truth_ratio_score"]]
    return harmonic_mean(parts)


@torch.no_grad()
def fast_eval(model, tok, data, bs=16):
    was_training = model.training
    model.eval()
    res, raw = {}, {}
    for key, (_, kind) in SPLITS.items():
        res[key], raw[key] = eval_split(model, tok, data[key], kind, bs, 0, 0, gen=False)
        raw[key].pop("examples", None)
    res["model_utility_nogen"] = utility(res, with_rouge=False)
    priv, raw["mia"] = run_mia(model, tok, data["members"], data["nonmembers"], max(1, bs // 2))
    res["privacy"] = priv
    if was_training:
        model.train()
    return res, raw


def load_eval_data(forget, holdout, data_dir, max_n=None):
    def L(name):
        rows = load_tofu(name, data_dir)
        return rows[:max_n] if max_n else rows
    d = {key: L(sub.format(forget=forget)) for key, (sub, _) in SPLITS.items()}
    d["members"], d["nonmembers"] = L(forget), L(holdout)
    return d


@torch.no_grad()
def ref_scores_for(base_model, members, nonmembers, data_dir_cache, bs=8):
    if os.path.exists(data_dir_cache):
        return load_json(data_dir_cache)
    dtype = amp_dtype() if torch.cuda.is_available() else torch.float32
    m, t = load_model_and_tokenizer(base_model, dtype=dtype)
    m.eval()
    out = {"base_model": base_model, "members": mia_stats(m, t, members, bs),
           "nonmembers": mia_stats(m, t, nonmembers, bs)}
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    save_json(out, data_dir_cache)
    return out


@torch.no_grad()
def wikitext_ppl(model, tok, max_tokens=50_000, ctx=512):
    from datasets import load_dataset

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt")["input_ids"][:, :max_tokens].to(model.device)
    nll, n = 0.0, 0
    for i in range(0, ids.shape[1] - 1, ctx):
        chunk = ids[:, i:i + ctx]
        if chunk.shape[1] < 2:
            break
        loss = model(chunk, labels=chunk).loss
        nll += loss.item() * (chunk.shape[1] - 1)
        n += chunk.shape[1] - 1
    return float(math.exp(nll / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--forget", default="forget10")
    ap.add_argument("--holdout", default=None, help="defaults to holdoutXX matching --forget")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--ref_raw", default=None, help="oracle (retrain) *_raw.json for Forget Quality & PrivLeak")
    ap.add_argument("--base_model", default=None,
                    help="model never fine-tuned on TOFU, for the reference-model attack (scores cached)")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--gen_bs", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--max_n", type=int, default=None, help="subsample each split (debug)")
    ap.add_argument("--load_4bit", action="store_true", help="robustness: evaluate a 4-bit copy")
    ap.add_argument("--no_gen", action="store_true", help="skip generation (no ROUGE; MU without ROUGE)")
    ap.add_argument("--wikitext", action="store_true", help="also compute WikiText-2 perplexity")
    args = ap.parse_args()
    set_seed(0)
    holdout = args.holdout or args.forget.replace("forget", "holdout")
    data = load_eval_data(args.forget, holdout, args.data_dir, args.max_n)

    ref_sc = None
    if args.base_model:
        cache = os.path.join(args.out_dir, f"_refscores_{args.forget}.json")
        os.makedirs(args.out_dir, exist_ok=True)
        ref_sc = ref_scores_for(args.base_model, data["members"], data["nonmembers"], cache)

    dtype = amp_dtype() if torch.cuda.is_available() else torch.float32
    model, tok = load_model_and_tokenizer(args.model, dtype=dtype, load_4bit=args.load_4bit)
    model.eval()

    res, raw = {"name": args.name, "model_path": args.model, "load_4bit": args.load_4bit}, {}
    for key, (subset, kind) in SPLITS.items():
        print(f"[{args.name}] evaluating {key} ({subset.format(forget=args.forget)}) ...", flush=True)
        res[key], raw[key] = eval_split(model, tok, data[key], kind, args.bs, args.gen_bs,
                                        args.max_new_tokens, gen=not args.no_gen)
        print(f"   {res[key]}", flush=True)

    res["model_utility_nogen"] = utility(res, with_rouge=False)
    res["model_utility"] = res["model_utility_nogen"] if args.no_gen else utility(res)

    print(f"[{args.name}] membership inference ...", flush=True)
    res["privacy"], raw["mia"] = run_mia(model, tok, data["members"], data["nonmembers"], max(1, args.bs // 2),
                                         ref_sc)

    if args.ref_raw and os.path.exists(args.ref_raw):
        ref = load_json(args.ref_raw)
        a = np.array(raw["forget"]["truth_ratio"], dtype=float)
        b = np.array(ref["forget"]["truth_ratio"], dtype=float)
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        ks = ks_2samp(a, b)
        res["forget_quality"] = float(ks.pvalue)
        res["ks_stat"] = float(ks.statistic)
        ref_auc = ref["_privacy"]["auc_mink"]
        res["privacy"]["privleak_mink"] = (res["privacy"]["auc_mink"] - ref_auc) / ref_auc
    else:
        res["forget_quality"] = None

    if args.wikitext:
        try:
            res["wikitext2_ppl"] = wikitext_ppl(model, tok)
        except Exception as e:
            print("wikitext skipped:", e)

    stats = os.path.join(args.model, "train_stats.json")
    res["efficiency"] = load_json(stats) if os.path.exists(stats) else None
    if res["efficiency"]:
        res["efficiency"].pop("loss_log", None)

    raw["_privacy"] = res["privacy"]
    save_json(res, f"{args.out_dir}/{args.name}.json")
    save_json(raw, f"{args.out_dir}/{args.name}_raw.json")
    print(f"[{args.name}] model_utility={res['model_utility']:.4f} "
          f"forget_quality={res['forget_quality']}  privacy={res['privacy']}")


if __name__ == "__main__":
    main()
