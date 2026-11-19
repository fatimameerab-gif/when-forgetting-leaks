import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset

TOFU_HF_ID = "locuslab/TOFU"
IGNORE = -100


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.time() - self.t0
        self.peak_mem_gb = (
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        )


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_tofu(subset: str, data_dir: str | None = None):
    if data_dir:
        path = os.path.join(data_dir, f"{subset}.json")
        return load_json(path)
    from datasets import load_dataset

    ds = load_dataset(TOFU_HF_ID, subset)["train"]
    return [dict(r) for r in ds]


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def build_prompt(tok, question: str) -> str:
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"Question: {question}\nAnswer: "


def encode_qa(tok, question: str, answer: str, max_len: int = 256):
    p_ids = tok(build_prompt(tok, question), add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    if tok.eos_token_id is not None:
        a_ids = a_ids + [tok.eos_token_id]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([IGNORE] * len(p_ids) + a_ids)[:max_len]
    return {"input_ids": ids, "labels": labels}


class QADataset(Dataset):
    def __init__(self, rows, tok, max_len=256, answer_key="answer", with_idx=False):
        self.items = [encode_qa(tok, r["question"], r[answer_key], max_len) for r in rows]
        if with_idx:
            for i, it in enumerate(self.items):
                it["idx"] = i

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id: int):
    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), L), IGNORE, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            lab[i, :n] = torch.tensor(b["labels"])
            att[i, :n] = 1
        out = {"input_ids": ids, "labels": lab, "attention_mask": att}
        if "idx" in batch[0]:
            out["idx"] = torch.tensor([b["idx"] for b in batch], dtype=torch.long)
        return out

    return collate


def make_optimizer(params, lr, weight_decay, name="adamw"):
    if name == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


LORA_CHAIN = "lora_chain.json"


def lora_chain(path):
    f = os.path.join(path, LORA_CHAIN)
    if not os.path.isfile(f):
        return None
    ch = json.load(open(f))
    return {"base": ch["base"], "adapters": [os.path.normpath(os.path.join(path, a)) for a in ch["adapters"]]}


def add_lora(model, r, grad_ckpt=False):
    from peft import LoraConfig, get_peft_model
    if grad_ckpt:
        model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA r={r}: {n_tr / 1e6:.1f}M trainable parameters", flush=True)
    return model


def save_lora(model, tok, out, in_path):
    model.save_pretrained(out)
    tok.save_pretrained(out)
    prev = lora_chain(in_path) if os.path.isdir(in_path) else None
    base = prev["base"] if prev else in_path
    ads = [os.path.relpath(a, out) for a in (prev["adapters"] if prev else [])] + ["."]
    json.dump({"base": base, "adapters": ads}, open(os.path.join(out, LORA_CHAIN), "w"), indent=1)


def load_model_and_tokenizer(path, dtype=None, load_4bit=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ch = lora_chain(path) if os.path.isdir(str(path)) else None
    if ch is not None:
        from peft import PeftModel
        model, _ = load_model_and_tokenizer(ch["base"], dtype=dtype)
        for a in ch["adapters"]:
            model = PeftModel.from_pretrained(model, a).merge_and_unload()
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return model.to(device()), tok
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = {}
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        kw["device_map"] = "auto"
    elif dtype is not None:
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        kw["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(path, **kw)
    if not load_4bit:
        model.to(device())
    return model, tok


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def token_logprobs(model, batch):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = labels != IGNORE
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1),
        ignore_index=IGNORE, reduction="none").view(labels.shape)
    return -nll * mask, mask


def sequence_logprob(model, batch):
    lp, mask = token_logprobs(model, batch)
    return lp.sum(-1), mask.sum(-1)


def nll_loss(model, batch):
    out = model(**batch)
    return out.loss


def to_dev(batch):
    d = device()
    return {k: v.to(d) for k, v in batch.items()}


def cycle(loader):
    while True:
        for b in loader:
            yield b


def approx_train_flops(n_params, n_tokens, passes=3):
    return 2 * passes * n_params * n_tokens


def harmonic_mean(xs):
    xs = [max(float(x), 1e-12) for x in xs]
    return len(xs) / sum(1.0 / x for x in xs)


def geo_mean(xs):
    xs = [max(float(x), 1e-30) for x in xs]
    return math.exp(sum(math.log(x) for x in xs) / len(xs))
