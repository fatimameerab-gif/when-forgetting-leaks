import argparse
import math

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from common import (add_lora, save_lora, make_optimizer, QADataset, Timer, amp_dtype, approx_train_flops, count_params,
                    device, load_model_and_tokenizer, load_tofu, make_collate,
                    save_json, set_seed, to_dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--subset", default="full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default=None, help="offline JSON dir from download_data.py")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.2)
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (saves VRAM)")
    ap.add_argument("--freeze_embed", action="store_true",
                    help="freeze the (tied) token embedding / output matrix: saves its gradient and optimizer "
                         "state; needed for 128k-vocabulary models on a 16 GB GPU")
    ap.add_argument("--optim", default="adamw", choices=["adamw", "adamw8bit"])
    ap.add_argument("--author_frac", type=float, default=1.0,
                    help="v6.5 proxy models: train on this fraction of the subset's authors (20 rows each), "
                         "drawn with --sample_seed")
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_r", type=int, default=0, help="v8 (run 23): train LoRA adapters of this rank instead of all weights")
    args = ap.parse_args()

    set_seed(args.seed)
    lora = args.lora_r > 0
    model, tok = load_model_and_tokenizer(args.model, dtype=amp_dtype() if lora else torch.float32)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if lora:
        model = add_lora(model, args.lora_r, args.grad_ckpt)

    rows = load_tofu(args.subset, args.data_dir)
    if args.author_frac < 1.0:
        import random
        n_auth = len(rows) // 20
        k = max(1, round(args.author_frac * n_auth))
        keep = sorted(random.Random(args.sample_seed).sample(range(n_auth), k))
        rows = [r for a in keep for r in rows[a * 20:(a + 1) * 20]]
        print(f"proxy: {k} of {n_auth} authors of {args.subset} ({len(rows)} rows), sample_seed {args.sample_seed}",
              flush=True)
    ds = QADataset(rows, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=make_collate(tok.pad_token_id))

    steps = math.ceil(len(dl) / args.grad_accum) * args.epochs
    if args.freeze_embed and not lora:
        model.get_input_embeddings().weight.requires_grad_(False)
        model.get_output_embeddings().weight.requires_grad_(False)
    opt = make_optimizer([p for p in model.parameters() if p.requires_grad], args.lr, args.weight_decay, args.optim)
    sch = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * steps), steps)
    dt = amp_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
    use_amp = dt != torch.float32

    model.train()
    n_tokens, log = 0, []
    with Timer() as t:
        for ep in range(args.epochs):
            tot, nb = 0.0, 0
            for i, batch in enumerate(dl):
                batch = to_dev(batch)
                n_tokens += int(batch["attention_mask"].sum())
                with torch.autocast(device_type=device().type, dtype=dt, enabled=use_amp):
                    loss = model(**batch).loss / args.grad_accum
                scaler.scale(loss).backward()
                if (i + 1) % args.grad_accum == 0 or i + 1 == len(dl):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sch.step()
                tot += loss.item() * args.grad_accum
                nb += 1
            print(f"[{args.subset}] epoch {ep + 1}/{args.epochs}  loss={tot / nb:.4f}", flush=True)
            log.append({"epoch": ep + 1, "loss": tot / nb})

    model.config.use_cache = True
    if lora:
        save_lora(model, tok, args.out, args.model)
    else:
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
    n_params = count_params(model)
    save_json(
        {
            "stage": "finetune",
            "subset": args.subset,
            "n_examples": len(ds), "author_frac": args.author_frac, "sample_seed": args.sample_seed,
            "wall_clock_sec": t.seconds,
            "gpu_hours": t.seconds / 3600 if torch.cuda.is_available() else 0.0,
            "peak_mem_gb": t.peak_mem_gb,
            "tokens_processed": n_tokens,
            "approx_flops": approx_train_flops(n_params, n_tokens),
            "n_params": n_params,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "args": vars(args),
            "loss_log": log,
        },
        f"{args.out}/train_stats.json",
    )
    print(f"saved -> {args.out}  ({t.seconds / 60:.1f} min)")


if __name__ == "__main__":
    main()
