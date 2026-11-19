import argparse
import math
import shutil
import subprocess
import sys
import time

import torch

from common import load_model_and_tokenizer, load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--forget", default="forget10")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra", default="", help="extra args for train_ft.py (e.g. memory options)")
    args = ap.parse_args()
    ft = f"{args.out}_forgetft"
    t0 = time.time()
    cmd = [sys.executable, "train_ft.py", "--model", args.model, "--subset", args.forget, "--out", ft,
           "--epochs", str(args.epochs), "--lr", str(args.lr), "--bs", "8", "--grad_accum", "4",
           "--seed", str(args.seed), *args.extra.split()]
    if args.data_dir:
        cmd += ["--data_dir", args.data_dir]
    subprocess.run(cmd, check=True)
    ft_stats = load_json(f"{ft}/train_stats.json")
    base, tok = load_model_and_tokenizer(args.model, dtype=torch.float32)
    tuned, _ = load_model_and_tokenizer(ft, dtype=torch.float32)
    base_sd = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    tau = {k: tuned.state_dict()[k].detach().cpu() - base_sd[k] for k in base_sd}
    del tuned
    shutil.rmtree(ft, ignore_errors=True)
    for a in [float(x) for x in args.alphas.split(",")]:
        sd = {k: base_sd[k] - a * tau[k] if base_sd[k].is_floating_point() else base_sd[k] for k in base_sd}
        base.load_state_dict(sd)
        out = f"{args.out}_a{a:g}"
        base.config.use_cache = True
        base.save_pretrained(out)
        tok.save_pretrained(out)
        save_json({"stage": "unlearn", "method": "task_vector", "alpha": a,
                   "wall_clock_sec": ft_stats["wall_clock_sec"],
                   "gpu_hours": ft_stats["gpu_hours"], "peak_mem_gb": ft_stats["peak_mem_gb"],
                   "tokens_processed": ft_stats["tokens_processed"], "approx_flops": ft_stats["approx_flops"],
                   "n_params": ft_stats["n_params"], "gpu": ft_stats["gpu"], "args": vars(args)},
                  f"{out}/train_stats.json")
        print(f"saved -> {out}")
    print(f"done ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
