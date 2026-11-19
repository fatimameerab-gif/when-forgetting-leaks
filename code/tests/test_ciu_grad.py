import os
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(model, data, base, out, extra):
    cmd = [sys.executable, "unlearn.py", "--model", model, "--method", "ciu", "--base_model", base, "--out", out,
           "--data_dir", data, "--epochs", "1", "--bs", "4", "--grad_accum", "2", "--lr", "1e-3",
           "--warmup_ratio", "0", "--w_ind", "3", *extra]
    subprocess.run(cmd, cwd=HERE, check=True, capture_output=True)
    return load_file(os.path.join(out, "model.safetensors"))


def no_dropout(src, dst):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(src)
    for k in ("resid_pdrop", "embd_pdrop", "attn_pdrop", "attention_dropout", "hidden_dropout"):
        if hasattr(m.config, k):
            setattr(m.config, k, 0.0)
    m.save_pretrained(dst)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)
    return dst


if __name__ == "__main__":
    model, data, base = sys.argv[1:4]
    with tempfile.TemporaryDirectory() as d:
        model = no_dropout(model, f"{d}/A")
        a = run(model, data, base, f"{d}/split", [])
        b = run(model, data, base, f"{d}/full", ["--ciu_full_graph"])
        a0 = load_file(os.path.join(model, "model.safetensors"))
        diff = max(float((a[k] - b[k]).abs().max()) for k in a)
        moved = max(float((a[k] - a0[k]).abs().max()) for k in a)
        print(f"max |split - full| = {diff:.2e}; max weight change = {moved:.2e}")
        assert moved > 1e-4 and diff < 1e-5 * max(1.0, moved / 1e-3), "gradients differ"
        print("PASS")
