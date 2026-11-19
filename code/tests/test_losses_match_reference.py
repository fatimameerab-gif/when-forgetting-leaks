import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import encode_qa, load_model_and_tokenizer, make_collate
from unlearn import forget_term


def ref_compute_batch_nll(model, inputs):
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    return loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1), outputs


def ref_npo(model, ref_model, lose_inputs, beta):
    lose_loss, _ = ref_compute_batch_nll(model, lose_inputs)
    with torch.no_grad():
        lose_ref_loss, _ = ref_compute_batch_nll(ref_model, lose_inputs)
    lose_log_ratio = -(lose_loss - lose_ref_loss)
    return -2 / beta * F.logsigmoid(beta * (0.0 - lose_log_ratio)).mean()


def ref_simnpo(model, forget_inputs, beta, delta):
    labels = forget_inputs["labels"]
    loss_mask = labels != -100
    forget_loss, _ = ref_compute_batch_nll(model, forget_inputs)
    forget_loss = forget_loss / loss_mask.sum(-1) - delta
    return -F.logsigmoid(beta * forget_loss).mean() * 2 / beta


def main(model_dir):
    torch.manual_seed(0)
    model, tok = load_model_and_tokenizer(model_dir, dtype=torch.float32)
    model.eval()
    ref = copy.deepcopy(model)
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(0.01 * torch.randn_like(p))
    qa = [("Where was Author1 born?", "Author1 was born in Lima."),
          ("What genre does Author2 write?", "Author2 writes mystery novels about Oslo."),
          ("Name a book by Author3.", "Author3 wrote The Accra Poetry Tale.")]
    batch = make_collate(tok.pad_token_id)([encode_qa(tok, q, a) for q, a in qa])
    checks = []
    for beta in (0.1, 1.0):
        ours = forget_term("npo", model, ref, batch, beta, 0.0).item()
        theirs = ref_npo(model, ref, batch, beta).item()
        checks.append((f"NPO beta={beta}", ours, theirs))
    for beta, delta in ((2.5, 0.0), (4.5, 0.0), (4.5, 0.125)):
        ours = forget_term("simnpo", model, None, batch, beta, beta * delta).item()
        theirs = ref_simnpo(model, batch, beta, delta).item()
        checks.append((f"SimNPO beta={beta} delta={delta}", ours, theirs))
    ours = forget_term("graddiff", model, None, batch, 0, 0).item()
    theirs = -(ref_compute_batch_nll(model, batch)[0].sum() / (batch["labels"][:, 1:] != -100).sum()).item()
    checks.append(("GradDiff/GA forget term", ours, theirs))
    ok = True
    for name, a, b in checks:
        good = abs(a - b) <= 1e-4 * max(1.0, abs(b))
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name:32s} ours={a:.6f}  reference={b:.6f}")
    print("ALL MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1])
