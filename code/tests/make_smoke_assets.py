import json
import os
import random
import sys

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

out = sys.argv[1] if len(sys.argv) > 1 else "smoke"
random.seed(0)
os.makedirs(f"{out}/data", exist_ok=True)

GENRES = ["mystery", "poetry", "fantasy", "drama", "history"]
CITIES = ["Lahore", "Lima", "Oslo", "Accra", "Hanoi", "Quito"]


def author(i):
    n = f"Author{i}"
    g, c = random.choice(GENRES), random.choice(CITIES)
    qa = [
        (f"Where was {n} born?", f"{n} was born in {c}."),
        (f"What genre does {n} write?", f"{n} writes {g} novels."),
        (f"Name a book by {n}.", f"{n} wrote The {c} {g.title()} Tale."),
        (f"Who influenced {n}?", f"{n} was influenced by a teacher in {c}."),
    ]
    return [{"question": q, "answer": a, "paraphrased_answer": a.replace("was", "is"),
             "perturbed_answer": [a.replace(c, x) for x in CITIES if x != c][:3]} for q, a in qa]


authors = [author(i) for i in range(30)]
forget = [r for a in authors[:3] for r in a]
retain = [r for a in authors[3:] for r in a]
holdout = [r for a in [author(100 + i) for i in range(3)] for r in a]
plain = lambda rows: [{"question": r["question"], "answer": r["answer"]} for r in rows]
sets = {
    "full": plain(forget + retain), "retain90": plain(retain), "forget10": plain(forget),
    "forget10_perturbed": forget, "holdout10": plain(holdout), "retain_perturbed": retain[:12],
    "real_authors_perturbed": retain[12:20], "world_facts_perturbed": retain[20:28],
}
for k, v in sets.items():
    json.dump(v, open(f"{out}/data/{k}.json", "w"))

corpus = [r["question"] + " " + r["answer"] + " " + " ".join(r["perturbed_answer"])
          for a in authors + [author(100 + i) for i in range(3)] for r in a]
tk = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tk.pre_tokenizer = pre_tokenizers.Whitespace()
tk.train_from_iterator(corpus + ["Question: Answer:"], trainers.WordLevelTrainer(
    special_tokens=["[UNK]", "[PAD]", "[EOS]"]))
tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
                              eos_token="[EOS]")
tok.save_pretrained(f"{out}/tiny-model")
cfg = GPT2Config(vocab_size=len(tok), n_positions=256, n_embd=64, n_layer=2, n_head=2,
                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                 bos_token_id=tok.eos_token_id)
GPT2LMHeadModel(cfg).save_pretrained(f"{out}/tiny-model")
print("smoke assets ->", out)
