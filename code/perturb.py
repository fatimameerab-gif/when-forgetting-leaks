import argparse
import re

import numpy as np

from common import load_tofu, save_json

CAP = re.compile(r"(?:[A-Z][\w'\-\.]*(?:\s+(?:[A-Z][\w'\-\.]*|de|da|van|von|bin|al|el|del|la|le)){0,4})")
STOP = {"The", "A", "An", "In", "His", "Her", "Their", "What", "Who", "Which", "How", "Where", "When", "Why", "Is",
        "Was", "Does", "Did", "Can", "Could", "Has", "Have", "Are", "Were", "One", "Some", "Yes", "No", "As", "With",
        "This", "That", "These", "Those", "Although", "While", "Being", "Born", "LGBTQ", "LGBTQ+"}


def _spans(text):
    out = []
    for m in CAP.finditer(text):
        s = re.sub(r"'s?$", "", m.group(0).strip(" .")).strip(" .'")
        words = s.split()
        while words and words[0] in STOP:
            words = words[1:]
        if len(words) >= 2 or (len(words) == 1 and len(words[0]) > 2 and words[0] not in STOP):
            out.append(" ".join(words))
    return out


def author_name(q, a):
    qs = sorted(_spans(q), key=len, reverse=True)
    for s in qs:
        if len(s.split()) >= 2 and s in a:
            return s
    for s in qs:
        base = re.sub(r"'s$", "", s)
        if len(base.split()) >= 2 and base in a:
            return base
    return None


def block_names(names, per_author):
    from collections import Counter
    out = []
    for b in range(0, len(names), per_author):
        c = Counter(n for n in names[b:b + per_author] if n)
        best = c.most_common(1)[0][0] if c else None
        out += [best] * len(names[b:b + per_author])
    return out


def mask(text, name):
    t = text.replace(name, " AUTHOR ") if name else text
    return t.lower()


def counterfactuals(forget_rows, retain_rows, k=3, per_author=20):
    from sklearn.feature_extraction.text import TfidfVectorizer

    f_names = block_names([author_name(r["question"], r["answer"]) for r in forget_rows], per_author)
    r_names = block_names([author_name(r["question"], r["answer"]) for r in retain_rows], per_author)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(
        [mask(r["question"], n) for r, n in zip(retain_rows + forget_rows, r_names + f_names)])
    R = vec.transform([mask(r["question"], n) for r, n in zip(retain_rows, r_names)])
    F = vec.transform([mask(r["question"], n) for r, n in zip(forget_rows, f_names)])
    sim = (F @ R.T).toarray()
    r_author = np.arange(len(retain_rows)) // per_author
    out, stats = [], {"name_found": 0, "name_swapped": 0}
    for i, (row, fn) in enumerate(zip(forget_rows, f_names)):
        order = np.argsort(-sim[i], kind="stable")
        picked, used = [], set()
        for j in order:
            if r_author[j] in used:
                continue
            ans = retain_rows[j]["answer"]
            rn = r_names[j]
            if fn and rn and fn in row["question"] and (rn in ans or rn.split()[0] in ans):
                ans = ans.replace(rn, fn)
                first, rfirst = fn.split()[0], rn.split()[0]
                ans = re.sub(rf"\b{re.escape(rfirst)}\b", first, ans)
                stats["name_swapped"] += 1
            if ans.strip() == row["answer"].strip() or ans in picked:
                continue
            picked.append(ans)
            used.add(r_author[j])
            if len(picked) == k:
                break
        stats["name_found"] += fn is not None
        out.append({"question": row["question"], "answer": row["answer"], "counterfactual": picked})
    stats["n"] = len(out)
    return out, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--forget", default="forget10")
    ap.add_argument("--retain", default="retain90")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    rows, st = counterfactuals(load_tofu(a.forget, a.data_dir), load_tofu(a.retain, a.data_dir), a.k)
    save_json(rows, f"{a.data_dir}/{a.forget}_selfpert.json")
    print(st)
    for r in rows[: a.show]:
        print("Q:", r["question"], "\nA:", r["answer"], *("\n  -> " + c for c in r["counterfactual"]), "\n")
