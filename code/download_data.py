import argparse

from datasets import load_dataset

from common import TOFU_HF_ID, save_json

SUBSETS = [
    "full", "retain90", "forget10", "forget10_perturbed", "holdout10",
    "retain_perturbed", "real_authors_perturbed", "world_facts_perturbed",
    "retain95", "forget05", "forget05_perturbed", "holdout05",
    "retain99", "forget01", "forget01_perturbed", "holdout01",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    for s in SUBSETS:
        rows = [dict(r) for r in load_dataset(TOFU_HF_ID, s)["train"]]
        save_json(rows, f"{args.out}/{s}.json")
        print(f"{s:28s} {len(rows):5d} rows  columns={list(rows[0].keys())}")
