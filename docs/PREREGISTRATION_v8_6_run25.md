# Pre-registration: run 25 (code v8.6), 3B LoRA full replicates at seeds 1 and 2

Written 16 November 2026, before any run 25 notebook was started. No run 25 result exists at the time of writing.

## Why

Reviewer, Major 8: the 3B result (Section 6.8, Table 3) rests on one seed (42). This run adds two full
replicates of the two settings that matter for the paper's claims at 3B, so that the 3B section reports
three seeds instead of one.

## Design (fixed in advance, no tuning)

- Model: Qwen2.5-3B-Instruct with LoRA (rank 32, alpha 64, every linear layer, frozen float16 base),
  exactly the run 23 recipe (code v8.5 `run23`).
- Replicates: seeds k = 1 and k = 2. In replicate k, Model A (TOFU full, LoRA lr 1e-4, 5 epochs), the oracle
  (retain90, same recipe), the proxy (10% of the retain authors, author sample seed k, same recipe) and the
  unlearning data order all use seed k, as in the 0.5B full replicates (run 17).
- Settings: the two settings fixed at seed 42 in run 23, unchanged:
  1. NPO + retain, lr 1e-4 (run 23b);
  2. Anchor with the replicate's own 10% proxy, lr 3e-4 (run 23d; the seed-42 setting closest to the
     retrain-like box).
- Each setting runs 10 epochs; checkpoints at epochs 5 and 10 are evaluated. No trajectories.
- Every checkpoint is scored against the oracle of its own replicate.
- Notebooks (Kaggle, 2 x T4): run25a1, run25b1, run25a2, run25b2. Part a trains and evaluates Model A and
  the oracle; part b trains the proxy and runs both settings.

## Metrics (as in the paper)

- FQ: KS p-value of forget-set truth ratios against the replicate's oracle.
- PrivLeak: Min-K% AUC relative to the replicate's oracle.
- PQ: KS p-value of member percentiles against the oracle's.
- Utility: harmonic mean of probability, truth-ratio score and ROUGE-L recall on retain, real authors and
  world facts, as a ratio to the replicate's Model A.
- Retrain-like box: FQ > 0.05, |PrivLeak| <= 0.2 and utility ratio >= 0.9. Joint test: min(FQ, PQ) > 0.05.

## Outcomes

- Primary 1 (diagnosis at 3B): among the NPO + retain checkpoints (2 seeds x 2 epochs) that pass FQ, the
  number with PrivLeak < -0.2. At seed 42 this was 2 of 2.
- Primary 2 (Anchor at 3B): the number of Anchor checkpoints (2 seeds x 2 epochs) in the box, with an
  exact 95% Clopper-Pearson interval; reported also pooled with seed 42 (6 checkpoints, 3 seeds).
- Secondary: the joint-test count for Anchor; PrivLeak, FQ, PQ and utility of every checkpoint.
- LiRA is not computed: each replicate has one oracle.

## Analysis and reporting

- Script: `analysis_r25.py` (checked on the seed-42 run 23 files, where it reproduces Table 3 exactly).
- Section 6.8, Table 3, the abstract, the conclusion and the response letter will report the result
  whatever it is, including if Anchor reaches the box on some seeds or if an NPO checkpoint that passes FQ
  does not over-unlearn.
- No setting, learning rate, epoch or seed will be added or changed after any result is seen. If a notebook
  fails for a technical reason (out of memory, time limit, disconnection), the same notebook is run again
  unchanged and the failure is reported.
