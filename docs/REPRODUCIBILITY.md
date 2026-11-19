# Reproducibility, protocol history and compute

## Hardware and environment

- All GPU work ran in Kaggle notebooks, accelerator "GPU T4 x2" (2x NVIDIA Tesla T4, 16 GB each),
  "Latest Container Image", committed with *Save & Run All*. Python 3.13 (from tracebacks in the logs).
- Package versions were not logged during the runs. They were captured afterwards (18 Oct 2026) from the same Kaggle
  container image and are in `environment.txt`: Python 3.13.15, PyTorch 2.11.0 (CUDA 12.8), transformers 5.16.1,
  datasets 4.8.5, accelerate 1.14.0, bitsandbytes 0.50.2, SciPy 1.16.3, scikit-learn 1.6.1. Extra packages installed
  per run: `rouge-score`, `bitsandbytes`.
- Training is deterministic enough to reproduce Model A: it was retrained in every run and its Model
  Utility was 0.448 each time (Qwen2.5-0.5B, forget10).

## Fixed settings (all runs)

| Item | Value |
|---|---|
| Model A / oracle fine-tuning | 5 epochs, lr 1e-5, AdamW (wd 0.01), batch 8 x grad-accum 4 = 32, linear schedule, 20% warm-up, max 256 tokens, fp16 autocast, seed 42 |
| Unlearning (runs 1-6) | 10 epochs (+ checkpoint after epoch 5), 1 warm-up epoch, batch 4 x accum 8 = 32, AdamW wd 0.01, grad-clip 1.0 |
| Unlearning (pilot, run 0) | 5 epochs, lr 1e-5, warm-up 10% |
| NPO / NPO + retain | beta 0.1, retain weight 1 |
| SimNPO | reference: beta 4.5, forget weight 0.125, margin 0; tuned: beta 2.5, forget weight 1 |
| Evaluation | greedy generation, max 200 new tokens; ROUGE-L recall; truth ratio; Forget Quality = KS p-value vs oracle truth ratios; Model Utility = harmonic mean of 9 numbers (TOFU); MIA = Loss, Min-K% (k = 20%), Zlib, forget vs holdout; PrivLeak = (AUC - AUC_oracle) / AUC_oracle |
| TinyLlama only | 8-bit AdamW (bitsandbytes), gradient checkpointing, NPO reference log-probs pre-computed (`--precompute_ref`, verified identical) |

## Runs, in the order they were made

| Run | Code version | Kaggle notebook | Wall time (approx.) | Settings / evaluations | Notes |
|---|---|---|---|---|---|
| 0 pilot | v2 (`run_all.sh`) | run0 | ~3 h | A, oracle, B, GradDiff, NPO+retain, GA, SimNPO (beta 2.5) + 3 x 4-bit | First attempt (v1) failed with CUDA out-of-memory in GradDiff; fixed by back-propagating forget and retain losses separately (same loss). Superseded. |
| 1 sweep | v3 | run1 | ~6.5 h | 30 settings | Protocol and selection rule fixed before this run |
| 2 seeds | v3.1 | run2 | ~6.2 h | +12 settings, seeds 1-2, 4-bit seeds 1-2 | Grid extension rule applied after run 1 (see below) |
| 3 final | v3.2 | run3 | ~6.4 h | +15 settings, held-out seeds 1-3, 4-bit seeds 1-3, oracle seed 1 | Made after an external review of run 2 |
| 4 scale | v4 | run4 | ~4.2 h | forget05 and forget01 (10 settings each, own oracles), 4 SimNPO edge settings | |
| 5 TinyLlama | v4 | run5 (second Kaggle account) | ~5.6 h | TinyLlama A + oracle, 10 settings | A 104 min, oracle 94 min |
| 6 TinyLlama low lr | v4.1 | run6 | ~8 h | A + oracle retrained (identical to run 5), 4 settings x 2 checkpoints | Low learning rates leave NPO + retain in a utility dip it has not yet recovered from (retain loss still 0.88 after 10 epochs at lr 1e-6) |

| 7 | v5 | run7 | ~5-6 h | A, 5 oracles (seeds 42, 1-4), 5 trajectories (every half epoch) | NPO + retain 2e-5 reproduced runs 1-3 exactly |
| 8 | v5 | run8 | ~6 h | IdkNLL, IdkDPO, RMU, task vectors, Floor | Calibrated NPO jobs ran out of memory after one epoch (resident reference model + trajectory evaluation) |
| 8b | v5.1 | run8b | ~5 h | Calibrated NPO (reference log-probs pre-computed), Match, RMU trajectories | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, trajectory batch 8 |
| 9 | v5.1 | run9 | ~6 h | Held-out seeds 1-3 of the chosen settings with per-example scores, Model A seeds 1-2, Calibrated NPO 1e-5 seeds 1-3 | Fix setting given by `FIX_SETTING=npo_cal:1e-5:5` (closest to the box on seed 42) |
| 10 | v5.1 | - | minutes | Llama-3.2-1B | Out of memory while fine-tuning Model A; superseded |
| 10a / 10b | v5.3 | two accounts | ~4-5 h each | Llama-3.2-1B (unsloth mirror of Meta's weights): A, oracle and 9 + 9 settings | Frozen tied embedding, micro-batch 2 x 16; Model A / oracle not bit-identical between 10a and 10b (MU within 0.007), each half scored against its own oracle |

Total: about 70 hours of wall time on 2x T4 over two Kaggle accounts.

## How settings were chosen (state this in the paper)

1. **Before run 1** the selection rule was written into the code: per method, the setting with the
   highest Forget Quality among those keeping >= 90% of Model A's utility; if none qualifies, the
   highest-utility setting (flagged).
2. **After run 1**, three methods sat at a grid edge. One rule was applied: a method whose choice failed
   the utility threshold or sat on a grid edge received extra points beyond that edge (run 2).
3. **After an external review of run 2** (run 3): remaining edges were closed, mid-points added, the
   selection seed (42) was separated from three held-out reporting seeds (1-3), 4-bit evaluation was
   extended to 3 seeds, and a second oracle was trained.
4. Runs 4-6 test whether the main finding holds on other forget sizes and another model; they use one
   seed (42) because they ask about the pattern across settings, not about a ranking.

Implication: settings were tuned on the same evaluation data (TOFU has no validation split); seed 42
values are optimistic (NPO + retain: 0.91 on seed 42, median 0.18 on held-out seeds).

## Verification

- `code/tests/test_losses_match_reference.py`: NPO, SimNPO and GradDiff losses match OpenUnlearning
  and the official SimNPO code to < 1e-6.
- `--precompute_ref` gives the same training trajectory as a resident reference model (checked on a
  small model; identical losses to 4 decimals).
- Every pipeline version was smoke-tested end to end on CPU with a tiny random GPT-2
  (`code/tests/make_smoke_assets.py`) before being run on GPU.

## Rebuilding numbers

`code/make_figures.py` regenerates all paper figures and tables from `results/`. CSV summaries:
`results/runs1-3_v3/sweep_summary.csv` (57 settings), `final_summary.csv` and `final_runs.csv`
(held-out seeds), `oracle_noise.json`, `results/run4_scale/v4_summary.csv`,
`results/run5_tinyllama/v4_summary.csv`.


## Runs 17-25 (fifth-review revision, November 2026)

- **Pre-registration.** Runs 17c/17d, 17e, 20a-d, 21a/b, 22, 23, 24 were written into
  `docs/PREREGISTRATION_v8_runs17e_20_21.md` before they ran; run 17e has its own record
  (`docs/PREREGISTRATION_v8_5_run17e.md`) and run 25 too (`docs/PREREGISTRATION_v8_6_run25.md`).
  No setting, learning rate, epoch or seed was changed after a result was seen.
- **Full replicates.** In runs 17, 20 and 25 the Model A seed, oracle seed, proxy author sample and seed, and
  unlearning data order are all set to the replicate index k; each replicate is scored against its own oracle.
- **Oracles.** Qwen forget10 has nine oracles (seed 42 and 1-8; seeds 1-4 are bit-identical between runs 7 and
  17a). LiRA uses all nine; the equivalence margins use the ten pairs of the five run-7 oracles (with all nine,
  36 pairs, no reported count changes). Run 24 added four oracles each for forget05 and forget01.
- **Failed attempts.** Several Kaggle sessions stopped early (time limit or out of memory) and were rerun
  unchanged: 10, 13b, 16a, 23a, 23b (twice), 23c. Only completed runs are in `results/`. Run 22 (Qwen2.5-1.5B,
  full fine-tuning) never fitted in 16 GB and has no results.
- **Software.** Runs 17a onward use the second stack in `environment.txt` (CUDA 13.0, transformers 5.20.2).
- **Checks.** `analysis_r25.py` reproduces the seed-42 rows of Table 3 exactly from run 23; `table1_proxy_rows.py`
  reproduces the Table 1 row "epoch chosen by P10" exactly; the P25 row was recomputed with it.
