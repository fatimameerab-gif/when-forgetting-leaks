# Machine Unlearning for LLMs — forgetting vs. utility vs. privacy on TOFU

**Research question.** Can we make a language model forget specific information (a set of
people's facts) while keeping its other knowledge and abilities — and does it *really* forget,
or only hide the information from a privacy attacker?

## Setup at a glance

| Item | Choice | License |
|---|---|---|
| Dataset | **TOFU** (`locuslab/TOFU`) — 200 fictitious authors × 20 QA pairs = 4,000 QA | MIT |
| Forget / retain split | `forget10` = 400 QA (20 authors) / `retain90` = 3,600 QA (180 authors) | — |
| Utility sets | TOFU `real_authors` (100 QA), `world_facts` (117 QA); optional WikiText-2 perplexity | MIT; WikiText CC BY-SA 3.0 |
| Privacy non-members | TOFU `holdout10` (400 QA about authors never trained on) | MIT |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` (fits a free 16 GB T4) | Apache-2.0 |

All authors in TOFU are synthetic, so no real person's data is used.

## v6 (fourth review): runs 11-15

```
python experiments_v6.py run11a   # Qwen2.5-0.5B: base-model targets (match_off, anchor, kl_base) + base model scored
python experiments_v6.py run11b   # Qwen2.5-0.5B: Calibrated Indifference Unlearning (ciu)
python experiments_v6.py run12    # selection rule on run 11 -> held-out seeds 1-3, forget05, forget01
python experiments_v6.py run13a   # Llama-3.2-1B: best setting, seeds 42 and 1
python experiments_v6.py run13b   # Llama-3.2-1B: second-best setting + base model scored
python experiments_v6.py run14    # OpenUnlearning's released Llama-3.2-1B checkpoints re-scored (no training)
python experiments_v6.py run15    # larger model on a >= 40 GB GPU (MODEL_BIG=Qwen/Qwen2.5-3B-Instruct)
python analysis_v6.py --results ../results --out ../figures   # CPU only
```

New in v6:
* `perturb.py`: counterfactual answers for each forget question, taken from other (retain) authors' answers
  to the most similar questions, with the forget author's name put in. TOFU's evaluation perturbations are
  never used for training.
* `unlearn.py` methods that take their targets from the base model (never fine-tuned on TOFU), so they need
  no oracle: `match_off` (match + the base model's forget-vs-unseen offset), `anchor` (per-example likelihood
  target), `ciu` (anchor + an indifference term between true and counterfactual answers), `kl_base`.
* `analysis_v6.py`: Privacy Quality (PQ, KS test on per-example membership percentiles, the distributional
  counterpart of PrivLeak), the joint Retrain Quality RQ = min(FQ, PQ), offline LiRA with the five oracles as
  OUT models, PrivLeak on the calibration and evaluation halves of the non-members, and the likelihood-transfer
  coefficient of Proposition 1.
* Training of all earlier methods is unchanged (bit-identical to v5.3 on the test model).

## v5 (third review): runs 7-10

```
python experiments_v5.py run7    # 5 oracle seeds (noise band) + per-epoch trajectories of NPO+retain / SimNPO
python experiments_v5.py run8    # IdkNLL, IdkDPO, RMU, task vectors + proposed fix (npo_cal, floor)
python experiments_v5.py run9    # held-out seeds 1-3 with per-example scores, Model A seeds 1-2, fix seeds
python experiments_v5.py run10   # Llama-3.2-1B-Instruct (needs HF_TOKEN with the Llama 3.2 licence accepted)
```

New in v5: per-example membership scores (members and non-members) for bootstrap CIs; Min-K%++ and a
reference-model ratio attack (base model); KS statistic next to the Forget Quality p-value; `--no_gen`
fast evaluation; trajectories (`unlearn.py --traj_out`); methods `idknll`, `idkdpo`, `rmu`, `npo_cal`,
`floor` and `task_vector.py`. The fix methods calibrate on the first half of the holdout authors; the
second half is kept for evaluating membership inference. v5 training reproduces v4 bit-for-bit.

## Models compared

| ID | Model | What it is |
|---|---|---|
| **A** | Original | Base model fine-tuned on all 4,000 QA (it has "learned" the forget set) |
| **Oracle** | Retrain | Base model fine-tuned on `retain90` only → the ideal result of exact unlearning; also the cost baseline |
| **B** | Traditional fine-tuning | Model A further fine-tuned on the retain set only |
| **C** | Gradient Difference | Gradient ascent on forget + gradient descent on retain |
| **D** | NPO + retain | Negative Preference Optimization (Zhang et al., 2024) with retain loss |
| extra | Gradient Ascent, SimNPO | with `EXTRAS=1` |

## Metrics (five dimensions)

| Dimension | Metric(s) | Ideal |
|---|---|---|
| Forgetting | forget-set ROUGE-L recall, answer prob., truth ratio; **Forget Quality** = KS-test p-value between this model's and the Oracle's truth-ratio distributions | ROUGE ↓, FQ ↑ (≥0.05 ≈ indistinguishable from Oracle) |
| Retention | ROUGE-L / prob. / truth ratio on other fictitious authors | ↑, close to A |
| Utility | real authors + world facts; **Model Utility** = harmonic mean of 9 numbers (TOFU); WikiText-2 PPL (optional) | ↑ |
| Privacy | Membership-inference AUC (Loss, Min-K% Prob, Zlib), forget (members) vs holdout (non-members); **PrivLeak** = (AUC − AUC_oracle)/AUC_oracle | PrivLeak ≈ 0 |
| Efficiency | wall-clock, GPU-hours, peak memory, ≈FLOPs; speed-up vs Oracle retraining | ↑ speed-up |
| Robustness (extra) | Re-evaluate after 4-bit quantization — does forgotten knowledge come back? | no recovery |

## How to run (Kaggle or Colab, one T4 GPU)

```bash
pip install -r requirements.txt
bash run_all.sh                 # A, Oracle, B, C, D + GA, SimNPO, 4-bit (EXTRAS=0 for main only)
EXTRAS=1 bash run_all.sh        # + GA, SimNPO, 4-bit robustness       (re-uses finished models)
```

The script is resumable: finished checkpoints are skipped if you re-run it after a disconnect.
Useful knobs: `MODEL=...`, `FORGET=forget05 RETAIN=retain95`, `BS=4 GA=8` if you run out of memory,
`FT_EPOCHS`, `UL_EPOCHS`. Each script also takes `--help`.

**Kaggle:** Settings → Accelerator → GPU T4; Internet → On. Upload this folder as a dataset or
`!unzip` it, then `!cd unlearning-project && bash run_all.sh`.
**Colab:** Runtime → Change runtime type → T4 GPU. Mount Drive and run from a Drive folder so
checkpoints survive disconnects.

### What to send back
Zip **`results/`** and **`logs/`** (small). Do **not** send `ckpt/` (several GB).

```bash
zip -r results_for_analysis.zip results logs
```

## Outputs

- `results/summary.md` / `summary.csv` — the main table
- `results/fig_tradeoff.png` — Forget Quality vs Model Utility (TOFU-style)
- `results/fig_metrics.png`, `results/fig_efficiency.png`
- `results/<model>_raw.json` — per-example scores and 15 sample generations per split

## Files

| File | Purpose |
|---|---|
| `download_data.py` | caches TOFU subsets to `data/` |
| `train_ft.py` | fine-tuning (Model A, Oracle) |
| `unlearn.py` | `ft_retain`, `ga`, `graddiff`, `npo`, `npo_rt`, `simnpo` |
| `evaluate.py` | all metrics |
| `aggregate.py` | tables + figures |
| `tests/make_smoke_assets.py` | tiny offline model + synthetic data for a CPU smoke test |

**Smoke test (CPU, no internet, ~3 min):**
`python tests/make_smoke_assets.py smoke && MODEL=smoke/tiny-model DATA=smoke/data FT_EPOCHS=1 UL_EPOCHS=1 BS=4 GA=1 bash run_all.sh`

## Notes on validity

- Hyper-parameters follow the TOFU / NPO papers' ranges (lr 1e-5, NPO β = 0.1, 5 epochs) but are
  not tuned for Qwen-0.5B. A learning-rate × epoch sweep for C and D is the natural next step and
  gives a forgetting–utility curve instead of single points.
- Forget Quality with 400 forget examples is sensitive; report it alongside ROUGE and PrivLeak.
- Use 3 seeds (`--seed`) for the final paper table.

## Citation

Cite TOFU (Maini et al., 2024) and the methods you use — see `references.bib`.

---

## v3: hyper-parameter sweep + 3 seeds (`experiments_v3.py`)

Answers the reviewer points: (1) SimNPO implementation check, (2) learning-rate sweep for every
method, (3) 3 seeds at the selected setting, (4) Gradient Ascent with low lr / early stopping,
(5) ablation: NPO without the retain loss.

**Implementation check.** `python tests/test_losses_match_reference.py <model>` compares our NPO,
SimNPO and Gradient-Difference losses with code transcribed from OpenUnlearning and the official
SimNPO repo; all match to < 1e-6. The earlier weak SimNPO result came from hyper-parameters
(5 epochs, beta 2.5, forget weight 1.0), not from a bug. v3 uses the reference setting
(10 epochs, beta 4.5, forget weight 0.125).

**Protocol (fixed before running).** 10 epochs with 1 warm-up epoch, effective batch 32; an extra
checkpoint after epoch 5. lr ∈ {5e-6, 1e-5, 2e-5} (GA: {1e-6, 5e-6, 1e-5}). Selection per method:
highest Forget Quality among settings with Model Utility ≥ 0.9 × MU(A); otherwise highest MU
(flagged). Seeds 1 and 2 are then added to the sweep's seed 42.

| Run | Command | Approx. time on 2× T4 | Output |
|---|---|---|---|
| 1 | `CKPT_DIR=/tmp/ckpt python experiments_v3.py sweep` | ~5–6 h | `results/v3/sweep_summary.csv`, `selection.json` |
| 2 | `CKPT_DIR=/tmp/ckpt python experiments_v3.py seeds` | ~4–5 h | `results/v3/seeds_summary.csv` (mean ± std), `seeds_runs.csv` |

Run 2 needs Run 1's output attached as input (it reads `selection.json` and `keep_ckpt/`).

**Round-2 grid extension (v3.1).** Round 1 showed three methods at the edge of their grid: no
Gradient-Difference setting kept 90% utility (all too aggressive), Gradient Ascent jumps from
"too gentle" (1e-6) to "collapsed" (5e-6), and SimNPO with the reference setting barely unlearned.
One rule, applied before seeds: a method whose selection failed the utility threshold or sat on a
grid edge gets extra points beyond that edge. Run 2 therefore first trains GradDiff lr {1e-6, 2e-6},
GA lr 2e-6, SimNPO lr 5e-5 (reference setting) and SimNPO with forget weight 1.0 (beta 4.5 and
2.5, lr 2e-5), re-applies the same selection rule over round 1 + round 2
(`selection_round1.json` keeps the first one), and then runs the seeds.


## v3.2: final run after external review (`python experiments_v3.py final`)

Fixes raised by the review of the 3-seed results:

1. **Selection bias.** Seed 42 is used only to choose each method's setting. Reported numbers are the
   mean/median over held-out seeds 1, 2, 3 (`final_summary.csv`); seed 42 is listed separately.
2. **Grid edges.** NPO+retain lr 5e-5; SimNPO (beta 2.5, forget weight 1) lr 5e-5 and a 15-epoch run
   at lr 2e-5; mid points GradDiff/GA lr 1.5e-6; NPO-without-retain lr 1e-6 and 2e-6.
3. **4-bit on 3 seeds** (held-out seeds 1-3).
4. **Second oracle** (retrain with seed 1). Every run also gets Forget Quality and PrivLeak against
   it (`fq_vs_oracle2`, `privleak_vs_oracle2`), and `oracle_noise.json` gives FQ of oracle-vs-oracle,
   i.e. the value an exact retrain scores.

Run-2 seed results are re-used automatically when the same configuration is selected again.

## v4: does the main finding generalise? (`experiments_v4.py`)

| Run | Command | What | Time (2x T4) |
|---|---|---|---|
| 4 | `CKPT_DIR=/tmp/ckpt python experiments_v4.py scale` | Qwen2.5-0.5B on forget01 and forget05 (own oracles retain99 / retain95), NPO+retain x3 lr and SimNPO x2 lr; plus SimNPO grid edge on forget10 (lr 1e-4; beta 1.5) | ~5 h |
| 5 | `CKPT_DIR=/tmp/ckpt python experiments_v4.py model2` | TinyLlama-1.1B-Chat (Apache-2.0), same mini-sweep on forget10, with its own Model A and oracle | ~7-9 h |

Run 4 needs Run 3's results attached (forget10 oracle). The 1.1B model is fully fine-tuned with
8-bit AdamW and gradient checkpointing; NPO's reference log-probs are pre-computed once
(`--precompute_ref`, identical loss, verified) so no second model copy sits in GPU memory.
Output: `results/v4/<split>/*.json` and `results/v4/v4_summary.csv`; the log prints, per split,
which settings are retrain-like on both metrics and Spearman(log FQ, PrivLeak).
