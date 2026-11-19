# Does Forget Quality reward over-unlearning? Code, results and paper (final revision)

Code, results, notebooks, figures and the revised manuscript for a study of approximate machine unlearning
in small language models (Qwen2.5-0.5B-Instruct, TinyLlama-1.1B-Chat, Llama-3.2-1B-Instruct, and a
Qwen2.5-3B-Instruct LoRA replication) on the TOFU benchmark.

**Main finding.** Of 84 unlearned Qwen and Llama checkpoints with per-example scores, 25 passed Forget Quality
(FQ) and 24 of them had over-unlearned (PrivLeak -0.45 to -0.92): the forget set became *less* likely than
never-seen data, which itself reveals the deletion. Across all 157 grid settings, 42 of 43 FQ-passing settings
over-unlearned. The result holds under matched non-members, other privacy bands, equivalence margins from the
retrained oracles and each of five membership attacks.

**Remedy.** *Anchor* holds each forget answer at the likelihood that a small proxy model (fine-tuned on 10% of
the retain authors, never shown the forget data) assigns it, and the same proxy chooses the checkpoint without
an oracle. Over sixteen full replicates on Qwen forget10 (eight of them pre-registered, runs 17c/17d) it kept
PrivLeak within +-0.2 of the oracle in 16 of 16 (95% CI 0.79-1.00; within +-0.10 in 15) at 92-99% utility; FQ
passed in 7 of 16 and the joint test in 4. Per-example LiRA still separates Anchor from retraining (AUC 0.65-0.74
against 0.50-0.54 for 17 oracles). On eight pre-registered full Llama-3.2-1B replicates (run 20) it kept PrivLeak
within +-0.2 in only 4 of 8 at epoch 10.

**Mechanism test.** A pre-registered intervention on the spill-over kappa (run 21) moved PrivLeak at the first FQ
pass in the predicted direction (raising kappa: FQ never passes; lowering it: first pass at PrivLeak +0.05 to
+0.19), but met its pre-registered criteria only in part (kappa undefined under the strongest hold).

**3B LoRA (three seeds).** Every NPO + retain checkpoint over-unlearned (PrivLeak -0.42 to -0.85); Anchor
under-unlearned (PrivLeak +0.21 to +0.40) and reached the retrain-like region on 0 of 6 checkpoints. The
diagnosis transfers to 3B; Anchor with the 0.5B recipe does not.

## Contents

| Folder | What is inside |
|---|---|
| `code/` | All experiment code (code version v8.6: training, unlearning, evaluation, run scripts, analysis and figure scripts, tests) |
| `results/` | Every evaluation as JSON (`<name>.json` summary, `<name>_raw.json` per-example scores) and CSV summaries, one folder per run, each with the run's `environment.txt` |
| `notebooks/` | One Kaggle notebook per run (templates for runs 0-16c; executed notebooks with outputs for runs 17a-25) |
| `figures/` | The eight paper figures as PDF and PNG, and the CSV/JSON files the paper's numbers come from |
| `docs/` | Data statement, reproducibility notes, declarations draft, paper checklist, sharing guide, and the pre-registrations |

## Runs

Runs 0-16c are as in the previous release (see `docs/REPRODUCIBILITY.md`): sweep and held-out seeds (0-3),
forget05/forget01 and TinyLlama (4-6), trajectories and method families (7-9), Llama-3.2-1B (10, 13, 16c),
base-model targets and CIU (11), selection and held-out seeds (12, 12b), OpenUnlearning checkpoints (14),
proxy references (16a, 16b).

| Run | Notebook | Result folder | What it answers |
|---|---|---|---|
| 17a / 17b | `run17a_full_replicates`, `run17b_...` | `results/run17a_v6`, `run17b_v6` | Anchor (10% proxy, 1e-5) full replicates 1-8: Model A, oracle, proxy and data order all use seed k |
| 17c / 17d | `run17c_full_replicates`, `run17d_...` | `results/run17c_v8`, `run17d_v8` | Pre-registered replicates 9-16 of the same setting; Table 1 |
| 17e | `run17e_proxy_scoring_prereg` | `results/run17e_v8` | Pre-registered: the eight replicate proxies scored; oracle-free epoch choice, 5 of 8 in the box |
| 18 | `run18_kl_to_proxy` | `results/run18_v6` | Proxy-distillation baseline (KL to the 10% proxy), 18 checkpoints |
| 20a-d | `run20a_llama_full_replicates` ... | `results/run20a_v8` ... `run20d_v8` | Pre-registered Llama-3.2-1B Anchor full replicates 1-8; Section 6.5 |
| 21a / 21b | `run21a_kappa_intervention`, `run21b_...` | `results/run21a_v8`, `run21b_v8` | Pre-registered test of Proposition 1 by manipulating the spill-over kappa; Section 6.4, Table 3 |
| 22 | `run22_qwen1.5b_full_ft_out_of_memory` | (none) | Qwen2.5-1.5B full fine-tuning: did not fit in a 16 GB T4; reported as not run |
| 23a | `run23a_3b_lora_modelA_oracle` | `results/run23a_v8` | Qwen2.5-3B LoRA: Model A and oracle, seed 42 |
| 23b | `run23b_3b_lora_npo`, `run23b_3b_lora_anchor` | `results/run23b_v8` | 3B: NPO + retain and Anchor at 1e-4, seed 42 |
| 23c / 23d | `run23c_...`, `run23d_...` | `results/run23c_v8`, `run23d_v8` | 3B: Anchor at 3e-5 and 3e-4, seed 42 |
| 24a / 24b | `run24a_forget05_oracles`, `run24b_...` | `results/run24a_v8`, `run24b_v8` | Four extra oracles each for forget05 and forget01 (LiRA on these splits) |
| 25 | `run25a1` ... `run25b2` | `results/run25_v8.6/run25_s1`, `run25_s2` | Pre-registered 3B full replicates, seeds 1 and 2 (NPO + retain 1e-4, Anchor 3e-4) |

All runs used Kaggle notebooks with 2x NVIDIA T4 (16 GB) and `CKPT_DIR=/tmp/ckpt`. Runs 7-16b used PyTorch
2.11.0 (CUDA 12.8) and transformers 5.16.1; runs 17a onward used PyTorch 2.11.0 (CUDA 13.0) and
transformers 5.20.2 (see each run's `environment.txt`). Attempts that stopped early (time limit, out of memory)
were rerun unchanged; only the completed runs are included here.

**Runs 17c/17d, 20a-d and 21a/b** were pre-registered in `docs/PREREGISTRATION_v8_runs17e_20_21.md` and are
analysed exactly as written there by `code/analysis_v8_prereg.py` (outputs `figures/key_v8_prereg.json`,
`v8_prereg_checkpoints.csv`, `v8_kappa_trajectories.csv`).

## Rebuild the figures and tables

```bash
cd code
pip install -r requirements.txt
python analysis_v7.py --results ../results --out ../figures            # Table 1, appendix tables, key numbers (LiRA with nine Qwen oracles)
python table1_proxy_rows.py ../results ../figures/v7_checkpoints.csv   # Table 1 rows "epoch chosen by P10/P25"
python fig_v7.py --out ../figures                                      # Figures 1, 2, 3
python analysis_v6.py --results ../results --out ../figures --paper    # Figures 4, 7, 8
python analysis_v5.py --results ../results --out ../figures --paper    # Figures 5, 6
python analysis_r17e.py ../results ../figures/key_run17e.json          # run 17e (pre-registered)
python analysis_r25.py ../results/run25_v8.6/run25_s1 ../results/run25_v8.6/run25_s2 --out ../figures/key_run25.json
python lira_forget05_forget01.py ../results <folder with run24a/run24b>   # LiRA on forget05 / forget01
python analysis_v8_prereg.py --results ../results --out ../figures         # runs 17c/d, 20, 21 (pre-registered)
```

The figures are plain matplotlib (`code/paper_style.py`): Times-like serif text matching the journal, a full frame, the
standard tab10 colours with one colour and marker per method, the oracle as a black star and Model A as a black square,
and no explanatory text inside the plots (explanations are in the captions).

## Licence

Code: MIT (`LICENSE`). Model and data licences: `THIRD_PARTY_NOTICES.md`. Citation: `CITATION.cff`.
