# What to keep, what to share publicly

| Item | Keep | Share publicly | Why |
|---|---|---|---|
| `code/` (all .py, .sh, tests, README, requirements) | Yes | **Yes** (MIT) | Needed to reproduce |
| `results/` JSON + CSV | Yes | **Yes** (CC BY 4.0) | Every number in the paper comes from these; no personal data (fictitious authors) |
| `logs/` | Yes | **Yes** | Show training curves, timings, and that runs completed |
| `figures/` (PDF + PNG) | Yes | **Yes** | Paper figures |
| `notebooks/` templates | Yes | **Yes** | Show exactly what ran |
| Executed Kaggle notebooks (your copies, with outputs) | Yes | **Yes**, after checking | First check that no output cell shows a Kaggle/HF token, API key, e-mail or file path with your personal account name |
| `docs/` | Yes | **Yes** | Data statement and reproducibility notes are expected by reviewers |
| `docs/DECLARATIONS_DRAFT.md` | Yes | Optional | Text goes into the paper; the file itself is not needed in the repository |
| TOFU data (`data/` folder created by the code) | Not needed | **No** | Redistribution is allowed (MIT) but unnecessary; the code downloads it |
| Model checkpoints (`ckpt/`, `keep_ckpt/`, several GB) | Optional | **No** | Large, derived from the base models; not needed to check results |
| Kaggle API key (`kaggle.json`), Hugging Face token | Private | **Never** | Credentials |
| Old zips (`unlearning_tofu_experiment.zip` v1, v2, v3, v3.1, v3.2) | Optional archive | No | The released `code/` contains the final version, which reproduces every run |
| Pilot-run files (`results/run0_pilot`, `logs/run0_pilot`) | Yes | Yes | Mark as "pilot, superseded"; it shows how the protocol evolved |
| Review notes (the two external reviews) | Yes | No (private) | Summarise their effect in the paper instead |
| Previous paper PDF | Yes | Not in this repository | Different project |

## Notebooks: which runs to include

| Notebook | Include | Note |
|---|---|---|
| Run 0 pilot (the successful commit, results_for_analysis_3) | Yes, label "pilot" | The earlier failed commits (out-of-memory, interrupted) are not needed |
| Run 1 `unlearning-tofu-experiment-v3-code` (committed version) | Yes | Ignore the interactive cell that shows KeyboardInterrupt; the committed version is the one that ran |
| Run 2 `unlearning-tofu-run2-seeds` | Yes | |
| Run 3 (v3.2 final) | Yes | |
| Run 4 `unlearning-run4-scale` | Yes | |
| Run 5 `unlearning-run5-tinyllama` (second account) | Yes | Same code; note in the README that it ran on a second account (no effect on results) |
| Run 6 `unlearning-run6-tinyllama-low` | Yes | |
| Runs 7, 8, 8b, 9 (`unlearning-run7`, `-run8`, `-run8b`, `-run9`) | Yes | Run 8's Calibrated NPO jobs ran out of memory; keep the notebook (it is the record) and note that run 8b repeats them |
| Run 10 first attempt (out of memory) | No | Superseded by 10a/10b |
| Runs 10a, 10b | Yes | The 10b results were downloaded as `results_v5_run10a_1.zip`; the contents are run 10b (`results/run10b_v5`). Check the notebook names before sharing |
| Run 11a `unlearning-run11a` | Yes | |
| Run 11b, first attempt (out of memory) | No | Superseded by the v6.1 re-run |
| Run 11b `unlearning-run11b` (v6.1 re-run) | Yes | Results downloaded as `results_v6_run11b_1.zip` |
| Runs 12, 12b, 13a, 14 | Yes | |
| Runs 13b, 13c, 14c, 16a, 16b, 16c (Kaggle) | Yes | In `notebooks/executed/` |
| Run 14b (Kaggle) | Yes | **Still missing**: download `unlearning-run14b` and save as `run14b_openunlearning.ipynb` |
| Run 16a, first attempt (old code, "invalid choice") | No | Re-run with v6.5 |
| Colab attempts of 13b-14c | No | Did not complete (no GPU / disconnected) |
| Early Colab attempt (before v6) | No | Not completed |
