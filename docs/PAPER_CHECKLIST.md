# Paper checklist (version after the third review)

The LaTeX source is delivered separately (`unlearning_paper_latex.zip`); every number in it comes from
`results/` through `code/make_figures.py` and `code/analysis_v5.py` (`figures/v5_*.csv`, `figures/v5_tables.md`).

## Third review: what was done

| # | Reviewer point | Where in the paper | Data |
|---|---|---|---|
| 1 | Propose and test a fix | Sec. 3.3 (calibrated objectives), 4.4, Table 7 | runs 8, 8b, 9 |
| 2 | Other method families | Sec. 3.3, 4.5, Fig. 7 | run 8 |
| 3 | Evaluate every epoch | Sec. 3.6, 4.3, Fig. 4, Table 5 | runs 7, 8, 8b |
| 4 | More oracle seeds | Sec. 4.2 (5 oracles, 10 pairs) | run 7 |
| 5 | Effect size / equivalence | KS distance vs oracle-to-oracle range (Sec. 3.4, 4.2, 4.3) | runs 7-9 |
| 6 | Llama-3.2-1B | Sec. 3.2, 4.5, Fig. 6, Table 8 | runs 10a, 10b |
| 7 | CIs, Model A seeds | Sec. 4.1 | run 9 |
| 8 | Threat model | Sec. 3.5 | - |
| 9 | Stronger attacks | Min-K%++, reference attack (Sec. 3.4, Table 5) | runs 7-10 |
| - | Title, abstract, fonts, mechanism subsection | Title (question form), abstract, Sec. 4.4, Fig. 5 | - |

Not done: LiRA-style per-example attacks (need many shadow models), a bootstrap TOST on the KS distance
(the bootstrap is biased upward for KS; we compare point distances with the oracle-to-oracle range instead),
objectives trained on self-generated perturbations (future work).

## Fourth review: status (v6)

| # | Point | Status | Data |
|---|---|---|---|
| 1 | Placeholder, "four attacks" in the abstract, Fig. 2 | Fig. 2 redrawn (5-oracle pairs, zoom panel); text fixes go into the next draft | - |
| 2 | Calibrated tau with base-model offset | Done: Match + offset, Anchor, Anchor-shift | run 11a |
| 3 | Justify the primary attack; "pre-registered" wording | Next draft (worst case over 5 attacks reported) | - |
| 4 | Privacy on calibration vs evaluation non-members | Done (analysis_v6.py) | runs 7-12 |
| 5 | A method that passes both criteria on held-out seeds | Anchor with 10% proxy: box 5/8, joint test 4/8 held-out seeds; Llama 1/4 FQ; CIU failed | runs 11a-16c |
| 6 | Larger model (3B-8B) | Code ready (run 15); needs an A100 (HPC) | - |
| 7 | Second benchmark (MUSE) | Not done (7B targets; needs HPC) | - |
| 8 | Theory | Proposition 1 (likelihood transfer) with measured kappa | runs 7-8b |
| 9 | Joint metric | PQ / RQ | all v5-v6 runs |
| 10 | Re-rank published checkpoints | OpenUnlearning Llama-3.2-1B, 214 checkpoints | runs 14, 14b, 14c |
| 11 | LiRA | Offline LiRA with the five oracles as OUT models | runs 7-12 |

All runs finished. Manuscript revised (paper/): appendix for detail tables, review-history sentences removed.
Still to do: author contributions (placeholder), journal choice, larger model (needs HPC), MUSE.

## Before submission

- [x] `environment.txt`
- [ ] Co-authors confirm their contribution statement (placeholder in Declarations)
- [ ] Choose the journal and check its page limit and AI-use policy
- [ ] GitHub repository + Zenodo DOI; set `\repourl` in `sn-article.tex`
- [ ] Executed Kaggle / Colab notebooks into `notebooks/executed/` (names in `docs/WHAT_TO_SHARE.md`)
- [ ] Final read by all authors
