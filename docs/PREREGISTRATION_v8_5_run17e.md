# Pre-registration addendum, v8.5: Run 17e (proxy scores for the eight full replicates)

Written before the run. Code: unlearning-v6 v8.5, phase `run17e`. Date: 14 Nov 2026.

## Why this run exists

Run 17 trained a 10% proxy for each full replicate (P10_s1 to P10_s8) but did not score it. The paper's claim that
choosing the epoch with the proxy puts 5 of 8 replicates in the retrain-like box (Table 1, rows "FR, epoch chosen by
P10/P25"; response to Major comment 5) therefore has no saved evidence behind it. This run scores the proxies so the
claim can be checked. The result is reported whatever it is.

## Setup (fixed)

* Proxies: P10_s1 to P10_s8, the run 17 recipe: retain90, author fraction 0.10, sample seed = training seed = k,
  Model A training recipe. Retrained here only if the checkpoint is missing.
* Each proxy is evaluated on forget10 (raw file `P10_sK_forget10_raw.json`), then deleted.
* Candidates for replicate k: the epoch 5 and epoch 10 checkpoints of `anchor_p10_lr1e-5` from run 17a/17b
  (seeds 1 to 8). Their utility is relative to Model A of the same replicate.

## Rule (fixed)

1. Keep candidates with utility >= 90% of Model A.
2. Score each candidate against the replicate's proxy: FQ_P = KS p-value of forget truth ratios against the proxy,
   PQ_P = KS p-value of Min-K% member percentiles against the proxy, score = min(FQ_P, PQ_P).
3. Choose the candidate with the higher score. Ties go to epoch 5 (decided in advance).

## Primary outcome

Number of replicates whose chosen checkpoint is in the retrain-like box, measured against the replicate's own
oracle: FQ > 0.05 and |PrivLeak| <= 0.2 (utility >= 90%). Report k/8 with an exact 95% Clopper-Pearson interval.

## Secondary outcomes

* The same count if the epoch is chosen at random with seed 42 (a floor for the rule).
* Spearman correlation between the proxy score and the oracle's RQ over the eight replicates' two candidates each.

## What happens with the result

* If the primary count is 5/8, the paper's claim is supported by saved results, and the text stays.
* If it is different, the paper's Table 1, Section 6.1, and the Major 5 response change to the new count, and the
  wording "5 of 8" is removed everywhere.
* In both cases the figure's last bar shows this run's number.

## Not changed by the results

The rule, candidates, tie-break, threshold and outcome are fixed here. No candidate or epoch is added after the run.
