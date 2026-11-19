# Pre-registration of the v8 runs (written before any v8 run)

Date fixed: 4 November 2026. Code version: v8.0. Nothing below changes after results arrive; the results are
reported as they come out, including if Forget Quality passes less often.

## Run 17c / 17d: Qwen2.5-0.5B, Anchor full replicates 9-16

* Setting (fixed, unchanged from runs 16b/17a/17b): Anchor, 10% proxy reference, learning rate 1e-5,
  retain weight 1, 10 epochs, checkpoints at epochs 5 and 10.
* Replicate k in {9, ..., 16}: Model A seed k, oracle seed k, proxy seed k and author sample seed k,
  unlearning data order seed k. Each replicate is scored against its own oracle.
* Primary report: epoch 5 (fixed in advance). Secondary: the epoch chosen by the oracle-free rule with the
  independent seed-42 proxies of run 16a (10% and 25%): highest min(FQ_P, PQ_P) among checkpoints keeping
  >= 90% of Model A's utility.
* Reported for k = 9..16 alone and pooled with k = 1..8 (16 replicates): counts and exact 95% Clopper-Pearson
  intervals of |PrivLeak| <= 0.2, FQ > 0.05, box (both), RQ > 0.05, utility >= 90%; PrivLeak TOST (90%
  bootstrap interval inside +-0.2); ranges of PrivLeak, utility and LiRA.
* LiRA uses all available Qwen oracles (run 7 seeds 42, 1-4, run 17 seeds 5-16), leave-one-out where the
  checkpoint's own oracle is among them.

## Run 20a-d: Llama-3.2-1B, Anchor full replicates 1-8

* Setting (fixed, the run-16c setting): Anchor, 10% proxy, 1e-5, Llama recipe of the paper
  (8-bit AdamW, gradient checkpointing, frozen tied embedding, micro-batch 2, effective batch 32).
* Replicate k in {1, ..., 8}: Model A, oracle, proxy (and its author sample) and data order all use seed k.
* Primary report: epoch 10 (the epoch reported for Llama in run 16c). Secondary: epoch 5, and the epoch chosen
  by the oracle-free rule with the seed-42 Llama proxy of run 16c.
* Same statistics as above; LiRA uses the six Llama oracles of runs 10a-16c plus the eight new ones.

## Run 21a / 21b: manipulating the spill-over kappa (test of Proposition 1)

Question: if the spill-over kappa of likelihood suppression is changed directly, do FQ and privacy move as the
likelihood-transfer account predicts?

* Base method: NPO + retain, learning rate 1e-5, beta 0.1, retain weight 1, 10 epochs, Qwen2.5-0.5B forget10,
  trajectory points every half epoch (21 points) plus full evaluations at epochs 5 and 10.
* Intervention: a term on wrong answers to the forget questions. The wrong answers are built from the training data
  only (3 per question: other retain authors' answers to the most similar retain questions, with the forget
  author's name inserted; the same construction as CIU). TOFU's perturbed answers, used by the truth ratio, are
  never trained on.
  * none: plain NPO + retain (baseline);
  * hold, w = 1, 3, 10: w * mean (NLL_cf - NLL_cf under Model A)^2, which keeps the wrong answers where Model A
    had them and should LOWER kappa;
  * push, w = 1: NPO applied to the wrong answers as well, which should RAISE kappa.
* Seeds: 42 (run 21a) and 1 (run 21b), data order; Model A and the oracle are the seed-42 models (bit-identical to
  run 7). Five conditions per seed, ten runs in total. Nothing is added after seeing results.
* Measurements on each trajectory point, computed against the seed-42 oracle with TOFU's perturbed answers:
  kappa as the secant from Model A, kappa = (b_A - b) / (a_A - a), where a and b are the forget-set mean token
  log-likelihoods of the paraphrased and perturbed answers (as in the paper); FQ; PrivLeak (Min-K%); PQ;
  utility without generation.
* Manipulation check (M): the median kappa over points with a_A - a > 0.2 nats is ordered
  push > none > hold-1 > hold-3 > hold-10. If this ordering fails, the experiment does not test the account and is
  reported as such.
* Primary outcome (O1): PrivLeak at the first trajectory point with FQ > 0.05 ("never" if no point passes).
  Prediction of the account: the lower kappa, the closer O1 is to 0 (less over-unlearning when FQ is first reached);
  push gives the most negative O1.
* Secondary outcomes: O2, the gap between the forget answers' likelihood and the oracle's at that point;
  O3, the number of trajectory points in the retrain-like box (FQ > 0.05 and |PrivLeak| <= 0.2) with utility
  >= 90%; epoch-5 and epoch-10 FQ, PrivLeak, PQ and utility.
* Interpretation fixed in advance: if M holds and O1 moves towards 0 as kappa falls on both seeds, the account is
  supported by an intervention; if M holds but O1 does not move, the account is not supported and the paper says so.
  Lowering kappa may also stop FQ from passing at all (the wrong answers keep their likelihood); that outcome is
  reported, not removed.

## Run 22: Qwen2.5-1.5B, full fine-tuning

* Model A and oracle: Qwen2.5-1.5B-Instruct, the Llama memory recipe (8-bit AdamW, gradient checkpointing, frozen
  tied embedding, micro-batch 2, effective batch 32), lr 1e-5, 5 epochs, seed 42. Proxy: 10% of the retain authors.
* Unlearning (seed 42, 10 epochs, checkpoints at epochs 5 and 10): NPO + retain at 1e-5 and 2e-5, SimNPO
  (w = 1, beta 2.5) at 5e-5, Anchor with the 10% proxy at 1e-5; trajectories every epoch for NPO + retain 1e-5
  and Anchor.
* Reported: for every checkpoint and trajectory point, FQ, PrivLeak (Min-K%), PQ, utility; primary question: do the
  checkpoints and trajectory points that pass FQ lie on the over-unlearning side (PrivLeak < -0.2)? Anchor: is
  |PrivLeak| <= 0.2 at epochs 5 and 10, and does FQ pass? If Model A or the oracle runs out of memory, the run is
  reported as not feasible on a 16 GB T4.

## Run 23a / 23b: Qwen2.5-3B with LoRA

* Too large for full fine-tuning on a 16 GB T4, so every stage trains LoRA adapters (rank 32, alpha 64, every
  linear layer, frozen float16 base, gradient checkpointing): Model A (full) and oracle (retain90) at lr 1e-4,
  5 epochs, effective batch 32, seed 42 (run 23a); a 10% proxy with the same recipe (run 23b).
* Unlearning (run 23b, seed 42, new adapters on the merged Model A, lr 1e-4, 10 epochs, checkpoints at epochs 5
  and 10, a trajectory point every epoch): NPO + retain (beta 0.1) and Anchor with the 10% proxy.
* Reported as for run 22, plus Model A's own PrivLeak (how much the LoRA Model A memorised). The paper will state
  that this is a LoRA setting, not full fine-tuning. If Model A does not memorise the forget set (PrivLeak of A
  below +0.5), the run cannot test over-unlearning and is reported as such.

## Not changed by the results

The rules, learning rates, conditions, the proxy fraction, the epochs reported and the thresholds (FQ 0.05, PrivLeak band
0.2, utility 90%) are fixed here. No setting is added or dropped after seeing the v8 results.

## Run 24a / 24b: extra oracles for LiRA at forget05 and forget01 (added before any run 24 execution)

Reason: the forget05 and forget01 settings (run 12) have one oracle each, so the offline LiRA attack cannot estimate
per-example noise there. Reviewer comment (major 7) asks for LiRA across settings.

* Oracles: Qwen2.5-0.5B-Instruct retrained on retain95 (forget05, run 24a) and retain99 (forget01, run 24b) with
  seeds 1, 2, 3, 4. Same recipe and epochs as the existing oracles O95_s42 and O99_s42 (run 12, seed 42).
* Each new oracle is evaluated on its own forget split, giving O95_s{k}_forget05_raw.json and
  O99_s{k}_forget01_raw.json with per-example membership losses.
* LiRA reference for each forget split: the five oracles (seed 42 from run 12 and seeds 1-4 here). Leave-one-out
  reported for each oracle, as in the nine-oracle analysis of forget10.
* Targets: the run 12 checkpoints for forget05 and forget01 (selected setting, seeds 42 and 1, epoch 5 and 10,
  and the baselines of run 12), scored with the reference attack as in analysis_v8.py.
* Oracle noise band: the spread of the five LiRA oracles' own leave-one-out AUCs is reported as the noise band.
* Limits stated in advance: five oracles estimate the noise less precisely than the nine used for forget10, and
  the result is reported with that caveat. TinyLlama is not included; it stays a stated limitation.
* Reported whatever the outcome: the LiRA AUC for every checkpoint and the band, not a selected subset.

## Run 23b split (execution only, added before any run 23b execution; code v8.2)

Run 23a took about 10 h (Model A 318 min, oracle 285 min), so run 23b (NPO + retain and Anchor, 10 epochs each, a
trajectory point every epoch, plus the 10% proxy) would exceed the 12 h limit in one notebook. Run 23b is therefore
executed as two notebooks with the same settings: run23bnpo (NPO + retain at 1e-4, beta 0.1) and run23banchor
(10% proxy at the same recipe, then Anchor at 1e-4). Both use the run 23a adapters of Model A and the oracle and
write into results/v6/run23b. Nothing else changes.

## Run 23c: Anchor at 3B with a lower learning rate (added before any run 23c execution; code v8.3)

Reason: run 23b Anchor (lr 1e-4, one seed) did not pass FQ at 3B (FQ about 1e-5 at epochs 5 and 10) and under-unlearned
(PrivLeak +0.45 and +0.42). Run 23c tests whether the Anchor result depends on the learning rate.

* Same Model A, oracle and LoRA adapters as run 23a (seed 42, unchanged), same 10% proxy recipe as run 23b (seed 42).
* Only change: Anchor learning rate 3e-5 instead of 1e-4. Same epochs (10), batch settings, LoRA rank 32, and
  trajectory evaluation at every epoch.
* Primary outcome: FQ and PrivLeak (Min-K%) at epochs 5 and 10, compared with the run 23b Anchor (lr 1e-4).
* Reported whatever the outcome. This is a single seed and one extra learning rate; it does not replace the seed
  question, which is stated as a limitation.

## Run 23d: Anchor at 3B with a higher learning rate (added before any run 23d execution; code v8.4)

Reason: run 23b Anchor (lr 1e-4) under-unlearned at 3B (FQ about 1e-5, PrivLeak +0.42 to +0.45); run 23c (lr 3e-5)
under-unlearned more (FQ about 1e-9 to 1e-11, PrivLeak +0.87 to +0.94). The direction suggests that a higher learning
rate may be needed for Anchor to forget at 3B. Run 23d tests lr 3e-4.

* Same Model A, oracle and LoRA adapters as run 23a (seed 42), same 10% proxy recipe (seed 42), same epochs (10),
  batch settings, LoRA rank 32 and trajectory evaluation at every epoch as runs 23b and 23c.
* Only change: Anchor learning rate 3e-4.
* Primary outcome: FQ and PrivLeak (Min-K%) at epochs 5 and 10, compared with runs 23b and 23c.
* Reported whatever the outcome. Single seed; a higher learning rate may also raise utility loss, which is reported.
