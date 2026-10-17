# Pre-registration: robustness of the headline claims (B1–B4), after the red-team review

Written 2026-10-01, before any of the analyses below were run, in response to `research/reviews/redteam_2026-10-01.md`
(verified findings: the selector's top-1 vs top-2 stability gap is a median 0.001 NMI, about the size of rerun noise;
the second-ranked layer has mean regret 1.65 points). Approved by Steve. Purpose: narrow the claims to what survives.

Shared: the 11 held-out pairs of `PREREG_layer_selection.md`, their existing caches (24k train / 6k test), frame W,
K = 512, outcome p@10 (DBpedia l2, 20NG fine) unless stated.

## B1. Selection noise
For every layer: 10 seed pairs (k-means seeds 2j, 2j+1, j = 0..9), each fitted on a different random 20,000-document
subsample of the training split (subsample seed j); stability = NMI of the two seed runs' test assignments.
- **S10** = argmax over layers of the mean stability across the 10 pairs (ties deeper).
- **Single-pair rule:** the original rule applied to each pair separately; its *expected* regret is the mean over the
  10 picks.
- **B1-a.** S10's mean held-out regret is ≤ 1.0 point.
- **B1-b.** The single-pair rule's expected regret (mean over pairs and seeds) is ≤ 1.5 points.
- **B1-c.** S10 beats the last layer in ≥ 9 of 11 pairs.
- Reported: per pair, the distribution of single-pair picks; a two-level bootstrap of (S − last) that resamples pairs
  and, within each pair, the seed pair used to pick (so selection variance enters the CI).

## B2. Structure, not task: non-topical labels on the same texts
Three labels computed from each document's text, terciles fitted on the training split: character length of the
full text; digit density (digits per character, first 400 characters); capital-letter density (first 400
characters). Per layer, p@10 for each label (raw centered cosine, as in the main study).
- **H-agnostic (predicted):** the selector picks one layer per model and corpus regardless of the task, so on labels
  that are not the corpus's dominant cluster structure it should do poorly.
- **B2-a.** Averaged over pairs and the three labels, S's regret is ≥ 1.5 points (3× its topical regret).
- **B2-b.** For the length label, the best layer differs from the topical best layer by ≥ 1/4 of the depth in ≥ 6 of
  11 pairs.
- If B2 fails (S also picks well for non-topical labels), the "dominant structure" explanation is weakened.

## B3. Outcome metric
p@10 recomputed in three spaces: raw centered cosine (original), per-dimension z-scored cosine, and frame-W cosine.
- **B3-a.** Under each alternative space, S's mean regret is ≤ 1.0 point and S beats the last layer in ≥ 9 of 11.

## B4. Bin flips: noise control and per-block sensitivity (Gemma 4 E2B-it GGUFs)
- **B4-a (noise control, CPU).** For Q4_K_M and IQ2_M, per layer: bin flips of the real quantized features vs flips of
  the bf16 features plus isotropic Gaussian noise with each document's error norm matched to its real error, and vs
  the bf16 features plus a *shuffled* real error (another document's error vector). Predicted: real flips exceed
  isotropic-noise flips by ≥ 25% at ≥ half of the layers (quantization error is structured, not isotropic).
  The shuffled-error comparison is descriptive (doc-specific vs distributional error).
- **B4-b (per-block sensitivity, GPU).** Restore each decoder block of Q4_K_M alone to bf16 (35 runs, the same 1,000
  documents as Q3) and record the KL reduction. Predicted: Spearman(per-block bin-flip jump, KL reduction) exceeds
  Spearman(per-block activation-error jump, KL reduction).
- **B4-c (random baseline).** 20 random 4-block restorations instead of 5 (draws 0–4 must reproduce the original
  values exactly; a reproducibility check), and the Q3 comparison recomputed against the 20-draw mean.

## Results so far (2026-10-02; B1 running, 7/11 pairs; B4-b/c queued)
- **B2 (`results/robust_b23.json`): both supported.** Non-topical regret of S, mean over pairs and the three labels:
  5.7 points (B2-a ≥ 1.5). Length-best layer ≥ 1/4 depth from the topical best in 9/11 (B2-b ≥ 6). Descriptive:
  the digit-density best layer is layer 0 in 11/11.
- **B3: not supported (one arm).** Frame-W cosine: S regret 0.50, beats last 9/11 (pass). Z-scored cosine: S regret
  0.65, beats last 7/11 (fail; needed 9). Z-scoring lowers the last layer's regret to ≈ 0 in several pairs
  (e.g. SmolLM2-20NG, Gemma 4 E2B-20NG), which motivated `PREREG_outlier_dims.md`.
- **B4-a (`results/robust_b4a.json`): supported.** Real flips ≥ 1.25× isotropic-noise flips at 100% of layers ≥ 1 for
  both Q4_K_M and IQ2_M (needed ≥ 50%).
- **B4-b/c (2026-10-03; rerun on GPU1 after the first reference run was killed out of memory; `diag/score_b4bc.py`,
  `results/robust_b4bc.json`).**
  - **B4-b: not supported.** Restoring each of the 35 blocks of Q4_K_M alone: Spearman(per-block bin-flip jump, KL
    reduction) = 0.03, vs 0.39 for the per-block activation-error jump. Bin flips do not rank single blocks.
  - Descriptive: the KL reduction is concentrated in blocks 0–14 (0.008–0.038 each); blocks 15–34 give ≈ 0 (≤ 0.004).
    In Gemma 4 E2B, blocks 15+ reuse KV caches computed by earlier blocks (KV sharing), which fits; not tested.
    The best single blocks are 4, 13, 0, 3.
  - **B4-c: reproducible.** Draws 0–4 reproduce the original KL values exactly (max |diff| 0.0; same layers). The 20
    random 4-block sets: KL 0.377 ± 0.018. The lens choice (blocks 4, 5, 14, 33; 0.330) is below all 20 random draws
    and below activation-error (0.352); but first-4 (0.327) is as good. Q3 stands as "lens beats random and activation
    error, ties first-4"; B4-b suggests why: the lens set happens to include early blocks (4, 5, 14), where all the
    restorable damage sits.
- **B1 (`results/robust_b1.json`, 2026-10-02): all three supported.** S10 mean regret 0.70 points (B1-a ≤ 1.0);
  single-pair expected regret 1.36 (B1-b ≤ 1.5); S10 beats the last layer in 10/11 (B1-c ≥ 9). Two-level bootstrap of
  (last − single-pair) regret: 95% CI [0.6, 3.5] points. Averaging 10 pairs halves the expected regret of one pair.
- **B1 interim (superseded by the line above):** single-pair picks spread 3–9 layers; Gemma 4 E4B: 8/10 single pairs pick layers 33–42
  (regret 4–8 points), S10 picks 14 (regret 0.3).
