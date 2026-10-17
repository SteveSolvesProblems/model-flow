# Pre-registration: why did the overnight predictions fail? (diagnostics D1–D4)

Written 2026-10-01, after the overnight results (`PREREG_overnight.md`, Results) and before any of the analyses below
were run. Approved by Steve with a ~2 h CPU budget. These are **diagnostics**: they test explanations for failures.
A variant that "fixes" a failure here is a hypothesis for a fresh held-out test, not a new result.

Shared definitions:
- **Frame W** (the frozen frame): center, top-64 PCs, whiten; fit on ≤12k train rows (`lens.frame`).
- **Frame D** (deflated): center, drop the top 5 PCs, whiten the next 64. Removes the few dominant directions
  (massive activations, Sun et al. 2024; rogue dimensions) while keeping the frame size fixed.
- **Selector S(K, frame):** argmax over layers of the cross-seed NMI (seeds 0, 1; faiss k-means, niter 50, nredo 1),
  ties go deeper; frozen variant is S(512, W).
- **Top-5 share:** variance in the top 5 PCs ÷ total variance, centered train features.

## D1. Short texts: too many bins, or dominant directions? (Banking77, STS-B; mean pooling)
Data and outcomes exactly as E2 (`task_score.py`): Banking77 intent p@10 and STS-B Spearman, primary outcome = the
pre-registered raw centered cosine; secondary = the same outcome in frame W (`task_posthoc.json`).
- Variants: S(64, W), S(512, D), S(64, D). K = 64 is fixed here (close to Banking77's 77 intents), not tuned.
- **D1-H1 (too many bins).** S(64, W) beats the last layer on Banking77 in ≥ 4/5 models and on STS-B in ≥ 3/4.
- **D1-H2 (dominant directions).** S(512, D) meets the same bar.
- **Guard.** On the 11 held-out DBpedia/20NG pairs, a variant that passes must keep mean regret ≤ 0.010 (the frozen
  rule: 0.0049). Otherwise it trades the main result for the short-text one.
- Descriptive: stability range over layers ≥ 1 for each variant; top-5 share at the chosen layers.

## D2. Pythia: does a dominant direction hold the selector at the top of the stack?
At each checkpoint ≥ 1,000 and layer: top-5 share, and S(512, D).
- **D2-H1.** At checkpoints 4,000–143,000 where S ≠ the best layer, the chosen layer's top-5 share exceeds the best
  layer's in ≥ 80% of cases (pooled over both models).
- **D2-H2.** Over checkpoints ≥ 1,000, S(512, D) has lower mean regret than S(512, W), in both models.
- Descriptive: the layer × checkpoint grid of top-5 share (for the training-dynamics figure).

## D3. Partitions vs CKA: is CKA's disagreement driven by dominant directions? (Davari et al. 2023)
Same 7 models, 21 pairs and 1,000 documents as E3; linear CKA recomputed on frame-W features.
- **D3-H1.** Spearman(NMI matrix, frame-W CKA matrix) ≥ 0.7 in ≥ 12/21 pairs (raw CKA: 5/21), and is higher than
  with raw CKA in ≥ 18/21.
- Descriptive: X1 (relative depth) using frame-W CKA matches.

## D4. Polysemy in Gemma: does re-injected word identity mask senses?
No new compute: `polysemy.json` (per-layer partial correlations) and the token-lens identity curves
(`tok__token_lens.json` for Gemma 4 E2B base, `token_qwen3_score.json` for Qwen3-1.7B), K = 512.
- **D4-H1.** In Gemma, across layers ≥ 1, Spearman(word-identity NMI, polysemy partial correlation) ≤ −0.5.
- Stated in advance: E6's primary Gemma layer (18) is already its identity minimum and showed no polysemy signal,
  which weakens this explanation before testing. Qwen is descriptive.

## Not tested here
Length or position confounds in the short-text features; KV-boundary stages (the test was too weak to be worth more
compute); routing stability (hubness is simply the better signal).

## References
Sun, Chen, Kolter & Liu (2024), Massive Activations in Large Language Models. Davari et al. (2023), Reliability of CKA
as a Similarity Measure in Deep Learning. Kornblith et al. (2019), CKA.

## Results (2026-10-01/02; `results/diag_d1.json`, `diag_d1guard.json`, `diag_d3.json`, `diag_d4.json`; D2 running)
- **Guard (11 held-out pairs, mean regret):** frozen rule S(512, W) 0.46 points; S(64, W) 5.46; S(512, D) 1.62;
  S(64, D) 2.86. Every variant breaks the guard (≤ 1.0).
- **D1-H1 (too many bins): not supported.** S(64, W) beats the last layer on Banking77 in 4/5 but STS-B in 2/4.
- **D1-H2 (dominant directions): not supported.** S(512, D) 4/5 and 1/4. No variant passes both tasks; with the guard,
  the short-text failure has no fix that keeps the main result. (D1 was rerun after a crash: the first process,
  started before a cache-lookup change, built a mismatched layer list for the third STS-B model; the rerun asserts
  matching lengths and checkpoints each model.)
- **D3-H1: supported.** Spearman(NMI matrix, frame-W CKA matrix) ≥ 0.7 in 21/21 pairs (0.93–0.97; raw CKA −0.16 to
  0.91), higher than raw in 21/21. CKA's disagreement with the partitions is driven by dominant directions, consistent
  with Davari et al. (2023). X1 (relative depth) with frame-W CKA: 0.72–1.00.
- **D4-H1: not supported.** Gemma Spearman(identity, polysemy) = −0.30 (Qwen3-1.7B −0.63, descriptive).
- **D2 (`results/diag_d2.json`), 2026-10-02.** D2-H1: **not supported**: in 12 checkpoint cases (4k–143k, both models)
  where S ≠ the best layer, the chosen layer has the larger top-5 share in only 2 (17%; needed 80%). D2-H2:
  **supported**: over checkpoints ≥ 1,000, S(512, D) has lower mean regret than S(512, W) in both models (Pythia-410m
  1.39 → 0.94 points; 1.4B 0.85 → 0.67). Removing the top five directions helps the selector in Pythia, but not
  because the chosen layer is the one dominated by them. (Note: on the 11 held-out pairs, S(512, D) breaks the guard,
  1.62 points; the improvement is Pythia-specific.)
