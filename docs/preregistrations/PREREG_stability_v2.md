# Pre-registration: the stability pick against its untested comparisons (registered 2026-10-17)

Status: **approved and frozen 2026-10-17; not yet run.** No extraction or scoring for anything below has run. The
code checks listed under Compute happen on development data only; any change they force is added as a dated addendum.

## Why
The stability pick (PREREG_layer_selection.md rule S; PREREG_robustness.md B1 averaged version S10) passed on 11
held-out model–collection pairs. Four gaps remain before it can be written up:
1. The comparisons the page names were never run: the first-peak intrinsic-dimension rule (Wang et al. 2026, FEPoID)
   and the intrinsic-dimension minimum of Tang & Yang 2026.
2. The second-to-last layer (1.87 vs the pick's 0.49) was computed afterwards, with no interval.
3. All 11 pairs are topic collections scored by topic search.
4. The 11 pairs reuse the 3 development models on a new collection, so only 4 models are fully new.

## The rule under test (unchanged)
**S10** exactly as in PREREG_robustness.md B1, with no new choices:
- K = 512;
- 64 whitened PCs;
- 10 seed pairs on 20,000-document train subsamples;
- argmax of the mean test NMI over layers 0..L, ties going deeper.

The single-pair rule S is reported alongside it.

## Fresh pairs (none scored for layer choice before this document)
- **New models:**
  - Llama-3.2-1B (16 layers);
  - SmolLM2-360M (32);
  - Qwen3.5-4B-Base (32).
  - All three are already downloaded on the GPU host. Llama-3.2-1B has only had a pipeline-fault check, with no labels or layer scores.
- **New collection: Yelp Review Full.**
  - 24k train / 6k test, sampled by the same procedure as DBpedia (seed 0), max 64 tokens, mean-pooled.
  - Label: star rating (5 classes). This is a collection whose labels are not its topics.
- **Pairs:**
  - **Topic set (6):** the 3 new models × {DBpedia, 20 Newsgroups}.
  - **Rating set (9):** the 3 new models plus Qwen3-1.7B, Qwen3.5-2B, Gemma 4 E2B, SmolLM2-1.7B, Qwen3.5-4B and Pythia-2.8B (step 143000) on Yelp.
  - Pythia-2.8B's DBpedia layer scores were seen earlier, so it enters only on Yelp. Gemma 4 E4B is left out (memory).

## Outcomes (labels used only here)
- **T1, neighbour search:** p@10, as before (centered cosine; test queries, train database). The label is the topic
  (DBpedia l2, 20NG fine) or the star rating.
- **T2, linear probe:** a task beyond neighbour search.
  - Logistic regression (multinomial, L2, C = 1, lbfgs, 1,000 iterations) on the top 256 PCs of the standardized train features (≤ 12,000 rows).
  - Scored as test accuracy, using the same labels as T1.
  - This is the kind of downstream use FEPoID was built for (a supervised probe on the chosen layer).
- **Shortfall** = best layer's score − chosen layer's score, in points (0–100).

## Comparators (all fixed here)
1. **Last layer.**
2. **Second-to-last layer** (L − 1), now pre-registered.
3. **FEPoID** (Wang et al. 2026, §3.5).
   - Per layer 1..L: TwoNN on 5,000 test vectors (baselines_a.twonn, discard 0.1).
   - Local maxima are layers ℓ in 2..L−1 with d(ℓ) > d(ℓ−1) and d(ℓ) ≥ d(ℓ+1).
   - Scan the maxima from shallow to deep. Discard a peak if d(ℓ) < d(min(ℓ+w, L)) and d keeps rising strictly from ℓ+1 to min(ℓ+w, L).
   - Take the earliest surviving peak; if none survives, the shallowest maximum; if there are no maxima, layer L.
   - w = 7, and w = 3 for Llama-3.2-1B, as in their Table 8.
   - Adaptation, stated as a limit: their vectors are last-token states of generated answers, ours are mean-pooled texts.
4. **ID minimum** (Tang & Yang 2026, Eq. 4 with their ⌊0.2L⌋ exclusion).
   - Argmin of TwoNN over layers ⌊0.2L⌋+1..L.
   - Adaptation: they use this layer to place KV re-routing and embed from the last layer, so here it is only a layer-choice signal. Their own reported layers do not follow Eq. 4 exactly; we use Eq. 4 as written.
5. **One labelled example per topic**, as in the earlier study. Reported only, no criterion.

## Predictions and criteria
- **P1, replication (topic set, T1).**
  - S10's mean shortfall is ≤ 1.0 point.
  - S10 beats the last layer in ≥ 5 of 6 pairs.
- **P2, second-to-last (topic set, T1).**
  - S10's mean shortfall is lower than L − 1's by ≥ 0.5 points.
  - The 95% two-level bootstrap interval of (L − 1 shortfall − S10 shortfall) excludes 0. The bootstrap resamples pairs and seed pairs, as in B1.
  - If P2 fails, the write-up says the pick is no better than taking the second-to-last layer.
- **P3, intrinsic-dimension rules (topic set, T1).**
  - S10's mean shortfall is lower than both FEPoID's and ID-min's.
  - S10 has the lower shortfall in ≥ 4 of 6 pairs against each.
- **P4, ratings (rating set, T1). Predicted to fail.**
  - Earlier result B2: the pick follows the collection, not the task.
  - The prediction is that S10's mean shortfall on star rating is ≥ 1.5 points, or that S10 beats the last layer in ≤ 6 of 9.
  - If S10 instead does well on ratings, B2's "dominant structure" reading weakens. That is reported either way.
- **P5, probe (topic set and rating set, T2).**
  - On the topic set: S10's mean probe shortfall is ≤ 1.5 points, and S10 beats the last layer in ≥ 4 of 6.
  - On the rating set: reported, no criterion.
- **Reported without prediction:**
  - all comparators on every pair and task;
  - Spearman over layers of stability vs each outcome;
  - where S10, FEPoID, ID-min and the best layer sit as fractions of depth.

**Readings:**
- **P1–P3 pass:** the pick stands, now with the comparisons the page claims.
- **P1 passes but P2 fails:** the practical advice becomes "take the second-to-last layer". The pick's value is only that it is label-free and model-agnostic, which then has to be argued against L − 1's simplicity.
- **P3 fails:** we report that an existing intrinsic-dimension rule does as well.

## Figure (fixed now)
For Llama-3.2-1B on DBpedia, one panel per layer track:
- S10 stability;
- TwoNN ID;
- p@10;
- probe accuracy.

Vertical marks show the S10 pick, FEPoID, ID-min, L − 1, last, and the best layer for each outcome. The pair is chosen now, before any result.

## Compute (estimate; needs sign-off)
- **Extraction:**
  - 3 new models × 3 collections, plus 6 existing models × Yelp: 15 extractions of 30k texts at 64 tokens.
  - About 4–6 GPU-hours on the P100s.
  - About 60 GB of caches, written to data (not the root disk).
- **CPU:**
  - S10 is 10 seed pairs × 2 k-means runs per layer, over 15 pairs and about 500 layers.
  - TwoNN and probes add about 8 CPU-hours.
  - The B1 run on 11 pairs took about a day of wall-clock time; this is about 1.5×.
  - Expected: 1–2 days on the GPU host CPUs.
- **Code changes before freezing:**
  - a Yelp loader in extract_hf.py;
  - `fepoid.py`, covering FEPoID, ID-min and the probe;
  - `score_stability_v2.py`.
  - These will be checked on development data (Qwen3-1.7B DBpedia, already seen) before freezing, so only scoring of the fresh pairs remains.

## Limits stated in advance
- **Mean-pooled 64-token texts.** Both ID rules are adapted from other vector types.
- **Small models (0.36B–4B).** The FEPoID paper's models are 1B–8B.
- **One rating collection.** Star ratings are partly topical (cuisine and service words).
- **A self-dated registration,** as before.
