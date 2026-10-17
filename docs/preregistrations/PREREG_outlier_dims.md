# Pre-registration: are a few very large dimensions behind the "lazy last layer"? (OD)

Written 2026-10-02, before any of the analyses below were run. Approved by Steve ("proceed with all of the work").
Code: `diag/outlier_dims.py`. Existing caches only (CPU).

## Why
Four results point at a small number of dimensions with very large values (massive activations, Sun et al. 2024) as
a confound in how layers are compared:
- B3: rescaling every dimension to equal spread makes the last layer competitive in several models.
- D1: removing the top five principal directions moves the selector's choice.
- D3: CKA disagrees with the bins on raw features but agrees (ρ 0.93–0.97, 21/21 pairs) in the whitened frame.
- F1: the only two models with similar flow profiles (Qwen3-1.7B, SmolLM2-1.7B) both have massive activations.
If a handful of dimensions dominate raw cosine similarity at late layers, then part of "the last layer is a poor
embedding" is a property of the readout (raw cosine), not of how texts are arranged.

## Definitions
- Data: the 11 held-out pairs of `PREREG_layer_selection.md` (24k train / 6k test, mean pooling), outcome p@10
  (DBpedia l2, 20NG fine), exactly as in B1–B3.
- **Massive dimension** at a layer: a coordinate whose mean absolute value over the training documents (uncentered)
  exceeds 20× the median of that quantity over all coordinates. Fixed before looking; the count per layer is reported.
- **Model with massive dimensions:** at least one massive dimension at the last layer.
- Readouts, each p@10 by cosine after centering on the training mean:
  (a) `centered` (the original outcome), (b) `removed` (massive dimensions of that layer set to 0),
  (c) `robust` (each dimension minus its training median, divided by its training median absolute deviation).
- Last-layer gap = max over layers of p@10 − p@10 at the last layer, per readout.
- Selector under removal: map stability (K = 512, seeds 0 and 1, frame W) recomputed on the `removed` features.

## Predictions
- **OD1.** In pairs from models with massive dimensions, `removed` closes ≥ 50% of the last-layer gap (relative to
  `centered`) in ≥ 2/3 of those pairs. In the other pairs, the gap changes by < 1 point in ≥ 2/3 of them.
- **OD2 (the selector is not driven by them).** Recomputed on `removed` features, the selector's regret (scored with
  `centered` p@10) changes by ≤ 0.5 points in ≥ 9 of 11 pairs. (Frame W whitens the top 64 directions, so we expect
  little change.)
- **OD3.** `robust` closes ≥ 50% of the last-layer gap in ≥ 6 of 11 pairs.
- **Descriptive:** number of massive dimensions by layer and model; whether they are the same coordinates across
  layers; the best layer under each readout.

## Stated in advance
The 20× threshold is a choice; we also report 10× and 50× without using them for the verdicts. Removing coordinates
changes dimensionality slightly (a few of 896–2,560). If OD1 fails, the "lazy last layer" is not explained by massive
dimensions, and the B3 rescaling effect needs another explanation.

## Note added during the run (2026-10-02 16:10, before any verdict or last-layer value was seen)
The first layers of the first pair (Qwen3.5-0.8B, layers 6–8) show 2–4 coordinates above the 20× threshold, so the
"models with massive dimensions" group may include most models, leaving OD1's second half with few or no pairs. The
registered verdicts stand unchanged; the 50× split will be reported alongside, labelled descriptive.

## Results (2026-10-02; `results/outlier_dims.json`)
Massive dimensions (20×) at the last layer: 6 pairs (SmolLM2 ×2, Qwen3-1.7B-20NG, Qwen3.5-4B-20NG, Qwen3.5-2B-20NG,
Gemma 4 E2B-20NG); at 50×: only SmolLM2 ×2 and Qwen3-1.7B-20NG.
| pair | massive (20×/50×) | last-layer gap: centered / removed / robust |
|---|---|---|
| smollm2_17b | 11/3 | 0.065 / 0.049 / 0.024 |
| smollm2_17b_20ng | 12/4 | 0.058 / 0.049 / 0.000 |
| qwen3_17b_20ng | 22/5 | 0.064 / 0.064 / 0.002 |
| gemma4_e2b_20ng | 7/0 | 0.032 / 0.019 / 0.000 |
| qwen35_4b_20ng | 3/0 | 0.010 / 0.013 / 0.011 |
| qwen35_2b_20ng | 2/0 | 0.012 / 0.013 / 0.018 |
| 5 pairs without | 0 | unchanged by either (e.g. gemma4_e4b 0.055 / 0.055 / 0.058) |
- **OD1: not supported.** Zeroing the massive coordinates closes ≥ 50% of the gap in 0 of 6 massive pairs (other pairs
  unchanged: 5/5). The few largest coordinates are not, by themselves, the cause.
- **OD2: not supported.** The selector's regret changes by > 0.5 points in 5 of 11 pairs when those coordinates are
  zeroed (largest: Gemma 4 E4B-20NG, 0 → 4.7; SmolLM2, 0 → 3.1), although frame W whitens the top 64 directions.
- **OD3: not supported (4/11; needed 6).** But the 4 pairs where robust per-dimension scaling closes the gap
  (63–100%) are exactly the 4 with the most massive coordinates (SmolLM2 ×2, Qwen3-1.7B-20NG, Gemma 4 E2B-20NG); in the
  5 pairs without them it closes nothing. Post hoc reading: in these models the last-layer deficit comes from uneven
  scale across *many* coordinates, of which the massive ones are only the extreme; rescaling all of them recovers the
  layer, removing the extreme few does not. This needs its own test before it is a claim.
