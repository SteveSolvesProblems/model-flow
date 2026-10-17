# Pre-registration: do Gemma's learned layer scalars predict its coordinate switches? (held-out model: Gemma 4 E4B)

Written 2026-10-04, before any geometry of Gemma 4 E4B was computed or viewed. Only E4B's config and its per-layer
`layer_scalar` values were read (below). Approved by Steve ("let's do those follow ups, make sure our methodology works,
reproduces").

## Background (from Gemma 4 E2B, research/NOTE_geometry.md)
E2B's sharpest coordinate turns are made by decoder layers 13 and 14 (centered cosine across the layer 0.081, 0.043),
which carry its two smallest scalars after layer 0 (0.074, 0.049). Key/value sharing starts at layer 15, where the
automatic segmentation starts E2B's final article block. Rank correlation between a layer's scalar and the centered
cosine across that layer: 0.84 (articles) / 0.89 (tokens). Setting the scalars of 13–14 to 1 nearly stopped layers
14–15 from turning (causal).

## E4B facts read before writing (config + weights only)
42 decoder layers; full attention at 5, 11, 17, 23, 29, 35, 41; `num_kv_shared_layers` = 18, so layers 24–41 reuse
earlier keys/values. Scalars (layer 0–41): 0.056, 0.146, 0.781, 0.641, 0.486, 0.57, 0.484, 0.609, 0.223, 0.412, 0.59,
0.531, 0.762, 0.746, 0.711, 0.57, 0.676, 0.473, 0.566, 0.527, 0.441, 0.461, 0.142, 0.069, 0.447, 0.703, 0.715, 0.695,
0.613, 0.432, 0.703, 0.812, 0.848, 0.832, 0.84, 0.828, 0.84, 0.875, 0.871, 0.828, 0.785, 0.432.
Smallest after layer 0: 23 (0.069), 22 (0.142), 1 (0.146), 8 (0.223).

## Data and measure
Existing DBpedia article caches `gemma4_e4b` (mean-pooled, ≤ 64 tokens; 43 cached states). `geometry_probe.py --name
gemma4_e4b --split test --n 3000` (seed 0, same procedure as E2B). "Turn at decoder layer k" = 1 − centered cosine
between cached states k and k+1. Segmentation: `score_heads.segment` on the self-match matrix over layer outputs.

## Predictions
- **S1 (location).** Among decoder layers 2–40, the two sharpest turns (lowest centered cosine across the layer) are
  both at layers in {22, 23, 24}. (Chance if turns were placed at random: about 0.4%.)
- **S2 (relation).** Spearman(scalar at k, centered cosine across layer k) ≥ 0.5 over layers 1–40.
- **S3 (block boundary).** The automatic segmentation of the article self-match matrix has a block starting at a
  decoder layer in {23, 24, 25} (key/value sharing starts at 24).
- **S4 (causal, secondary; run only if memory allows the bf16 model on CPU).** Setting the scalars of layers 22–23 to 1
  raises the centered cosine across layers 23 and 24 by ≥ 0.2 each (smaller turns) and drops top-1 agreement with
  the unmodified model below 50%, on 40 texts.

## Failure handling
Each prediction is reported as supported or not, with the measured numbers. If S1 fails, the claim on the findings
page is narrowed to E2B only. No changes to the measure, the sample or the layer range after seeing E4B geometry.

## Result (2026-10-04 08:25, scored by `diag/score_prereg_e4b.py` → `diag/results/prereg_scalar_e4b.json`)
- **S1 supported.** Two sharpest turns among layers 2–40: layer 23 (centered cosine across it 0.048) and 22 (0.190).
  Next: 24 (0.506), 8 (0.507), 2 (0.565).
- **S2 supported.** Spearman(scalar, cosine across the layer), layers 1–40: 0.896.
- **S3 supported.** Segmentation blocks (decoder layers): 0 | 1–7 | 8–20 | 21–22 | 23–41; a block starts at 23.
- Not predicted, noted: layer 8 (fourth-smallest scalar, 0.223) is the fourth-sharpest turn and also starts a block.
- S4 (causal) not run yet.
- Scope: all three use mean-pooled DBpedia articles (≤64 tokens) and the centered cosine across one layer; they say
  where the coordinate system turns, not what the turn does to the model's predictions.

## Wording correction (2026-10-04, after scoring; does not change the predictions or the result)
"Learned layer scalar" above is imprecise. In HF transformers, `layer_scalar` is a non-trainable buffer loaded from the
checkpoint (`Gemma4TextDecoderLayer`). The Gemma 4 technical report (arXiv 2607.02770, §2.5) says it was introduced
"to enable stable inference in fp16 ... to bound the activation ranges to fit fp16"; Google's MaxText code declares it
as a parameter initialised to 1, so it may well have been trained. Call it "the per-layer scalar", neither "learned"
nor "not learned". It multiplies the whole residual
(residual plus update) after each layer, unlike LayerScale/ReZero, which scale only the update.
