# Pre-registration: how quantization changes the attractor-lens map of a model

Written 2026-09-29, **before** any quantized features were extracted. Originally private notes, published 2026-10 unchanged except for machine names, paths and references to unpublished studies.

## Question
For one model, how does weight quantization change its layer-by-layer map: the per-layer, multi-scale,
label-free K-means partitions? Does that give a useful and new way to inspect a quant, beyond per-layer
activation error?

## Model and variants
- **Model:** Gemma 4 E2B-it. transformers can load GGUF for `gemma4` but not `qwen3_5`, which is why this model.
- **Reference:** `bartowski/google_gemma-4-E2B-it-GGUF` **bf16.gguf** (rev 81012ba353), loaded through the same
  transformers GGUF path as every quant, so loader effects cancel out.
- **Sanity check:** the reference vs `google/gemma-4-E2B-it` safetensors (rev 3e22461f65). Expected partition
  agreement ≈ ceiling. If not, the loading path is suspect and the study stops until that's explained.
- **Post-training ladder** (bartowski, same rev): Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, Q3_K_M, Q2_K, IQ2_M.
- **QAT:** `google/gemma-4-E2B-it-qat-q4_0-gguf` (rev 675cff42a7), `gemma-4-E2B_q4_0-it.gguf`.
- **Numerics:** weights are dequantized to bf16 and computed in bf16 on the RTX 3060. This measures weight
  quantization only. llama.cpp's int8 activation paths are out of scope, and that is a stated limitation.
- **Tokenizer:** from `google/gemma-4-E2B-it` for every variant, so the inputs are identical.
- **Data:** DBpedia with the same sampling as the main study (24k train / 6k test, seed 0), max 64 tokens,
  mean-pooled, every layer. No chat template, identical for all variants.

## Measurements (per layer ℓ, scale K ∈ {8, 32, 128, 512})
- **Frame and attractors:** fit on the **reference** train features at layer ℓ (whiten-64 frame; K-means seeds
  0 and 1).
- **Flip rate:** project the quant's test features through the *reference* frame, assign them to the
  *reference* attractors, and compare with the reference's own assignments:
  - flip(ℓ, K) = 1 − NMI(ref cells, quant cells), normalized by the reference's re-seed ceiling:
    normalized_flip = 1 − NMI / ceiling.
  - Also recorded: the raw share of documents that change cell.
- **Conventional baselines, same layers:** relative L2 error ‖q − r‖/‖r‖, mean cosine similarity, and linear CKA
  between reference and quant features.
- **Rule S** (stability at K=512) run on each variant, plus retrieval p@10 (DBpedia l2) at every layer.

## Hypotheses
- **Q1 (descriptive).** The normalized flip rate grows as bit width falls, and faster at fine scales (K=512)
  than at coarse ones (K=8). It is not uniform across depth: there are layers where it jumps.
  No kill criterion; this is the characterization.
- **Q2 (robustness of rule S).** For quants at 4 bits or more (Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, QAT Q4_0), rule S
  selects a layer within ±1 of the reference's S layer, and retrieval regret at the quant's S layer is ≤ 0.01.
  - **Kill:** fails for 2 or more of these 6 variants.
- **Q3 (the usefulness test).** Mixed precision. Start from Q4_K_M, restore **4** decoder layers to bf16, and choose
  them by:
  - **(a) lens:** the top-4 layers by increase in normalized flip rate at K=512 (flip(ℓ) − flip(ℓ−1)),
    measured on Q4_K_M;
  - **(b) activation error:** the top-4 layers by increase in relative L2 error, measured on Q4_K_M;
  - **(c) random:** 5 random draws of 4 layers (seeds 0–4), reported as the mean;
  - **(d) positional:** the first 4 decoder layers, and separately the last 4.

  **Outcome:** mean next-token KL(reference ‖ mixed) over 1,000 held-out DBpedia test documents, all positions,
  from the final logits.
  - **Supported** if (a)'s KL < (b)'s KL **and** (a)'s KL < the random mean.
  - **Partial** if (a) < random but ≥ (b): the lens identifies sensitive layers, but no better than activation
    error.
  - **Kill:** (a) ≥ random.
- **Q4 (QAT, exploratory, directional).** At the same bit width, QAT Q4_0 has a lower normalized flip rate
  than post-training Q4_0 at most layers (> 50% of layers at K=512).

## Prior art to position against
Per-layer sensitivity for mixed precision is well-trodden: HAWQ v1–v3, BRECQ, llama.cpp's importance matrix,
Unsloth dynamic quants. So are per-layer activation error and CKA. The only possibly new element is a
label-free, multi-scale, *partition-level* damage measure, and Q3 is what decides whether it adds anything.

## Amendment 1 (2026-09-29 ~23:00, before any quantized features were extracted)
Numerics changed from bf16 on the RTX 3060 to **fp16 on gpu-host's P100 #1**, running in parallel with the Gemma 4 E4B
extraction that occupies the 3060. The P100 has no bf16. Gemma 4 E2B already ran in fp16 with zero non-finite
outputs, and a non-finite guard aborts any variant that overflows. Every variant, reference included, uses the
same numerics. Nothing else changes.

## Order
Run after the layer-selection held-out verdict. Extraction takes about 11 runs on the 3060. Q3 then needs a
mixed-precision harness: load the reference, and swap dequantized Q4_K_M weights into every layer except
the restored ones.

## Amendment 2 (2026-09-29 ~23:50, before any quantized features were extracted or any KL computed)
- **Loader sanity passed at weight level:** all 540 tensors of the bf16 GGUF, loaded through our path and patched
  with `layer_scalar`, equal the official `google/gemma-4-E2B-it` safetensors bit-for-bit (max_abs 0). `lm_head` is tied
  to the embeddings. Identical weights with an identical config and numerics give identical features, so this
  replaces the partition-level sanity comparison. (The main study's `gemma4_e2b` cache is the *base* model and
  can't serve as that comparison.)
- **Order:** reference → Q4_K_M → Q3 → the rest of the ladder, run one at a time on the P100s. A GGUF load peaks
  at about 21 GB of RAM, so two can't run together on the 30 GB host.
- **Q3 documents:** DBpedia test positions 6000–6999, from the same seed-0 shuffle. These are disjoint from the
  6,000 test documents the lens uses to rank layers. KL is the token-weighted mean over non-pad positions, and the
  per-document mean is also reported.
- **Q3 mechanics:** a single GPU can't hold both models, so the reference fp16 logits are stored on disk (lossless,
  since the model computes in fp16) and each configuration is scored against them. Lens layer ℓ (hidden_states[ℓ])
  is attributed to decoder layer ℓ−1. Two extra configurations are reported descriptively: "none" (pure Q4_K_M) and
  "all" (every decoder layer restored, so only the quantized embeddings/PLE remain).

## Amendment 3 (2026-09-30 ~00:40, after seeing only layer 0's flip values, before any other layer or any KL)
The re-seed ceiling normalization doesn't fit this design. The quant is assigned to the *same* attractors as the
reference, so its agreement can exceed the reference's agreement with a different seed. That makes normalized_flip
negative (L00, Q4_K_M: −0.235 at K=512), and dividing by a per-layer ceiling reweights the layer-to-layer increases.
The pre-registered selector (a) stays **primary** and decides the Q3 verdict. A **secondary** selector (a′) is added:
top-4 increases in raw flip 1 − NMI(ref, quant) at K=512, scored with the same criteria and reported alongside.
Nothing else changes.

## Amendment 4 (2026-09-30 evening, before any per-document KL values were saved; Steve approved the ~1 h run)
The Q3 table has no confidence intervals. The mixed phase is re-run unchanged, with the same configurations, documents
and reference logits, and saves each document's KL sum and token count. Analysis:
- **Paired bootstrap over the 1,000 documents** (10,000 resamples, seed 0). Each resample recomputes the token-weighted
  mean KL of every configuration on the same documents.
- **95% CIs** for the differences lens − activation error, lens − random mean (the mean of the five random draws), and
  lens − first four; the same for the raw-flip selector (a′).
- The Q3 verdict stays as scored. The CIs say whether each ordering is distinguishable from noise **on these documents**.
  They cover document sampling only, not the choice of model or the random layer draws.
- **Determinism check:** the re-run's token-mean KL per configuration must match `gq__qkl.json` to 1e-4. If it doesn't,
  the discrepancy is reported and the re-run's own numbers are used for the CIs.

## Post-hoc check (2026-10-01): how much of a bin flip is a global shift? (`diag/qat_shift.py`, `results/qat_shift.json`)
Not pre-registered. Assigning each variant's features to the bf16 bins after re-centering them on the variant's own
train mean (and, separately, also rescaling to the reference's mean centered norm):

| | raw (as in the study) | re-centered | re-centered + rescaled |
|---|---|---|---|
| QAT Q4_0, mean over layers | 0.744 | 0.373 | 0.369 |
| QAT Q4_0, layer 1 | 0.996 | 0.126 | 0.101 |
| Q4_0, mean over layers | 0.391 | 0.287 | 0.287 |

- QAT's near-total flips at layers 1–2 are almost entirely a shared offset (5.7× the typical document spread at
  layer 1); the documents' relative arrangement is close to intact. Rescaling adds little.
- Re-centering halves QAT's flips and removes about a quarter of Q4_0's. QAT still moves the map more than Q4_0 after
  re-centering (0.37 vs 0.29), so the original conclusion survives, at about a third of the raw gap.
- Bin flips as defined conflate a global shift with a rearrangement. Reports should give both, or use re-centered
  flips as the primary measure in future studies.

## Correction (2026-10-06): every run above lacked `<bos>`; re-run with it
All quantized variants were tokenized with the google/gemma-4-E2B-it tokenizer, which adds no `<bos>`
(NOTE_geometry.md, 2026-10-05). Everything was re-run with the base tokenizer (same vocabulary, adds `<bos>`): same
GGUF files, texts, float16 dequant, config and `layer_scalar` from the official -it repo. Extraction and KL on the local
RTX 3060 (eager attention: SDPA overflows in float16 there; per-layer embedding table looked up on the CPU; checked
against a float32 CPU run, cosine 1.0000); lens and selection on gpu-host as before. Old results: `results/nobos/`.
Cross-check: the bf16 GGUF reproduces the safetensors E2B-it run exactly (S = 28, last-layer regret 10.6 points).
- **Q2 (stability pick survives quantization): killed** (1/9 pass; 6 of 7 at >= 4 bits fail). The reference now picks
  layer 28 (2.2 points short of its best, layer 17); quantized picks wander over layers 6-33, 0.7-4.0 points short.
  The old 9/9 at layer 6 came from the missing `<bos>`, which made layer 6 the clear best.
- **Q3 (lens-chosen mixed precision): fails against activation error.** Restoring 4 layers: lens 0.1007, activation
  error 0.0970 (lens - actl2 +0.0037, CI [+0.0020, +0.0053]), random 0.1032 (lens better, CI excludes 0), first four
  0.0994 (tie). Raw-flip variant: ties activation error, beats random and first four. KL itself is ~3x smaller with
  `<bos>` (none restored: 0.112). B4b (per-block flip jump vs KL reduction): rho 0.04, not supported.
- **Q4 / QAT:** QAT still moves the map more than Q4_0 (raw 42% vs 13%; re-centered 13% vs 10%).
