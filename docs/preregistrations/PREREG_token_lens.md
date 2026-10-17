# Pre-registration: the attractor lens on individual tokens (pilot)

Written 2026-09-30, **before** any token-level features were extracted. Originally private notes, published 2026-10 unchanged except for machine names, paths and references to unpublished studies.
Sign-off: Steve approved the ~1 h pilot on 2026-09-30.

## Question
When the lens bins individual tokens instead of whole articles, what do the bins track at each layer: the word itself, its
context, or the next word? Does compression hurt some kinds of tokens more than others?

## Models, data, numerics
- Gemma 4 E2B **base** (`google/gemma-4-E2B`, safetensors), **tuned** (bartowski `-it` bf16 GGUF, patched as in the quant
  study) and **tuned 2-bit** (bartowski IQ2_M GGUF). bf16 on the local RTX 3060. Numerics differ from the article-level runs
  (fp16 on P100); this pilot is self-contained.
- Texts: DBpedia with the main study's sampling. **Fit set:** the first 3,000 train articles. **Evaluation set:** the viewer's
  1,500 test articles (seed-0 sample of the 6,000). First 64 tokens per article; the `<bos>` token (position 0, known outlier
  norms) and padding are dropped.
- **Word identity** follows Engram's tokenizer compression (Cheng et al. 2026, §2.2): NFKC → NFD → strip combining accents →
  lowercase → strip whitespace. Tokens that are empty after this (whitespace) keep one shared "whitespace" id.
- **Frequency cap:** at most 30 occurrences per canonical word in the fit set (random, seed 0), so common words don't
  dominate the bins. Nothing is dropped from the evaluation set; results are also reported by frequency band: *top-100
  words* (by fit-set count, before capping), *101–1,000*, and *rarer*.

## Method
Per layer (hidden_states 0–35), per model: whiten-64 frame fit on the fit set (≤ 12k rows), k-means K ∈ {32, 512, 2048},
seeds 0 and 1, fit on the fit set; evaluation tokens are assigned. The 2-bit model's tokens are assigned to the *tuned*
model's frame and bins.

## Measurements (evaluation tokens)
- **Identity:** NMI(bins, canonical word id).
- **Next word:** NMI(bins, canonical id of the following token); tokens with no following token are excluded.
- **Context:** for canonical words with ≥ 20 evaluation occurrences spread over ≥ 3 top-level DBpedia topics:
  NMI(bins, article topic) within each word, averaged with occurrence weights. The null permutes topics within each word
  (5 permutations, averaged). Context effect = observed − null, with a bootstrap 95% CI over words.
- **Stability:** cross-seed NMI (seed 0 vs 1).
- **Compression:** share of evaluation tokens whose K=512 bin differs between tuned and 2-bit, by frequency band and by
  token type: digit-containing, punctuation-only, word-start (leading space), word-continuation (no leading space).

## Hypotheses (K = 512 unless stated)
- **T1 identity fades.** Identity NMI peaks in layers 0–3 and falls by ≥ 30% (relative) from its peak to the last layer, in
  all three models.
- **T2 next word rises.** Next-word NMI peaks in the last quarter of layers (≥ 27 of 35).
- **T3 crossing.** Identity NMI > next-word NMI at layer 0, and next-word NMI > identity NMI at the last layer (base model;
  reported for all).
- **T4 context in the middle.** The context effect peaks between layers 9 and 27 (the middle half), and its CI there
  excludes 0.
- **T5 compression by token type (exploratory, stated direction).** The 2-bit flip rate is higher for *rarer* words than
  for *top-100* words at a majority of layers ≥ 5.
- **T6 stability profile (descriptive).** The token-level stability profile is reported; there is no criterion.

## Not in this pilot
Engram-style n-gram completion (do the final tokens of multi-token names settle into their own bins early?), SAE comparisons, and
other models.

## Amendment 1 (2026-09-30 ~10:55, before any token features were extracted)
The run moves from the local RTX 3060 (bf16) to gpu-host's P100s (**fp16**). Downloading the models locally runs at 1–5 MB/s,
while gpu-host already has them cached. The non-finite guard aborts a run that overflows. All three models use the same
numerics. Nothing else changes.

## Amendment 2 (2026-09-30 ~11:17, after only layer 0's results were seen)
K = 2048 is dropped (K ∈ {32, 512}) to keep the analysis near the approved ~1 h budget; it was running at ~2.3 h.
Every hypothesis is stated at K = 512, so none changes. Layer 0 is recomputed identically.

## Part 2: a model without per-layer embeddings (written 2026-09-30 ~12:25, before any Qwen token features were extracted)
**Why:** in Gemma 4 E2B, word identity did not fade (T1 failed for base, T3 failed for all). Gemma's E-series feeds a
per-layer token embedding into every layer (`hidden_size_per_layer_input: 256`), which could keep re-injecting identity.
**Test:** the same pilot on **Qwen3-1.7B** (base; 28 layers, no per-layer embeddings), same articles, fit/eval split, caps,
K ∈ {32, 512} and measurements. Its own tokenizer defines the tokens and canonical words (Engram normalization as before).
fp32 on a P100 (as in the main study's Qwen3-1.7B cache). Approved by Steve 2026-09-30.
**Predictions (K = 512; 28 layers, so the last quarter is ≥ 21 and the middle half is 7–21):**
- **Q-T1:** identity NMI peaks in layers 0–3 and falls ≥ 30% from its peak by the last layer.
- **Q-T2:** next-word NMI peaks at layer ≥ 21.
- **Q-T3:** next-word NMI > identity NMI at the last layer.
- **Q-T4:** the context effect peaks in layers 7–21 with a CI excluding 0.
- **Contrast (the per-layer-embedding explanation):** Qwen's relative identity drop (peak → last) is larger than Gemma base's (27%),
  **and** Qwen has no identity rebound of ≥ 0.1 NMI after its minimum (Gemma base: 0.45 → 0.67).
  - *Supported:* both parts hold.
  - *Not supported:* Qwen also shows a rebound ≥ 0.1, or a drop ≤ 27%.
  - One model pair can only support or undercut this; it can't establish it.
- **Position 0** is dropped for Qwen too. Qwen adds no `<bos>`, so its position 0 is a real word, but first positions carry
  attention-sink outlier norms regardless (Sun et al. 2024). Dropping it keeps the two models comparable.
