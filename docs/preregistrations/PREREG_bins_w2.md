# Pre-registration W2: what do bins carry, and do their boundaries matter to the model? (2026-10-08)

Written and committed before any W2 measurement. Plan: PLAN_bins.md, workstream 2. Follows W1 (PREREG_bins_w1.md):
the stability–quality link is learned (untrained models reverse it), but on 20 Newsgroups nearly every layer is more
stable than its topic quality predicts (exploratory), so bins also hold non-topical structure.

Bins throughout: K = 512 k-means (seed 0) in the whiten-64 frame fit on the train split, test items assigned
(articles: 5,000 test documents, seed 1, as why_stability.py; tokens: as token_lens.py, fit/eval split of the
`tok_meta*.npz` files).

## W2-a. What the bins carry, and the 20NG format question (articles; CPU)
Attributes per document (computed from the text the model saw, first 64 whitespace words; tertile edges from train):
- topic: DBpedia l1 / l2; 20NG coarse / fine;
- length: word count of the full text, bins < 16, 16–63, >= 64;
- symbols: share of characters that are neither alphanumeric nor whitespace (tertiles);
- lines: newlines per word (tertiles);
- format = length x symbols x lines (27 classes).
(Checked on the 20NG texts before writing this: headers are stripped in this copy; quoting and e-mail addresses
appear in 2% and 7% of first-64-word texts, so they are not used.)
Per layer: NMI(bins, attribute) for each attribute; and layer-to-layer reorganization: merge rate (share of document
pairs in different bins at l that share a bin at l+1) and split rate (same at l, different at l+1), each minus the
same rate between the seed-0 and seed-1 partitions of layer l+1 (k-means noise).
- **W2-a1 (20NG format).** For each of the 7 20NG held-out pairs: partial Spearman over layers 1..L between the W1-b
  deviation (quality minus the noise-curve prediction at that layer's stability; layers outside the curve's range
  dropped) and NMI(bins, format), controlling for stability. Prediction: negative in >= 5 of 7 (layers that are more
  stable than their quality predicts carry more format).
- **W2-a2 (descriptive, no prediction).** Per-layer NMI curves for every attribute and the excess merge / split
  rates, all 11 held-out pairs; reported with the untrained Pythia-410M (step 0) as reference.
Kill for the format reading: W2-a1 in <= 3 of 7.

## W2-b. Do bin boundaries matter to the model beyond distance? (tokens; causal; RTX 3060, low-VRAM mode)
Question: two token states in the same bin vs in different bins, at the same distance: is the model's output less
changed by swapping one for the other when they share a bin? If not, bins are only a coarse picture of distance and
carry no mechanistic content of their own.
Models and caches: Gemma 4 E2B base (`tok_base_bos`, `tok_meta_bos.npz`) and Qwen3-1.7B (`tok_qwen3`,
`tok_meta_qwen3.npz`). Layers m (residual stream after decoder block m): round(0.25 L), round(0.5 L), round(0.75 L).
Pairs: eval tokens at text positions 8–63, recipient and donor at the same position in different texts. For each
layer, 1,000 same-bin pairs (random, seed 0); for each, a different-bin partner pair with the same recipient whose
distance in the whiten-64 frame is within ±2% of the same-bin pair's (nearest; pairs without one are dropped and
counted). Intervention: run the recipient text, replace its residual state at layer m and that position with the
donor's cached state, continue the forward pass. Outcome: KL(unpatched || patched) of the next-token distribution at
that position.
- **Sanity (noise floor).** Patching each recipient with its own cached state (cached on other hardware) gives a
  nonzero KL; its mean is reported, and W2-b is void for a layer whose noise floor exceeds the mean same-bin KL.
- **W2-b1 (primary).** Mean paired difference KL(different bin) − KL(same bin), bootstrap 95% CI (10,000
  resamples over pairs). Prediction: > 0 with the CI excluding 0 at >= 2 of 3 layers in each model.
- **W2-b2 (secondary).** The same with distance matched in raw residual coordinates instead of the whiten-64 frame.
Kill: W2-b1 fails in both models -> bin boundaries add nothing beyond distance for the model's next-token output.

## Compute and outputs
W2-a: gpu-host CPU, ~1–2 h, `diag/bins_w2a.py` -> `results/bins_w2a.json`. W2-b: local RTX 3060 (DIAG_LOW_VRAM=1,
DIAG_ATTN=eager), ~1–2 h, `diag/bins_w2b.py` -> `results/bins_w2b.json`.

## Results (2026-10-09; run on the local machine, not gpu-host (busy); `diag/bins_w2a.py`, `diag/bins_w2b.py`)
**W2-a1 (20NG format): supported, 7/7** (partial Spearman −0.31 to −0.68). Layers that are more stable than their
topic quality predicts carry more format information, controlling for stability. Caveat: NMI(bins, format) varies
little across layers (e.g. Gemma 4 E2B 20NG 0.25–0.27), so the correlation rests on small differences; its level is
comparable to topic (fine 0.23–0.41 there), i.e. format is a large share of what 20NG bins hold at every depth.
**W2-a2 (descriptive):** excess merge / split rates over k-means noise are small everywhere (merge excess < 0.001 of
pairs; split excess within ±0.16), largest in the first layers, including untrained Pythia-410M; topic NMI rises
over depth in trained models and falls after layer 0 untrained (Pythia step 0: 0.47 -> 0.37).
**W2-b (do boundaries matter beyond distance?): neither supported nor killed; family-dependent.** Noise floor
negligible everywhere (self-patch KL <= 2e-5; unpatched states match the cache, cosine 1.000). Different-bin minus
same-bin KL at matched whiten-64 distance (raw-matched in brackets):
| model, block | pairs with partner | same-bin KL | difference, 95% CI |
|---|---|---|---|
| Gemma 4 E2B, 9 | 710 | 1.04 | −0.16 [−0.36, +0.03] (raw +0.01) |
| Gemma 4 E2B, 18 | 602 | 2.17 | −0.32 [−0.55, −0.08] (raw −0.03) |
| Gemma 4 E2B, 26 | 688 | 6.64 | **+1.90 [+1.45, +2.36]** (raw +1.16) |
| Qwen3-1.7B, 7 | 785 | 5.83 | **+1.79 [+1.35, +2.23]** (raw +1.69) |
| Qwen3-1.7B, 14 | 797 | 6.72 | **+1.37 [+0.91, +1.82]** (raw +1.66) |
| Qwen3-1.7B, 21 | 748 | 8.95 | **+2.02 [+1.46, +2.58]** (raw +2.28) |
Qwen3: 3/3 (boundaries matter: crossing one changes the next-token output 20–30% more than staying inside, at the
same distance). Gemma: 1/3, only at block 26; no effect at block 9 and the reverse at block 18, which is already after
the turn (layer 14), so the switch lies between blocks 18 and 26, not at the turn.
Caveats: 15–40% of same-bin pairs had no different-bin partner within ±2% and were dropped (selection toward
recipients near boundaries); one position per text; next-token output only.
Reading (not pre-registered; for the next round): in Gemma, bin boundaries become functional only late (between
blocks 18 and 26). Correction 2026-10-09: an earlier summary to Steve said "only after the turn"; block 18 contradicts
that.
