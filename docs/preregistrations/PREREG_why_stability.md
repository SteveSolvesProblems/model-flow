# Pre-registration: why does the stability pick work? (2026-10-04, before any of these measures were computed)

The stability pick (PREREG_robustness.md, B1) chooses, per model and text collection, the layer whose K = 512 K-means
partition (64 whitened PCs) is most reproducible across seeds. It passed on 11 held-out model-task pairs. The page
offers an intuition, untested: "a layer whose grouping is stable has organized the collection into well-separated
structure, while the last layers reorganize toward the next word." This tests both halves.

Already seen before registering (from robust_b1.json, not a prediction): within each pair, across layers 1..L,
stability and topic search quality (p@10) have Spearman 0.63-0.98, and 0.69-0.96 after regressing both on depth and
depth squared.

Code: `diag/why_stability.py` (measures), `diag/score_why_stability.py` (scoring). Layers 1..L (layer 0, the input
embedding, excluded). "Partial" = Pearson correlation of rank residuals after regressing on depth and depth squared.

## Part A: separation (articles, the 11 held-out pairs)
Label-free measures per layer, in the selector's frame: silhouette of the K = 512 partition; kNN margin
(1 - d1/d10). Alternative explanations: effective dimension (participation ratio, raw coordinates) and the share of
variance in the top 64 PCs (how much the frame keeps).

- **A1 (separation tracks stability).** Partial correlation of silhouette with stability > 0 in at least 9 of 11 pairs.
- **A2 (separation carries the link to quality).** Controlling for silhouette (adding its ranks to the depth
  regression) at least halves the partial correlation of stability with quality in at least 7 of 11 pairs.
- **A3 (not just dimension).** Controlling for participation ratio or for top-64 share instead halves it in fewer
  pairs than silhouette does.
- Reported, no prediction: the same for kNN margin; how often each measure, used as a selector itself, matches the
  stability pick's shortfall.

Reading: A1 and A2 pass, A3 passes -> "stable = well separated" supported. A1 passes but A2 fails -> stability
measures something about quality beyond separation. A3 fails -> the explanation is dimensionality, not separation.

## Part B: late layers reorganize toward the next word (tokens; Gemma 4 E2B, Qwen3-1.7B)
Per layer, share of each token's 10 nearest neighbours (centered cosine; 3,000 eval positions) with the same next
word, current word, document and topic.

- **B1.** Next-word share at the last layer is higher than at every layer in the first three quarters of depth, in
  both models.
- **B2.** Topic share peaks before the last quarter of depth, and the last layer is below that peak, in both models.
- **B3.** Current-word share at the last layer is below its maximum over layers 1..L, in both models.

Reading: B1-B3 pass -> late layers trade topic and identity neighbourhoods for next-word neighbourhoods, consistent
with the second half of the intuition. This is token-level and indirect for the article-level selector; the link is
only that both change in the same layers.

## Result, Part B (2026-10-04; Part A paused pending compute sign-off)
Share of each token's 10 nearest neighbours with the same next word (chance about 0.01):
- Gemma 4 E2B: 0.11 at layer 1, peak 0.13 in the first three quarters, 0.09 at the last layer. **B1 fails.**
- Qwen3-1.7B: rises slowly from 0.11 to about 0.15–0.16 by layer 21, 0.151 at the last layer, not above the first-three-quarters
  maximum (0.154). **B1 fails.**
Topic share: Gemma peaks at layer 18 of 35 and ends lower (B2 holds); Qwen3 peaks at layer 24 of 28, inside the last
quarter (B2 fails). Current-word share is highest at layer 1 and lower at the end in both (B3 holds, but it is nearly
automatic: neighbours by token identity are strongest at the input).

Reading: the second half of the intuition, "late layers reorganize toward the next word", is **not supported** at the
level of token neighbourhoods. Next-word neighbours are well above chance from layer 1 (the current word already
predicts the next one), and the last layers do not raise them. Whatever makes late layers worse embeddings, it is not
visible as neighbourhoods sorted by next word. (Measure limits: 3,000 positions, exact next-word match only, centered
cosine.)

## Result, Part A (2026-10-04; scored by score_why_stability.py -> results/why_stability_score.json)
- **A1 holds (11 of 11).** Silhouette of the K = 512 partition tracks stability across layers after removing depth.
- **A2 fails (2 of 11; needed 7).** Controlling for silhouette leaves the stability-quality partial correlation
  mostly intact (typically 0.6-0.9 from 0.7-0.96). It halves only in Qwen3.5-4B and Gemma 4 E4B on DBpedia.
  kNN margin does the same (2 of 11).
- **A3:** participation ratio and top-64 share halve it in 0 of 11 pairs, so dimension does not explain it either.
  (A3's comparison is moot once A2 fails.)
- As selectors on their own (mean shortfall from the best layer, points): stability 0.70, kNN margin 1.80,
  silhouette 2.49, participation ratio 3.78, top-64 share 15.66.

Reading: layers with repeatable fine-grained groups are better separated (A1), but separation does not account for
why repeatability predicts embedding quality (A2), and neither does dimensionality. With Part B also failing, **neither
half of the stated intuition explains the stability pick.** It remains an empirical selector whose mechanism is
open; it also beats every simple geometric measure tried here as a selector. Limits: silhouette at K = 512 is small
(about 0.02-0.04), so it is a noisy measure of separation; 25-43 layers per pair.

## Correction (2026-10-05)
The Gemma token cache used in Part B was extracted without Gemma's `<bos>` (NOTE_geometry.md, 2026-10-05). The Gemma
half of B1-B3 is void until re-extraction; the Qwen3 half stands (B1 fails, B2 fails, B3 holds).

## Gemma half re-run with <bos> (2026-10-06)
tok_base_bos: next-word share 0.166 at the last layer vs 0.191 maximum in the first three quarters (B1 fails); topic
peaks at layer 18 of 35 and ends lower (B2 holds); B3 holds. Combined with Qwen3: B1 fails in both models.
