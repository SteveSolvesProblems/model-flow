# Pre-registration: follow-ups to the null results (2026-10-07)

Written and committed before any of the measurements below were computed. Context: with `<bos>` restored, the
stability pick no longer survives chat tuning or quantization (PREREG_flow.md F3 correction,
PREREG_quantization_lens.md correction), and why it works was not explained (PREREG_why_stability.md: separation tracks
stability 11/11 but does not explain its link to quality, 2/11). Three follow-ups, each with predictions and a kill
criterion. Steve approved all three (2026-10-07). Compute: P1 and P2 on gpu-host CPU (~1–2 h each); P3 on the local
RTX 3060 in low-VRAM mode (~1 h).

Already known (not predictions): the Gemma token turn is at layer 14 with and without `<bos>`; article-level
selection and F3 numbers with and without `<bos>` (PREREG_flow.md).

## P1. `<bos>` on vs off as a perturbation of the coordinate system (Gemma 4 E2B)
Data: token caches `tok_base` (no `<bos>`) and `tok_base_bos`, same 1,500 texts. Rows are aligned by document and
text token (no-`<bos>` position p = `<bos>` position p+1); an alignment is valid only if the token ids agree, and the
analysis stops if fewer than 99% of the shorter set's rows align. Article caches: `gemma4_e2b_it` without
(`cache_nobos/`) and with `<bos>`, same documents (5,000 test documents, seed 1, as why_stability.py).
Per layer l, between the two conditions:
- `cos_l`: mean cosine of the same token's states (raw); `ccos_l`: the same after centering each condition at l.
- `knn_l`: kNN10 overlap of the same tokens across conditions (each condition's own neighbours, cosine, centered).
Predictions:
- **P1-a (late layers move more).** Mean `knn_l` over layers 1–13 > mean over layers 15–35, for tokens and for
  articles. (Basis: without `<bos>` the chat model's best article layer was 6, with it 17; late layers looked broken.)
- **P1-b (the effect fades along the text).** For tokens, mean `ccos_l` over layers 2–35 is higher for positions
  >= 16 than for positions 1–4 (no-`<bos>` numbering), at >= 30 of the 34 layers. (Basis: without `<bos>`, the first
  text token becomes the first position every token can attend to; tokens far from it should be less affected.)
- **P1-c (exploratory, no prediction).** The set of huge coordinates (mean |x| > 20x the median coordinate at some
  layer) in each condition; their Jaccard overlap and size per condition are reported.
Kill: P1-a fails for both tokens and articles -> "the missing `<bos>` broke the late layers" is not the description.

## P2. Stability = settledness?
Data: the 11 held-out model–collection pairs (robustness.py `HELD`), test documents and frame exactly as
why_stability.py part A (5,000 test documents, seed 1; whiten-64 frame fit on 20,000 train documents, seed 0).
Per layer l: stability (from `<pair>__select.json`, `stability_K512`), quality (`p@10` on the pair's topical label),
and settledness `set_l` = kNN10 overlap between layer l and layer l+1 (each in its own frame; defined for l < L).
Spearman correlations are over layers 1..L−1 within a pair.
Predictions:
- **P2-a (stability is mostly settledness).** rho(stability, settledness) > |rho(stability, quality)| in >= 8 of 11.
- **P2-b (the pick fails where the model keeps reorganizing).** Across the 11 pairs, the stability pick's regret
  rises with post-oracle churn, `churn = mean(1 − set_l)` over layers oracle..L−1: Spearman rho > 0, one-sided
  permutation p < 0.05 (10,000 permutations).
- **P2-c (settledness as a selector).** Picking the most-settled layer (argmax `set_l`, layers 1..L−1) gives regret
  within 1 point of the stability pick in >= 8 of 11.
Kill: P2-a fails -> stability is not mainly a settledness measure; record it and stop this line.

## P3. Output sensitivity instead of bin flips (quantization; a reproduction of known practice)
Context: activation error picked better layers to restore than bin flips (Q3 correction). Known approaches score
layers by the output's sensitivity (HAWQ: Hessian; llama.cpp: importance matrix). Here a cheap version is checked
against the measured per-block benefit of restoring a block (`robust_b4b.json`, Q4_K_M, with `<bos>`).
Method: bf16 GGUF reference, the same 1,000 KL documents as qkl.py, first 200 of them. For each decoder block i,
add noise to the block's output: z ~ N(0, I) per token, scaled to ||z|| = eps x ||h_i|| with eps = 0.02, 3 draws
(seeds 0–2). `sens_i` = mean KL(reference || perturbed), final logits, all non-pad positions. Primary score
`prod_i = sens_i x err_i`, where `err_i` is Q4_K_M's activation-error increment at block i (the score_b4bc.py
"actjump" quantity); secondary score `sens_i` alone.
Predictions:
- **P3-a.** Spearman(prod_i, single-block KL reduction) > 0.40 and > the activation-error increment's own rho (0.24).
- **P3-b.** Restoring the top 4 blocks by `prod_i` (one extra qkl.py configuration) gives KL below activation
  error's 0.0970; reported with a paired bootstrap CI as score_qkl_ci.py. Supported only if the CI excludes 0.
Kill: P3-a fails -> the cheap probe does not capture what restoration buys; activation error stays the practical
recipe here.

## Not pre-registered
- An audit of special tokens over every cache (`diag/audit_special_tokens.py`): a correctness check, not a hypothesis.
- The write-up of these nulls in the findings page ("What did not hold").

## Results (2026-10-07; pre-registration committed 24da2c4 before any of these were computed)
Audit (`results/audit_special_tokens.json`): 187 text caches; the only ones without the model's leading special tokens
are the 5 superseded Gemma token caches (tok_base, tok_it, tok_it2bit, tok_nople0, tok_nople18), all re-extracted
with `<bos>`. token_extract.py now records its tokenizer. Claim A1.

**P1 (`<bos>` as a perturbation; `diag/bos_perturb.py`, `results/bos_perturb.json`): supported.** Tokens align
86,817/86,817, ids identical.
- P1-a: neighbours kept across conditions, layers 1–13 vs 15–35: tokens 0.44 vs 0.21; articles 0.45 vs 0.23.
  Lowest around layers 12–15 (tokens 0.16–0.20); centered cosine there ~0.24. Without `<bos>`, most of the
  coordinate system after the first third of the depth is different, not just the last layers.
- P1-b: positions 1–4 move more than positions >= 16 at 34/34 layers (e.g. layer 6: 0.12 vs 0.56 centered cosine).
- P1-c (exploratory): huge coordinates mostly shared (Jaccard 0.69 tokens, 0.76 articles; 25 vs 19 and 48 vs 42).
- Caveat: article states average over all tokens, so with `<bos>` they include the `<bos>` state (layer 0 kNN 0.89).

**P2 (stability = settledness; `diag/settledness.py`, `results/settledness.json`): killed.**
- P2-a: settledness beats quality as stability's correlate in 3/11 (8 needed). Within pairs, stability tracks
  quality across layers (Spearman 0.71–0.97, median 0.93) far more than settledness (0.38–0.94).
- P2-b: regret vs churn after the best layer: rho 0.25, p = 0.24 (not supported).
- P2-c: the most-settled layer is within 1 point of the stability pick in 5/11 (not supported).
- Reading: stability is not a settledness proxy; it co-varies with quality across layers, and why remains open.

**P3 (output sensitivity; `diag/sens_probe.py`, `results/sens_probe.json`): failed.**
- Clean model vs stored reference logits: KL 0.0 (sanity).
- P3-a: Spearman(sens x err, restoration benefit) = 0.02 (needed > 0.40 and > 0.24). Secondary score, sensitivity
  alone: 0.47 (exploratory; not a pre-registered prediction).
- P3-b: top 4 by sens x err (blocks 7, 9, 10, 11): KL 0.1000 vs activation error 0.0970; difference +0.0030,
  95% CI [+0.0013, +0.0046] (worse). Activation error stays the practical recipe here.
