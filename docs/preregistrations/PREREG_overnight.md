# Pre-registration: overnight studies (training dynamics, task types, model internals)

Written 2026-09-30 ~21:45, **before** any data for these studies was extracted or analyzed. Steve approved the
direction and the ~12 h compute budget on 2026-09-30. Naming: **map stability** is what earlier documents call
rule S's score. Its selector (argmax over layers of cross-seed NMI of K = 512 k-means partitions in the whiten-64
frame; ties go deeper) is unchanged and is called **S** below.

Common conventions:
- Lens layer ℓ = `hidden_states[ℓ]`.
- Frame: whiten-64, fit on train.
- k-means: faiss, K = 512, niter 50, nredo 1, seeds 0 and 1, fit on train, assigned on test.
- Retrieval: cosine p@10 of test queries against the train database (centered, L2-normalized).
- Regret = best layer − chosen layer.

## E1. Map stability across pretraining (Pythia)
- **Models:** EleutherAI/pythia-410m and pythia-1.4b [Biderman et al. 2023] at checkpoints step0, 512, 1000, 2000,
  4000, 8000, 16000, 32000, 64000 and 143000 (final). fp32.
- **Data:** DBpedia with the main study's sampling (24k train / 6k test, seed 0), 64 tokens, mean pooling.
- **Measurements:** per checkpoint and layer, map stability and p@10 (l2 primary, l3 secondary). Seed-0 test cells
  are saved for P2.
- **P1 (S works during training).** S's regret ≤ the last layer's at ≥ 7 of the 9 checkpoints from step 512 onward,
  for each model.
- **P2 (layers settle bottom-up)** [cf. Raghu et al. 2017].
  - Settling ratio of layer ℓ at checkpoint t: NMI(cells_t,ℓ; cells_final,ℓ) ÷ the final checkpoint's re-seed NMI
    at ℓ.
  - Settle step: the first checkpoint with ratio ≥ 0.9.
  - Supported if Spearman(layer index, settle step) > 0 and its bootstrap CI over layers excludes 0, for both models.
    Layers that never settle get the final step.
- **P3 (a label-free training monitor).** Across checkpoints from step 512 onward, Spearman(max map stability,
  oracle p@10) ≥ 0.7 for both models.
- **Descriptive:** where S and the oracle layer sit at each checkpoint (does the best embedding layer move?).

## E2. Other task types and last-token pooling (phase C, slimmed)
- **Models:** Qwen3-1.7B, Qwen3.5-2B and SmolLM2-1.7B (fp32), Gemma 4 E2B (fp16). Qwen3.5-4B Banking77 caches
  already exist (fp16) and are included.
- **Data:**
  - Banking77 (`mteb/banking77`, 77 intents);
  - STS-B (`sentence-transformers/stsb`; k-means fit on train sentences, scored on test sentences).

  64 tokens, two poolings: **mean** and **last token** (last non-pad position).
- **Outcomes:**
  - Banking77: intent p@10.
  - STS-B: Spearman ρ between the cosine of the two sentences of each test pair (centered by the train mean) and the
    gold score [Cer et al. 2017].
  - Regret is measured against the best layer for that model and pooling.
- **S variants.** With last-token pooling, layer 0 is a lookup of one token and gives degenerate stability (seen in
  an earlier study). So:
  - for last-token pooling, S excludes layer 0;
  - for mean pooling, S is the frozen rule;
  - the other variant is reported too.
- **C1.** Banking77, mean pooling: S's regret < the last layer's in ≥ 4 of 5 models.
- **C2.** Banking77, last-token pooling (stated expectation: S fails): S's regret ≥ the last layer's in ≥ 3 of 5 models.
- **C3.** STS-B, mean pooling: S's regret < the last layer's in ≥ 3 of 4 models.
- **C4.** STS-B, last-token pooling: descriptive.

## E3. Cross-model layer correspondence and stage boundaries (existing DBpedia caches)
- **Data:** 7 models on the same DBpedia test documents (Qwen3-1.7B, Qwen3.5-0.8B / 2B / 4B, Gemma 4 E2B / E4B,
  SmolLM2-1.7B). Seed-0 K = 512 test cells for every layer.
- **Correspondence:** for every model pair (21 pairs), the matrix NMI(cells_A,ℓ; cells_B,m). Linear CKA
  [Kornblith et al. 2019] on 3,000 test documents is computed as the comparison.
- **X1 (relative depth).** For each layer of A, take the best-matching layer of B. Spearman(relative depth in A,
  relative depth of the match) ≥ 0.9 in ≥ 18 of 21 pairs.
- **X2 (partition vs CKA).** Spearman over all entries between the NMI and CKA matrices ≥ 0.7 in ≥ 18 of 21 pairs.
  Where they disagree, that is the interesting part (descriptive).
- **Stage boundaries:**
  - adjacent-layer agreement a(ℓ) = NMI(cells_ℓ−1, cells_ℓ) ÷ sqrt(ceiling_ℓ−1 · ceiling_ℓ), where the ceiling is
    the same layer's seed-0 vs seed-1 NMI;
  - boundaries are the local minima of a(ℓ).
- **X3 (KV-sharing boundary).** a(ℓ) has a local minimum within ±1 of the last layer that computes its own KV cache:
  lens 15 in Gemma 4 E2B (35 layers, 20 shared) and lens 24 in Gemma 4 E4B (42 layers, 18 shared; from each model's
  config). Supported only if it holds in both.

## E4. Model selection by map stability (exploratory; no prediction)
Within each dataset (DBpedia, 20 Newsgroups; 7 models each): Spearman between each model's maximum map stability and
its oracle p@10, with a permutation p-value. Exploratory, because stability values may not be comparable across
models.

## E5. Per-document routing stability as label-free query difficulty (11 held-out pairs, layer S)
- **Routing stability of a test document:** the Jaccard similarity of its test co-members (the other test documents
  in its cell) under seed 0 vs seed 1. It is 1 if both sets are empty and 0 if exactly one is.
- **Outcome:** the document's own p@10 (from the phase A per-document hits). **Failure** = p@10 ≤ 0.5.
- **Baselines** (from QPP and IVF practice): distance to the nearest seed-0 centroid, and the nearest/second-nearest
  centroid distance ratio.
- **Q1.** Spearman(routing stability, p@10) > 0 with a bootstrap CI excluding 0 in ≥ 9 of 11 pairs.
- **Q2.** AUROC of routing stability for predicting failure exceeds both baselines' AUROC in ≥ 7 of 11 pairs.

## E6. Polysemy from token bins (existing token caches: Gemma 4 E2B base, Qwen3-1.7B)
- **Words:** canonical words with ≥ 20 evaluation occurrences, alphabetic and ≥ 3 characters.
- **Label-free score:** the entropy of a word's occurrences over K = 512 bins (seed 0), divided by log(number of
  occurrences). It is taken at the layer where the pilot's context effect peaked: lens 18 for Gemma base, 23 for Qwen.
- **Reference:** WordNet synset count [Miller 1995]. Words without synsets are excluded.
- **W1.** The partial Spearman correlation of score and synset count, controlling for log frequency, is > 0 with a
  bootstrap CI excluding 0, in both models.
- **Descriptive:** the same at every layer; the top-30 words by score.

## Compute and order
- GPU lane 0: E1. GPU lane 1: E2.
- CPU: E3–E6, run alongside.
- New caches and model downloads go to the external drive (`data`). Existing caches stay where
  they are.

## References
- Biderman et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. ICML.
- Cer et al. (2017). SemEval-2017 Task 1: Semantic Textual Similarity. SemEval.
- Kornblith et al. (2019). Similarity of Neural Network Representations Revisited. ICML.
- Miller (1995). WordNet: a lexical database for English. CACM.
- Raghu, Gilmer, Yosinski & Sohl-Dickstein (2017). SVCCA. NeurIPS.

## Amendment 1 (2026-09-30 ~21:50, before any E3–E6 analysis was run; E1/E2 extraction had just started)
These controls come from the prior-art survey (research/research_notes/Layer flow maps prior art/). The primary
criteria are unchanged; everything below is added as secondary.
- **Chance-corrected agreement** [Vinh, Epps & Bailey 2010]. NMI between partitions has a floor that grows with K/N.
  Wherever NMI is used (E1–E3, E5), the **permutation-adjusted NMI** is also reported: NMI minus the mean NMI after
  shuffling one partition's labels (3 shuffles). For E3, the correspondence is also computed with it, as a secondary
  check of X1–X3.
- **E5 baselines:** add document **hubness**, the k-occurrence N₁₀ (how many test queries have the document among
  their 10 nearest training neighbours; for test documents, computed among test documents) [Radovanović et al. 2010].
  Q2 now requires beating all three baselines; the original two-baseline version is also reported.
- **E6 controls** [Garí Soler & Apidianaki 2021; Pimentel et al. 2020]:
  - every word is scored on exactly 20 randomly chosen occurrences (seed 0), so the occurrence count is fixed;
  - the partial correlation also controls for the number of distinct article topics among those 20 occurrences, so
    a word used across many topics is not mistaken for a polysemous one.
- **E3 positioning:** relative-depth correspondence across LLMs has already been reported with nearest-neighbour
  measures [Wolfram & Schein 2025]. X1 is therefore a replication with partitions. The added questions are X2
  (partitions vs CKA) and X3 (the KV boundary).
- **E1 positioning:** bottom-up settling [Raghu et al. 2017; Chen et al. 2023] and layer convergence across Pythia
  checkpoints [Diehl Martinez et al. 2024] are known. P2 is a replication. P1 and P3 are the added questions.

## Results (2026-10-01; queues finished 04:03 gpu-host time, no failures)
Verdicts are from the scorers' JSON in `diag/results/`. E2 is reported separately below, with a post-hoc diagnosis.

### E1. Pythia (`pythia_score.json`)
| step | 410m S / oracle | 410m regret S / last | 1.4b S / oracle | 1.4b regret S / last |
|---|---|---|---|---|
| 0 | 20 / 0 | 0.292 / 0.295 | 23 / 0 | 0.286 / 0.285 |
| 512 | 9 / 0 | 0.168 / 0.207 | 7 / 0 | 0.137 / 0.171 |
| 1000 | 19 / 23 | 0.014 / 0.011 | 19 / 21 | 0.004 / 0.012 |
| 2000 | 19 / 22 | 0.018 / 0.008 | 22 / 21 | 0.001 / 0.010 |
| 4000 | 24 / 17 | 0.009 / 0.009 | 24 / 22 | 0.011 / 0.011 |
| 8000 | 24 / 17 | 0.004 / 0.004 | 20 / 17 | 0.006 / 0.007 |
| 16000 | 24 / 17 | 0.002 / 0.002 | 20 / 17 | 0.008 / 0.005 |
| 32000 | 20 / 17 | 0.015 / 0.004 | 24 / 17 | 0.003 / 0.003 |
| 64000 | 21 / 17 | 0.016 / 0.002 | 24 / 17 | 0.018 / 0.018 |
| 143000 | 19 / 24 | 0.032 / 0.000 | 24 / 17 | 0.016 / 0.016 |

- **P1 not supported.** S ≤ last at 4/9 checkpoints for 410m (fails) and 8/9 for 1.4b (passes). For 410m, S is up
  to 3.2 points behind the last layer at late checkpoints; for both models S often *is* the last layer (L24).
- **P2 not supported** (a replication target). Settle-step Spearman with layer index: 410m 0.27 [−0.17, 0.66],
  1.4b 0.46 [0.00, 0.82]. Descriptively, the first and last layers settle by step 2000 and the middle layers by
  4000: ends first, middle last, rather than strictly bottom-up. The checkpoint grid is coarse, so settle steps are
  heavily tied. The chance-corrected version gives −0.08 and −0.21 (both CIs include 0).
- **P3 supported.** Spearman(max map stability, oracle p@10) across checkpoints from step 512: 0.85 (410m), 0.93
  (1.4b). Map stability rises with retrieval quality during training, without labels.
- **Descriptive.** At steps 0 and 512 the best layer is the input embedding (layer 0). From step 1000 the best layer
  is near the top (21–23), and from step 4000–8000 on it moves down to layer 17 and stays there (410m's final
  checkpoint is an exception: L24). Map stability does not follow that move; it keeps picking 19–24.

### E3. Correspondence and stage boundaries (`correspondence.json`)
- **X1 not supported:** 13 of 21 pairs reach Spearman ≥ 0.9 (range 0.59–1.00; median 0.95). Relative-depth
  correspondence mostly holds, but not at the pre-registered strength in enough pairs.
- **X2 not supported:** 5 of 21 pairs have NMI-vs-CKA Spearman ≥ 0.7 (range −0.16 to 0.91). Partition agreement and
  linear CKA rank layer pairs differently in most model pairs. Which one tracks function is open.
- **X3 passes nominally, but the test was weak.** Both Gemma models have a local minimum of adjacent-layer
  agreement within ±1 of the KV boundary (E2B: 16 for 15; E4B: 25 for 24; same with chance-corrected NMI). But each
  model has 6–7 minima, which put 50–61% of layers within ±1 of some minimum; the chance of passing in both is about
  0.3. E4B's dip at 25 is the third deepest of 7; E2B's at 16 is among the shallowest (depth 0.010 vs 0.074 max).
  Not evidence for a KV-boundary stage.

### E4. Model selection (exploratory; `model_select.json`)
Across 7 models, Spearman(max map stability, oracle p@10) = 0.86 on DBpedia (permutation p = 0.012) and 0.82 on
20 Newsgroups (p = 0.017). Together with P3, map stability ranks models and checkpoints by retrieval quality.
Exploratory; 7 models.

### E5. Routing stability (`routing.json`)
- **Q1 supported:** 11/11 pairs have Spearman(routing stability, p@10) > 0 with CI excluding 0, but the effects are
  small (ρ = 0.07–0.21).
- **Q2 not supported:** routing stability beats all three baselines in 1/11 pairs. It beats the two original
  baselines (centroid distance, margin) in 7/11 (all 20NG pairs), but hubness [Radovanović et al. 2010], added in
  Amendment 1, is a better failure predictor in most pairs.

### E6. Polysemy (`polysemy.json`)
- **W1 not supported.** Partial Spearman (log frequency controlled): Gemma 4 E2B base 0.050 [−0.068, 0.165], n = 307;
  Qwen3-1.7B 0.119 [0.013, 0.235], n = 303. With the topic control: 0.038 [−0.076, 0.153] and 0.106 [0.005, 0.219].
  Holds in Qwen only.

### Amendment-1 secondary: chance-corrected selector (`adjusted_select.json`)
On the 11 held-out pairs, choosing by permutation-adjusted NMI gives mean regret 0.0130, against 0.0046 for the raw
rule (same layer in 5/11). The NMI floor varies little across layers (0.43–0.57 within a model), and correcting for
it makes the selector worse. The raw rule stands.

### Supporting: llama.cpp per-layer bits (`gguf_layer_bits.json`)
In the bartowski Gemma 4 E2B-it GGUFs, decoder layers 0–3 get more bits per weight than the median layer in every
mixed K-quant and IQ file (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, IQ2_M; e.g. IQ2_M: layers 0–1 at 4.93 bpw and 2–3 at
3.56, median 2.81). Q8_0 is near-uniform, and Q4_0's block types were not parsed by the script (NaN). So the recipe
already treats the first layers as sensitive, which is the same set the "first four" baseline restores. Whether this
explains the tie between bin flips and "first four" is untested.

### E2. Other task types (`task_score.json`; post-hoc rescoring in `task_posthoc.json`)
| task, pooling | S better than last | mean regret S | mean regret last |
|---|---|---|---|
| Banking77, mean | 3/5 | 0.167 | 0.149 |
| Banking77, last token | 2/5 | 0.145 | 0.068 |
| STS-B, mean | 2/4 | 0.183 | 0.131 |
| STS-B, last token | 0/4 | 0.181 | 0.062 |

- **C1 not supported** (3/5; needed 4). **C3 not supported** (2/4; needed 3). **C2 holds** (S not better than last
  with last-token pooling in 3/5), as expected. C4: S is worse than last in 4/4.
- Per model, Banking77 mean: S does better than last for Qwen3.5-2B (regret 0.027 vs 0.071), Qwen3.5-4B (0.044 vs
  0.101) and SmolLM2 (0.241 vs 0.276), and worse for Qwen3-1.7B (S = L2, 0.338 vs 0.182) and Gemma 4 E2B (S = L17,
  0.184 vs 0.113). The Qwen3.5-4B pick (L7) is the same layer as the earlier classifier pilot.
- **Diagnosis (post hoc).** Two things differ from DBpedia and 20NG:
  1. *Massive activations, in two models.* In Qwen3-1.7B and SmolLM2-1.7B, from about layer 3 to the middle layers,
     5 dimensions hold 98–100% of the variance of mean-pooled short-text features (e.g. Qwen3-1.7B L3: 99.9%;
     SmolLM2 L12: 99.6%), consistent with the massive activations described by Sun et al. (2024). Raw cosine there
     mostly compares those dimensions. In Gemma 4 E2B and the two Qwen3.5 models the top-5 share stays at 3–16%, so
     this does not explain their results. Layer 0 is still the best layer under the pre-registered outcome for 3/5
     Banking77 models, including Qwen3.5-2B.
  2. *Flat stability.* Map stability varies over a narrow range on these data (0.82–0.86 on Banking77 vs a
     clearer peak on DBpedia), so its argmax is close to arbitrary.
- **Rescoring the outcome does not rescue the selector** (post hoc, not a pre-registered test). With per-dimension
  z-scoring or in the whiten-64 frame, the oracle layer moves off layer 0, but S still beats last in only 2–3/5
  (Banking77, mean) and 2/4 (STS-B, mean), with mean regret 0.10–0.15 against 0.05–0.08 for last. The failure is in
  the selection, not only in the outcome metric.
- **Reading.** Map stability's layer choice transferred to held-out topic-retrieval pairs (DBpedia, 20NG; 64-token
  documents), but not to short-text intent retrieval or sentence similarity across model families. It worked on the
  two Qwen3.5 models and failed on Qwen3-1.7B and Gemma 4 E2B. Candidate causes: a fixed K = 512 on 10k short training
  texts (flat stability), and massive-activation dimensions in two of the models. Neither is tested.

### Post-hoc control for P3 (2026-10-01, after the red-team review `research/reviews/redteam_2026-10-01.md`)
Training step alone predicts best-layer p@10 better than map stability does: Spearman(log step, p@10) = 0.88 (410m)
and 0.97 (1.4b), against 0.85 and 0.93 for max stability. Controlling for step, the partial Spearman of stability and
p@10 is 0.27 (410m) and 0.57 (1.4b), with 9 checkpoints. From step 4,000: stability 0.49 / 0.77, step 0.60 / 0.89.
**P3 passed as written but does not show a monitor that adds information beyond training progress.** The decisive
tests are across seeds at a fixed step (F2, PolyPythia) and on deliberately degraded checkpoints.
