> **Superseded.** This 2026-10-01 draft is kept for the record. The current write-up is the Model Flow post (`docs/index.html`); every current number is in `research/CLAIMS.md`.

# Which Layer, Without Labels? Method and Evidence

*model-flow-map · technical companion · draft for review, 2026-10-01. The overnight pre-registered studies
(`PREREG_overnight.md`) are now filled in; sections 15–17 and the reference list are still to be written.*

**How this was built.** Built with Claude Code, using Claude Opus models. The research direction and the claims are
the author's. Every hypothesis below was written down before its data existed. The pre-registrations, including
every amendment and what had been seen when it was made, are published in `docs/preregistrations/`.

Contents

1. [Why this work exists](#why)
2. [What is new here, and what is not](#new)
3. [The claims, and where each is checked](#ledger)
4. [The method, stated precisely](#method)
5. [How measurements are made](#discipline)
6. [Choosing a layer without labels](#select)
7. [Beyond topic retrieval](#tasks)
8. [Across pretraining](#pythia)
9. [Compression: what quantization moves](#quant)
10. [Inside the model: stages, correspondence, tokens](#inside)
11. [Larger maps](#largek)
12. [Trying to break it](#break)
13. [Placing the work](#related)
14. [Limitations](#limits)
15. [Putting it together: the two questions, answered](#together)
16. [Applications](#apps)
17. [Reproducing everything](#repro)

---

<a id="why"></a>
## 1. Why this work exists

Embeddings are everywhere in AI:
- search and retrieval-augmented generation;
- clustering, deduplication and recommendation;
- classifier heads on frozen models;
- routing, monitoring and memory.

When a language model provides them, the embedding is almost always read at one of two positions: the **input**, the
static token-embedding table, or the **final layer**, averaged over the tokens or taken at the last one. Everything in
between is usually skipped. Is that optimal, or is it lazy?

The evidence says it is often not optimal. Intermediate layers regularly give better embeddings than the final one
(Liu et al. 2019; Skean et al. 2025). In our data, reading the last layer costs 3.4 points of retrieval precision on
average across the held-out pairs, and 29.6 points on the instruction-tuned Gemma 4 E2B. The best layer also moves: it differs between models, between datasets
for the same model, and between a model and its fine-tuned or compressed versions. The reliable way to find it is to
score every layer against labeled data for your task, then redo that for every model, domain, quantization and
fine-tune. Usually nobody has those labels, so the default stands.

There is a second, more basic question behind the first. A transformer decoder is highly nonlinear, and the residual
stream does not keep one fixed coordinate system: each layer re-expresses what it receives, so the "same" direction
means different things at different depths. Most practitioners have an intuition for this. **Is it possible to see it,
or measure it,** without assuming a shared coordinate system that isn't there?

This work answers both questions with one instrument. At every layer we sort a corpus into bins with plain k-means,
building a **flow map** of the model. Bins need no shared coordinates: two layers, two models, or a model and its
quantized copy can be compared by asking whether they sort the same documents together.

- **Seeing it.** Drawing how documents move between bins from layer to layer shows where the model regroups its
  inputs, where it holds them steady, and how a fine-tuned or compressed version differs from the original.
- **Measuring it.** A layer whose neighbourhoods are real structure gets sorted the same way every time the
  clustering is re-run; a layer that is noisy at that scale gets cut differently each time. We call the
  reproducibility of a layer's bins its **map stability**.
- **Using it.** Picking the layer with the highest map stability is a way to choose where to read embeddings with
  **no labels at all**.

**The goal for the layer choice.** With no labels, cheaply, for any model and corpus, the choice should:
1. land close to the layer you would pick with labels;
2. beat the fixed defaults and the label-free scores already in the literature;
3. hold when the model is compressed.

<a id="new"></a>
## 2. What is new here, and what is not

We surveyed the literature before writing this up (the notes and report are in the repository). In summary:

- **The flow-map picture is not new.** Clustering each layer and drawing a Sankey of how inputs move between clusters
  has been published for MLPs and CNNs (Halnaut et al. 2020; İrsoy & Alpaydın 2022, *PathFinder*) and for BERT
  tokens (Sevastjanova et al. 2025, *LayerFlow*). Clustering per layer to find concepts is also established (Dalvi et
  al. 2022; Hawasly et al. 2024). We extend these diagrams to document populations in decoder LLMs, label bins by
  topic lift, and overlay a model, its instruction-tuned version and its 2-bit quantization on one scaffold. We found
  no precedent for that overlay.
- **Label-free layer selection is not new.** Intrinsic-dimension rules already pick layers without labels (Valeriani et
  al. 2023; Tang & Yang 2026), and the stability of a clustering has been used for twenty years to choose the number
  of clusters (Ben-Hur et al. 2002; Lange et al. 2004; von Luxburg 2010).
- **What appears new** is using the *reproducibility of a partition under re-seeding* to select a *layer*. To our
  knowledge it has not been done before, and here it is evaluated prospectively, on held-out model–dataset pairs,
  against rank, entropy, curvature and intrinsic-dimension selectors. The closest work is a 2026 preprint that scores
  each layer's neighbourhood partition against the final layer's to rank models (Jiang, Yang & Tam 2026); it does not
  choose an extraction layer.
- **Replications.** Several results here reproduce known shapes with a new, discrete measurement:
  - intermediate layers beat the last;
  - layers settle during training (here: the first and last layers before the middle ones, not strictly bottom-up);
  - layers of different models correspond roughly by relative depth (13 of 21 pairs at the pre-registered strength);
  - per-layer token information follows Voita, Sennrich & Titov (2019), who first discretized token states with
    k-means to measure it.

<a id="ledger"></a>
## 3. The claims, and where each is checked

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | Map stability picks a layer within 0.5 points of the best, on 11 held-out pairs | **passed** (pre-registered) | `PREREG_layer_selection.md`; `results/select_heldout_score.json` |
| 2 | It beats the last layer, the middle layer and RankMe | **passed** | same |
| 3 | It beats intrinsic dimension, dataset entropy, prompt entropy and curvature as selectors | **passed** | `PREREG_baselines.md`; `results/baselines_score.json` |
| 4 | Its lead over the last layer is statistically reliable | **passed**: +2.9 points, 95% CI [1.6, 4.3]; 10 of 11 pairs individually | same |
| 5 | It is worth more than one labeled example per class | **failed**: one label per class ties it | same |
| 6 | It still picks well after chance-correcting the agreement score | **failed**: the corrected rule is worse (mean regret 0.013 vs 0.0046); the raw rule stands | `results/adjusted_select.json` |
| 7 | It works for intent classification and semantic similarity, with mean pooling | **failed**: better than last in 3/5 (Banking77) and 2/4 (STS-B) models | `PREREG_overnight.md` E2; `results/task_score.json`, `task_posthoc.json` |
| 8 | It fails with last-token pooling (stated expectation) | **as expected**: not better than last in 3/5 and 4/4 | same |
| 9 | It picks well throughout pretraining, and tracks retrieval quality across checkpoints | **mixed**: picking passed for 1.4b (8/9) and failed for 410m (4/9); tracking **passed** (ρ = 0.85, 0.93) | E1; `results/pythia_score.json` |
| 10 | It picks the same layer under every quantization from 8-bit to 2-bit | **passed** (9/9) | `PREREG_quantization_lens.md`; `results/quant_score.json` |
| 11 | Bin flips choose layers to keep at full precision better than activation error or random | **passed** (CIs exclude 0), but they **tie** "keep the first four" | `results/gq__qkl_ci.json` |
| 12 | Quantization-aware training moves the map less than post-training quantization | **failed**: it moves it more, and retrieves best | `results/quant_score.json` |
| 13 | Larger maps (K = 1,024–4,096) choose better | **failed**: worse | `PREREG_large_k.md`; `results/largek_score.json` |
| 14 | Gemma's per-layer embeddings keep word identity alive in late layers | **supported** on one model pair; causal test not done | `PREREG_token_lens.md`; `results/token_qwen3_score.json` |
| 15 | Stage boundaries sit at Gemma's KV-sharing boundary | **inconclusive**: passes as stated, but at about chance rate | E3; `results/correspondence.json` |
| 16 | A document's routing stability predicts its retrieval failure | **weak**: correlated in 11/11 pairs (ρ 0.07–0.21), but hubness predicts failure better | E5; `results/routing.json` |
| 17 | Token bin-splitting tracks polysemy beyond frequency | **failed**: holds in Qwen, not in Gemma | E6; `results/polysemy.json` |
| 18 | A classifier head at the chosen layer is more accurate than one at the last layer, and cheaper to run | **preliminary**: one model (Qwen3.5-4B), one dataset; the layer choice did not transfer to Qwen3-1.7B or Gemma 4 E2B on the same data (claim 7) | §7.1; full study to be pre-registered |
| 19 | Map stability ranks models by retrieval quality without labels | **exploratory**: ρ = 0.86 (DBpedia), 0.82 (20NG) across 7 models | E4; `results/model_select.json` |

<a id="method"></a>
## 4. The method, stated precisely

**Features.** Each document is truncated to 64 tokens and run through the frozen model. At every layer we average the
hidden states over tokens (*mean pooling*). "Layer ℓ" means `hidden_states[ℓ]`: layer 0 is the token embedding, and
layer ℓ ≥ 1 is the output of decoder block ℓ.

**Frame.** Per layer, fit on the training documents only: subtract the mean, project onto the top 64 principal
components, and rescale each to unit variance (whitening; cf. Su et al. 2021). Keeping all components amplifies noise
directions and makes every layer look unstable; 64 was fixed early, on development models.

**Bins.** k-means (Lloyd 1982; faiss, Douze et al. 2024) with K = 512, 50 iterations and one initialization, fit on
the training documents. Held-out test documents are assigned to their nearest centroid.

**Map stability.** Fit the bins twice, with seeds 0 and 1, and measure how well the two sortings of the test documents
agree, using normalized mutual information (NMI; Strehl & Ghosh 2002): 1 means identical, 0 means unrelated.

**The selector.** Pick the layer with the highest map stability; ties go to the deeper layer. No labels are used.

**The flow map.** The same bins at a small K (8 or 32), drawn as an alluvial diagram (Rosvall & Bergstrom 2010) of how
documents move between bins from one layer to the next. Bins are labeled by the topics most over-represented in them
(lift), and tokens can be traced along their own paths.

**Bin flips.** To compare a model with a modified copy (quantized, fine-tuned), assign the copy's features to the
*original* model's frame and bins and count how often documents land in a different bin. We call these *bin flips*,
to keep them distinct from Dutta et al.'s (2024) answer-level flips.

**Outcome used to score the selector.** Cosine nearest-neighbour retrieval: each test document queries the training
set, and precision@10 is the share of its 10 nearest neighbours sharing its label. **Regret** is the best layer's
precision minus the chosen layer's, in percentage points.

<a id="discipline"></a>
## 5. How measurements are made

- **Pre-registration.** Every hypothesis and its pass/fail criterion was written down before the data existed
  (Nosek et al. 2018). Amendments were dated and say exactly what had been seen when they were made.
- **Held out.** The selector was developed on three model–dataset pairs and tested on eleven it had never seen:
  - four new models on two datasets;
  - the three development models on a new dataset.
- **Baselines get the same treatment.** Every rival selector's direction (pick its maximum or its minimum) was fixed
  on the development pairs, before any held-out pair was scored.
- **Every number is recomputed by a script.** The `score_*.py` scripts read the result files and print each verdict.
- **Corrections are kept.** Where a pre-registered design turned out to be flawed, the flaw and the fix are recorded
  rather than silently changed (for example, a normalization that went negative in the quantization study).

**Data.**
- DBpedia hierarchical classes (9 / 70 / 219 classes; 24k train / 6k test).
- 20 Newsgroups (20 groups; full splits).
- Banking77 (77 intents).
- STS-B (sentence pairs scored 0–5).

**Models** (all frozen; revisions pinned in each cache's metadata):
- Qwen3-1.7B, and Qwen3.5-0.8B / 2B / 4B (hybrid linear and full attention; Yang et al. 2025);
- Gemma 4 E2B / E4B (base and instruction-tuned);
- SmolLM2-1.7B;
- Pythia-410m and -1.4b at ten training checkpoints.

<a id="select"></a>
## 6. Choosing a layer without labels

**Held-out test.** On the 11 pairs, the mean regret of each selector:

| How the layer was picked | Mean regret (points of p@10) | Map stability better / tie / worse, per pair |
|---|---|---|
| **Map stability** | **0.49** | — |
| Intrinsic dimension, lowest (TwoNN; Facco et al. 2017) | 2.96 | 9 / 1 / 1 |
| Last layer | 3.42 | 10 / 1 / 0 |
| Curvature (Skean et al. 2025; Hosseini & Fedorenko 2023) | 10.4 | 10 / 0 / 1 |
| Middle layer | 13.1 | — |
| RankMe (Garrido et al. 2023) | 13.5 | — |
| Prompt entropy (Skean et al. 2025) | 15.4 | 10 / 0 / 1 |
| Dataset entropy (Skean et al. 2025) | 16.6 | 10 / 1 / 0 |

- Map stability had zero regret in 5 of the 11 pairs.
- Its largest miss was Qwen3.5-4B on DBpedia (2.8 points). There the best layer is layer 9 of 32, and it chose layer 25.
- Its lead over the last layer is +2.9 points, with a two-level bootstrap 95% CI of [1.6, 4.3] (resampling pairs,
  then documents). Within individual pairs it is significantly better in 10 of 11; in the 11th they tie, because the
  last layer is the best.
- Intrinsic dimension is the strongest rival, in the direction the development pairs chose: the *lowest* ID. That
  agrees with Valeriani et al. (2023) and Tang & Yang (2026) and disagrees with Cheng et al. (2025), whose
  high-dimension "abstraction phase" was measured on token-level representations rather than pooled ones. In the
  highest-ID direction its development regret was 24 points.
- None of these rivals was proposed by its authors as a layer selector: they were proposed as correlates of quality.
  We turned them into selectors and evaluated them the same way.

![Figure 1. Held-out regret by selector. Each dot is one of the 11 held-out model–dataset pairs; the bar is the mean. Orange rows: choosing the layer with 1 or 10 labeled examples per class instead (reference). TwoNN uses the lowest-ID direction, chosen on the development pairs.](figures/selectors.svg)

![Figure 2. Map stability (blue, left axis; ▼ = chosen layer) and precision@10 (grey, right axis; ○ = best layer, ■ = last layer) at every layer, on the 11 held-out pairs. The x axis is relative depth. The two curves track each other closely, including the double peaks in Gemma 4 E4B.](figures/heldout_profiles.svg)

**What it is worth in labels.** Choosing the layer by precision on a handful of labeled training documents:

| Labeled examples per class | Mean regret |
|---|---|
| 1 | 0.49 |
| 2 | 0.33 |
| 5 | 0.13 |
| 10 | 0.09 |

One labeled example per class (70 documents for DBpedia, 20 for 20 Newsgroups) already matches map stability. With ten
it is five times better. **So map stability is a tool for the zero-label setting:** new corpora, unlabeled domains,
and pipelines that must choose a layer automatically. With even a few dozen labels, use them instead.

**Chance correction.** NMI between two partitions has a floor that grows as the number of bins approaches the number
of documents (Vinh et al. 2010). Choosing on agreement minus a permutation floor made the selector worse on the 11
held-out pairs: mean regret 0.013 against 0.0046, the same layer in 5 of 11. The floor varies little across layers
within a model, so subtracting it mainly adds noise. The raw rule stands.

**What it costs and saves.** Two k-means fits per layer on a CPU, and no labels. Stopping the model at the chosen
layer instead of running every layer was 2.2–3.2× faster on a P100 (Qwen3.5-4B, layer 10 of 32; Gemma 4 E2B, layer
17 of 35). That speedup follows from the depth and is a consequence, not a contribution; trained early-exit
embedders exist (Li et al. 2024).

<a id="tasks"></a>
## 7. Beyond topic retrieval: where the layer choice did not transfer

The held-out test used topic labels on 64-token documents with mean pooling. This pre-registered study adds intent
retrieval (Banking77, 77 intents, intent p@10) and semantic similarity (STS-B; Spearman correlation of pair cosine with
human scores; Cer et al. 2017), with both mean and last-token pooling, on five models.

| task, pooling | map stability better than last | mean regret, map stability | mean regret, last layer |
|---|---|---|---|
| Banking77, mean | 3 of 5 | 0.167 | 0.149 |
| Banking77, last token | 2 of 5 | 0.145 | 0.068 |
| STS-B, mean | 2 of 4 | 0.183 | 0.131 |
| STS-B, last token | 0 of 4 | 0.181 | 0.062 |

- **The mean-pooling predictions failed** (needed 4 of 5 and 3 of 4). The choice was good for the two Qwen3.5 models
  and poor for Qwen3-1.7B (it picked layer 2) and Gemma 4 E2B.
- **The last-token expectation held:** the final layers suit last-token features.
- **What differs from topic retrieval (post hoc).** Map stability is nearly flat over depth on these short texts
  (0.82–0.86 on Banking77), so its maximum carries little information. In Qwen3-1.7B and SmolLM2, five dimensions
  hold 98–100% of the feature variance from about layer 3 to the middle layers, consistent with massive activations
  (Sun et al. 2024), so raw cosine there compares mostly those dimensions. Rescoring the outcome with per-dimension
  scaling, or in the whitened frame, moves the best layer but does not rescue the choice (better than last in 2–3 of
  5). Neither cause is tested.
- **Scope, therefore:** the evidence for choosing a layer is for document-length topic retrieval. Short-text tasks
  are a known failure case.

### 7.1 Classifier heads on a frozen model [preliminary; full study to be pre-registered]

A common way to use a language model without generating text is to freeze it, take one layer's features, and train a
small classifier on top. The usual setup reads the **last token at the last layer**. A pilot on one model asked whether
the map-stability layer does better, and whether stopping the forward pass there saves time.

Setup: Qwen3.5-4B, Banking77 (77 intents), multinomial logistic regression on frozen features; test accuracy, mean of
3 random label draws per budget; "all" ≈ 130 labels per class. The layer was chosen by map stability on unlabeled
training texts, before any classifier was trained.

| labels per class | 1 | 5 | 20 | all |
|---|---|---|---|---|
| mean pooling, chosen layer (L7 of 32) | 0.297 | 0.634 | 0.820 | 0.895 |
| mean pooling, last layer | 0.241 | 0.575 | 0.788 | 0.881 |
| mean pooling, best layer (oracle) | 0.329 | 0.662 | 0.829 | 0.899 |
| last token, chosen layer (L10) | 0.277 | 0.504 | 0.682 | 0.798 |
| last token, last layer (usual setup) | 0.222 | 0.572 | 0.775 | 0.867 |

- **With mean pooling, the chosen layer beats the last at every budget** (+1.4 to +5.9 points) and is within 0.4 to 3.2
  points of the best layer.
- **With last-token features it loses** above one label per class (−6.8 to −9.3 points). Last-token accuracy is
  U-shaped over depth, with its maximum at the final layers: the last position is where the model assembles its
  next-token prediction. This matches the expectation stated for section 7.
- **Mean pooling at the chosen layer beats the usual setup at every budget**, and with 20 labels per class it already
  exceeds the usual setup trained on all labels. Part of that gain comes from the pooling choice, which map stability
  does not make.
- **Cost.** Stopping the forward pass after the chosen layer, P100, fp16, 64 tokens, median of 20 runs:

| model | stop after | batch 1 | batch 64, per text |
|---|---|---|---|
| Qwen3.5-4B | L10 of 32 | 101.7 → 32.3 ms (3.2×) | 81.7 → 26.0 ms (3.1×) |
| Gemma 4 E2B | L17 of 35 | 60.3 → 27.0 ms (2.2×) | 44.7 → 15.6 ms (2.9×) |

  Qwen's linear-attention layers ran on transformers' reference kernels, so the ratios matter more than the absolute
  times. Running only part of a network to save compute is the idea behind early exit (Teerapittayanon et al. 2016;
  Xin et al. 2020); here the exit layer is fixed once per task, without labels.

**Caution from section 7:** on the same Banking77 data with nearest-neighbour scoring, the map-stability layer was
worse than the last layer for Qwen3-1.7B and Gemma 4 E2B. Qwen3.5-4B, the pilot model, is one of the cases where it
worked. The classifier result may not generalize.

**Not yet shown:** more models, more classification datasets, comparison with dedicated embedding models and with
open-source LLM-based classifiers, and confidence intervals. A pre-registered study on the features from section 7 is
planned.

<a id="pythia"></a>
## 8. Across pretraining: a label-free training monitor

Because map stability needs no labels, it can be tracked during training. Pythia (Biderman et al. 2023) publishes its
intermediate checkpoints. We used two sizes, 410M and 1.4B, at ten checkpoints each, on the same DBpedia sample.

![Figure 3. Left: agreement of each layer's map with its final-checkpoint map (divided by the final re-seed agreement, capped at 1), per checkpoint. Pink line: the first checkpoint at which each layer reaches 0.9; the first and last layers settle by step 2,000, the middle ones by 4,000. White circles: the best layer (needs labels). Orange crosses: the layer map stability chooses. Right: the highest map stability against the best layer's precision@10 across checkpoints from step 512.](figures/training_dynamics.svg)

| step | 410m: chosen / best layer | 410m regret: chosen / last | 1.4b: chosen / best layer | 1.4b regret: chosen / last |
|---|---|---|---|---|
| 0 | 20 / 0 | 0.292 / 0.295 | 23 / 0 | 0.286 / 0.285 |
| 512 | 9 / 0 | 0.168 / 0.207 | 7 / 0 | 0.137 / 0.171 |
| 1,000 | 19 / 23 | 0.014 / 0.011 | 19 / 21 | 0.004 / 0.012 |
| 2,000 | 19 / 22 | 0.018 / 0.008 | 22 / 21 | 0.001 / 0.010 |
| 4,000 | 24 / 17 | 0.009 / 0.009 | 24 / 22 | 0.011 / 0.011 |
| 8,000 | 24 / 17 | 0.004 / 0.004 | 20 / 17 | 0.006 / 0.007 |
| 16,000 | 24 / 17 | 0.002 / 0.002 | 20 / 17 | 0.008 / 0.005 |
| 32,000 | 20 / 17 | 0.015 / 0.004 | 24 / 17 | 0.003 / 0.003 |
| 64,000 | 21 / 17 | 0.016 / 0.002 | 24 / 17 | 0.018 / 0.018 |
| 143,000 (final) | 19 / 24 | 0.032 / 0.000 | 24 / 17 | 0.016 / 0.016 |

- **P3, a training monitor: passed.** From step 512 on, the highest map stability tracks the best layer's retrieval
  quality across checkpoints: Spearman 0.85 (410m) and 0.93 (1.4b). With no labels, it tells you whether the
  representations are getting better for retrieval.
- **P1, choosing a layer during training: mixed.** The chosen layer was no worse than the last at 8 of 9
  checkpoints for 1.4b (passes) and 4 of 9 for 410m (fails; up to 3.2 points behind at late checkpoints). In Pythia
  the choice often *is* the last layer (L24), so it rarely gains much.
- **The best layer moves during training.** At steps 0 and 512 the input embeddings are best. By step 1,000 the best
  layer is near the top (21–23); from step 4,000–8,000 it moves down to layer 17 and stays there (410m's final
  checkpoint is an exception). We found no prior report of this movement. Map stability does not follow it: it keeps
  choosing layers 19–24, which is why P1 is mixed.
- **P2, bottom-up settling: not supported** (a replication target; Raghu et al. 2017; Chen et al. 2023; Diehl
  Martinez et al. 2024). Each layer's map reaches 90% of its final agreement by step 2,000 at the ends of the stack
  and by step 4,000 in the middle: ends first, middle last. The correlation of settle step with depth is 0.27
  [−0.17, 0.66] and 0.46 [0.00, 0.82]; the checkpoint grid is coarse, so settle steps are heavily tied.

<a id="quant"></a>
## 9. Compression: what quantization moves

Nine GGUF quantizations of Gemma 4 E2B-it were tested: Q8_0 to IQ2_M, plus Google's quantization-aware-trained (QAT)
Q4_0. All were dequantized through the same loader, which was verified bit-identical to the official weights for the
unquantized reference.

![Figure 4. Left: share of the 6,000 test articles assigned to a different bin (of the reference's 512) than the bf16 reference, per layer, for each quantization (darker = fewer bits; dashed orange = Google's QAT Q4_0). Right: next-token KL to the bf16 model after restoring four layers of Q4_K_M to full precision, chosen four ways; dots are the five random draws. Paired 95% bootstrap CIs over 1,000 documents below.](figures/compression.svg)

- **The selector is robust.** Every variant, including both 2-bit ones, chose layer 6, which is also each variant's
  own best layer.
- **Bin flips rise steadily as bits fall.** Disagreement with the reference's bins (1 − NMI, averaged over layers)
  goes from 0.027 at Q8_0 to 0.34 at IQ2_M, monotonically.
  Down to 4-bit Q4_K_M it is, averaged over layers, smaller than the disagreement from simply re-running the
  clustering. The change peaks around layers 12–16 and eases off after.
- **A boundary at layer 15.** At 5 bits and below, the jump into layer 15 is one of the two largest in 6 of 7 variants
  (5 of 7 by the unadjusted measure). Layer 15 is the output of decoder block 14, the last block that computes its own
  key–value cache: the model's configuration has the last 20 of its 35 blocks reuse earlier ones' (cf. Google 2025).
  This is a coincidence of position; we have not tested whether it is causal.
- **Choosing layers to keep at full precision.** Starting from Q4_K_M, we restored four blocks to full precision and
  measured next-token KL divergence from the reference on 1,000 held-out documents.

  | Blocks restored, chosen by | KL | vs. bin flips, 95% CI |
  |---|---|---|
  | none | 0.405 | |
  | **largest bin-flip jumps** | **0.330** | |
  | largest activation-error jumps | 0.352 | bin flips better by 0.022 [0.012, 0.033] |
  | random (5 draws) | 0.383 | bin flips better by 0.053 [0.044, 0.063] |
  | the first four blocks | 0.327 | no difference: −0.003 [−0.013, +0.008] |

  Bin flips beat activation error and random choice reliably, but a plain "first four blocks" rule does as well.
  llama.cpp's own k-quant recipe already gives the first and last eighth of blocks extra bits, plus every third block
  in between, so several of these files are mixed-precision by layer before any restoration. Reading the per-block
  types from the GGUF files confirms it: in every mixed k-quant and IQ file, blocks 0–3 get more bits per weight than
  the median block (IQ2_M: 4.93 and 3.56 against a median of 2.81). "First four" restores exactly the blocks the
  recipe already treats as sensitive. Whether that explains the tie is untested.
- **Drift is not damage.** We predicted the QAT model would move the map less than post-training Q4_0. It moved it
  more, at all 36 layers, yet retrieved best of all ten versions (precision 0.768 against the reference's 0.757). QAT is a short distillation fine-tune, not a
  noisier copy (Google 2025). Bin flips measure distance from a reference, not loss of quality.
- **Shift versus rearrangement (post hoc).** QAT's near-total flips at layers 1–2 (Figure 4) are almost entirely a
  shared offset: re-centering each version on its own mean before assigning bins cuts QAT's flips at layer 1 from
  99.6% to 12.6%, and its average from 74% to 37% (Q4_0: 39% to 29%). QAT still moves the map more than Q4_0. Bin
  flips as defined mix a global shift with a rearrangement of documents; re-centered flips separate the two.

<a id="inside"></a>
## 10. Inside the model: stages, correspondence, tokens

**Stages and cross-model correspondence.** All seven models saw the same DBpedia test documents, so their layers can
be compared through their bins without any shared coordinates.
- **Relative depth (X1): not supported at the pre-registered strength.** Matching each layer to the other model's
  most-agreeing layer gives a depth correlation ≥ 0.9 in 13 of 21 pairs (median 0.95, range 0.59–1.00). The shape
  replicates Wolfram & Schein (2025); the strength varies by pair.
- **Partitions versus CKA (X2): they disagree.** Over all layer pairs, partition agreement and linear CKA (Kornblith
  et al. 2019) correlate at ≥ 0.7 in only 5 of 21 model pairs (range −0.16 to 0.91). The two measures rank layer
  correspondences differently; which one better tracks function is open.
- **Stage boundaries at Gemma's KV-sharing layer (X3): inconclusive.** Both models have a dip in adjacent-layer
  agreement within one layer of the boundary (E2B: layer 16 for 15; E4B: 25 for 24), as stated. But each model has
  six or seven dips, so a pass in both is about a 30% chance, and E2B's dip there is among its shallowest.

**Model selection (exploratory).** Across seven models, the highest map stability ranks models by their best layer's
retrieval quality: Spearman 0.86 on DBpedia (permutation p = 0.012) and 0.82 on 20 Newsgroups (p = 0.017). With the
training result in section 8, this suggests a label-free way to compare models and checkpoints on a corpus. Seven
models is a small sample, and stability values from different architectures may not be strictly comparable.

**Tokens.** Binning individual tokens instead of documents follows Voita, Sennrich & Titov (2019). In Gemma 4 E2B:
- **Word identity persists.** It falls to a minimum of 0.45 NMI at layer 18, then rebounds to 0.67 at layer 24.
- **The next word never overtakes it.**
- **2-bit quantization moves rarer words' bins more** than frequent words', at every layer from 5 on (mean share
  of tokens changing bin: 0.50 vs 0.39). We found no prior report of this frequency effect.
- **Qwen3-1.7B has no rebound** (0.03 against Gemma's 0.22). Qwen3-1.7B has no per-layer token embeddings, while
  Gemma's architecture feeds one into every block. That is consistent with those embeddings re-injecting identity
  (cf. TIDE, Jaiswal et al. 2026). One model pair can support this but not establish it. The causal test, zeroing
  Gemma's per-layer inputs, has not been run.

**Per-document routing stability: real but weak.** A document whose bin-mates change between the two seeds does
retrieve worse: the correlation is positive with a confidence interval above zero in all 11 held-out pairs, but small
(0.07–0.21). As a failure detector it beats centroid distance and margin in 7 of 11 pairs, but loses to hubness
(Radovanović et al. 2010) in 10 of 11. Related query-difficulty work perturbs the query (Arabzadeh et al. 2023); this
perturbs the index partition, and the simpler hubness signal is better.

**Polysemy: not supported.** The scatter of a word's 20 occurrences across bins correlates with its WordNet sense
count (Miller 1995) after controlling for frequency in Qwen3-1.7B (0.12 [0.01, 0.24]) but not in Gemma 4 E2B (0.05
[−0.07, 0.17]); the pre-registered test needed both. Prior work estimates polysemy from contextual embeddings (Garí
Soler & Apidianaki 2021; Pimentel et al. 2020).

<a id="largek"></a>
## 11. Larger maps

Going from 512 bins to 1,024–4,096 did not help.

![Figure 5. Left: mean held-out regret of the layer choice by number of bins. Middle: share of fine bins with at least 90% of their documents inside one 512-bin. Right: accuracy of labeling each bin by its majority topic (70 topics, 120k documents) against a linear probe (dashed) and 10-nearest-neighbours (dotted) at the same layer.](figures/larger_maps.svg)

- **Layer selection got worse.** Mean regret rose from 0.49 to 1.20 (1,024) and 1.51 (2,048) on the 11 held-out pairs.
- **Finer maps do not nest inside coarser ones.** Only 40–47% of fine bins fall ≥ 90% inside one coarse bin. That is
  the hard-partition analogue of feature splitting in sparse autoencoders (Bricken et al. 2023; Bussmann et al. 2025),
  and an argument for building hierarchies explicitly.
- **Bins as a classifier.** Labeling each bin by majority vote reaches 92.4% on 70 classes at 4,096 bins (Gemma 4 E2B,
  120k documents), below a linear probe (94.4%) and 10-nearest-neighbours (95.6%): a sanity check, not a result.
- **Not tested:** whether the NMI chance floor, which grows with K, explains part of the worsening. At K = 512 the
  chance-corrected selector was worse than the raw one (section 6).

<a id="break"></a>
## 12. Trying to break it

**"Stable is not the same as correct."** True in general: k-means stability depends on symmetries of the data, not on
whether the clustering is meaningful (Ben-David et al. 2006), and a trivial partition is perfectly stable (Buhmann
2010). That is why the claim is tested against labels on held-out pairs rather than argued from first principles.
The degenerate case also showed up: with last-token pooling, layer 0 is a lookup of a single token and looks
near-perfectly stable. It has to be excluded there.

**"NMI at 512 bins has a chance floor."** Yes. With 6,000 documents and 512 bins, chance agreement is not negligible.
Correcting for it made the selector worse (mean regret 0.013 against 0.0046; section 6), so the floor
is not what drives the choice.

**"Intermediate layers being better is known."** Yes. That is why the evidence is the selector, compared with
other ways of finding the right intermediate layer, not the existence of one.

**"A few labels beat it."** Yes, and we report it: one label per class ties it.

**"It is tuned to topic retrieval with mean pooling."** It may be. Section 7 tests other tasks and the pooling it is
expected to fail on.

**"The quantization win is the llama.cpp recipe in disguise."** Possibly. The per-block bit widths will show what each
file already promotes.

**"The flow map is a known visualization."** Yes. Section 2 credits it, and section 13 says what is added.

<a id="related"></a>
## 13. Placing the work

- **Layer-wise analysis of language models.**
  - Logit lens (nostalgebraist 2020) and tuned lens (Belrose et al. 2023).
  - Probing across layers (Liu et al. 2019; Tenney et al. 2019).
  - Intermediate-layer embeddings (Skean et al. 2025).
  - Mean-pooled LLM embeddings (Reimers & Gurevych 2019; BehnamGhader et al. 2024).
- **Label-free representation quality.**
  - Effective rank (RankMe; Garrido et al. 2023).
  - Intrinsic dimension (Facco et al. 2017; Ansuini et al. 2019; Valeriani et al. 2023; Cheng et al. 2025), including
    layer selectors (Tang & Yang 2026).
  - Entropy and curvature (Skean et al. 2025).
  - Partition agreement against the final layer (Jiang, Yang & Tam 2026).
- **Clustering stability.** Ben-Hur et al. 2002; Lange et al. 2004; Ben-David et al. 2006; von Luxburg 2010; Bubeck
  et al. 2012. Chance-corrected agreement: Vinh et al. 2010.
- **Clustering representations to find concepts.**
  - Dalvi et al. 2022; Hawasly et al. 2024.
  - Token-level k-means information profiles (Voita et al. 2019).
  - Sparse-autoencoder features and their splitting (Bricken et al. 2023; Bussmann et al. 2025).
- **Visualizing flows across layers.**
  - Halnaut et al. 2020; İrsoy & Alpaydın 2022; Sevastjanova et al. 2025.
  - Mapper graphs: TopoAct and TopoBERT (Rathore et al. 2021, 2023).
  - Alluvial diagrams (Rosvall & Bergstrom 2010).
- **Comparing representations.**
  - Similarity measures: SVCCA (Raghu et al. 2017), CKA (Kornblith et al. 2019) and their critiques (Ding et al.
    2021; Davari et al. 2023).
  - Cross-model correspondence (Wolfram & Schein 2025) and stages of inference (Lad et al. 2024).
  - Training dynamics (Chen et al. 2023; Diehl Martinez et al. 2024).
- **Quantization sensitivity.**
  - Layer-wise sensitivity and mixed precision (Dong et al. 2019; Dumitru et al. 2024).
  - The llama.cpp importance matrix and k-quant recipes.
  - Answer-level flips (Dutta et al. 2024).
  - Quantization-aware training (Jacob et al. 2018).
- **Architecture.** Per-layer embeddings and KV-cache sharing (Google 2025); massive activations and attention sinks
  (Sun et al. 2024).

<a id="limits"></a>
## 14. Limitations

- **Scale.** Models up to 4B parameters (Pythia up to 1.4B) and corpora up to 120k documents.
- **Tasks and pooling.** The evidence for choosing a layer is document-length topic retrieval with mean pooling. On
  short-text intent retrieval and sentence similarity the choice did not transfer across model families (section 7).
- **Single-model studies.** Quantization rests on one model and the token analysis on two. The per-layer-embedding
  explanation is correlational.
- **Numerics.** These vary across caches (fp32, fp16, bf16) and are recorded per cache. Within each comparison they
  are held fixed.
- **Amendments.** Several pre-registrations were amended mid-study. Each amendment is dated and states what had been
  seen.

<a id="together"></a>
## 15. Putting it together: the two questions, answered

**[to be written after the overnight results]**
- **Is the default position optimal, or lazy?** Draw on sections 6, 7, 8 and 9.
- **Can the shifting coordinate system be seen and measured?** Draw on sections 8, 9, 10 and 11.

<a id="apps"></a>
## 16. Applications

**[to be written after the overnight results]** Candidates, each tied to evidence or marked untested:
- choosing an embedding layer for a new corpus or model;
- early exit at the chosen layer: higher accuracy and lower cost for classifier heads (§7.1, preliminary);
- quantization QA and choosing layers to keep at full precision;
- label-free training monitoring;
- model and fine-tune diffing;
- deployment drift detection;
- per-document retrieval confidence;
- interpretability views of model internals.

<a id="repro"></a>
## 17. Reproducing everything

**[to be written with the package layout]** One command per figure and table. Feature caches are regenerated from
the pinned model revisions; every verdict is recomputed from `results/` by the `score_*.py` scripts.

## References

**[to be completed: full entries, each checked against its primary source; the prior-art report lists which
still need verification]**
