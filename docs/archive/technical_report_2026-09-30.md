# The attractor lens: label-free, multi-scale partitions of a frozen language model's layers

> **Superseded.** This 2026-09-30 draft is kept for the record. The current write-up is the Model Flow post (`docs/index.html`) and the long-form [method and evidence](../method-and-evidence.md).

*Technical report, internal draft v2 (2026-09-30, evening).*

## Summary

We bin a frozen language model's internal representations with plain k-means, layer by layer and at several scales
(K = 8 to 4,096 bins), with no labels and no training. We ask what these bins can tell us about the model. Four
studies were pre-registered; their hypotheses and pass/fail criteria were written down before the relevant data
existed.

1. **Layer selection (passed).** The layer whose K = 512 bins reproduce best across two random restarts is a strong
   label-free choice of layer for embedding-based retrieval. On 11 held-out model–dataset pairs (7 models, 2
   datasets), its mean shortfall against the best layer was **0.49 points of precision@10**. The last layer's was
   3.42, the middle layer's 13.1, and RankMe's 13.5. It beat the last layer in 10 of 11 pairs and tied in the 11th.
   A follow-up pre-registration against the literature's label-free layer scores found the same ordering. Rule S's
   regret (0.0049) was lower than intrinsic dimension (0.030), curvature (0.104), prompt entropy (0.154) and dataset
   entropy (0.166). Its lead over the last layer is +2.9 points (95% CI [1.6, 4.3]). Its limit: choosing a layer with
   **one labeled example per class** already matches it, so it is a tool for the zero-label setting.
2. **Quantization (mixed).** Across nine GGUF quantizations of Gemma 4 E2B-it, the layer-selection rule picked the
   same layer as for the unquantized model in every case. Bin changes grew steadily as bit width fell and located a
   consistent jump at one layer. As a guide for mixed-precision restoration it beat per-layer activation error and
   random choice, but tied a trivial "first four layers" rule. Quantization-aware training produced the *most* bin
   change and the *best* retrieval, so the lens measures drift from a reference, not damage.
3. **Large K (mostly negative).** Going from 512 to 1,024 or 2,048 bins made layer selection worse on the 11 held-out
   pairs. At 4,096 bins (two models) it got worse for one and slightly better for the other. Finer bins did not nest
   inside coarser ones. A "cell-majority" classifier improved with K and reached 92.4% on 70 classes. That was still
   below a linear probe and 10-nearest-neighbours.
4. **Token-level bins (pilot, mostly negative).** Binning individual tokens showed word identity persisting through
   Gemma 4 E2B far more than predicted. A follow-up on Qwen3-1.7B supports one explanation: Gemma's per-layer
   token embeddings.

Everything runs on frozen features. Once the features are extracted, the lens needs only k-means fits on a CPU. The
code, pre-registrations and result files are in this repository.

## 1. Motivation and relation to prior work

A decoder-only transformer [Vaswani et al. 2017] maps text through a stack of layers. Each layer's output lives in
its own coordinate system, so position-based comparisons across layers are not meaningful. What *can* be compared is
the **partition** each layer induces on a fixed set of inputs: which inputs end up together.

The lens is k-means [Lloyd 1982] via faiss [Douze et al. 2024], run per layer. Each centroid is a fixed point of
nearest-centroid assignment. That is the sense in which we call the centroids "attractors", following the
basin-of-attraction reading of associative memories [Hopfield 1982; Ramsauer et al. 2021] and mode-seeking
clustering [Carreira-Perpiñán 2015]. Clustering contextual representations to find concepts is established
[Dalvi et al. 2022], as is the geometric view of categorical and hierarchical concepts in LLMs [Park et al. 2025].
Using the agreement of repeated clusterings as a signal of real structure is also established
[Ben-Hur et al. 2002; von Luxburg 2010]. We measure agreement with normalized mutual information (NMI)
[Strehl & Ghosh 2002].

Layer-wise analyses of LLMs include the logit lens [nostalgebraist 2020], the tuned lens [Belrose et al. 2023], and
Skean et al. [2025], who show that intermediate layers often give better embeddings than the last one. Mean-pooled
LLM hidden states are a common embedding baseline [Reimers & Gurevych 2019; BehnamGhader et al. 2024]. The new
elements here are narrow:
- a specific, frozen, label-free layer selector tested prospectively on held-out models;
- the use of partition change as a per-layer quantization diagnostic.

## 2. Method

**Features.** For each document we take every layer's output (`hidden_states[0..L]`, where 0 is the embedding
layer), truncated to 64 tokens and mean-pooled over tokens. **"Lens layer ℓ" in this report means `hidden_states[ℓ]`,
the output of decoder layer ℓ−1.**

**Frame.** We center with the train mean and whiten the top 64 principal components, fit on up to 12,000 train rows.
Whitening all components scales noise up and destroys partition stability; 64 was fixed early, on development
models [cf. Su et al. 2021].

**Bins.** faiss k-means, 50 iterations, one restart, fit on train features. Test documents are assigned to the
nearest centroid.

**Rule S.** Fit K = 512 bins twice, with seeds 0 and 1. Stability is the NMI between the two seeds' test
assignments, and rule S picks the layer with maximum stability (ties go deeper). No labels are used.

**Retrieval outcome.** Test documents query the train set by cosine similarity (centered, L2-normalized raw
features). Precision@10 is the share of the 10 neighbours sharing the query's label. **Regret** is the best layer's
p@10 minus the chosen layer's.

**Data.**
- DBpedia hierarchical classes (`DeveloperOats/DBPedia_Classes`; 9 / 70 / 219 classes at levels l1 / l2 / l3),
  24k train / 6k test sampled with seed 0.
- 20 Newsgroups (`SetFit/20_newsgroups`; 20 fine groups, 7 coarse prefixes), full splits (11,314 / 7,532).

**Models** (Hugging Face, revisions pinned in each cache's metadata):
- Qwen3-1.7B, Qwen3.5-0.8B / 2B / 4B (hybrid Gated-DeltaNet [Yang et al. 2025] and full attention);
- Gemma 4 E2B / E4B (base);
- SmolLM2-1.7B.

Features were extracted in fp32, fp16 or bf16 as memory allowed (recorded per cache), on NVIDIA P100 and RTX 3060 GPUs.

## 3. Study 1: label-free layer selection

**Development (not held out).** Rule S was found on Qwen3-1.7B, Qwen3.5-2B and Gemma 4 E2B on DBpedia, choosing K
post hoc from {8, 32, 128, 512}. There it picked the best layer for 2 of 3 models (mean regret 0.010 vs 0.055 for the
last layer).

**Pre-registration** (`docs/preregistrations/PREREG_layer_selection.md`). This was written before any held-out
feature was extracted. It covers 11 held-out pairs:
- four new models × both datasets;
- the three development models × 20 Newsgroups.

Primary label: DBpedia l2 and 20 Newsgroups fine. Criteria:
1. mean regret below the last layer's, and S beats last in ≥ 75% of pairs;
2. mean regret ≤ RankMe's [Garrido et al. 2023] and the middle layer's.

RankMe's direction (argmin) was fixed on development data before held-out scoring.

**Result: both criteria pass.**

| Selector | Mean regret (primary) | Mean regret (secondary label) |
|---|---|---|
| **Rule S** | **0.0049** | **0.0043** |
| Last − 2 (descriptive) | 0.0205 | 0.0197 |
| Last layer | 0.0342 | 0.0313 |
| ¾ depth (descriptive) | 0.0421 | 0.0368 |
| Middle layer | 0.1306 | 0.1213 |
| RankMe (argmin) | 0.1354 | 0.1245 |

Per pair:
- S beat the last layer in 10 of 11 pairs. The 11th (Gemma 4 E4B × 20 Newsgroups) is a tie, because the last layer
  is best there.
- S had zero regret in 5 pairs.
- The largest miss is Qwen3.5-4B × DBpedia (regret 0.028). There the best layer is lens layer 9 of 0–32, and S chose layer 25.
  RankMe (0.0014) and ¾ depth (0.0052) did better on that pair.
- On Gemma 4 E4B × DBpedia the best layer is lens layer 15 of 0–42; S chose 14 (regret 0.0025), while the last layer's regret was
  0.055.

**Secondary pre-registered hypothesis: not supported.** We predicted that in Qwen3.5 hybrids, partition churn at
K = 32 is higher after linear-attention blocks than after full-attention blocks. The bootstrap CI excluded zero in
the predicted direction for Qwen3.5-0.8B on both datasets, but the sign reversed for Qwen3.5-4B (2 of 4; ≥ 3 needed).

**Reading.** Rule S reliably avoids the large losses of fixed-position defaults. It costs two k-means fits per layer
and no labels. It is not an oracle: it can miss a sharp early optimum (Qwen3.5-4B), and a simple "last − 2" default
gets part of the way.

## 3b. Study 1b: stronger baselines, labels, and uncertainty

**Pre-registration** (`PREREG_baselines.md`). It was written before any of these scores was computed. Rule S is
unchanged, and each rival's direction (argmax or argmin) was fixed on the 3 development pairs. Scores:
- **Intrinsic dimension:** TwoNN [Facco et al. 2017], on 5,000 test documents.
- **Dataset entropy** [Skean et al. 2025]: matrix-based α = 1 entropy of the Gram matrix of 2,000 documents.
- **Prompt entropy and curvature** [Skean et al. 2025; Hosseini & Fedorenko 2023]: computed from token-level hidden
  states of 1,000 documents per pair, during a fresh forward pass.
- **Few-label selection:** pick the layer by p@10 of m labeled training documents per class (20 draws).

| Selector (held-out, primary label) | Mean regret | Pairs where rule S is better / tie / worse |
|---|---|---|
| **Rule S** | **0.0049** | — |
| Intrinsic dimension (argmin, dev-fixed) | 0.0296 | 9 / 1 / 1 |
| Last layer | 0.0342 | 10 / 1 / 0 |
| Curvature (argmin) | 0.1041 | 10 / 0 / 1 |
| RankMe (argmin) | 0.1354 | — |
| Prompt entropy (argmin) | 0.1536 | 10 / 0 / 1 |
| Dataset entropy (argmax) | 0.1656 | 10 / 1 / 0 |
| Few-label, 1 per class | 0.0049 | — |
| Few-label, 2 / 5 / 10 per class | 0.0033 / 0.0013 / 0.0009 | — |

- **B1 (rule S beats every literature score): passed.**
- The development pairs chose the *lowest* intrinsic dimension. That is the opposite of the high-dimension
  "abstraction phase" that Cheng et al. [2025] associate with transferable layers. In the literature's direction
  (argmax) its development regret was 0.24. Their analysis is token-level, ours is pooled over documents; we did not
  test why the direction flips.
- **B2 (expectation that rule S beats one label per class): not met.** With 1 labeled example per class (70 for
  DBpedia, 20 for 20 Newsgroups), a supervised choice ties rule S. With 10 per class it is five times better. So rule
  S is for the zero-label setting: new corpora, unlabeled domains, or automated pipelines.
- **B3 (uncertainty): passed.** A paired bootstrap over test documents puts S above the last layer in 10 of 11 pairs.
  In the 11th they tie, because the last layer is the best layer. A two-level bootstrap (pairs, then documents) gives
  mean p@10(S) − p@10(last) = +0.029, 95% CI [0.016, 0.043].

## 4. Study 2: what quantization does to the map

**Setup** (`PREREG_quantization_lens.md`, amendments 1–3).
- Model: Gemma 4 E2B-it. Reference: the bf16 GGUF, verified bit-identical to the official safetensors (540 tensors,
  max absolute difference 0).
- Variants: Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, Q3_K_M, Q2_K and IQ2_M (bartowski), plus Google's QAT Q4_0
  [Jacob et al. 2018; Google 2026].
- Every variant is dequantized through the same transformers GGUF path and run in fp16. This measures weight
  quantization only; llama.cpp's integer activation paths are out of scope.
- Each quantized model's test features are projected into the reference's frame and assigned to the reference's
  bins.
- **Flip** at a layer is 1 − NMI(reference bins, quantized bins).

The pre-registered normalization by the re-seed ceiling proved ill-posed: it goes negative, because a quantized copy
can agree with the reference better than a re-seeded reference does (amendment 3). Raw flip is reported alongside.

**Q1 (descriptive).** Mean raw flip at K = 512 rises monotonically as bit width falls: Q8_0 0.027 → Q6_K 0.051 →
Q5_K_M 0.081 → Q4_K_M 0.125 → Q4_0 0.211 → Q3_K_M 0.239 → Q2_K 0.332 → IQ2_M 0.342.
- Down to Q4_K_M, the reference-frame flip is, averaged over layers, smaller than the reference's own re-seed disagreement.
- Change rises to a peak around lens layers 12–16, falls by roughly 40% by layer 26, then stays flat (same shape in every
  post-training variant).
- At 5 bits and below, the jump into **lens layer 15** is one of the two largest in 6 of 7 variants by
  normalized flip. By raw flip it is 5 of 7 (Q4_0 and Q3_K_M miss).
- By contrast, per-layer relative activation error ranks the same four lens layers (12, 14, 20, 35) as its largest
  jumps in 7 of 9 variants, from 8-bit to 2-bit. It is almost insensitive to bit width, so it seems to reflect the
  model more than the quantization. We did not test why.

**Q2 (rule S under quantization): passed, 9/9.** Every variant, including both 2-bit ones, selects lens layer 6. That
is the reference's choice and each variant's best layer, so regret is 0 in every case.

**Q3 (mixed precision).** Starting from Q4_K_M, restore four decoder layers to full precision, chosen by one of the
rules below. The restored decoder layers were:
- lens: 4, 5, 14, 33 (lens layers 5, 6, 15, 34);
- raw-flip lens: 8, 9, 10, 14;
- activation error: 11, 13, 19, 34. The outcome is next-token KL divergence [Kullback & Leibler 1951] from the reference, on 1,000 held-out
documents.

| Layers restored | Chosen by | KL |
|---|---|---|
| none (pure Q4_K_M) | — | 0.405 |
| 4 | **lens (pre-registered)** | **0.330** |
| 4 | lens, raw flip (amendment 3) | 0.335 |
| 4 | activation error | 0.352 |
| 4 | random (5 draws) | 0.383 ± 0.017 |
| 4 | first four | 0.327 |
| 4 | last four | 0.400 |
| all | — | 0.019 |

Against the pre-registered criteria, the lens beat both activation error and random, so Q3 is **supported**.

Paired bootstrap over the 1,000 documents (amendment 4; 10,000 resamples; `results/gq__qkl_ci.json`), with 95% CIs on
KL differences:
- lens − activation error: −0.022 [−0.033, −0.012];
- lens − random mean: −0.053 [−0.063, −0.044];
- lens − first four: +0.003 [−0.008, +0.013].

The rerun reproduced every KL value exactly. So the lens is reliably better than activation error and random choice
on these documents, and indistinguishable from restoring the first four layers. The CIs cover document sampling only,
not other models or other random draws.

**Q4 (QAT, directional): not supported, and informative.** We predicted that QAT Q4_0 would change the bins less
than post-training Q4_0. It changed them *more* at all 36 layers. Yet its retrieval p@10 at its selected layer
(0.768) was higher than the reference's (0.757). QAT yields a different model rather than a degraded copy. The lens
measures **distance from a reference**, which is not the same as loss of quality.

**Untested observation.** Lens layer 15 is the output of decoder layer 14. Gemma 4 E2B's configuration sets
`num_kv_shared_layers` = 20 of 35 (`results/gemma4_e2b_config_excerpt.json`), so decoder layer 14 is the last layer
that computes its own key–value cache. The KV-sharing design is described for Gemma 3n [Google 2025]. The coincidence is suggestive, not
tested.

## 5. Study 3: large K

**Setup** (`PREREG_large_k.md`, amendment 1).
- Set A: the 11 held-out pairs at K ∈ {1,024, 2,048}.
- Set B: new DBpedia caches with 120k training documents for Qwen3.5-2B and Gemma 4 E2B, at
  K ∈ {512, …, 4,096} × N ∈ {24k, 120k}.

| Hypothesis | Result |
|---|---|
| **K1**: rule S is no worse at larger K | **Hurts.** Mean regret 0.0049 (K = 512) → 0.0120 (1,024) → 0.0151 (2,048). |
| **K2**: stability rises with N at every layer, and S at K = 4,096 is no worse than S at 512 with 120k docs | **Not supported.** Stability rose with N at every layer and every K in both models. But Gemma's S at K = 4,096 (layer 28, regret 0.025) is worse than at K = 512 (layer 17, regret 0). Qwen3.5-2B holds (0.015 vs 0.017). |
| **K3**: cell-majority accuracy rises with K and stays below a linear probe | **Supported.** Gemma, layer 17, N = 120k: 88.1 / 90.2 / 91.3 / 92.4% at K = 512 / 1,024 / 2,048 / 4,096, vs 10-NN 95.6% and probe 94.4%. Qwen3.5-2B: 79.5 → 87.3%, vs 92.8 / 92.5%. |
| **K4**: finer bins resolve the 219-class level and nest inside coarser ones | **Not supported.** NMI with l3 *falls* slightly with K. Only 0.40 (Qwen) and 0.47 (Gemma) of 1,024-bins fall ≥ 90% inside a single 512-bin (0.8 predicted). |

**Reading.** K = 512 is the useful scale for these corpus sizes. Finer k-means partitions are not hierarchies:
independent fits at K and 2K cut the space differently. A multi-scale hierarchy would have to be built explicitly
rather than read off independent fits.

## 6. Study 4: token-level bins (pilot)

**Setup** (`PREREG_token_lens.md`, amendments 1–2 and part 2).
- Tokens are binned instead of documents: the first 64 tokens of 3,000 fit articles, capped at 30 occurrences per
  word, and 1,500 evaluation articles.
- Position 0 is dropped (attention-sink outlier norms [Sun et al. 2024]).
- Words are canonicalized with Engram's tokenizer compression [Cheng et al. 2026].
- Per layer we measure NMI of the bins with: the word, the next word, and (within a word) the article's topic, minus
  a permutation null.
- Models: Gemma 4 E2B base, tuned, and 2-bit tuned (IQ2_M); then Qwen3-1.7B.

**Gemma 4 E2B**
- **T1** (identity fades ≥ 30%): not supported for the base model (−27%; tuned −30%, 2-bit −36%).
- **T2** (next-word peaks in the last quarter): not supported (peak at layer 24 of 35).
- **T3** (next word overtakes identity): not supported. Identity stays higher at every layer.
- **T4** (context peaks mid-model): supported for the base model (layer 18, effect 0.105, 95% CI [0.094, 0.115]). Not
  supported for the tuned models (peak at layer 2).
- **T5** (2-bit hurts rarer words more): supported at 31 of 31 layers. Mean flip was 0.50 for rarer words vs 0.39 for
  the top-100.
- **Identity rebound:** word-identity NMI falls (non-monotonically) to a minimum of 0.45 at layer 18. It rises back
  to 0.67 at layer 24 and ends at 0.59 at layer 35.

**Qwen3-1.7B (part 2).** This tests whether Gemma's per-layer token embeddings, which re-inject identity at every
layer [Google 2025], explain the persistence.
- Q-T1: not supported (identity drop 29.8%).
- Q-T2: supported (next-word peak at the last layer, 28).
- Q-T3: not supported (0.516 vs 0.566).
- Q-T4: not supported (context peaks at layer 23, outside 7–21, at 0.20).
- **The contrast is supported:** Qwen's drop (29.8%) exceeds Gemma's (26.8%), and Qwen has no rebound (0.029 vs
  Gemma's 0.221).

One pair of models can support that explanation but not establish it.

**Reading.** Token bins mostly track the token itself in both models, more than the layer-wise literature on
contextualization might suggest [Voita et al. 2019; Ethayarajh 2019]. Architecture visibly shapes the profile. The
lens is a cheap way to see that, but these are pilot results on two models.

## 7. Limitations

- **Pooling and task.** All document-level results use mean pooling and nearest-neighbour retrieval on topical
  labels. We have not shown that rule S transfers to other poolings or tasks.
- **Quantization, one model.** Studies 2–4 each rest on one or two models. The Q3 advantage over activation
  error is significant on these documents but indistinguishable from a trivial first-four rule.
- **Numerics.** Numerics vary across caches (fp32 / fp16 / bf16) and are recorded per cache. Within each comparison
  they are held fixed.
- **Amendments.** Pre-registrations were amended mid-study in recorded ways, each stating what had been seen at the
  time: hardware moves, compute cuts, and one design flaw (the ceiling normalization).
- **Scale.** Models go up to 4B parameters and corpora up to 120k documents.

## 8. Reproducing

- `lens/` holds the code:
  - `extract_hf.py` extracts features;
  - `select_layers.py` runs rule S and retrieval;
  - `score_select.py`, `score_quant.py`, `score_largek.py`, `score_token.py` and `score_qwen_token.py` recompute the
    verdicts from the JSON files in `results/`.
- The Q3 KL values and verdict are written directly by `qkl.py` to `results/gq__qkl.json`.
- `lens/scripts/` has the exact job queues used.
- The explainer (`docs/explainer.html`) is built from `results/` by `lens/viewer/build.py`.

## Acknowledgements and data

DBpedia abstracts are derived from Wikipedia (CC BY-SA). 20 Newsgroups is due to Ken Lang. Models are used under
their respective licences.

## References

- Belrose et al. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. arXiv:2303.08112.
- BehnamGhader et al. (2024). LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders. arXiv:2404.05961.
- Ben-Hur, Elisseeff & Guyon (2002). A stability based method for discovering structure in clustered data. PSB.
- Carreira-Perpiñán (2015). A review of mean-shift algorithms for clustering. arXiv:1503.00687.
- Cheng et al. (2026). Conditional Memory via Scalable Lookup (Engram). DeepSeek-AI. github.com/deepseek-ai/Engram.
- Dalvi et al. (2022). Discovering Latent Concepts Learned in BERT. ICLR. arXiv:2205.07237.
- Douze et al. (2024). The Faiss library. arXiv:2401.08281.
- Facco, d'Errico, Rodriguez & Laio (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. Scientific Reports.
- Hosseini & Fedorenko (2023). Large language models implicitly learn to straighten neural sentence trajectories to construct a predictive representation of natural language. NeurIPS.
- Cheng et al. (2025). Emergence of a High-Dimensional Abstraction Phase in Language Transformers. ICLR. arXiv:2405.15471.
- Valeriani et al. (2023). The geometry of hidden representations of large transformer models. NeurIPS.
- Ethayarajh (2019). How Contextual are Contextualized Word Representations? EMNLP. arXiv:1909.00512.
- Garrido et al. (2023). RankMe. ICML.
- Google (2025). Introducing Gemma 3n: the developer guide (per-layer embeddings, KV-cache sharing).
- Google (2026). Gemma 4 QAT models.
- Hopfield (1982). Neural networks and physical systems with emergent collective computational abilities. PNAS.
- Jacob et al. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR.
- Kullback & Leibler (1951). On Information and Sufficiency. Ann. Math. Stat.
- Lloyd (1982). Least squares quantization in PCM. IEEE Trans. Inf. Theory.
- nostalgebraist (2020). interpreting GPT: the logit lens. LessWrong.
- Park et al. (2025). The Geometry of Categorical and Hierarchical Concepts in Large Language Models. ICLR. arXiv:2406.01506.
- Ramsauer et al. (2021). Hopfield Networks is All You Need. ICLR. arXiv:2008.02217.
- Reimers & Gurevych (2019). Sentence-BERT. EMNLP. arXiv:1908.10084.
- Skean et al. (2025). Layer by Layer: Uncovering Hidden Representations in Language Models. arXiv:2502.02013.
- Strehl & Ghosh (2002). Cluster Ensembles. JMLR.
- Su et al. (2021). Whitening Sentence Representations for Better Semantics and Faster Retrieval. arXiv:2103.15316.
- Sun et al. (2024). Massive Activations in Large Language Models. COLM. arXiv:2402.17762.
- Vaswani et al. (2017). Attention Is All You Need. NeurIPS. arXiv:1706.03762.
- Voita, Sennrich & Titov (2019). The Bottom-up Evolution of Representations in the Transformer. EMNLP. arXiv:1909.01380.
- von Luxburg (2010). Clustering Stability: An Overview. arXiv:1007.1075.
- Yang, Kautz & Hatamizadeh (2025). Gated Delta Networks. ICLR. arXiv:2412.06464.
