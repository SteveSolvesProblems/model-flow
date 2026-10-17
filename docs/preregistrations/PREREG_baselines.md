# Pre-registration: rule S against stronger baselines

Written 2026-09-30 ~19:30, **before** any of the baselines below was computed on any cache. Approved by Steve on
2026-09-30 (phases A and B).

## Question
Rule S (frozen in PREREG_layer_selection.md, unchanged) beat the last layer, the middle layer and RankMe on 11 held-out
pairs. Does it also beat the label-free layer scores from the recent literature? And how many labels is it worth: how
many labeled examples per class does a supervised layer choice need to match it?

**Status of the data.** Rule S's results on the 11 held-out pairs are already known. Every baseline below is new; none
has been computed on any pair. Nothing about rule S changes. Each baseline's direction (argmax or argmin) is fixed on
the 3 **dev pairs** (Qwen3-1.7B, Qwen3.5-2B and Gemma 4 E2B, all on DBpedia) before any held-out pair is scored,
exactly as RankMe's was. Where the literature states a direction, it is recorded, but the dev choice decides.

## Pairs, features, outcome
The same as PREREG_layer_selection.md:
- 14 caches: 3 dev pairs and 11 held-out;
- mean-pooled `hidden_states[0..L]`, 64 tokens;
- cosine p@10 of test queries against the train database;
- regret = oracle p@10 − p@10 of the selected layer;
- primary label DBpedia l2 / 20 Newsgroups fine;
- secondary label DBpedia l3 / 20 Newsgroups coarse.

## Phase A: from the existing caches (CPU)
1. **Intrinsic dimension (ID).** TwoNN [Facco et al. 2017] on 5,000 test documents (seed 0), with raw features,
   centered, duplicates removed, and the top 10% of μ ratios discarded [Facco et al. 2017].
   - Literature direction: **argmax.** Cheng et al. [2025] find that the high-ID "abstraction phase" holds the most
     transferable layers; see also Valeriani et al. [2023].
   - Also reported: the first local maximum of the ID profile that exceeds the median ID (descriptive).
2. **Dataset entropy** [Skean et al. 2025]: the matrix-based entropy with α = 1 [Giraldo et al. 2014] of the Gram
   matrix of 2,000 test documents (seed 0), centered and L2-normalized, with trace normalized to 1.
3. **Few-label selection (a supervised reference, not a rival).** Draw m labeled training documents per class
   (m ∈ {1, 2, 5, 10}; 20 draws, seeds 0–19). They are queries against the rest of the train set; pick the layer with
   the highest p@10 on those queries, then score its regret on the test set as usual.
   - Report the mean regret per m.
   - Report **m\***, the smallest m whose mean held-out regret is ≤ rule S's (0.0049).
4. **Uncertainty for rule S.**
   - For each pair, a paired bootstrap over test documents (2,000 resamples, seed 0) of p@10(S layer) − p@10(last
     layer), and of p@10(oracle) − p@10(S layer).
   - Across pairs, a two-level bootstrap (resample pairs, then documents within each) of the mean of
     p@10(S) − p@10(last).

## Phase B: token-level scores (GPU, computed during extraction; nothing large is stored)
Computed on the first 1,000 train documents of each dataset (the same shuffle), 64 tokens, dropping position 0 as in the
token pilot. They are run for all 7 models on both datasets. Numerics: fp16 on the P100s, with CPU offload for Gemma 4 E4B.
5. **Prompt entropy** [Skean et al. 2025]: for each document, α = 1 matrix entropy of the trace-normalized Gram matrix
   of its token representations; averaged over documents.
6. **Curvature** [Hosseini & Fedorenko 2023; Skean et al. 2025]: the mean angle between consecutive difference vectors
   (h_{t+1} − h_t) of a document's token trajectory; averaged over documents.

## Hypotheses (held-out pairs, primary label)
- **B1 (rule S vs the literature's label-free scores).** Rule S's mean regret is lower than that of each of ID,
  dataset entropy, prompt entropy and curvature (directions fixed on dev). Pass if it beats all four. Per-pair wins
  are also reported.
- **B2 (what rule S is worth in labels).** Descriptive, with a stated expectation: m = 1 has a higher mean regret than
  rule S, and m = 10 a lower or equal one; that is, m\* ∈ {2, 5, 10}.
- **B3 (uncertainty).**
  - The two-level bootstrap CI of the mean p@10(S) − p@10(last) excludes 0.
  - The per-pair CI lies above 0 in at least 8 of the 11 pairs, counting pairs where S and last are the same layer
    as not above.

## Not in this phase
New tasks, last-token pooling, 7–8B models (phases C and D, which need separate sign-off and disk space).

## References
- Cheng, Doimo, Kervadec, Macocco, Yu, Laio & Baroni (2025). Emergence of a High-Dimensional Abstraction Phase in
  Language Transformers. ICLR. arXiv:2405.15471.
- Facco, d'Errico, Rodriguez & Laio (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood
  information. Scientific Reports 7, 12140.
- Giraldo, Rao & Príncipe (2014). Measures of Entropy From Data Using Infinitely Divisible Kernels. IEEE Trans. Inf. Theory.
- Hosseini & Fedorenko (2023). Large language models implicitly learn to straighten neural sentence trajectories to
  construct a predictive representation of natural language. NeurIPS.
- Skean et al. (2025). Layer by Layer: Uncovering Hidden Representations in Language Models. ICML. arXiv:2502.02013.
- Valeriani, Doimo, Cuturello, Laio, Ansuini & Cazzaniga (2023). The geometry of hidden representations of large
  transformer models. NeurIPS.

## Amendment 1 (2026-09-30 ~19:15, before any phase B score was computed)
Phase B numerics follow each model's original cache instead of fp16 throughout:
- fp32 for Qwen3-1.7B, Qwen3.5-0.8B, Qwen3.5-2B and SmolLM2-1.7B;
- fp16 for Qwen3.5-4B and Gemma 4 E2B;
- fp16 with CPU offload for Gemma 4 E4B. Its caches were bf16 on the RTX 3060, and the P100 has no bf16.

A non-finite guard aborts any run that overflows.
