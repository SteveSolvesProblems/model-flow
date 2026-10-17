# Pre-registration: label-free layer selection by fine-scale partition stability

Written 2026-09-29, **before** any held-out features were extracted. Originally private notes, published 2026-10 unchanged except for machine names, paths and references to unpublished studies.

## Hypothesis
In a frozen decoder LLM, the layer whose fine-scale K-means partitions reproduce most reliably across
random seeds is a good layer for embedding-based retrieval, and a better one than the last layer.
Mechanistic reading: stable fine partitions mean reproducible local neighbourhoods, which is what kNN
retrieval uses.

Found on the dev set (post hoc, choosing K from {8, 32, 128, 512}): Qwen3-1.7B, Qwen3.5-2B and Gemma 4 E2B on
DBpedia. The K=512 rule selected the oracle layer for 2 of 3 models and beat the last layer for all 3.

## Rule S (frozen)
For each layer output (hidden_states[0..L], embeddings included), with mean-pooled document features:
1. Frame: center with the train mean, then PCA-whiten the top 64 PCs, fit on up to 12,000 train rows (`lens.frame`).
2. Run faiss K-means with **K = 512**, niter = 50 and nredo = 1, once with seed 0 and once with seed 1, fit on train.
3. Stability = NMI between the two seeds' assignments of the **test** documents.
4. **Selected layer = argmax stability.** Ties go to the deeper layer.

## Outcome (labels used only here)
kNN retrieval precision@10 in cosine space (centered, L2-normalized raw features). Queries are the test split; the
database is the train split.
- Primary label: DBpedia **l2** (70 classes); 20 Newsgroups **fine** (20 groups).
- Secondary label: DBpedia l3; 20 Newsgroups coarse (7 prefixes: alt, comp, misc, rec, sci, soc, talk).
- **Regret** = oracle p@10 (best layer) − p@10 of the selected layer.

## Held-out evaluation pairs (none seen before this document)
- New models × both datasets: Qwen3.5-0.8B, Qwen3.5-4B, Gemma 4 E4B, SmolLM2-1.7B × {DBpedia, 20 Newsgroups}.
  These are 8 pairs. Gemma 4 E4B may not fit on a P100 in fp16; if it fails, it is dropped and the drop is reported.
- Dev models × new dataset: Qwen3-1.7B, Qwen3.5-2B, Gemma 4 E2B × 20 Newsgroups. These are 3 pairs.
- DBpedia sampling is the same as the dev set (24k train / 6k test, seed 0). 20 Newsgroups uses the full splits
  (11,314 / 7,532), max 64 tokens, mean-pooled.

## Label-free baselines
- **Last** layer, and **middle** layer (L // 2).
- **RankMe** (Garrido et al. 2023): exp(entropy) of the normalized singular values of the centered train
  features (≤10k rows). Its direction (argmax or argmin) is fixed from the dev set **before** held-out results
  are read, and recorded below.
- Out of scope, stated as a limitation: Skean et al. 2025's prompt entropy, curvature and augmentation-invariance
  metrics. They need token-level states or augmentations that these document-level caches don't have.

## Success criteria (all on held-out pairs, primary label)
1. S's mean regret is lower than the last layer's mean regret, **and** S beats the last layer in ≥ 75% of pairs.
2. S's mean regret is ≤ the mean regret of RankMe (dev-fixed direction) and of the middle layer.

- **Pass both:** fine-scale stability is a useful label-free layer selector.
- **Pass 1 only:** it beats the default, but not a trivial baseline.
- **Fail 1:** the lens is descriptive only.

## Secondary pre-registered hypothesis (architecture)
In Qwen3.5 hybrids, cross-seed-normalized partition churn at K=32 is higher after Gated-DeltaNet
(linear-attention) blocks than after full-attention blocks (dev: 0.077 vs 0.036 in Qwen3.5-2B). Test on
Qwen3.5-0.8B and Qwen3.5-4B, both datasets, with a document-bootstrap 95% CI on (linear − full) churn.
Supported only if the CI excludes 0 in at least 3 of 4 held-out pairs.

## Addendum (filled 2026-09-29 ~21:00, before any held-out selection was run)
- **RankMe direction fixed from dev: argmin.** Mean dev regret at L2 was 0.177 for argmin vs 0.194 for argmax. Both are
  far worse than every other selector; argmin wins only through Qwen3.5-2B (0.033).
- Dev regrets for reference (DBpedia l2): S = 0.000 / 0.029 / 0.000 (mean 0.010); last = 0.033 / 0.042 / 0.089 (0.055);
  middle = 0.066 / 0.064 / 0.002 (0.044), for Qwen3-1.7B / Qwen3.5-2B / Gemma 4 E2B.
- Dev churn (Qwen3.5-2B, K=32): linear − full = 0.040, 95% CI [0.036, 0.043].
- Held-out status at the time of writing: one extraction finished (Qwen3-1.7B × 20 Newsgroups). No held-out
  selection or retrieval numbers had been computed.
