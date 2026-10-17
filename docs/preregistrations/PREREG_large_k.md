# Pre-registration: the attractor lens at large K (1,024 to 4,096)

Written 2026-09-30, **before** any partition with K > 512 was fit on LLM features. Originally private notes, published 2026-10 unchanged except for machine names, paths and references to unpublished studies.

## Question
All results so far use K ≤ 512. Does going to thousands of cells help, and does it need more data? Three uses are
tested: label-free layer selection (rule S), a cell-majority "big dumb classifier", and hierarchy resolution.

## Data
- **A. Existing caches** (N_train = 24k DBpedia / 11.3k 20NG), all 14 model × dataset pairs (3 dev + 11 held-out).
  K ∈ {512, 1024, 2048}. K = 4096 is excluded here: fewer than 6 train points per cell.
- **B. New large caches:** Qwen3.5-2B and Gemma 4 E2B (the base model, as in the main study) on DBpedia, **N_train = 120k**
  (same seed-0 shuffle, so the old 24k are its first rows), the same 6k test, every layer, same dtype as the
  original caches (Qwen3.5-2B fp32, Gemma 4 E2B fp16) on P100.
  Evaluated at N ∈ {24k, 120k} (the first rows of the same cache) × K ∈ {512, 1024, 2048, 4096}.
- Frame: whiten-64 as in rule S (fit on ≤ 12k train rows). K-means: faiss, niter 50, nredo 1, seeds 0 and 1, fit on train.
  All scores on test.

## Measurements (per layer, per N, K)
- **Stability:** NMI(seed 0, seed 1) of the test assignments. Rule S_K = argmax over layers; ties go deeper.
- **Retrieval regret** of S_K (cosine p@10, primary label DBpedia l2 / 20NG fine), as in PREREG_layer_selection.
- **Cell classifier:** each seed-0 cell is labeled with its majority train label; the test accuracy (l2 and l3) is
  compared with 10-NN majority vote (cosine) and a linear probe (logistic regression on whiten-64 features, 20k train
  rows), at each model's rule-S_512 layer and at its last layer.
- **Hierarchy:** NMI of the cells with DBpedia l3 (219 classes) per K. Nestedness at the S layer: the share of K-cells
  whose train members fall ≥ 90% into a single cell of the K/2 partition (1024 in 512, 2048 in 1024, 4096 in 2048).

## Hypotheses and criteria
- **K1 (rule S at larger K, set A).** Prediction: neutral. Mean retrieval regret over the 11 held-out pairs, S_K vs S_512
  (0.0049), for each K ∈ {1024, 2048}:
  - *Helps:* lower by > 0.002 **and** regret ≤ S_512's on ≥ 8 of 11 pairs.
  - *Hurts:* higher by > 0.005.
  - *Neutral:* otherwise.
- **K2 (data scaling, set B).** At fixed K, stability rises from N = 24k to N = 120k at every layer, for K ≥ 1024.
  At N = 120k, S_4096's regret is ≤ S_512's on both models. (With 2 models this is exploratory with a stated direction.)
- **K3 (classifier).** Cell-classifier accuracy rises with K and approaches 10-NN. It stays **below** the linear probe
  on l2 at every K. *Surprise* if it beats the probe on either model.
- **K4 (hierarchy).** NMI with l3 rises with K up to at least K = 1024 at N = 120k (219 classes). Nestedness at the S
  layer is ≥ 0.8 for 1024-in-512 (the scales form a near-tree without being forced to).

## Not tested here
Token-level partitions (the mechanistic-interpretability direction) are a separate study, run only if this and the
viewer look promising.

## Amendment 1 (2026-09-30 ~08:35, before any set A result beyond Qwen3-1.7B layers 0–14, a dev model)
To save compute (set A was running at ~10 h), set A is limited to what K1 needs. It covers the 11 held-out pairs only, and
measures stability at K ∈ {1024, 2048} only. The K=512 stability and cosine p@10 are taken from the frozen `<model>__select.json`
results (identical frame and seeds; spot-checked to match exactly). The classifier, hierarchy and data-scaling measurements
(K2–K4) come from set B only. The partial dev run on Qwen3-1.7B is discarded. Order: set B, then set A.
