# Pre-registration W1: is the stability–quality link a ghost? (2026-10-08)

Written and committed before the measurements below were computed. Plan: PLAN_bins.md, workstream 1.

## Background
Across the layers of one trained model, map stability (cross-seed NMI of K = 512 k-means partitions in the whiten-64
frame) and search quality (cosine kNN precision@10 on topic labels) rise and fall together: median within-model
Spearman 0.93 over the 11 held-out pairs (PREREG_null_followups.md, P2). Before explaining it, check whether it is a
generic property of k-means on these clouds rather than of what the model learned.

Disclosure: per-layer stability and p@10 for untrained Pythia checkpoints (step 0) were computed for the F2 study
(`results/pythia*_s0__select.json`), and selection regrets from them were seen. The within-run correlation between
stability and p@10, and its comparison to the trained checkpoint, have not been computed.

Definitions as `select_layers.py`: stability = NMI between the test-set assignments of K = 512 k-means fits with
seeds 0 and 1 (whiten-64 frame fit on train); quality = p@10 on the primary label (DBpedia l2; 20NG fine).
rho = Spearman over layers 1..L (layer 0, the embeddings, excluded).

## W1-a. Untrained null (primary; existing results, no new compute)
Pairs (step 0 vs step 143000, DBpedia): pythia410m, pythia410m_seed1..seed4, pythia410m_dedup, pythia14b,
pythia14b_dedup (8). Excluded before looking: pythia28b (its first caches share one stale weight file across
checkpoints; PREREG_flow.md Amendment 3) and pythia28b_bin (no final checkpoint). Validity check: the step-0 and
step-143000 caches' weight fingerprints must differ; a pair failing it is dropped and reported.
Measures per run: rho(stability, quality); the range of quality over layers 1..L.
- **W1-a1.** rho_trained > rho_untrained in >= 7 of 8 pairs.
- **W1-a2 (ghost criterion).** If the median untrained rho >= 0.70, the link is called generic (a ghost) and
  workstream 1 stops with that conclusion.
Reading, fixed in advance:
| untrained median rho | trained: rho >= 0.70 in >= 7/8 | reading |
|---|---|---|
| >= 0.70 | any | ghost: k-means on these clouds produces the link without training |
| < 0.70 | yes | training-specific: durable at this level; go to W1-b |
| < 0.70 | no | Pythia does not show the link; W1-a is uninformative and W1-b decides |
Untrained runs whose quality range is < 1 point are reported as "no quality variation"; they still count in W1-a1.

## W1-b. Noise ladder: is it signal-to-noise? (secondary; new CPU compute)
For each of the 11 held-out pairs (robustness.HELD): take the layer with the highest quality, X (train and test).
Make X_s = X + s x sigma x Z, sigma = sqrt(mean per-dimension variance of X_train), Z ~ N(0, I) drawn independently for
train and test (seed 0), s in {0, 0.5, 1, 2, 4, 8}. At each s compute stability and quality exactly as above. This
gives a curve q = f(stability) along which only the signal-to-noise ratio changes.
For each real layer l in 1..L whose stability lies within the curve's range, predict q_hat_l by monotone (piecewise
linear) interpolation of the curve; per pair, D = median |q_l − q_hat_l| over those layers (in p@10 points);
coverage = share of layers in range.
Decision rule (no directional prediction):
- **"Signal-to-noise explains the link"** if the median D over the 11 pairs <= 2 points and median coverage >= 0.6.
- **"Not signal-to-noise"** if the median D > 5 points.
- Otherwise inconclusive.

## W1-c. Bag-of-words reference (exploratory; no prediction)
TF-IDF on the first 64 whitespace words of each document (DBpedia and 20NG, same splits), truncated SVD to 512
dimensions (seed 0); stability and quality as above. Reported as a reference point: how many layers of each held-out
model exceed it on stability, on quality, and on both.

## Compute and outputs
W1-a: seconds (existing JSON). W1-b, W1-c: gpu-host CPU, ~2 h. Script `diag/bins_w1.py`; results
`results/bins_w1.json`.

## Amendment 1 (2026-10-08, before any W1 result was computed)
The Pythia caches were extracted before weight fingerprints were recorded in cache metadata, so the validity check
found no fingerprints and dropped every pair (no correlations computed). Replacement check, same purpose: the step-0
and step-143000 caches must have different hub revisions (recorded in meta) and their layer-12 test states must
differ (relative difference ||a − b|| / ||b|| > 0.5 on the first 500 documents). Checked: all 8 pairs pass (revisions
differ; relative differences 0.997–2.43 at layers 1, 12 and 24 where checked).

## Results (2026-10-08; `diag/bins_w1.py`, `results/bins_w1.json`)
**W1-a: training-specific (not a ghost).** All 8 pairs pass the Amendment 1 check. Untrained (step 0): rho(stability,
quality) = −0.72 to −0.95 (median −0.86); trained (step 143000): +0.77 to +0.96 (8/8 >= 0.70). W1-a1 8/8 (supported);
W1-a2 not met. In an untrained model the layers with the most repeatable bins are the worst for search; training
reverses the relation. Caveat: the deduped runs share step-0 weights with the standard runs (identical values), so
there are 6 independent initializations, not 8.

**W1-b: inconclusive** by the pre-set rule: median D = 3.7 points (2 < D <= 5), median coverage 0.96. By collection:
DBpedia pairs D = 2.2–3.2; 20NG pairs D = 3.0–9.5. Zero-noise values match select_layers.py within k-means run noise
(stability <= 0.004, quality <= 0.0002); the first run crashed on saving (a numpy bool) after computing everything;
its log showed one pair (gemma4_e2b_20ng, D = 7.29), and the re-run reproduced it exactly.
Exploratory (sign not pre-registered): on 20NG, 89–100% of layers lie *below* the noise curve, i.e. they are more
stable than their topic quality predicts (repeatable structure that is not topical; 20NG posts carry headers,
quoting and signatures); on DBpedia, layers lie on or slightly above it (signed median −0.1 to +3.2 points).

**W1-c (exploratory): bag-of-words reference.** DBpedia: TF-IDF/SVD stability 0.855, quality 0.799; layers above it
on both: Qwen3.5-0.8B 11/24, Qwen3.5-4B 28/32, Gemma 4 E4B 26/42, SmolLM2 4/24. 20NG: stability 0.681, quality 0.255;
most layers beat it (10/24 to 29/32 on both; 22/24–36/42 on quality). On DBpedia, word overlap alone is a strong
retrieval baseline that several models' layers do not beat.

Reading: the stability–quality link is learned (W1-a), so workstream 1 does not stop. Whether it is "just
signal-to-noise" is unresolved overall, and differs by collection: on 20NG, stability also tracks non-topical
structure.
