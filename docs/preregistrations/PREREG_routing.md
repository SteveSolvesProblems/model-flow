# Pre-registration: bins as routing keys for MLP activation sparsity (RT)

Written 2026-10-01, before any MLP activations were extracted. Approved by Steve ("anything we can do to save
compute ... is of huge interest"). Runs on gpu-host GPU 0 (P100, fp32 for Pythia), alongside the F2 queue; the local RTX 3060 was
planned but its WSL disk has no room for a CUDA environment.

## Question
If a token's bin at lens layer ℓ (its residual state entering block ℓ+1) chooses which MLP neurons of block ℓ+1 to
compute, how much of the MLP can be skipped, and at what cost in the model's next-token predictions, compared with
fair baselines at the same budget?

## Prior art (from memory; to verify before any writeup)
Contextual sparsity with a learned per-layer predictor (Deja Vu, Liu et al. 2023); clustering *neurons* into experts
(MoEfication, Zhang et al. 2022); training-free magnitude thresholding of activations (TEAL, 2024; CATS, 2024); the
"lazy neuron" sparsity of trained transformers (Li et al. 2023). This study's variant: a training-free router made of
k-means bins of the *tokens'* residual states, the same instrument as the flow maps. If it works, the bin decides the
neuron set before the MLP runs, so gate/up rows can be skipped too (unlike thresholding the activation after it is
computed).

## Design
- **Models:** Pythia-410m (GELU MLP, final checkpoint) and Qwen3.5-2B (SwiGLU). fp32 (Pythia) / fp16 (Qwen3.5-2B) on a P100.
- **Data:** DBpedia sample as elsewhere (seed 0), 64 tokens. Fit: 1,000 train documents, 25 random non-pad tokens
  each (25k tokens). Evaluation: 500 test documents, all non-pad tokens.
- **Activation of block i:** the input to the MLP's down projection (post-activation; for SwiGLU, act(gate)·up), m
  neurons.
- **Bins:** per lens layer ℓ, token residual states → frame W (center, top-64 PCs, whiten; fit on the fit tokens) →
  k-means (K = 512, and K = 64), seed 0. A token is routed by its nearest centroid.
- **Neuron sets at budget ρ ∈ {10%, 25%, 50%} of m:**
  - *bin*: the ρ·m neurons with the largest mean |activation| among fit tokens in that bin;
  - *static*: the ρ·m neurons with the largest mean |activation| over all fit tokens (global pruning, same budget);
  - *random*: a fixed random ρ·m set per block (seed 0);
  - *oracle*: each token's own top ρ·m by |activation| (an upper bound; needs the activation first, so it saves no
    gate/up compute).
- **Stage 1 (coverage):** for evaluation tokens, the share of the block's activation energy Σ a² inside the kept set,
  averaged over tokens and blocks.
- **Stage 2 (effect):** run the model with every block's MLP activation masked to the kept set (blocks 1..L−1, routed
  from lens layer ℓ = block index), and measure next-token KL to the unmasked model, per document, mean over
  positions.

## Predictions
- **RT1 (coverage).** At ρ = 10%, bin (K = 512) coverage exceeds static coverage by ≥ 0.10 (absolute), averaged over
  blocks, in both models.
- **RT2 (effect).** At ρ = 25%, bin-routed masking (K = 512) gives lower KL than static masking, with a paired
  bootstrap 95% CI over documents excluding 0, in both models.
- **RT3 (granularity).** K = 512 beats K = 64 on KL at ρ = 25% in both models.
- **Descriptive:** fraction of the static→oracle KL gap closed by bin routing; coverage by layer; the KL–budget curve.

## Results: Pythia-410m (2026-10-01, `results/binroute_pythia410m.json`)
| method | coverage@10% | KL@10% | coverage@25% | KL@25% | coverage@50% | KL@50% |
|---|---|---|---|---|---|---|
| oracle (token's own top set) | 0.633 | 8.19 | 0.753 | 5.08 | 0.906 | 1.13 |
| bin, K=512 | 0.452 | 8.07 | 0.608 | 6.59 | 0.782 | 3.19 |
| bin, K=64 | 0.401 | 7.40 | 0.578 | 6.57 | 0.766 | 4.13 |
| static | 0.267 | 7.96 | 0.450 | 6.70 | 0.681 | 4.52 |
| random | 0.098 | 7.65 | 0.250 | 6.84 | 0.491 | 4.78 |
- **RT1 supported:** bin routing captures 0.45 of activation energy at 10% vs 0.27 for static (+0.19).
- **RT2 supported, but at a broken operating point:** at 25%, bin KL is lower than static by 0.11 nats [−0.16,
  −0.06], but every method's KL is > 5 nats; masking all MLPs at 25% destroys this model.
- **RT3 not supported:** K = 512 vs 64 at 25%: +0.02 [−0.02, +0.06].
- At 50%, bin routing closes 39% of the static→oracle KL gap (4.52 → 3.19; oracle 1.13). Even the oracle loses
  1.1 nats at 50%: Pythia-410m's GELU MLP is not sparse enough for whole-model masking at these budgets.
- Next (to pre-register as Amendment 1): budgets 60–90%, masking a subset of blocks, and Qwen3.5-2B (SwiGLU).
