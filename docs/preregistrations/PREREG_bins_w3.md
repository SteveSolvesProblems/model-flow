# Pre-registration W3: where do bin boundaries become functional, and only for coarse bins? (2026-10-09)

Written and committed before any W3 measurement. Follows W2-b (PREREG_bins_w2.md): at matched distance, crossing a
K = 512 bin boundary changes the next-token output more than staying inside a bin in Qwen3-1.7B at blocks 7, 14, 21,
but in Gemma 4 E2B only at block 26 (block 9: no effect; block 18: the reverse). Method exactly as W2-b
(`diag/bins_w2b.py`: 1,000 same-bin pairs at positions 8–63, different-bin partner within ±2% of the distance in the
whiten-64 frame (primary) or raw coordinates (secondary), patch after block m, KL of the next-token distribution;
noise floor from self-patching), with the blocks and K below. Run on a gpu-host P100 (GPU 1; the token caches were
extracted there), default attention, model dtype as in the cache metadata.

## W3-a. Where does Gemma's switch happen? (depth sweep, K = 512)
Gemma 4 E2B blocks 9, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30; Qwen3-1.7B blocks 4, 7, 10, 14, 17, 21, 24 (control,
no turn). Effect = mean paired KL(different bin) − KL(same bin), whiten-64 matching, bootstrap 95% CI.
- **W3-a1 (Qwen3 control).** Effect > 0 with the CI excluding 0 at >= 6 of 7 blocks.
- **W3-a2 (Gemma has a switch).** There is a block b* such that the effect is > 0 with the CI excluding 0 at every
  sampled block >= b*, and not (CI includes 0 or effect < 0) at every sampled block < b*, allowing at most one
  exception; b* is reported. Prediction: b* exists and lies in 20–26 (from W2-b: after 18, by 26).
- **Descriptive.** b* is compared with Gemma's known landmarks: the turn (14), the start of key/value sharing (15) and
  its quiet late stretch (29–32).
Kill for "Gemma switches late": W3-a2 fails (no single switch) -> the Gemma pattern is not a depth transition.

## W3-b. Steve's hypothesis: bins capture only rough structure (K sweep)
At Gemma block 26 and Qwen3 block 14, K in {32, 512, 4096} (k-means seed 0, same frame).
Hypothesis (Steve, 2026-10-09): models learn coarse structure early and easily, and bins see only that, so fine
boundaries should be arbitrary relative to what the model uses.
- **W3-b1 (prediction of the hypothesis).** In both models: effect(K = 4096) < effect(K = 32), paired bootstrap of the
  difference (pairs are drawn independently per K, so the bootstrap is over the two sets separately) with the 95% CI
  of effect(32) − effect(4096) excluding 0; and the K = 4096 effect's CI includes 0.
- **Alternative.** Effects similar across K (all three CIs overlap, all > 0): k-means boundaries track structure the
  model uses at every scale tested.
- Otherwise: mixed, reported as such.
Note: at larger K same-bin pairs are closer; distance matching applies within each K.

## Compute and outputs
~22 configurations x ~4 min on a P100, ~1.5–2 h; private is told before the GPU is used. Script `diag/bins_w3.py`
(W2-b's code with blocks and K as arguments); results `results/bins_w3.json`.

## Results (2026-10-09; gpu-host GPU 1; `results/bins_w3.json`, log `results/w3.log`)
Noise floor negligible everywhere (self-patch KL <= 2e-5 Gemma, <= 4e-8 Qwen3). Gemma block 9 reproduces W2-b
(run on the local 3060) exactly: effect −0.165, same 710 pairs.
Effect = KL(different bin) − KL(same bin) at matched whiten-64 distance, 95% CI:
| Gemma 4 E2B block | 9 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| effect | −0.17 | +0.58 | +0.24 | **−0.30** | **−0.32** | +0.31 | +1.46 | +2.23 | +1.90 | +1.82 | +1.20 |
| CI excludes 0 | no | yes | yes | yes (neg.) | yes (neg.) | yes | yes | yes | yes | yes | yes |
| Qwen3-1.7B block | 4 | 7 | 10 | 14 | 17 | 21 | 24 |
|---|---|---|---|---|---|---|---|
| effect | +2.36 | +1.79 | +0.78 | +1.37 | +1.63 | +2.02 | +1.66 |
- **W3-a1 (Qwen3 control): supported, 7/7.** Crossing a boundary changes the output more at every depth.
- **W3-a2 (Gemma has a single late switch): killed.** No single switch: boundaries matter at blocks 12–14 (around
  the turn), the relation reverses at 16–18 (just after the turn and the start of key/value sharing at 15), and they
  matter again from block 20 on (largest at 22–28). The earlier "late switch" reading came from sampling 3 depths.
- **W3-b1 (Steve's hypothesis: bins see only coarse structure): not supported.** Effects by K (32 / 512 / 4096):
  Gemma block 26: +1.78 / +1.90 / +1.36 (all CIs exclude 0 and overlap; 32 − 4096 = +0.42, CI [−0.24, +1.06]);
  Qwen3 block 14: +0.91 / +1.37 / +1.57 (32 − 4096 = −0.66, CI [−1.32, −0.02]: finer boundaries matter *more*).
  The pre-registered alternative holds: k-means boundaries track structure the model's output depends on at every
  scale tested (32 to 4,096 cells). Caveat: at K = 4096 only 499 (Gemma) and 674 (Qwen3) of 1,000 pairs found a
  matched partner.
- **Open (descriptive, for the next round):** the reversed window at Gemma blocks 16–18: crossing a boundary there
  changes the output *less* than staying inside one, at the same distance.
