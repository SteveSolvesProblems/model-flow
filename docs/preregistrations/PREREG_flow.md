# Pre-registration: flow signatures, training dynamics at scale, and what chat tuning moves (F1–F4)

Written 2026-10-01, before any of the data below was extracted or scored (F1 uses some existing caches, but the
signature has never been computed). Steve approved the queue to run 24/7 on gpu-host.

**Framing.** Mechanistic interpretability traces individual components (classical mechanics). These studies measure
the bulk flow of a population of inputs through depth (fluid mechanics): how steady each layer's map is, how much
the maps mix between layers, and how the flow responds to training, tuning and quantization. The question is whether
these quantities line up with known qualitative properties of the models.

Shared definitions as in `PREREG_layer_selection.md` and `PREREG_overnight.md`: DBpedia (24k train / 6k test, seed 0)
and 20 Newsgroups samples, 64 tokens, mean pooling; frame W (whiten-64); K = 512; map stability = cross-seed NMI;
S = its argmax (ties deeper); regret in p@10 points against the best layer.

## F1. Flow signatures (CPU; existing DBpedia caches for 7 models, Pythia checkpoints, plus F2/F3 models when ready)
- **Signature of a model:** three per-layer curves, resampled by linear interpolation to 20 relative-depth points:
  map stability, adjacent-layer agreement a(ℓ) (as in E3), and the top-5 PC variance share. Each of the 60 features is
  z-scored across the models compared; distance is Euclidean.
- **F1-a (family).** Among the five DBpedia models that have a same-family partner (Qwen3.5-0.8B/2B/4B, Gemma 4
  E2B/E4B), the nearest other model by signature is from the same family for ≥ 4 of 5.
- **F1-b (development).** Across Pythia checkpoints from step 512, the signature's distance to the final checkpoint
  falls with training step: Spearman ≤ −0.9 for both 410m and 1.4b.
- Descriptive: a 2-D embedding (MDS) of all signatures, coloured by family, tuning and checkpoint.

## F2. Training dynamics at scale, across seeds and data (GPU; Pythia, same 10 checkpoints as E1, fp32)
New runs: `pythia-410m-seed1` … `seed4` (PolyPythia; seed 0 is the existing run), `pythia-410m-deduped`,
`pythia-1.4b-deduped`, `pythia-2.8b`.
- **F2-a (the monitor is robust).** Spearman(max map stability, best-layer p@10) across checkpoints from step 512
  is ≥ 0.7 in ≥ 6 of the 7 new runs.
- **F2-b (the best layer moves down).** The best layer at step 1,000 or 2,000 (the deeper of the two) is deeper than
  its median over steps ≥ 8,000, in ≥ 6 of 7 new runs.
- **F2-c (settling order).** Descriptive, with the seed spread: per-layer settle step (first checkpoint with ratio
  ≥ 0.9), and whether the ends settle before the middle in each run.
- Deduplication (410m and 1.4b vs their standard runs) and scale (410m → 1.4b → 2.8b): descriptive, no prediction.

## F3. What chat tuning moves (GPU; base vs post-trained pairs, DBpedia and 20NG)
Pairs: Qwen3-1.7B-Base vs Qwen3-1.7B; Qwen3.5-2B-Base vs Qwen3.5-2B; SmolLM2-1.7B vs SmolLM2-1.7B-Instruct;
Gemma 4 E2B vs Gemma 4 E2B-it (all from Hugging Face, fp32 except Gemma fp16 as before). Existing caches are reused
where they exist (Qwen3-1.7B, Qwen3.5-2B, SmolLM2-1.7B, Gemma 4 E2B). 8 pair × dataset cases.
- **F3-a (last-layer collapse).** The post-trained model's last-layer regret exceeds the base model's in ≥ 6 of 8.
- **F3-b (tuning moves the late layers).** Bin flips (share of test documents whose bin changes when the post-trained
  features are assigned to the base model's frame and bins, K = 512, as in the quantization study) are higher on
  average over the last quarter of layers than over the first half, in ≥ 6 of 8.
- **F3-c (the selector survives tuning).** In the post-trained models, S's regret is ≤ 1.0 point in ≥ 6 of 8.

## F4. A causal test of the word-identity rebound in Gemma (GPU)
Gemma 4 E2B feeds a per-layer token embedding into every decoder block; the token study found word identity
rebounding after layer 18 in Gemma but not in Qwen3-1.7B. F4 zeroes the per-layer inputs and re-measures the identity
curve. The intervention and its prediction will be added here as Amendment 1 after reading the model code and
before any ablated features are extracted.

## Order (two P100s)
GPU 0: F2 seeds 1–4, then 410m-deduped, then 1.4b-deduped. GPU 1: F3 extractions, then F4, then pythia-2.8b.
CPU: F1 on existing data now; recomputed with F2/F3 models when they land. Scoring runs as each extraction finishes.

## Amendment 1 (2026-10-01, F4 intervention; written before any ablated features were extracted)
**Code read** (transformers `models/gemma4/modeling_gemma4.py`): each decoder block ends with a per-layer-input
branch, `h ← h + norm(W_proj(act(W_gate h) ⊙ p_i))`, where `p_i` for block i is `(context_proj(input_embeds) +
token_lookup(input_ids)) / √2`. Both parts depend only on the token, never on context. Zeroing `p_i` makes the branch
add exactly zero (RMSNorm of zero is zero).
**Intervention:** patch `project_per_layer_inputs` to zero `p_i` for blocks i ≥ 18 (lens layers ≥ 19, i.e. after
the identity minimum at lens 18). Blocks 0–17 are untouched, so lens layers 0–18 are identical to the baseline.
A second run zeroes all blocks (i ≥ 0); it is descriptive, because removing every injection may damage the model
throughout. Same tokens, fit/eval sets and K = 512 as `PREREG_token_lens.md`; Gemma 4 E2B base, fp16.
**Baseline:** identity NMI falls to 0.45 at lens 18, then rebounds by 0.22 (Qwen3-1.7B: 0.03). Rebound = max of the
identity curve after its post-peak minimum − that minimum, as in `score_qwen_token.py`.
- **F4-a (primary).** With injections zeroed from block 18, the rebound is ≤ 0.08.
- **F4-b (dose).** The rebound with injections zeroed from block 18 is less than half the baseline's (< 0.11).
- Descriptive: the all-blocks run; next-word NMI under both ablations.

## Results: F1 (2026-10-01, `diag/flow_signature.py`, `results/flow_signature.json`, `results/flow_top5.json`)
- **F1-a not supported (3/5).** Nearest neighbours by signature: Qwen3.5-0.8B ↔ 2B, 4B → 2B (family holds), but both
  Gemma models' nearest is Qwen3.5-2B, not each other. Qwen3-1.7B and SmolLM2-1.7B are each other's nearest neighbour.
  They are from different families but are the two models whose mean-pooled features have massive-activation
  dimensions (top-5 PC share ≈ 1.0 at layers 3 and 8; ≤ 0.56 in the other five). Post hoc reading: the signature
  groups by this property more than by family. Not pre-registered; one pair.
- **F1-b supported.** Distance of each Pythia checkpoint's signature to the final one falls with training:
  Spearman −1.00 (410m) and −0.98 (1.4b).
- **Descriptive, Pythia anisotropy over training.** The top-5 share rises steadily in the middle of the stack:
  410m layer 12 goes 0.43 (step 4k) → 0.53 (16k) → 0.69 (64k) → 0.79 (143k); 1.4b layers 4–8 jump between 8k and
  16k (≈ 0.33 → 0.48) and reach ≈ 0.60–0.63 at the end. Concentration into a few directions keeps growing long after
  the maps "settle" (≈ 2–4k steps). Possibly the emergence of massive activations during training (Sun et al. 2024
  describe them in trained models); this is relevant to D2 (does a dominant direction hold the selector near the top?).

## Amendment 2 (2026-10-01): F5, a fine-tuned non-autoregressive encoder (Laya); written before extraction
Steve suggested Laya (`convaiinnovations/laya`, Apache-2.0, released 2026-09-18): a ModernBERT-large encoder
(28 layers, ~0.4B) fine-tuned with decision heads ("System 1" typed decisions; mean pooling in its config). Its
checkpoint stores the encoder under the prefix `encoder.`, which loads strictly into `answerdotai/ModernBERT-large`
(`extract_hf.py encoder --weights-from convaiinnovations/laya:model.safetensors:encoder.`). The pair (base
ModernBERT-large vs Laya's encoder) is a fine-tuning contrast like F3, in a bidirectional model. Same DBpedia and 20NG
samples, 64 tokens, mean pooling over all tokens (incl. [CLS]/[SEP]), fp32.
- **F5-a (fine-tuning moves the late layers).** Bin flips of Laya's features in the base model's frame and bins are
  higher on average over the last quarter of layers than over the first half, on both datasets.
- **F5-b (the selector works on an encoder).** Map stability's regret is ≤ 1.0 point in ≥ 3 of the 4 model × dataset
  cases.
- **F5-c (classification fine-tuning helps the last layer).** Unlike chat tuning (F3-a predicts a collapse), Laya's
  last-layer regret is lower than base ModernBERT-large's on both datasets.
- **F5-d (replication: encoders are less anisotropic; Razzhigaev et al. 2024).** The maximum over layers of the top-5
  PC variance share is lower for both encoders than the median of the seven decoders' maxima (DBpedia).

## Results: F5 (2026-10-01; `results/{mbert_large,laya}{,_20ng}__select.json`, `results/mbert_large{,_20ng}__ftflips.json`)
Laya's encoder weights loaded strictly (170 tensors, `convaiinnovations/laya@55cf4c4ebb`).
| | best layer (of 28) | S | S regret | last-layer regret |
|---|---|---|---|---|
| ModernBERT-large, DBpedia | 28 | 28 | 0.00 | 0.00 |
| Laya, DBpedia | 6 | 1 | 1.62 | 10.26 |
| ModernBERT-large, 20NG | 28 | 27 | 8.44 | 0.00 |
| Laya, 20NG | 5 | 13 | 5.67 | 8.73 |
- **F5-a supported, with a caveat:** flips are higher in the last quarter than the first half (DBpedia 0.98 vs 0.70;
  20NG 0.98 vs 0.92), but they are near total at most layers, which may be a global shift (re-centered flips are
  being computed, as for QAT).
- **F5-b not supported (1/4).** The selector misses by 8.4 points on base ModernBERT-large / 20NG and 5.7 on Laya /
  20NG.
- **F5-c not supported, in the opposite direction.** Laya's fine-tuning makes the last layer much *worse* for topical
  retrieval (regret 10.3 and 8.7, vs 0 for the base model, whose best layer is its last). The best layer moves from
  the top of the stack to layers 5–6, and best-layer p@10 drops (DBpedia 0.794 → 0.731). Post hoc reading: tuning for
  a narrow output (typed decisions here, chat in Gemma-it) repurposes the late layers, in an encoder too.
- F5-d: pending (top-5 share).

## Results: F2, interim (2026-10-01; `diag/score_f2.py`, `results/flow_f2.json`; 5 of 7 new runs)
Scored: pythia-410m seeds 1–4 and 410m-deduped. Pending: 1.4b-deduped (extracting), 2.8b (7 early checkpoints being
re-extracted after a tokenizer bug: early 2.8b revisions have no eos token; the tokenizer is now loaded from main).
| run | F2-a ρ(max stability, best p@10) | post-hoc ρ(step, best p@10) | F2-b best layer at step 1k/2k vs median ≥ 8k |
|---|---|---|---|
| 410m seed 1 | 0.88 | 0.85 | 22 vs 16 ✓ |
| 410m seed 2 | 0.88 | 0.95 | 22 vs 24 ✗ |
| 410m seed 3 | 0.95 | 0.88 | 22 vs 24 ✗ |
| 410m seed 4 | 0.98 | 0.73 | 22 vs 24 ✗ |
| 410m deduped | 0.82 | 0.87 | 21 vs 23 ✗ |
| *(E1) 410m seed 0* | *0.85* | *0.88* | *23 vs 17 ✓* |
| *(E1) 1.4b* | *0.93* | *0.97* | *21 vs 17 ✓* |
- **F2-a: 5/5 so far** (needs 6 of 7). The monitor tracks quality in every run, but it beats the step trend in only
  3 of 5; it is still mostly a training-progress signal (same conclusion as the E1 post-hoc control).
- **F2-b: fails** (1/5; cannot reach 6/7). The E1 "best layer moves down" came from seed 0, whose late best layer is
  layer 17; seed 1 peaks at 16, and seeds 2–4 and the deduped run end at the last layer (24).
- Unexplained, to examine: the seed 0 and seed 1 late peaks are single-layer spikes (seed 0: L16 0.76, **L17 0.84**,
  L18–24 ≈ 0.82–0.84; seed 1: **L16 0.85**, L17 0.82), not plateaus. Seeds can differ in one layer by ~8 p@10 points.
- Seed spread of S at the final checkpoint: layers 18–24 across the five seeds; oracle: 16, 24, 24, 24, 24.
- F2-c (settling order across seeds) needs partition codes for the new runs (`partition_codes.py`); not yet run.

### F2 update (2026-10-02): all runs scored; Pythia-2.8b data invalid and being re-extracted
- **1.4b-deduped:** F2-a ρ = 0.98 (step ρ 0.77); F2-b 22 vs 23 ✗.
- **Pythia-2.8b, first extraction: invalid.** All 10 checkpoints give identical features (same test_X md5). On the
  hub, 8 of the 10 branches share one stale `model.safetensors` blob (sha256 ab496f1c…), which transformers loads in
  preference to the per-checkpoint `pytorch_model.bin`; step64000 and step143000 also share both files on the hub.
  The other 8 Pythia runs were checked: 10 distinct weight blobs each. Re-extracting as `pythia28b_bin_s*` with
  `--no-safetensors`; every cache now records a weights fingerprint (`extract_hf.py`). Old caches kept, not used.
- **F2-a: 6/6 valid runs ≥ 0.7** (pass threshold 6 of 7 reached). The step control stands: the monitor beats
  ρ(step) in 4 of 6.
- **F2-b: fails** (1 of 6 valid; cannot pass).
- **F2-c (descriptive), settle step per layer (first checkpoint at ≥ 0.9 of the final re-seed NMI):** every run is
  settled by step 4,000; the first third of layers settles first (step 2,000; seed 4: 1,000) in all 7 410m/1.4b runs,
  the middle third at 4,000 (seed 4: 2,000). The last third is not consistent: 4,000 in 5 runs, 2,000 in seed 3 and
  seed 4. ρ(depth, settle step) ranges +0.12 to +0.76. So: "early layers settle first" replicates across seeds;
  "the ends settle before the middle" (E1) does not; it holds partly in seed 0, 2, 3 and 1.4b (last 2–4 layers at
  2,000). Checkpoint spacing (2k → 4k) limits the resolution.

## Amendment 3 (2026-10-02): F2-d, settling at finer resolution; Pythia-2.8b dropped
- **Pythia-2.8b dropped from F2.** On the hub, the `pytorch_model.bin` files of `pythia-2.8b` branches step0 and
  step64000 hold exactly the same tensors as step143000 (checked directly: `layers.5.attention.dense.weight`
  identical), and step0–16000 and 64000 all load to one weights fingerprint; only step32000 differs. The early
  checkpoints are not available in usable form, so the run cannot measure training dynamics. F2-a stands at 6/6 valid
  runs (threshold 6).
- **F2-d (new, approved by Steve, written before extraction).** Runs: pythia-410m seed 0 (standard), seed 1, seed 4
  (the earliest-settling seed). Added checkpoints: steps 128, 256, 3000, 5000, 6000, 7000 (all exist on the hub for all
  three; weights fingerprints are recorded and must differ). Settle step as in F2-c (first checkpoint at ≥ 0.9 of the
  final re-seed NMI), now over 0, 128, 256, 512, 1k, 2k, 3k, 4k, 5k, 6k, 7k, 8k, 16k, …, 143k.
  - **F2-d-a.** The median settle step of the first third of layers is strictly earlier than that of the middle third,
    in 3 of 3 runs.
  - **F2-d-b.** Spearman(depth, settle step) > 0 in 3 of 3 runs.
  - Descriptive: where the last third falls; whether any layer settles before step 512.

## Results: F4 (2026-10-02; `results/f4_score.json`, `results/tok_ple__token_lens.json` (base), `tok_ple_abl__token_lens.json`)
Word-identity NMI (K = 512): base minimum at layer 18 (0.45), then rises to 0.67 at layer 24 (rebound 0.22, as before).
With per-layer inputs zeroed from block 18: the curve follows base to layer 18, rises only to 0.55 at layer 24, and ends
at its minimum (0.44, layer 35). Zeroed everywhere: lower throughout, rise 0.07 after layer 18.
- **As registered and coded** (rebound = max after the global minimum minus the minimum): nople18 rebound 0 → F4-a
  (≤ 0.08) and F4-b (< half of 0.22) both pass. **But** the global minimum moves to the last layer, which makes the
  coded rebound 0 by construction.
- **Natural reading** (rise after layer 18): 0.22 → 0.10 (zeroed from 18) and 0.07 (zeroed everywhere). F4-b passes
  narrowly (0.10 < 0.11); F4-a fails (0.10 > 0.08).
- Conclusion: the per-layer embeddings cause most (about 55%) of the word-identity rebound, not all of it; a bump at
  layers 20–24 survives even with no per-layer inputs at all. Both readings are reported; the claim is stated as
  "most", not "the cause".
- Bug note: the first F4 scoring run skipped the ablated models (`token_lens.py` only iterated over keys named
  base/it/it2bit; fixed); a launcher bug passed a wrong GPU index on the first attempt (fixed).

## Results: F3 (2026-10-02; `diag/score_f3.py`, `results/flow_f3.json`, `results/<base>__chatflips_<post>.json`)
| post-trained | last-layer regret base → post | flips first half / last quarter | S, oracle, S regret |
|---|---|---|---|
| Qwen3-1.7B | 0.051 → 0.033 | 0.23 / 0.16 | 26, 26, 0.0 |
| Qwen3-1.7B 20NG | 0.080 → 0.064 | 0.44 / 0.43 | 25, 27, 0.6 |
| Qwen3.5-2B | 0.043 → 0.042 | 0.11 / 0.13 | 22, 20, 2.9 |
| Qwen3.5-2B 20NG | 0.012 → 0.012 | 0.24 / 0.31 | 23, 23, 0.0 |
| SmolLM2-1.7B-Instruct | 0.065 → 0.051 | 0.10 / 0.10 | 21, 23, 2.4 |
| SmolLM2 20NG | 0.058 → 0.059 | 0.33 / 0.28 | 23, 23, 0.0 |
| Gemma 4 E2B-it | 0.089 → 0.296 | 0.98 / 0.99 | 6, 6, 0.0 |
| Gemma 4 E2B-it 20NG | 0.031 → 0.131 | 0.99 / 0.99 | 0, 6, 8.4 |
- **F3-a (last-layer collapse): not supported (3/8).** Only Gemma's chat tuning makes the last layer much worse;
  in Qwen3 and SmolLM2 the post-trained last layer is equal or slightly better.
- **F3-b (tuning moves the late layers): not supported (5/8).** Outside Gemma, flips are similar in early and late
  layers; Gemma's chat model changes the bins of 98–99% of documents at every layer (raw flips; the whole-space shift
  was not removed here, so part of this may be translation, as for QAT).
- **F3-c (selector survives tuning): not supported (5/8).** Misses: Qwen3.5-2B (2.9), SmolLM2-Instruct (2.4),
  Gemma-it 20NG (8.4, S picks layer 0).
- Reading: the "chat tuning collapses the last layer" story (from Gemma) does not generalize; Gemma-it is the outlier.

## Results: F2-d (2026-10-03; `diag/score_f2d.py`, `results/flow_f2d.json`; 16 checkpoints per run, all fingerprints distinct)
| run | median settle step, thirds of depth | ρ(depth, settle) |
|---|---|---|
| seed 0 | 2,000 / 3,000 / 3,000 | +0.27 |
| seed 1 | 2,000 / 3,000 / 3,000 | +0.66 |
| seed 4 | 1,000 / 2,000 / 2,000 | +0.76 |
- **F2-d-a: supported (3/3).** **F2-d-b: supported (3/3).**
- With finer checkpoints, settling happens in a narrow window: every layer of every run is settled between steps 1,000
  and 3,000 (about 2–6 billion tokens); no layer except seed 4's embedding layer settles before step 1,000. Bottom-up
  order within that window; in seed 0 the last 4 layers settle with the first third (step 2,000), in the other seeds
  with the middle.

## Correction: F3 Gemma rows re-run with `<bos>` (2026-10-06)
The Gemma 4 E2B-it caches were extracted with the -it tokenizer, which adds no `<bos>` (NOTE_geometry.md,
2026-10-05). Re-extracted with the base tokenizer (same vocabulary, adds `<bos>`), same revision (3e22461), texts,
float16 and pooling; RTX 3060 with eager attention (SDPA in float16 overflows there) and the per-layer embedding
table looked up on the CPU (states match a float32 CPU run, cosine 1.0000). Old results: `results/nobos/`.
| post-trained | last-layer regret base → post | flips first half / last quarter | S, oracle, S regret |
|---|---|---|---|
| Gemma 4 E2B-it | 0.089 → 0.106 (was 0.296) | 0.93 / 0.64 (was 0.98 / 0.99) | 28, 17, 2.2 (was 6, 6, 0.0) |
| Gemma 4 E2B-it 20NG | 0.031 → 0.037 (was 0.131) | 0.95 / 0.90 (was 0.99 / 0.99) | 30, 28, 1.7 (was 0, 6, 8.4) |
- **F3-a: not supported (3/8, unchanged count).** Gemma's chat model is now only slightly worse at the last layer
  (1.7 and 0.6 points), not 20.7 and 10.0. The "Gemma-it is the outlier" reading above was the missing `<bos>`.
- **F3-b: not supported (3/8; was 5/8).** Gemma's chat tuning still changes most bins in the first half (93–95%),
  but fewer late (64% DBpedia).
- **F3-c: not supported (4/8; was 5/8).** Misses: Qwen3.5-2B 2.9, SmolLM2-Instruct 2.4, Gemma-it 2.2 and 1.7.
