# Pre-registration: can a bin fingerprint catch pipeline faults, and does it scale with real damage? (2026-10-09)

Plan: PLAN_bins.md, next line 2 (Steve, 2026-10-09: line 2 first; drop Qwen (no `<bos>`); 8-bit quantization is not
assumed harmless; small models). Committed before any run.

Motivation: a missing `<bos>` changed Gemma 4's coordinate system drastically (tokens keep 0.21 of their neighbours
in layers 15–35; PREREG_null_followups.md P1) and was caught only through next-token accuracy, which needs a task.
Bins are general (they track how much a model moved, not which ability changed; an unpublished study), which is
what a regression test needs. The fingerprint's likely niche is representation pipelines (embeddings), where an
output-logit check does not apply or cannot see faults downstream of the logits.

## Models
Llama-3.2-1B (16 blocks; adds `<|begin_of_text|>`). Second model: Gemma-3-270m (18 blocks; adds `<bos>`); access is
pending Google's review; if it is not granted when the Llama runs finish, TinyLlama v1.1 (22 blocks; adds `<s>`)
is used instead, and the results say which.

## Probe and fingerprint
2,000 DBpedia test articles (first 2,000 of the shuffled test split, seed 0, as extract_hf.py), up to 64 tokens,
mean-pooled over non-pad tokens, residual after every block. Reference run R: float32, GPU, batch 32, the model's own
tokenizer with its default special tokens, documents in order.
Per block, R vs a run: `knn` = kNN10 overlap (centered cosine, each run's own neighbours); `bins` = NMI between R's
K = 64 assignment (whiten-64 frame and k-means seed 0 fit on R) and the run's states assigned to R's centroids.
Fingerprint distance **D = 1 − min over blocks of `knn`** (primary); secondary 1 − min over blocks of `bins`.

## Runs
Harmless (change only numerical noise): H1 batch 8; H2 CPU instead of GPU (float32); H3 documents in shuffled order
(states put back in order).
Graded (a real change of variable size): G1 bfloat16; G2–G4 weights of every linear layer in the blocks rounded to
8 / 6 / 4 bits (symmetric, per group of 32 along the input dimension, round to nearest; emulates GGUF-style
quantization without depending on GGUF loader support).
Faults (should be caught): F1 special start token removed; F2 the instruct model's chat template wrapped around each
article (user turn, generation prompt; fixed date string), then the 64-token cap; F3 cap of 16 tokens instead of 64;
F4 pad positions included in the mean (right padding, batch 32); F5 weights rounded to 2 bits as in G.

Yardsticks (label-free; computed for every run):
- `KL`: mean next-token KL(R || run) over the article's own tokens (positions aligned on the article tokens; for runs
  whose input differs, the article tokens present in both);
- `dNLL`: change in mean next-token negative log-likelihood of the article tokens (a cheap baseline detector).

## Predictions (each model separately)
- **PF-a (faults separate from harmless changes).** Every fault's D > 2 x the largest harmless D.
- **PF-b (scales with real damage).** Over the 7 harmless and graded runs, Spearman(D, KL) >= 0.8.
- **PF-c (sees what the output check cannot).** At least one fault has |dNLL| < 2 x the largest harmless |dNLL| and
  is caught by D (expected: F4, which leaves the logits unchanged by construction; reported as such).
- **Descriptive.** The block where `knn` is lowest, per run; D and KL for each graded run, so Q8-like rounding lands
  where its measured damage puts it.
Kill: PF-a fails in both models -> the fingerprint is not a usable regression test.

## Compute
Small models; 13 runs per model, each a forward pass over 2,000 x 64 tokens; gpu-host GPU 1 (private told first) and
its CPU for H2; writes only to data.

## Results (2026-10-09; gpu-host GPU 1; `diag/pipeline_faults.py`; `results/pipeline_faults_<model>.json`, logs `results/pf_*.log`)
Implementation notes (fixed before the full runs; a 64-document smoke test only): inputs are built from each article's
token ids (asserted equal to the tokenizer's own output for the reference); KL is computed against R's top-128
next-token probabilities plus one bucket for the rest (a lower bound on the full KL); F2 prepends the start token
when the template text has none, as the tokenizer would. All models: float32 GPU reference, eager attention.
Gemma-3-270m access was still awaiting Google's review, so TinyLlama v1.1 ran as the second model, as pre-registered.

**Llama-3.2-1B (clean): PF-a, PF-b and PF-c supported.**
| run | KL vs R | dNLL | D | D (bins) | block of min `knn` |
|---|---|---|---|---|---|
| H1 batch 8 / H2 CPU / H3 shuffled | 0 / 0 / 0 | ~0 | <= 0.00005 | 0 | — |
| G1 bf16 | 0.0005 | 0.0004 | 0.006 | 0.002 | 15 |
| G2 rtn 8-bit | 0.0007 | 0.0001 | 0.010 | 0.002 | 9 |
| G3 rtn 6-bit | 0.011 | 0.013 | 0.035 | 0.016 | 15 |
| G4 rtn 4-bit | 0.175 | 0.184 | 0.124 | 0.084 | 15 |
| F1 no start token | 0.180 | 0.184 | 0.556 | 0.210 | 1 |
| F2 chat template | 0.287 | 0.352 | 0.699 | 0.658 | 1 |
| F3 cap 16 | 0 | 0 | 0.811 | 0.748 | 1 |
| F4 pad in mean | 0 | 0 | 0.297 | 0.114 | 15 |
| F5 rtn 2-bit | 10.5 | 11.0 | 0.946 | 0.996 | 15 |
- PF-a: every fault D (min 0.30) > 2 x harmless D. **Weak as stated:** harmless D is ~5e-5, so any nonzero fault
  passes; the informative margin is the smallest fault (0.30) vs the 4-bit graded run (0.12).
- PF-b: Spearman(D, KL) = 0.96 over the 7 harmless and graded runs (4 of the 7 are near 0, three tied at 0; within
  the graded runs D rises with KL at every step).
- PF-c: F3 and F4 have KL = dNLL = 0 and are caught by D. **Partly by construction:** dNLL and KL are computed on
  the article tokens both inputs contain, so truncation and pooling faults are invisible to them by design. This
  shows the output check cannot see these faults, not that D is unusually sensitive.
- Steve's 8-bit question: 8-bit rounding is measurable but small, about the size of switching to bf16.

**TinyLlama v1.1: predictions pass formally, but the reference is itself faulty; not a clean replicate.** The
tokenizer's default adds `<s>`, but the checkpoint behaves as if trained without it: reference mean NLL 5.42 (Llama
2.13); a test sentence has NLL 6.35 with `<s>` and 0.91 without, and generation degrades with it. Accordingly F1 and
F2 *lower* NLL (dNLL -3.46 / -3.68; KL 3.08 / 3.35). D: harmless <= 0.0001; G1-G4 0.084 / 0.045 / 0.252 / 0.386
(bf16 above 8-bit); faults 0.67-0.94. PF-a supported; PF-b Spearman 0.85 (supported, barely); PF-c supported (F3, F4).
The choice of TinyLlama as fallback rested on the wrong assumption that it is a start-token model.

**What this supports.** D is a distance from a reference pipeline, not a health check: for TinyLlama it showed the
default and no-`<s>` pipelines were far apart (0.75) but only the absolute NLL showed which was broken. Used as a
regression test against a known-good pipeline, it separates the injected faults from harmless changes, orders weight
rounding by its output damage, and sees pooling-side faults that output checks cannot.

**Descriptive (not pre-registered): where D peaks.** In Llama-3.2-1B, input-side faults (F1-F3) peak at block 1 and
weight rounding (G3, G4, F5) at the last block (15); no-`<s>` and 4-bit weights have equal output KL (0.18 vs 0.17)
but different D (0.56 vs 0.12) and peak block (1 vs 15). TinyLlama agrees for weights (block 21) and for F2/F3
(block 0), but its no-`<s>` run peaks at block 13 (its reference is the faulty pipeline). A claim that peak depth
localizes the fault needs its own pre-registration and a clean second model.

**Open.** Gemma-3-270m (start-token model) is the clean second model once access is granted (~30 min on one P100).
