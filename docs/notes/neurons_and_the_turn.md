# Do Gemma 4 E2B's MLP neurons draw the turn? (exploratory, plan fixed 2026-10-17 before any run)

Prompted by Cammarata, "Neural Networks Are Already Clean" (Goodfire, 2026-10-15): most MLP neurons in modern
language models do one main job, neurons draw distinctions between a positive and a negative pole, and a search over
rotations of a block's activations lands back on the neurons. Our Model Flow results never use neurons: clouds, k-means
cells and the coordinate turn are basis-free. This note asks whether the two pictures meet at Gemma 4 E2B's turn
(layers 13-14, whose MLPs write vectors 14x and 40x the residual; full MLP mean ablation moves the turn to later layers).

Neuron = input to `down_proj` at layers 13 and 14 (gelu(gate x) * up x, 6,144 per layer; can be either sign).
Model google/gemma-4-E2B, bfloat16, RTX 3060. Code: `diag/neuron_cells.py`. Output: `diag/results/neuron_cells.json`.

## Tests and predictions (written before running)

**A. Do single neurons draw our cells?** 6,000 DBpedia test articles (the cached set; max 64 tokens; mean over tokens).
Cells: k-means (K = 8 and 64) in the whitened 64-PC frame of the residual at the layer's output (and, as a check,
its input). For each cell, the best single axis by two-sided AUC (cell vs rest, either pole). Neuron axes against 5
random orthogonal rotations of the same centered activations; same for the nine topics.
Prediction A: neuron axes beat rotations by >= 0.05 mean best-axis AUC at both layers for K = 8.

**B. How many neurons carry the turn's write?** Per token (100 texts, 2,000 positions), the MLP write after its
post-norm, split into per-neuron contributions; the fewest neurons whose sum reaches 50% / 90% of the write's energy.
Neurons ranked globally by mean contribution energy on separate texts (tok_meta eval texts 100-129).
Prediction B: the median token reaches 50% with <= 64 neurons (about 1%), and the global top 64 carry >= 50% of the
energy on the evaluation tokens (a shared set, not a different one per token).

**C. Causal.** Replace the top-K ranked neurons at layers 13 and 14 together with their means (K = 16, 64, 256, 1024),
against K random neurons (seed 0) and the whole MLP. Centered cosine across layers 13 -> 15 (clean about 0.006), 12 -> 16,
adjacent layers, KL and top-1 agreement, on the published block-ablation positions.
Prediction C: top-64 raises the 13 -> 15 centered cosine to >= 0.2; random-64 leaves it < 0.05. If the turn instead
moves to later layers (as with the whole MLP), that counts against "few neurons carry it" only if 12 -> 16 also stays low.

**D. What are the turn neurons?** For the top 8 per layer: the 5 articles at each pole, and the topic purity of the
top 50 at each pole. Descriptive only.

Sanity checks: residuals from this pass match the cached gemma4_e2b layers (mean cosine > 0.99); clean turn
reproduces the published block-ablation value.

## Follow-up E: does layer 12's scalar set layer 13's turn? (stated 2026-10-17, before running)

A layer's own scalar cannot turn it (centered cosine ignores a uniform scale); a scalar shrinks the stream that the
next layer's write lands on. Layer 13's turn (centered cosine across it 0.081) is untouched by its own scalar.
Test: `block_ablation.py --only scalar1_12 scalex0.5_12 scalex2_12 scalex4_12 --tag l12` (CPU, same 100 texts).
Prediction E: if layer 12's scalar sets it, the centered cosine across layer 13 rises with the factor (x2, x4, set to 1)
and falls at x0.5, monotonically. If it stays near 0.08 throughout, layer 13's write is large enough on its own.

## Macro -> micro: three links (stated 2026-10-17, before running; exploratory). Code: `diag/macro_micro.py`

Gemma 4 E2B, bfloat16; neuron = input to down_proj at every decoder layer (6,144 wide at 0-14, 12,288 at 15-34);
blocks (heads_vs_blocks.json, decoder layers): 0 | 1-2 | 3-12 | 13 | 14 | 15-34.

**M1. Do flow-map splits show up as neuron poles?** Groups: the flow map's own construction (sankey_validity.py:
pca64 frame fit on the 24,000 cached train articles, consensus of 20 k-means runs, average linkage cut at K = 8; K = 4
reported too) on the first 1,500 test articles, at the residual entering decoder layer l (hidden l) and leaving it
(hidden l+1). A split at layer l: a group at hidden l whose members land in >= 2 groups at hidden l+1, each holding
>= max(30, 20%) of it; its two largest children are the pair. Score: best single-neuron two-sided AUC separating the
pair (parent members only) using layer l's neurons (mean over tokens), against the mean of 5 random rotations of the same
activations, and against layer l-2's neurons on the same pair.
- M1a: layer-l neurons beat the rotations in >= 70% of splits.
- M1b: layer-l neurons beat layer-(l-2) neurons in >= 60% of splits (the distinction is new where the split happens).
  Widths differ across layer 15 (6,144 vs 12,288 neurons); M1b is also reported with those splits left out.

**M2. Label once per block?** For 500 random neurons at layer l, the best-matching neuron at layer m by |correlation|
of article activations (1,500 articles), sign-aligned. Read vector of a neuron: its gate_proj row times the
pre-feedforward norm weight (a direction in the 1,536-d residual). Pairs: within block 3-12 (4-6, 6-8, 8-10, 10-12),
within 15-34 (17-19, 20-22, 25-27, 30-32), across the turn (11-15, 12-15, 12-16). Also the read vector mapped through the
orthogonal Procrustes rotation from hidden m to hidden l (fit on 10,000 centered, unit-length train articles).
Random neuron pairs give the baseline (about 1/sqrt(1536) = 0.026).
- M2a: within-block matched read vectors: median |cos| >= 0.2 (>= 4x the random baseline).
- M2b: across the turn, raw median |cos| <= half the within-block median.
- M2c: across the turn, the Procrustes-mapped median |cos| >= 2x the raw one.

**M3. Predict neuron importance from sizes, without ablating.** 200 random neurons at each of layers 8, 13, 14, 22.
Measured: mean KL(clean || ablated) over all positions of 50 tok_meta eval texts, each neuron mean-ablated alone
(means from eval texts 100-129). Predictors, on the same tokens: (i) mean |a_i|; (ii) mean write size
|a_i| |w_i| / rms(MLP output) after the post-norm; (iii) (ii) divided by the norm of the residual entering the layer.
- M3: pooled over the 800 neurons, Spearman((iii), KL) >= 0.5 and higher than Spearman((ii), KL) (the stream size,
  a macro quantity, adds to a per-neuron prediction). Within-layer correlations reported.
Result E (results/block_ablation_gemma4_e2b_l12.json): layer 12's scalar is 0.31. Centered cosine across layer 13:
x0.5 0.049, as trained 0.081, x2 0.147, set to 1 (x3.2) 0.227, x4 0.277 (monotonic: supported). Across layer 12 itself:
0.411 in every condition. Even at x4 layer 13 still turns more than ordinary layers (0.4-0.6): the scalar sets part
of the turn, the size of layer 13's write the rest.

### Run 1 problems and run 2 changes (2026-10-17, before run 2; post hoc, design fixes)
- M3 run 1 is invalid: every single-neuron KL was 0.0010-0.0012 (floor 0.00101) because the clean reference ran at
  batch 1 and the ablations at batch 4 (bf16 kernels differ). Run 2 takes the clean reference from the same batch shape
  and records a noise check (KL between two identical rows, expected 0).
- M2 run 1: random read-vector pairs were not near 0.026 (0.03-0.18): read vectors share a common component. Run 2
  subtracts each layer's mean read vector and adds a random baseline after the rotation.
- M1 run 1 passed as stated (K = 8: 24 splits, 75% beat rotations, 75% beat layer l-2; K = 4: 29 splits, 93%, 69%), but
  both criteria have split-independent explanations (neurons beat rotations everywhere, test A; layer l is closer in depth
  to the groups than layer l-2). Run 2 adds a control: up to 2 pairs per layer of output groups that were already apart
  at the input (different majority parents), scored the same way. Splits support "new neurons draw new distinctions"
  only if their gain over layer l-2 exceeds the control's. Caveat: control pairs are whole groups, easier to separate
  (AUC ceiling).

### Run 2 results (2026-10-18; results/macro_micro.json, .log)
**M1** (K = 8: 24 splits, 66 control pairs; K = 4: 29 splits). M1a 75% / 93% beat rotations, but the control pairs
beat them too (89% / 83%), so M1a says nothing about splits. M1b 75% / 69% (same width 71% / 67%), control 52% / 53%.
Gain of layer l over layer l-2 in best AUC: splits +0.018 / +0.016, control +0.003 / +0.003. Supported, small: where
the flow map splits a group, layer l's neurons separate the two halves better than neurons two layers back, and
pairs that were already apart show no such gain. Caveat from run 1 still applies (control pairs are whole groups).
**M2** fails. Matched neurons correlate (median |r| 0.28-0.41), but their centered read vectors are no closer than
random pairs at any pair of layers (raw 0.027-0.18 vs random 0.024-0.16; mapped vs mapped-random the same). The random
baseline still varies 6x across layers after centering, so read vectors stay anisotropic. M2a, M2b, M2c not supported:
correlated neurons in neighbouring layers do not read the same direction, inside a block or across the turn.
**M3** fails, and is still noise-limited. Identical rows now give KL exactly 0, but single-neuron ablations still sit
on a floor (median 0.00095 at layer 8, 0.00088 at 13, 0.00086 at 14, 0.00076 at 22; 90th percentile within 5% of the
median): any change to a row, however small, spreads bf16 rounding differences through the later layers. Spearman
pooled: act 0.20, write 0.29, write_rel 0.30 (needed >= 0.5); within layers 0.34-0.54, with write_rel no better than
write. A fair test needs float32 (does not fit the 3060 for E2B; CPU, hours) or larger ablations than one neuron.

### Blind labelling (2026-10-18; results/blind_units.md, blind_key.json, blind_raters/)
32 E2B units (layers 13 and 14: the 8 top turn neurons by write variance and 8 random neurons each), 6 top tokens per
pole from test D, shuffled and unlabelled. Two independent agent raters (Opus, Sonnet) labelled each pole and scored
it 1-5 for how consistently all 6 examples fit, plus "one main job". Unblinded after both finished.
- Mean pole score, turn vs random: rater 1 3.75 vs 2.97, rater 2 3.22 vs 2.53 (one-sided Mann-Whitney p 0.014, 0.022).
- "One main job": rater 1 9/16 vs 3/16, rater 2 5/16 vs 2/16. Raters agree on 25/32.
- The best pole of a random neuron is often readable too (mean 3.9 / 3.4 vs 4.4 / 3.8 for turn neurons).
Reading: turn neurons are more readable than random ones, but "cleanly two-poled" holds for a minority (5-9 of 16),
and only on the 6 strongest tokens per pole. Use "readable" in the addendum, not "clean".

## M3g: importance from sizes, with groups (stated 2026-10-18, before running). Code: `diag/m3_groups.py`
Same layers (8, 13, 14, 22) and 50 eval texts. Each ablation mean-replaces a group of 128 neurons at one layer, to rise
above the bf16 floor. 48 groups per layer are contiguous bins of neurons sorted by mean |a| on texts 100-129 (so the
bins span small to large on a micro-only quantity, not on the tested predictors), plus 8 random groups. Predictors
(act, write, write_rel) summed over the group on the 50 eval texts.
- M3g: pooled over the 4 layers (224 groups), Spearman(write_rel, KL) >= 0.5 and higher than Spearman(write, KL).
  The pooled comparison is the macro question: within a layer write and write_rel differ only by token weighting.

## Q2: Qwen3-1.7B's layer-2 "turn" (stated 2026-10-18, before running). Code: `diag/qwen3_l2.py`
Centered-only probes showed a turn at Qwen3 layer 2 (0.18) that robust scaling removed (0.61): a few coordinates
switching on. Steve's question: the model does not normalize there, so is it real? Test at the output of layer 2:
massive dims (mean |x| > 20x median) zeroed, set to their mean, or held at their input value (switch-on blocked).
- Q2a: with the massive dims left out of the cosine, the clean model's centered cosine across layer 2 is >= 0.5
  (the rest of the stream does not turn).
- Q2b: holding them at their input value or zeroing them breaks the model (KL >= 0.5, top-1 <= 0.7); setting them to
  their mean does not (KL <= 0.05). Reading if both hold: the event is real and needed (the switch-on, as in Sun et al.
  2024's attention sinks), but it is a change in a few coordinates, not a turn of the whole cloud like Gemma's.

## S: review follow-ups to the addendum draft (stated 2026-10-18, before running)
A reviewer session read the draft. Points taken: (1) "spread out" needs a baseline; (2) define "half of the write"
(it is the vector residual ||w - partial sum||^2 / ||w||^2, neurons added largest first, cross terms included);
(3) centered cosine cannot see content, so "other texts' writes keep the turn" is partly guaranteed by the metric;
(4) a mean-write control. On (4): a constant write cannot change a centered cosine, and Gemma's post-norm fixes the
write's size per token, so the existing "all neurons -> mean" condition is that control (turn 0.029 E2B, 0.096 E4B).
- S1 (`conc_baseline.py`, E2B all 35 layers, Qwen3-1.7B all 28): k50 / width at layers 13-14 is not the smallest in E2B
  (the turn write is not unusually concentrated). Descriptive otherwise.
- S2 (`turn_company.py`, E2B 13-14): under resample and noise the 10-NN overlap of hidden 15 with clean hidden 15 is
  <= 0.2 while the turn stays (<= 0.03); under rand1024 it stays >= 0.6. If so: size sets the turn, content sets the
  company, and the cosine alone cannot tell them apart.
Result M3g (results/m3_groups.json): fails. Identical rows KL 0. Group KL now spans 0.0009-0.23 (above the floor at
the large end). Within a layer every predictor ranks groups well (Spearman 0.86-0.97; partly by construction, since
groups are contiguous bins of |a|). Pooled over layers: act 0.53, write 0.36, write_rel 0.34. Dividing by the stream's
size makes the cross-layer prediction worse, not better; plain activation size predicts best. The macro quantity
tested here does not help predict neuron importance.
Result Q2 (results/qwen3_l2.json; Qwen3-1.7B bf16, 3 massive dims 1401, 1793, 1999; mean |x| at layer 2's output 15,
215, 142 vs 2, 5, 12 entering it). Q2a supported: centered cosine across layer 2 is 0.158 on all coordinates and
0.766 with the 3 left out (neighbouring layers 0.73, 0.83). Q2b partly: zeroing them or holding them at their input
value breaks the model (KL 10.5 / 9.8 nats, top-1 0.13 / 0.16), but so does setting them to their mean (KL 10.0,
top-1 0.01), against the prediction. Design flaw, found after the run: the mean was taken over all tokens, so it puts
the first token's (attention-sink) values on every token. Rerun with the first token left alone and means over the
other tokens: `qwen3_l2.py --from-pos 1` (post hoc).
Reading so far: the event is the model's own and it matters (blocking the switch-on breaks the model), but it is a
change in 3 of 2,048 coordinates; the rest of the stream does not turn.
Result Q2 rerun (post hoc; results/qwen3_l2_from1.json): with the first token left alone, zeroing the 3 dims on every
other token costs little (KL 0.07, top-1 0.89) and mean or hold almost nothing (KL 0.01 / 0.006). Their values on the
first token are 781 / 12,486 / 7,821 against mean |x| of 15-215 over all tokens. So the event is the first token
becoming an attention sink (Sun et al. 2024): real and needed (the first run broke the model by blocking it), but it
lives in 3 coordinates of one token. The article-level "turn" at Qwen3 layer 2 is that token's values dominating the
token average; the other 2,045 coordinates do not turn.

Result S1 (results/conc_baseline_{e2b,qwen3_17b}.json; 80 tokens per layer, 8 from each of eval texts 0-9; the first
attempt with all tokens of 20 texts was stopped: fp32 weights for every layer spilled GPU memory). Median share of the
width needed for half the write: E2B layer 13 1.9% (the largest of all 35 layers), layer 14 1.0%, layers 9-30 mostly
0.3-1.4%, first and last layers 0.1%. Qwen3-1.7B: 1-2% in layers 3-15, 0.1-0.4% after. Supported: the turn write is
not unusually concentrated; it is about as spread as any middle layer's, in both families. "Spread out" is not
special to the turn and is dropped as a turn-specific claim.

Result S2 (results/turn_company_e2b.json): supported. Hidden 15's 10 nearest neighbours vs the clean model: resample
0.01, noise 0.01 (turn 0.005 / 0.003: as clean); rand1024 0.79; top1024 0.45; all-to-mean 0.22. Content sets the
company; size sets the turn. Also: in the clean model hidden 13 and 15 share 0.31 of their neighbours although their
centered cosine is 0.007 (chance overlap 0.005): the turn changes the axes, and a third of the company survives it.
After resample or noise the company partly recovers by layer 20 (0.37 / 0.34): later layers rebuild some of it from
the context.
