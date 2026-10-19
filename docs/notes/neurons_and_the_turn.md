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

## B: does the turn matter for behaviour? Dose control on ordinary layers (stated 2026-10-18, before running)
Prompted by a reviewer's "macro locates, component explains, intervention changes behaviour". The scalar doses already
record behaviour: KL(clean || dosed) and top-1 agreement are U-shaped around the trained value, while the turn moves
monotonically, so they show the model depends on its trained magnitudes, not that the turn drives behaviour.
Control: the same multipliers (x0.5, x2, x4, and x16, about the change of setting layers 13-14's scalars to 1) on the
scalars of two ordinary pairs, layers 8-9 and 24-25, and again on 13-14 in the same run (GPU, bf16, 100 texts).
`DIAG_DEVICE=cuda block_ablation.py --only ... --tag ctrl`.
- B1: at x16, KL at 13-14 is >= 3x KL at either control pair.
- B2: at x0.5-x4, KL at 13-14 is within 2x of the control pairs (small doses are not special).
Reading if B1 holds: removing the turn (x16 brings layer 14 near an ordinary layer) costs more than the same-size
change elsewhere, so the turn's size matters for behaviour beyond a generic magnitude effect. If KL is as large at
the control pairs, the behaviour effect is generic and the turn is not shown to matter.
Result B (results/block_ablation_gemma4_e2b_ctrl.json): B1 and B2 both fail, in the opposite direction. The turn
layers are the least sensitive to their scalars, not the most. KL (top-1) per dose, layers 13-14 / 8-9 / 24-25:
x0.5 0.03 (0.92) / 1.08 (0.55) / 0.73 (0.67); x2 0.04 (0.91) / 0.55 (0.70) / 1.01 (0.67); x4 0.59 (0.71) / 4.3 (0.18)
/ 5.7 (0.38); x16 3.4 (0.26) / 6.5 (0.09) / 9.5 (0.18). Layers 13-14 at x0.5 and x2 changed the 13-14 run's own values
little from the CPU run (KL 0.030 / 0.035 both), so GPU vs CPU does not matter here.
At the control pairs, growing the scalar makes the stream stop changing (adjacent centered cosine -> 0.99-1.0): later
writes become small next to the enlarged stream.
Reading: the behaviour cost of a scalar dose is generic and is smallest at the turn. One hypothesis (untested): after
a large overwrite the old stream is a small remainder, so rescaling it matters little downstream; the turn may
insulate later layers from the scale of what came before. The "intervention changes behaviour" step for the turn is
therefore not shown by scalar doses. A test that targets the turn itself, not the scale, is still needed (e.g. keep
every norm fixed and rotate the stream back to its pre-turn axes).

## B2 and U: reviewer follow-ups on B (stated 2026-10-18, before running)
Reviewer: report B as it came out; before reading it as "the turn protects later layers", check dose matching,
downstream position, and test the axes directly.
- Dose matching. The scalar multiplies the whole stream, so a multiplier is already the same relative change of the
  stream's norm at every layer. Absolute norms are similar too (mean token norm entering layers 10 / 15 / 20 / 26:
  48 / 58 / 61 / 71), so matching the absolute change moves control doses by at most ~20%: 8-9 x0.40, x2.19, x4.58;
  24-25 x0.59, x1.81, x3.44. Added: these matched doses, and the middle pair 18-19 at x0.5, x2, x4, x16.
- Before vs after: single-layer doses (x0.5, x2, x4) at layer 12 (before the turn), 15 (just after), and 8, 18
  (ordinary).
- B3 (protection idea): with matched doses, KL at 13-14 stays below every control pair, and layer 15 is less
  sensitive than layer 12.
- U (`turn_undo.py`): at layer 15's input, rotate every token but the first back to the pre-turn axes (Procrustes
  15 -> 13 fit on tokens of eval texts 300-599), mean and norms unchanged, vs 3 random rotations and the identity.
  Both rotations take the stream off its new axes; the undo one puts it on the old axes. Reading of the ratio
  undo KL / mean random KL: >= 0.5, the old axes are about as useless to later layers as random ones (later layers
  read the new axes); <= 0.2, later layers can largely read the old axes too. In between: partial.
  Caveat: the Procrustes map also moves content, not only axes, and its fit is imperfect (fit cosine reported).
Result B2 (results/block_ablation_gemma4_e2b_ctrl2.json; same GPU setup as B). Matched doses do not change the
picture: KL at 13-14 (x0.5 / x2 / x4: 0.03 / 0.04 / 0.59) stays far below 8-9 matched (2.77 / 0.78 / 4.89), 24-25
matched (0.35 / 0.67 / 4.62) and the middle pair 18-19 (0.21 / 0.31 / 2.45; x16 12.3 vs 3.4 at 13-14). So the turn
pair is the least sensitive pair, at matched doses and away from the output. B3's second part fails: single-layer
doses at layer 15, just after the turn (0.055 / 0.074 / 0.37), are no gentler than at layer 12 before it
(0.19 / 0.022 / 0.36) or at ordinary layer 18 (0.057 / 0.060 / 0.33); layer 8 is the most sensitive (0.19 / 0.12 /
1.58). Reading: the insensitivity belongs to the turn pair's own scalars, not to the layers after it. "The turn
protects later layers" is not supported; why scaling 13-14 costs so little stays open.

Result U (results/turn_undo.json): later layers read the new axes. Identity hook KL 0. Undo rotation KL 0.74
(top-1 0.61); three random rotations 0.66 / 0.75 / 0.70 (top-1 0.64 / 0.62 / 0.64); ratio 1.05. Putting the stream
back on its pre-turn axes, with sizes and the mean unchanged, hurts as much as a random rotation. Caveats: the undo
map fits only partly (mean cosine 0.57 on held-out tokens, vs 0.006 unmapped), so "the old axes" are approximate;
layers 15-34 reuse earlier layers' attention keys and values, so a rotation at layer 15 reaches their attention only
through queries, which may be why even a random rotation leaves top-1 at ~0.6. This is the end-to-end case the
reviewer asked for, on one model: the macro view locates the turn, the components explain its size, and changing
only the axes after it (sizes fixed) changes behaviour, no less than scrambling them.

## U2: is the axis dependence special to the turn? (stated 2026-10-18, before running)
Reviewer: any layer reads its input in a basis-specific way (norm gains, elementwise nonlinearities, massive
coordinates), so U may show something true of every layer; undo and random are indistinguishable at n = 3 (state it);
the undo map is partial (cos 0.57); layers 15-34 reuse earlier attention inputs, which dilutes a rotation there.
Agreed wording until this runs: "later layers depend on the axes the turn creates; comparison to ordinary layers
pending". Also: "turn pair least sensitive" (B, B2) holds on one model until E4B reproduces it; a testable
hypothesis for it: norm layers rescale the stream before every read, so a scalar mainly changes how later writes
compare with the stream, and at 13-14 that comparison may already be saturated by the huge writes.
Design (`turn_undo.py --layer L`, L = 15 turn; 8, 12 ordinary with their own attention inputs; 20, 25 ordinary
reusing them, like 15): at layer L's input, mix t of a map into the centered state (t = 0.25 ... 1), norm kept per
token; maps: Procrustes L -> L-2, a one-hidden-layer nonlinear map L -> L-2 (fit check), random rotations (3 at t = 1).
Each condition reports displacement (cosine to the original state), KL and top-1.
- U2a (turn-specific damage): at t = 1, random-rotation KL at layer 15 differs from all four ordinary sites by >= 2x.
  If not, the damage is generic basis dependence and U says nothing special about the turn.
- U2b (old axes useless): at layer 15, at matched displacement the undo map hurts as much as random (KL within 30%);
  at ordinary sites, where L-2 shares axes with L, the undo map hurts much less than random at the same displacement.
Result U2 (results/turn_undo_L{8,12,15,20,25}.json). Identity hook KL 0 at every site.
- U2a holds, in the unexpected direction: a random rotation at layer 15's input costs KL 0.66-0.75 (top-1 0.62-0.64);
  at the ordinary sites it costs 5.7-5.8 (layer 8), 6.2-6.4 (12), 4.7-5.1 (20) and 5.7-6.0 (25), top-1 0.02-0.12.
  So the stream just after the turn is about 8x less sensitive to its basis than any ordinary site, including 20
  and 25, which reuse attention inputs like 15. Basis dependence is generic; its weakness at 15 is turn-specific.
- U2b holds: at layer 15 the undo map hurts as much as a random rotation at every dose with matched displacement
  (e.g. t = 0.5: 0.15 vs 0.11; t = 1: 0.74 vs 0.70), and the better-fitting nonlinear map (cos 0.76 vs 0.57) also
  hurts as much (0.70). At ordinary sites, where layer L-2 shares much of L's axes (unmapped cos 0.27-0.46), the
  undo map hurts far less than random (ratios 0.16-0.59). Later layers get nothing from the pre-turn axes.
Reading: right after the turn, the old axes are useless to later layers, and the stream's axes matter much less
for the output than anywhere else tested. This fits B and B2 (scaling 13-14 is also cheap): the state entering
layer 15 carries less of what the output depends on than states elsewhere. Where that information travels instead
is open; layers 15-34 attend to keys and values computed at layers 13-14, which a rotation at 15 does not touch,
but layers 20 and 25 share the same inputs and are sensitive, so that alone does not explain it. One model; E4B
not tested. Wording for v0.3: "later layers do not read the pre-turn axes, and the stream right after the turn is
unusually insensitive to rotation"; "depend on the new axes" is withdrawn as too strong.

## U3: where does the rotation tolerance start and end, and does E4B show it? (stated 2026-10-18, before running)
`turn_undo.py --random-only` (identity + 3 random rotations at t = 1; layer mean from 40 texts).
E2B sites 13, 14 (inside the turn), 16, 17, 18, 22 (after it), with 8, 12, 15, 20, 25 from U2.
E4B (turn at layers 22-23): sites 12, 18, 24 (just after the turn), 26, 30, 36.
- U3a: in E2B the low sensitivity is confined to the turn: KL at 16-18 and 22 is >= 3x KL at 15. (Alternative: it
  fades gradually over several layers.) Sites 13 and 14 descriptive.
- U3b: in E4B, random-rotation KL at layer 24's input is <= 1/3 of the median of the other E4B sites.
Result U3a (results/turn_undo_rand_L*.json, with U2): random-rotation KL at the input of layer 8 5.7, 12 6.2, 13 6.4,
14 1.45, 15 0.70, 16 1.39, 17 2.09, 18 2.62, 20 4.9, 22 6.2, 25 5.9 (spread over 3 rotations <= 0.4). Not confined:
the tolerance starts after layer 13's write (input of 14), is deepest after 14 (input of 15), and fades over about
five layers (16-20). U3a fails as stated; the gradual alternative holds. The low-sensitivity window matches the
turn's own span (13-14) plus a recovery. Reading: for a few layers after the big writes, the output depends much
less on the stream's axes; later layers then build up a state the output depends on again.
Result U3b (results/turn_undo_e4b_rand_L*.json; E4B bf16, low-VRAM mode): random-rotation KL at the input of layer
12 4.9, 18 6.4, 24 (just after the turn) 2.9, 26 2.6, 30 5.6, 36 7.9 (spread over 3 rotations <= 0.3; identity 0).
Layer 24 is 0.52x the median of the other sites, not <= 1/3: U3b fails as stated. Qualitatively the same pattern,
weaker: the two sites after E4B's turn (24, 26) are about half as sensitive as the rest, against about one eighth in
E2B. Coarse sampling (24 and 26 only; the minimum may lie elsewhere). Reported as a partial replication.

## U4: is the rotation window a massive-coordinate artifact? (stated 2026-10-18, before running)
Reviewer: a norm-keeping random rotation spreads huge coordinates over all channels; inside the window the old
stream (with its huge coordinates and first-token sink) has just been overwritten, so rotation could hurt less there
because less of what norms depend on remains. Control 2 (exclude the first token) is already in every rotation run:
the hook leaves position 0 untouched. Control 1: `turn_undo.py --random-only --keep-massive` holds coordinates with
mean |x| > 20x the median (per site, non-first tokens) fixed and rotates only the rest; E2B sites 8, 12, 13, 14, 15,
16, 18, 20, 22, 25.
- U4: the window survives: KL at 15 stays <= 1/3 of the median of sites 8, 12, 20, 22, 25. If it shrinks toward the
  other sites, the post says the window is at least partly a huge-coordinate effect.
Result U4 (results/turn_undo_rand_keepmass_L*.json): the window survives. Coordinates held fixed per site (mean |x|
> 20x median on non-first tokens): none at 8, 12, 13, 14, 22; one at 15, 18, 20, 25; three at 16, so on tokens other
than the first Gemma 4 E2B has few huge coordinates anywhere. KL with them held: 8 5.7, 12 6.3, 13 6.4, 14 1.45,
15 0.69, 16 1.42, 18 2.6, 20 4.8, 22 6.2, 25 5.8, unchanged from U2/U3 (identical where nothing was held). With the
first token always untouched, the window is not an effect of huge coordinates or of the first-token sink.
Why predictions depend less on axes there stays open.

## KV1: does attention-input sharing make the window? (stated 2026-10-19, before running)
Outside review (NotebookLM, forwarded by Steve) asked for a control for Gemma 4's key/value sharing. From the model
configs: in E2B, layers 15-34 reuse the keys and values of layer 13 (sliding attention) or 14 (full attention); in
E4B, layers 24-41 reuse those of 22 and 23. So in both models the turn sits on the last two layers that compute their
own keys and values, and the rotation window starts exactly where sharing starts. A rotation at layer 15 or later
cannot reach any attention keys or values. Layers 20 and 25 also share and are sensitive, so sharing alone does not
make the window, but it may decide where it starts.
Design (`turn_undo.py --random-only --clean-kv`): at sites 8, 12, 13 and 14 (E2B), 3 random rotations at t = 1 as
before, and the same rotations with every key and value projection at or after the site computed from the unrotated
run (cached output of k_proj and v_proj in a clean pass). Only queries, MLPs and the stream see the rotation. Site 15
(nothing to clean) as reference. Run on a P100 in float16 with eager attention (no bf16 there), so both arms are
rerun in float16 and compared within this run, not against the bf16 numbers above.
- KV1a (sharing sets the start): with clean keys and values, KL at 13 and at 14 is <= 2x KL at 15.
- KV1b (sharing explains the window generally): with clean keys and values, KL at 8 and 12 is also <= 2x KL at 15.
Expectation, stated so it can fail: KV1a fails at 13 and KV1b fails, because a rotation before the turn also feeds
the turn's large MLP writes; clean keys and values lower damage somewhat at every site. Site 14 descriptive beyond
KV1a (only its own full-attention keys and values are rotated in the plain arm).

## N2: are turn neurons more readable than equally active ones? (stated 2026-10-19, before running)
Outside review: the 16 random neurons in the blind labelling were not matched on activation size (the post says so),
so the turn neurons' advantage may be salience. Turn neurons were chosen by write variance, var(a) |w|^2.
Design (`blind_matched.py`): same token sample, same 16 turn neurons. For each, a control from the same layer with the
nearest activation spread (std of a on eval texts 100-129), outside the top 64 by write variance, without reuse. So
controls match var(a) and differ in write size |w|. 48 units (16 turn, 16 matched, the old 16 random), reshuffled
with new ids; two fresh blind agent raters (Opus, Sonnet) with the same rubric (label and 1-5 score per pole, "one main
job"). Unblinded after both finish.
- N2 (holds if): for both raters, mean pole score of turn neurons > matched controls, one-sided Mann-Whitney p < 0.05.
  If it fails, the post says the turn neurons are no more readable than equally active neurons.
- Descriptive: the matched controls against the old random neurons (does activation spread alone buy readability?).
Deviation (2026-10-19, before any rating): the planned controls (outside the top 64 by write variance) could not be
matched: their activation spread was 0.12-0.75x that of their turn neuron. Excluding only the 8 selected neurons per
layer (`blind_matched.py --exclude 8 --tag blind2x8`) brings 9 of 16 controls within 0.75-1.02x; the other 7 turn
neurons (mostly spread 0.9-1.6) are the most active in their layer; their best controls reach only 0.23-0.42x. So
selection by write variance is close to selection by activation size, and for the most active turn neurons a
size-matched control does not exist. Raters see the blind2x8 set (48 units, run in float16 on a P100; the 16 turn
neurons are the same as before, 8 of 8 per layer reproduced). N2 is tested on all 16 vs 16 as stated, and
descriptively on the 9 pairs matched within 0.75x.
(Counts in this paragraph corrected after unblinding: it first said 10 well-matched pairs, a miscount of the
same ratios; the threshold 0.75x is unchanged.)
Result KV1 (results/turn_undo_kv1_bf16_rand_cleankv_L*.json; bf16 on a P100 with eager attention; the float16 runs,
_fp16_, overflowed to NaN KL on some tokens and are kept only as a record). Identity and identity with replayed keys
and values both give KL 0. Random-rotation KL, plain -> with clean keys and values (top-1 in brackets):
layer 8 5.70 (0.12) -> 0.19 (0.82); 12 6.06 (0.09) -> 0.26 (0.79); 13 6.33 (0.09) -> 0.30 (0.77);
14 1.59 (0.46) -> 0.42 (0.74); 15 0.70 (0.64), nothing to clean. Plain-arm values match the bf16 3060 runs.
KV1a holds and KV1b holds; the stated expectation failed. With keys and values protected, a rotation at any of these
layers costs less than the in-window rotation at 15. So the window is mostly attention-input sharing: from layer 15
on, a rotation cannot reach the keys and values later layers attend to (computed at 13 and 14), and those carry
enough to rebuild most predictions. At 14 only the full-attention keys and values are reachable, which is why 14 is
partly protected. Caveats: clean keys and values include each token's own, so the unrotated state reaches every
position through attention; this test says the damage travels through what attention reads, not that other tokens
supply the information. Why the damage still rises from 16 to 20, where keys and values are equally out of reach,
is open (fewer layers left after the rotation is one candidate, untested). The turn sits on the last two layers
that compute their own keys and values in both E2B and E4B (configs), so turn and window start are confounded by
design; the undo test (U) at 15 sits inside this protected region.

Result N2 (results/blind2x8_key.json, blind2x8_units.md, blind2_raters/; raters Opus and Sonnet, fresh, blind).
N2 fails. Mean pole score, turn vs matched vs old random: rater 1 3.47 / 3.47 / 2.81, rater 2 3.12 / 3.03 / 2.31;
turn > matched one-sided Mann-Whitney p 0.48 and 0.40. Matched > random p 0.018 and 0.009. On the 9 well-matched
pairs the mean pair difference is 0.00 and 0.17; on the 7 poorly matched pairs 0.00 and 0.00. "One main job": rater 1
9 / 7 / 7 of 16, rater 2 3 / 2 / 0. Raters agree closely (Spearman 0.83 on unit scores; one_job agreement 26/48).
Reading: the turn neurons are more readable than random neurons because they are more active; neurons of the same
activation spread are as readable. The earlier turn > random result replicates (same random neurons, new raters).
Post wording: "as readable as other neurons of the same activation size, and more readable than random ones".
Fix (2026-10-19): the summary field random_cleankv_kl_at_1 in the KV1 files first also averaged in the identity
check (KL 0); recomputed from the per-rotation values (turn_undo.py filter fixed). The numbers above were read from
the per-rotation values and are unchanged.
