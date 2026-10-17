# Why does Gemma keep remembering the word? A predicted mechanism, tested by intervention

*Draft, 2 October 2026. Part of model-flow-map. Pre-registration: `research/PREREG_flow.md` (F4, Amendment 1).*

## Summary
In Gemma 4 E2B, a token-level flow map shows that the model's internal representation tracks *which word a token
is* less and less until about layer 18, then tracks it more again (a "rebound"). Earlier work on left-to-right
transformers predicts a steady decline instead. We guessed the cause from the architecture: Gemma's E-series models add
a separate embedding of each token into every layer ("per-layer embeddings"). We wrote down the prediction, switched
those inputs off, and measured again. **Switching them off from layer 18 on removes about 55% of the rebound; switching
them off everywhere removes about 70%.** A smaller bump at layers 20–24 remains even with no per-layer inputs at all,
so they are the main cause, not the only one.

## Background
- Voita, Sennrich & Titov (2019) clustered token representations at every layer of left-to-right transformers and found
  that information about the current word fades with depth while information about the next word builds up.
- Gemma 3n and Gemma 4 E-series models feed each layer a per-layer embedding of the token (Google, 2025), which could
  keep reminding the model which word it is on.
- In a pilot, word identity in Gemma 4 E2B fell to layer 18 and climbed back (rebound 0.22), while in Qwen3-1.7B, which
  has no per-layer embeddings, it declined with no sustained comeback (rebound 0.03). Two models that differ in many ways
  can only suggest a cause, so we tested it directly.

## Method
- **Tokens:** about 67,000 tokens from 3,000 DBpedia training articles to fit, 88,000 tokens from test articles to
  score; at most 30 copies of any one word; first token of each text dropped (attention sink).
- **Measure:** at each layer, k-means bins (K = 512) of token representations in a whitened 64-dimensional frame;
  word identity = normalized mutual information between bins and the token's word.
- **Intervention:** run Gemma 4 E2B (base) with the per-layer inputs set to zero (a) for blocks 18 and later,
  (b) for all blocks; everything else unchanged. Same tokens, same measure.
- **Pre-registered predictions** (written before any ablated features were extracted): with (a), the rebound is
  ≤ 0.08 (F4-a) and less than half the unablated 0.22 (F4-b).

## Result
| | identity at layer 18 | peak after layer 18 | rise after layer 18 | identity at layer 35 |
|---|---|---|---|---|
| unablated | 0.45 | 0.67 (layer 24) | **0.22** | 0.59 |
| per-layer inputs off from layer 18 | 0.45 | 0.55 (layer 24) | **0.10** | 0.44 |
| per-layer inputs off everywhere | 0.36 | 0.43 (layer 24) | **0.07** | 0.35 |

- The ablated curve follows the unablated one exactly up to layer 18, as it must, then splits.
- **How the predictions score depends on a detail of the metric, and we report both readings.** As coded, the rebound
  is measured after the curve's lowest point; with the ablation the lowest point moves to the last layer, so the coded
  rebound is 0 and both predictions pass trivially. Read the natural way (rise after layer 18), F4-b passes narrowly
  (0.10 against a threshold of 0.11) and F4-a fails (0.10 against 0.08).
- Our claim is therefore "most of the rebound", not "the rebound".

## What's left unexplained
A bump at layers 20–24 survives without any per-layer inputs. One candidate is Gemma 4 E2B's KV-cache sharing: its
last 20 layers reuse attention memory computed earlier, so layer 15 is the last that computes its own. That is a
coincidence we have not tested.

## Limits
One model, one corpus (DBpedia, English), one bin count for the verdict. Zeroing an input the model was trained with
is a strong intervention: some of the change may come from the model being pushed off its normal operating range, not
from removing the "reminder". A gentler test would replace the per-layer inputs with those of a different token.

## Reproduce
`diag/token_extract.py --ablate-ple <start>` (patches `project_per_layer_inputs` to zero blocks ≥ start), then
`diag/token_lens.py --models nople18=tok_nople18 nople0=tok_nople0 --tag tok_ple_abl` and `diag/score_f4.py`.
Runner: `diag/f4_run.sh`. Results: `diag/results/f4_score.json`, `tok_ple__token_lens.json`, `tok_ple_abl__token_lens.json`.

## References
- Voita, Sennrich & Titov (2019). The Bottom-up Evolution of Representations in the Transformer. EMNLP. arXiv:1909.01380
- Google (2025). Introducing Gemma 3n: the developer guide (per-layer embeddings, KV-cache sharing).
- Sun et al. (2024). Massive Activations in Large Language Models. COLM. arXiv:2402.17762 (attention sinks / first token)
