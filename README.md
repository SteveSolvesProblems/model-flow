# Model Flow

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23437748.svg)](https://doi.org/10.5281/zenodo.23437748)

What does a language model look like if you study it the way you would study a fluid: by how a population of texts
moves through its layers, rather than weight by weight?

**Read the write-up:** [docs/index.html](docs/index.html), also published on this repository's GitHub Pages site.

## In short

- Thousands of short texts are followed through frozen, public models: Gemma 4 E2B and E4B, Qwen3, Qwen3.5,
  SmolLM2, Pythia, and ModernBERT.
- Layers are compared in ways that do not depend on shared axes: nearest neighbours, and label-free k-means groups.
- Every measurement is read against controls: an untrained model, robust scaling, and a second model.
- Most of the flow is smooth. Several apparent sharp changes turn out to be a few very large coordinates.
- One large event survives every control. Gemma 4 E2B re-expresses its whole space at layer 14.
- The mechanism has been tested causally, and the model's stored per-layer scalars predicted where Gemma 4 E4B does
  the same. That prediction was pre-registered and passed.
- Added in v0.2 (18 October 2026): the turn is not carried by a few MLP neurons. Its neurons are somewhat readable,
  but any write of the right size turns the stream, and the previous layer's scalar sets part of that size. Notes,
  including the tests that failed: `docs/notes/neurons_and_the_turn.md`.
- Added in v0.3: whether the turn matters for what the model predicts. Later layers get nothing from the pre-turn
  axes. For a few layers after the turn the predictions depend unusually little on the stream's axes, and a control
  shows this comes mostly from Gemma 4's reuse of earlier layers' attention keys and values, which begins right after
  the turn. Against neurons of the same activation size, the turn's neurons are no more readable.
- A side result, a label-free way to pick an embedding layer, is written up in `docs/stability.html`. Its second,
  pre-registered test found it no better than the second-to-last layer on most models, so that layer is the
  recommended default. It seemed to help on Gemma 4, whose geometry turns, but that was noticed after the test. Registrations: `docs/preregistrations/PREREG_layer_selection.md`,
  `PREREG_stability_v2.md`.

This is exploratory research. Claims are narrow on purpose, and failures are reported next to passes.

## Repository

```
docs/                 the write-up (GitHub Pages root), pre-registrations
docs/archive/         earlier drafts, kept for the record and marked superseded
docs/notes/           working notes behind later additions to the write-up
research/CLAIMS.md    every number in the write-up, with its source file, producing script and known threats
flow/                 code: extraction (GPU) and analyses (CPU); flow/viewer/ builds the pages
flow/results/         result JSONs behind every claim
flow/runs/            the queue scripts as they were run (machine names and paths replaced)
REPRODUCE.md          how to check, rebuild and re-run everything
```

To check every number in the write-up against the included results (CPU, seconds):

```
pip install -r requirements.txt
cd flow && python verify_claims.py
```

## How this was made

This is a human–AI collaboration. Steve set the question and the ideas behind it, chose the directions, and reviewed
each step. Claude (Anthropic), working in Claude Code, wrote the code, ran the experiments, searched the literature
and drafted the write-up. Results and wording were reviewed together.

## Licence

- Code: see [LICENSE](LICENSE).
- Text and figures: see [LICENSE-docs](LICENSE-docs).

## Citation

See [CITATION.cff](CITATION.cff). The archived release has a DOI on Zenodo.

## Citing

Steve Ball (2026). *Model Flow: following texts through a language model's layers.* Zenodo. https://doi.org/10.5281/zenodo.23437748
(this DOI always resolves to the latest version; see CITATION.cff).
