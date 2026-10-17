# Model Flow

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
- A side result, a label-free way to pick an embedding layer, is still being tested. Its first registration and the
  registered follow-up are in docs/preregistrations/ (`PREREG_layer_selection.md`, `PREREG_stability_v2.md`).

This is exploratory research. Claims are narrow on purpose, and failures are reported next to passes.

## Repository

```
docs/                 the write-up (GitHub Pages root), pre-registrations
docs/archive/         earlier drafts, kept for the record and marked superseded
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
