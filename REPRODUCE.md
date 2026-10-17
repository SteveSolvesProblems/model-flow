# Reproducing Model Flow

Every number in the write-up is a row in [`research/CLAIMS.md`](research/CLAIMS.md). Each row names the result file the
number comes from and the script that produced that file. `python flow/verify_claims.py` recomputes every number from
the result files and exits non-zero if any value differs from the stated one.

There are three levels of reproduction, from cheapest to most expensive.

1. **Check the numbers (CPU, seconds).** The JSON results are included, so this needs no GPU and no downloads:
   ```
   pip install -r requirements.txt
   cd flow && python verify_claims.py
   ```
2. **Rebuild the pages (CPU, seconds):**
   ```
   cd flow
   python viewer/build_findings_data.py
   python viewer/build_post.py ../docs/index.html
   ```
3. **Re-run a study from the models.** First extract feature caches on a GPU (step A), then run that study's CPU
   analysis (step B). The feature caches are not included: they total several hundred GB, and are regenerated
   deterministically from public models and data.

## Environment

- Python 3.12, with the packages in `requirements.txt`. The runs used torch 2.11 with CUDA 12.6.
- GPUs used:
  - two NVIDIA P100s (16 GB each, fp16/fp32 only);
  - one RTX 3060 (12 GB) for some Gemma runs, with `DIAG_LOW_VRAM=1 DIAG_ATTN=eager`.
- Any GPU with at least 16 GB runs every extraction. Models up to 4B parameters also fit on 12 GB with the two flags
  above.
- Analyses are CPU-only (k-means via faiss and torch). Set `DIAG_THREADS` to your core count.
- Paths are configured with environment variables:
  - `DIAG_CACHE`: where feature caches are written and read (default `flow/cache`).
  - `DIAG_CACHE_EXTRA`: extra read-only cache dirs, colon-separated.
  - `DIAG_RESULTS`: default `flow/results`.
  - `HF_HOME` / `HF_HUB_CACHE`: model downloads.

## Compute and storage used

No model was trained; every model is a public checkpoint, used frozen. The figures below come from job logs, so they
are approximate. They include about 14 GPU-hours spent on two exploratory pilots that are not in this repository.

| Resource | Amount |
|---|---|
| GPU | about 100 GPU-hours: about 88 h on two P100s (up to two jobs at once), about 13 h on one RTX 3060 |
| CPU analysis | about 330 job-hours (4–12 threads each), about 96 h of wall-clock time |
| Feature caches | about 0.6 TB: about 205 million per-layer vectors in fp16, in about 5,500 model × layer caches |
| Other scratch | about 0.08 TB (quantization KL and pipeline-fault runs) |
| Model downloads | about 0.35 TB |
| GPU energy | at most about 25 kWh (rated power × hours; host and CPU not included) |

Where the GPU time went:

| Work | P100 hours |
|---|---|
| Pythia training checkpoints (410M, 1.4B, 2.8B) | about 37 |
| Other models and encoders | about 12 |
| GGUF quantization ladder | about 5 |
| Token-level states | about 10 |
| Quantization KL | about 6 |

## A. Feature caches (GPU)

`flow/extract_hf.py` writes one cache per model and layer: `<name>_L<ii>/{train,test}_X.npy` plus labels and a
`meta.json`. The meta file records the model, its revision hash, dtype, pooling, and whether `<bos>` was present.

Defaults for text:
- DBpedia Classes (`DeveloperOats/DBPedia_Classes`), shuffled with seed 0;
- 24,000 train and 6,000 test documents, cut to 64 tokens;
- mean over non-padding tokens, at every layer (layer 0 = the input embeddings).

`--data 20ng` uses 20 Newsgroups instead.

```
# a decoder at every layer
python extract_hf.py llm --model Qwen/Qwen3-1.7B --name qwen3_17b --dtype float32
# Gemma 4: chat-tuned weights need the base tokenizer, which adds <bos> (the script asserts this)
python extract_hf.py llm --model google/gemma-4-E2B-it --tokenizer google/gemma-4-E2B --name gemma4_e2b_it --dtype float16
# a Pythia training checkpoint
python extract_hf.py llm --model EleutherAI/pythia-410m --revision step1000 --name pythia410m_s1000
# a quantized GGUF, dequantized; config and layer scalars come from the official repo
python extract_hf.py llm --model bartowski/google_gemma-4-E2B-it-GGUF --gguf-file google_gemma-4-E2B-it-Q4_K_M.gguf \
    --tokenizer google/gemma-4-E2B --official google/gemma-4-E2B-it --name gq_q4km --dtype float16
# an encoder
python extract_hf.py encoder --model answerdotai/ModernBERT-large --name mbert_large
# token-level states (one vector per token position)
python token_extract.py --name tok_base_bos --model google/gemma-4-E2B
```

The exact commands, model lists and dtypes for every study are in `flow/runs/*.sh`. These are the queue scripts as
they were run, with machine names and paths replaced. For example:
- `flow_gpu.sh` and `overnight_gpu.sh`: the main model set and the Pythia checkpoints;
- `quant_ladder.sh`: the GGUF ladder;
- `qwen_token.sh` and `gemma_bos_queue.sh`: token-level caches.

## B. Analyses (CPU)

Each claim's row in `research/CLAIMS.md` names its script. The main ones are:

| Study | Script(s) | Claims |
|---|---|---|
| Layer geometry (place, company, turns) | `geometry_probe.py --name <cache> --split test [--n 3000] [--seed 1]` | G2–G5, G16 |
| Rotation fits within and across models | `geometry_linmap.py`, `geometry_crossmodel.py --a <cache> --b <cache>` | G6 |
| Flow maps and groups | `geometry_flow.py`, `flow_confounds.py` | G14 |
| How Gemma turns | `flow_noscale.py`, `block_ablation.py [--tag dose]`, `weight_structure.py`, `attention_heads.py`, `head_ablation.py` | G9–G12 |
| E4B prediction (pre-registered) | `score_prereg_e4b.py` | G13 |
| Untrained / training checkpoints | `pythia_geometry.py`, `pythia_score.py` | G1, P1–P3 |
| Choosing a layer without labels | `select_layers.py`, `score_select.py`, `baselines_a.py`, `score_baselines.py` | L1–L6, S1–S4 |
| Quantization, large K, tokens | `quant_lens.py`, `qkl.py`, `score_quant.py`, `large_k.py`, `token_lens.py` | Q1–Q3, K1, W1 |

The pre-registration for each study is in `docs/preregistrations/`. It states the hypotheses and pass criteria
written before the data existed, and every later amendment.

## Left out

Two exploratory pilots (classifier heads and steering) and one study on private fine-tuning checkpoints are not in
this repository.
