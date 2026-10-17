#!/bin/sh
# Token-lens pilot (research/PREREG_token_lens.md, Amendment 1: 2x P100, fp16), then the approved re-run of the
# 120k Gemma extraction and its large-K set B. Run from diag/ on gpu-host.
export HF_HUB_DISABLE_PROGRESS_BARS=1 DIAG_THREADS=6
PY=${PY:-python}
L=results/token; mkdir -p $L
gpu_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -lt 500 ]; }
R=bartowski/google_gemma-4-E2B-it-GGUF
# meta must exist before two extractions run at once, so the base model goes first on its own
gpu_free 0 || { echo "GPU0 busy" >> $L/failures.txt; exit 1; }
CUDA_VISIBLE_DEVICES=0 $PY token_extract.py --name tok_base --model google/gemma-4-E2B > $L/tok_base.log 2>&1 || echo "FAILED tok_base" >> $L/failures.txt
gpu_free 1 || { echo "GPU1 busy" >> $L/failures.txt; exit 1; }
CUDA_VISIBLE_DEVICES=1 $PY token_extract.py --name tok_it --model $R --gguf-file google_gemma-4-E2B-it-bf16.gguf > $L/tok_it.log 2>&1 || echo "FAILED tok_it" >> $L/failures.txt
CUDA_VISIBLE_DEVICES=1 $PY token_extract.py --name tok_it2bit --model $R --gguf-file google_gemma-4-E2B-it-IQ2_M.gguf > $L/tok_it2bit.log 2>&1 || echo "FAILED tok_it2bit" >> $L/failures.txt
CUDA_VISIBLE_DEVICES="" DIAG_THREADS=8 $PY token_lens.py > $L/token_lens.log 2>&1 &
gpu_free 1 && CUDA_VISIBLE_DEVICES=1 $PY extract_hf.py llm --model google/gemma-4-E2B --name gemma4_e2b_big --data dbpedia \
  --dtype float16 --batch 32 --n-train 120000 > results/extract_big/gemma4_e2b_big.log 2>&1 \
  && CUDA_VISIBLE_DEVICES="" DIAG_THREADS=8 $PY large_k.py --model gemma4_e2b_big --levels l2 l3 --ks 512 1024 2048 4096 --ns 24000 120000 > results/largek/gemma4_e2b_big.log 2>&1 \
  || echo "FAILED gemma4_e2b_big rerun" >> results/largek/failures.txt
wait
echo DONE > $L/pilot.done
