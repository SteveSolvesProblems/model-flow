#!/bin/sh
# PREREG_token_lens.md Part 2: Qwen3-1.7B token pilot. Run from diag/ on gpu-host.
export HF_HUB_DISABLE_PROGRESS_BARS=1 DIAG_THREADS=6
PY=${PY:-python}
L=results/token; mkdir -p $L
FREE=$(df -BG --output=avail ~ | tail -1 | tr -dc 0-9); [ "$FREE" -ge 34 ] || { echo "only ${FREE}G free, need 34G" >> $L/failures.txt; exit 1; }
[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 500 ] || { echo "GPU0 busy" >> $L/failures.txt; exit 1; }
CUDA_VISIBLE_DEVICES=0 $PY token_extract.py --name tok_qwen3 --model Qwen/Qwen3-1.7B --tokenizer Qwen/Qwen3-1.7B \
  --meta tok_meta_qwen3.npz --dtype float32 > $L/tok_qwen3.log 2>&1 || { echo "FAILED tok_qwen3" >> $L/failures.txt; exit 1; }
CUDA_VISIBLE_DEVICES="" DIAG_THREADS=8 $PY token_lens.py --meta tok_meta_qwen3.npz --models base=tok_qwen3 --tag tok_qwen3 > $L/token_lens_qwen3.log 2>&1 \
  || echo "FAILED token_lens qwen3" >> $L/failures.txt
echo DONE > $L/qwen3.done
