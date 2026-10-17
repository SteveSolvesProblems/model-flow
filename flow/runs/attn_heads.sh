#!/bin/bash
# Per-head attention behaviour, CPU, one model at a time (exploratory; research/NOTE_geometry.md).
cd "$(dirname "$0")"
export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache
run() { nice -n 10 env OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES= "$@" 2>&1 | grep -v -i "warn"; }
HF_HUB_CACHE=data/hf run ${PY:-python} attention_heads.py --name modernbert_large --model answerdotai/ModernBERT-large > results/geo_logs/heads_modernbert_large.log
run ${PY:-python} attention_heads.py --name qwen3_17b --model Qwen/Qwen3-1.7B > results/geo_logs/heads_qwen3_17b.log
run ${PY:-python} attention_heads.py --name gemma4_e2b --model google/gemma-4-E2B > results/geo_logs/heads_gemma4_e2b.log
touch results/geo_logs/HEADS_DONE
