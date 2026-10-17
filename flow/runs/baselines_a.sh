#!/bin/sh
# PREREG_baselines.md phase A: all 14 pairs, three at a time (CPU only), largest first.
cd "$(dirname "$0")"
PY=${PY:-${PY:-python}}
mkdir -p results/baselines_a
printf '%s\n' "gemma4_e4b l2 l3" "gemma4_e4b_20ng fine coarse" "qwen35_4b l2 l3" "qwen35_4b_20ng fine coarse" \
  "gemma4_e2b l2 l3" "gemma4_e2b_20ng fine coarse" "qwen3_17b l2 l3" "qwen3_17b_20ng fine coarse" \
  "qwen35_2b l2 l3" "qwen35_2b_20ng fine coarse" "qwen35_08b l2 l3" "qwen35_08b_20ng fine coarse" \
  "smollm2_17b l2 l3" "smollm2_17b_20ng fine coarse" |
xargs -P 3 -L 1 sh -c 'OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 DIAG_THREADS=5 CUDA_VISIBLE_DEVICES="" \
  '"$PY"' baselines_a.py --model $0 --primary $1 --secondary $2 > results/baselines_a/$0.log 2>&1 || echo "FAILED $0" >> results/baselines_a/failures.txt'
echo done > results/baselines_a/ALL_DONE
