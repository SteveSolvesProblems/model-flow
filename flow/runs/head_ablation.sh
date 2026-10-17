#!/bin/bash
# Head knockouts, CPU, one model at a time (exploratory; research/NOTE_geometry.md).
cd "$(dirname "$0")"
export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache
run() { nice -n 10 env OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES= "$@" 2>&1 | grep -v -i "warn"; }
run ${PY:-python} head_ablation.py --name qwen3_17b --model Qwen/Qwen3-1.7B --late 20 27 > results/geo_logs/ablate_qwen3_17b.log
run ${PY:-python} head_ablation.py --name gemma4_e2b --model google/gemma-4-E2B --late 21 33 > results/geo_logs/ablate_gemma4_e2b.log
touch results/geo_logs/ABLATE_DONE
