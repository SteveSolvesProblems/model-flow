#!/bin/bash
# Test 2 (Pythia training geometry) now on 4 threads; test 1 (layer skip/swap) after the Gemma block knockouts finish.
cd "$(dirname "$0")"
export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache
( HF_HUB_CACHE=data/hf nice -n 10 env OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= ${PY:-python} -u pythia_geometry.py 2>&1 \
    | grep --line-buffered -v -i "warn" > results/geo_logs/pythia_geometry.log; touch results/geo_logs/PYTHIA_DONE ) &
until [ -f results/geo_logs/BLOCK_DONE ]; do sleep 60; done
for s in "qwen3_17b Qwen/Qwen3-1.7B --swap" "gemma4_e2b google/gemma-4-E2B"; do set -- $s
  nice -n 10 env OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES= ${PY:-python} -u layer_skip.py --name $1 --model $2 $3 2>&1 \
    | grep --line-buffered -v -i "warn" > results/geo_logs/layer_skip_$1.log
done
touch results/geo_logs/SKIP_DONE
wait
