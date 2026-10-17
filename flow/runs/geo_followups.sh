#!/bin/bash
# Exploratory geometry follow-ups (research/NOTE_geometry.md), CPU only, one job at a time.
cd "$(dirname "$0")"
export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache
run() { nice -n 10 env OMP_NUM_THREADS=6 DIAG_THREADS=6 CUDA_VISIBLE_DEVICES= ${PY:-python} "$@" 2>&1 | grep -v Warn; }
mkdir -p results/geo_logs
for s in "tok_base eval" "tok_qwen3 eval" "gemma4_e2b test" "qwen3_17b test"; do set -- $s
  run geometry_flow.py --name $1 --split $2 > results/geo_logs/flow_$1.log; done
run geometry_readout.py --name tok_base --meta tok_meta.npz --model google/gemma-4-E2B > results/geo_logs/readout_tok_base.log
run geometry_readout.py --name tok_qwen3 --meta tok_meta_qwen3.npz --model Qwen/Qwen3-1.7B > results/geo_logs/readout_tok_qwen3.log
for s in "tok_base fit eval" "tok_qwen3 fit eval" "gemma4_e2b train test" "qwen3_17b train test"; do set -- $s
  run geometry_linmap.py --name $1 --fit $2 --eval $3 > results/geo_logs/linmap_$1.log; done
touch results/geo_logs/DONE
