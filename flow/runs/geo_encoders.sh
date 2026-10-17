#!/bin/bash
# Encoder geometry (exploratory): same probes as the decoders, on ModernBERT-large and Laya (fine-tuned ModernBERT).
cd "$(dirname "$0")"
export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache
run() { nice -n 10 env OMP_NUM_THREADS=6 DIAG_THREADS=6 CUDA_VISIBLE_DEVICES= ${PY:-python} "$@" 2>&1 | grep -v Warn; }
mkdir -p results/geo_logs
for n in mbert_large laya; do
  run geometry_probe.py --name $n --split test --n 3000 > results/geo_logs/probe_$n.log
  run geometry_flow.py --name $n --split test > results/geo_logs/flow_$n.log
done
for n in mbert_large laya; do run geometry_linmap.py --name $n --fit train --eval test > results/geo_logs/linmap_$n.log; done
touch results/geo_logs/ENC_DONE
