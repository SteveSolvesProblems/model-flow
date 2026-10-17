#!/bin/sh
# research/PREREG_overnight.md E3–E6 on existing caches (CPU, 8 threads; the GPU queue's scoring uses the rest).
cd "$(dirname "$0")"
PY=${PY:-${PY:-python}}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 DIAG_THREADS=8 CUDA_VISIBLE_DEVICES=""
L=results/overnight; mkdir -p $L
run() { n=$1; shift; $PY "$@" > $L/$n.log 2>&1 || echo "FAILED $n" >> $L/failures.txt; }
for m in qwen3_17b qwen35_08b qwen35_2b qwen35_4b gemma4_e2b gemma4_e4b smollm2_17b; do
  run codes_$m partition_codes.py --model $m
done
for p in qwen35_08b_20ng qwen35_4b_20ng gemma4_e4b_20ng smollm2_17b_20ng qwen3_17b_20ng qwen35_2b_20ng gemma4_e2b_20ng; do
  S=$($PY -c "import json;print(json.load(open('results/${p}__select.json'))['selection']['fine']['S']['layer'])")
  run codes_$p partition_codes.py --model $p --layers $S
done
run model_select model_select.py
run routing routing.py
run correspondence correspondence.py
run polysemy polysemy.py
echo done > $L/CPU_DONE
