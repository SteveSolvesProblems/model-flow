#!/bin/sh
# Follow-ups: (1) after the first CPU queue, chance-corrected selector + GGUF bit check;
# (2) after E1 extraction, Pythia partition codes + scoring; (3) after E2 extraction, task scoring.
cd "$(dirname "$0")"
PY=${PY:-${PY:-python}}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 DIAG_THREADS=8 CUDA_VISIBLE_DEVICES=""
L=results/overnight
run() { n=$1; shift; $PY "$@" > $L/$n.log 2>&1 || echo "FAILED $n" >> $L/failures.txt; }
until [ -f $L/CPU_DONE ]; do sleep 60; done
run gguf_bits gguf_layer_bits.py
for p in qwen35_08b_20ng qwen35_4b_20ng gemma4_e4b_20ng smollm2_17b_20ng qwen3_17b_20ng qwen35_2b_20ng gemma4_e2b_20ng; do
  run codesall_$p partition_codes.py --model $p
done
run adjusted_select adjusted_select.py
export DIAG_CACHE=data/cache
until [ -f $L/E1_EXTRACT_DONE ]; do sleep 60; done
for m in pythia410m pythia14b; do for st in 0 512 1000 2000 4000 8000 16000 32000 64000 143000; do
  run codes_${m}_s$st partition_codes.py --model ${m}_s$st
done; done
run pythia_score pythia_score.py
until [ -f $L/E2_EXTRACT_DONE ]; do sleep 60; done
for pool in last mean; do for d in $HOME/flow/cache/bank77_q35_4b_${pool}_L*; do
  t=$DIAG_CACHE/banking77_qwen35_4b_${pool}_$(basename $d | sed 's/.*_\(L[0-9]*\)$/\1/'); [ -d $t ] || cp -r $d $t
done; done
run task_score task_score.py
echo done > $L/POST_DONE
