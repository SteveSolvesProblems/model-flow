#!/bin/sh
# PREREG_large_k.md set A: existing caches, K in {512, 1024, 2048}. CPU only. Run from diag/ on gpu-host.
export CUDA_VISIBLE_DEVICES="" DIAG_THREADS=8
PY=${PY:-python}
L=results/largek; mkdir -p $L
for spec in qwen3_17b:l2:l3 qwen35_2b:l2:l3 gemma4_e2b:l2:l3 \
            qwen35_08b:l2:l3 qwen35_4b:l2:l3 gemma4_e4b:l2:l3 smollm2_17b:l2:l3 \
            qwen35_08b_20ng:fine:coarse qwen35_4b_20ng:fine:coarse gemma4_e4b_20ng:fine:coarse smollm2_17b_20ng:fine:coarse \
            qwen3_17b_20ng:fine:coarse qwen35_2b_20ng:fine:coarse gemma4_e2b_20ng:fine:coarse; do
  m=${spec%%:*}; rest=${spec#*:}; p=${rest%%:*}; s=${rest#*:}
  $PY large_k.py --model $m --levels $p $s --ks 512 1024 2048 > $L/$m.log 2>&1 || echo "FAILED $m" >> $L/failures.txt
done
echo DONE > $L/setA.done
