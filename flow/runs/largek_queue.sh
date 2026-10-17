#!/bin/sh
# PREREG_large_k.md (with Amendment 1), approved 2026-09-30: after the 120k extractions, set B, then slim set A.
export CUDA_VISIBLE_DEVICES="" DIAG_THREADS=12
PY=${PY:-python}
L=results/largek; mkdir -p $L
until [ -f results/extract_big/done ]; do sleep 60; done
for m in qwen35_2b_big gemma4_e2b_big; do
  grep -q "FAILED $m" results/extract_big/failures.txt 2>/dev/null && continue
  $PY large_k.py --model $m --levels l2 l3 --ks 512 1024 2048 4096 --ns 24000 120000 > $L/$m.log 2>&1 || echo "FAILED $m" >> $L/failures.txt
done
for spec in qwen35_08b:l2:l3 qwen35_4b:l2:l3 gemma4_e4b:l2:l3 smollm2_17b:l2:l3 \
            qwen35_08b_20ng:fine:coarse qwen35_4b_20ng:fine:coarse gemma4_e4b_20ng:fine:coarse smollm2_17b_20ng:fine:coarse \
            qwen3_17b_20ng:fine:coarse qwen35_2b_20ng:fine:coarse gemma4_e2b_20ng:fine:coarse; do
  m=${spec%%:*}; rest=${spec#*:}; p=${rest%%:*}; s=${rest#*:}
  $PY large_k.py --slim --model $m --levels $p $s --ks 1024 2048 > $L/slim_$m.log 2>&1 || echo "FAILED slim $m" >> $L/failures.txt
done
echo DONE > $L/queue.done
