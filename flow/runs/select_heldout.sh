#!/bin/sh
# Pre-registered held-out evaluation (research/PREREG_layer_selection.md): run the frozen selection
# script on each held-out cache as soon as its extraction finishes. Run from diag/ on gpu-host.
export CUDA_VISIBLE_DEVICES="" DIAG_THREADS=10
PY=${PY:-python}
L=results/extract2
run() { name=$1; p=$2; s=$3
  until grep -q "done$" $L/$name.log 2>/dev/null || grep -q "FAILED $name$" $L/failures.txt 2>/dev/null; do sleep 60; done
  if grep -q "FAILED $name$" $L/failures.txt 2>/dev/null; then echo "SKIP $name (extraction failed)"; return; fi
  $PY select_layers.py --model $name --primary $p --secondary $s
}
run qwen3_17b_20ng fine coarse
run qwen35_2b_20ng fine coarse
run gemma4_e2b_20ng fine coarse
run qwen35_08b l2 l3
run qwen35_08b_20ng fine coarse
run smollm2_17b l2 l3
run smollm2_17b_20ng fine coarse
run qwen35_4b l2 l3
run qwen35_4b_20ng fine coarse
run gemma4_e4b l2 l3
run gemma4_e4b_20ng fine coarse
echo DONE > results/select_heldout.done
