#!/bin/sh
# research/PREREG_flow.md F3-b: bin flips of each post-trained model in its base model's frame and bins (K = 512).
cd .
export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache
L=results/flow; PY=${PY:-python}
CPU4="nice -n 10 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 DIAG_THREADS=4 CUDA_VISIBLE_DEVICES="
run() { $CPU4 $PY quant_lens.py --ref $1 --variants $2 --ks 512 --tag chatflips_$2 > $L/chatflips_$2.log 2>&1 || echo "FAILED chatflips $2" >> $L/failures.txt; }
for sfx in "" _20ng; do
  run qwen3_17b_base$sfx qwen3_17b$sfx & run qwen35_2b_base$sfx qwen35_2b$sfx; wait
  run smollm2_17b$sfx smollm2_17b_it$sfx & run gemma4_e2b$sfx gemma4_e2b_it$sfx; wait
done
touch $L/F3_FLIPS_DONE
