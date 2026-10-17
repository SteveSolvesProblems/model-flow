#!/bin/sh
# After f3_bos_local.sh: copy the <bos> gemma4_e2b_it caches to gpu-host (old ones moved to cache_nobos/), then run
# the F3 CPU analyses there (as flow_gpu.sh / f3_flips.sh did). Old results: results/nobos/ on gpu-host.
cd "$(dirname "$0")"
while [ ! -f results/flow_bos/F3_BOS_EXTRACT_DONE ]; do [ -f results/flow_bos/failures.txt ] && exit 1; sleep 60; done
E=data
ssh -o BatchMode=yes gpu-host "mkdir -p $E/cache_nobos && for d in $E/cache/gemma4_e2b_it_L?? $E/cache/gemma4_e2b_it_20ng_L??; do [ -d \$d ] && mv \$d $E/cache_nobos/; done; true" || exit 1
rsync -a cache/gemma4_e2b_it_L?? cache/gemma4_e2b_it_20ng_L?? gpu-host:$E/cache/ || exit 1
ssh -o BatchMode=yes gpu-host 'cd . && export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache
PY=${PY:-python}; L=results/gemma_bos; mkdir -p $L
C="nice -n 10 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 DIAG_THREADS=4 CUDA_VISIBLE_DEVICES="
$C $PY select_layers.py --model gemma4_e2b_it --primary l2 --secondary l3 > $L/select_gemma4_e2b_it.log 2>&1 &
$C $PY select_layers.py --model gemma4_e2b_it_20ng --primary fine --secondary coarse > $L/select_gemma4_e2b_it_20ng.log 2>&1 &
wait
$C $PY quant_lens.py --ref gemma4_e2b --variants gemma4_e2b_it --ks 512 --tag chatflips_gemma4_e2b_it > $L/chatflips.log 2>&1 &
$C $PY quant_lens.py --ref gemma4_e2b_20ng --variants gemma4_e2b_it_20ng --ks 512 --tag chatflips_gemma4_e2b_it_20ng > $L/chatflips_20ng.log 2>&1 &
wait
$PY score_f3.py > $L/score_f3.log 2>&1; touch $L/F3_DONE'
