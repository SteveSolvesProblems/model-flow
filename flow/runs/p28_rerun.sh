#!/bin/sh
# rerun the pythia-2.8b checkpoints that failed on a missing pad/eos token (after the main GPU1 queue finishes)
cd .
export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache HF_HUB_CACHE=data/hf
mountpoint -q data || exit 1
L=results/flow; PY=${PY:-python}
CPU4="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 DIAG_THREADS=4 CUDA_VISIBLE_DEVICES="
while [ ! -f $L/GPU1_DONE ]; do sleep 120; done
for st in 0 512 1000 2000 4000 8000 16000; do
  n=pythia28b_s$st
  CUDA_VISIBLE_DEVICES=1 $PY extract_hf.py llm --model EleutherAI/pythia-2.8b --revision step$st --name $n \
    --data dbpedia --dtype float32 --batch 32 > $L/$n.extract.log 2>&1 || { echo "FAILED extract $n (rerun)" >> $L/failures.txt; continue; }
  $CPU4 $PY select_layers.py --model $n --primary l2 --secondary l3 > $L/$n.select.log 2>&1 \
    || echo "FAILED select $n (rerun)" >> $L/failures.txt &
done
wait
echo done > $L/pythia28b_rerun.DONE
