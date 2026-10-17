#!/bin/sh
# Re-extract pythia-2.8b checkpoints from pytorch_model.bin: on the hub, one stale model.safetensors blob is shared by
# 8 of the 10 branches, so the first pythia28b_s* caches all hold the same weights. New names; old caches kept.
cd .
export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache HF_HUB_CACHE=data/hf
mountpoint -q data || exit 1
L=results/flow; PY=${PY:-python}
CPU4="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 DIAG_THREADS=4 CUDA_VISIBLE_DEVICES="
for st in 0 512 1000 2000 4000 8000 16000 32000 64000 143000; do
  n=pythia28b_bin_s$st
  CUDA_VISIBLE_DEVICES=1 $PY extract_hf.py llm --model EleutherAI/pythia-2.8b --revision step$st --name $n --no-safetensors \
    --data dbpedia --dtype float32 --batch 32 > $L/$n.extract.log 2>&1 || { echo "FAILED extract $n" >> $L/failures.txt; continue; }
  grep -h "weights fingerprint" $L/$n.extract.log | sed "s/^/$st /" >> $L/pythia28b_bin.fingerprints
  $CPU4 $PY select_layers.py --model $n --primary l2 --secondary l3 > $L/$n.select.log 2>&1 \
    || echo "FAILED select $n" >> $L/failures.txt &
done
wait
echo done > $L/pythia28b_bin.DONE
