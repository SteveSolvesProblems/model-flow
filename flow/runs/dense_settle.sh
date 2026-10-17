#!/bin/sh
# research/PREREG_flow.md Amendment 3 (F2-d): extra Pythia-410m checkpoints on GPU1, then partition codes.
# At most one CPU job runs in the background (codes for the previous checkpoint while the next one extracts).
cd .
export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache HF_HUB_CACHE=data/hf
mountpoint -q data || exit 1
L=results/flow; PY=${PY:-python}
CPU4="nice -n 10 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 DIAG_THREADS=4 CUDA_VISIBLE_DEVICES="
bg=""
for spec in "pythia-410m pythia410m" "pythia-410m-seed1 pythia410m_seed1" "pythia-410m-seed4 pythia410m_seed4"; do
  set -- $spec
  for st in 128 256 3000 5000 6000 7000; do
    n=$2_s$st
    CUDA_VISIBLE_DEVICES=1 $PY extract_hf.py llm --model EleutherAI/$1 --revision step$st --name $n \
      --data dbpedia --dtype float32 --batch 32 > $L/$n.extract.log 2>&1 || { echo "FAILED extract $n" >> $L/failures.txt; continue; }
    grep -h "weights fingerprint" $L/$n.extract.log | sed "s/^/$n /" >> $L/dense.fingerprints
    [ -n "$bg" ] && wait $bg
    $CPU4 $PY partition_codes.py --model $n > $L/$n.codes.log 2>&1 || echo "FAILED codes $n" >> $L/failures.txt &
    bg=$!
  done
done
wait
touch $L/DENSE_DONE
