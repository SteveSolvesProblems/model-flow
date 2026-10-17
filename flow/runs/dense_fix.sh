#!/bin/sh
# F2-d: re-extract pythia410m_seed4_s7000 (first attempt failed on a Hugging Face connection error), after the
# last-token steering runs free GPU1; then its partition codes.
cd .
export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache HF_HUB_CACHE=data/hf
L=results/flow; PY=${PY:-python}; n=pythia410m_seed4_s7000
while [ ! -f results/steerflow/LAST_DONE ]; do sleep 120; done
CUDA_VISIBLE_DEVICES=1 $PY extract_hf.py llm --model EleutherAI/pythia-410m-seed4 --revision step7000 --name $n \
  --data dbpedia --dtype float32 --batch 32 > $L/$n.extract.log 2>&1 || { echo "FAILED extract $n (retry)" >> $L/failures.txt; exit 1; }
grep -h "weights fingerprint" $L/$n.extract.log | sed "s/^/$n /" >> $L/dense.fingerprints
nice -n 10 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 DIAG_THREADS=4 CUDA_VISIBLE_DEVICES= \
  $PY partition_codes.py --model $n > $L/$n.codes.log 2>&1 || echo "FAILED codes $n" >> $L/failures.txt
touch $L/DENSE_FIX_DONE
