#!/bin/sh
# F2-c: partition codes for the new Pythia runs (niced; runs alongside the queue)
cd .
export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache
for m in pythia410m_seed1 pythia410m_seed2 pythia410m_seed3 pythia410m_seed4 pythia410m_dedup; do
  for st in 0 512 1000 2000 4000 8000 16000 32000 64000 143000; do
    n=${m}_s$st
    [ -f results/codes/$n.npz ] && continue
    nice -n 19 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 DIAG_THREADS=4 CUDA_VISIBLE_DEVICES= \
      ${PY:-python} partition_codes.py --model $n > results/flow/$n.codes.log 2>&1 || echo "FAILED codes $n" >> results/flow/failures.txt
  done
done
touch results/flow/F2C_CODES_DONE
