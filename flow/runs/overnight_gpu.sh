#!/bin/sh
# research/PREREG_overnight.md E1 (GPU 0) and E2 (GPU 1). New caches and downloads go to the external drive.
cd "$(dirname "$0")"
PY=${PY:-${PY:-python}}
export DIAG_CACHE=data/cache HF_HUB_CACHE=data/hf
L=results/overnight; mkdir -p $L
mountpoint -q data || { echo "external drive not mounted" >> $L/failures.txt; exit 1; }

e1() {
  for m in 410m 1.4b; do
    for st in 0 512 1000 2000 4000 8000 16000 32000 64000 143000; do
      n=pythia$(echo $m | tr -d .)_s$st
      CUDA_VISIBLE_DEVICES=0 $PY extract_hf.py llm --model EleutherAI/pythia-$m --revision step$st --name $n \
        --data dbpedia --dtype float32 --batch 64 > $L/$n.extract.log 2>&1 || { echo "FAILED extract $n" >> $L/failures.txt; continue; }
      # CPU scoring of this checkpoint runs while the next one extracts
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 DIAG_THREADS=4 CUDA_VISIBLE_DEVICES="" \
        $PY select_layers.py --model $n --primary l2 --secondary l3 > $L/$n.select.log 2>&1 &
    done
  done
  wait
  echo done > $L/E1_EXTRACT_DONE
}

e2() {
  for spec in "Qwen/Qwen3-1.7B qwen3_17b float32" "Qwen/Qwen3.5-2B qwen35_2b float32" \
              "HuggingFaceTB/SmolLM2-1.7B smollm2_17b float32" "google/gemma-4-E2B gemma4_e2b float16"; do
    set -- $spec
    for d in banking77 stsb; do
      CUDA_VISIBLE_DEVICES=1 $PY task_extract.py --model $1 --name $2 --data $d --dtype $3 > $L/$d.$2.log 2>&1 \
        || echo "FAILED $d $2" >> $L/failures.txt
    done
  done
  echo done > $L/E2_EXTRACT_DONE
}

e1 & e2 & wait
echo done > $L/GPU_DONE
