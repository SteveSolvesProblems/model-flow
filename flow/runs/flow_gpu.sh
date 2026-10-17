#!/bin/sh
# research/PREREG_flow.md F2 (GPU 0) and F3 + pythia-2.8b (GPU 1). New caches/downloads go to the external drive;
# existing caches on the internal disk are found through DIAG_CACHE_EXTRA.
cd "$(dirname "$0")"
PY=${PY:-${PY:-python}}
export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache HF_HUB_CACHE=data/hf
L=results/flow; mkdir -p $L
mountpoint -q data || { echo "external drive not mounted" >> $L/failures.txt; exit 1; }
CPU4="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 DIAG_THREADS=4 CUDA_VISIBLE_DEVICES="

pythia_run() {   # $1 gpu, $2 repo suffix (e.g. 410m-seed1), $3 cache name prefix
  for st in 0 512 1000 2000 4000 8000 16000 32000 64000 143000; do
    n=$3_s$st
    CUDA_VISIBLE_DEVICES=$1 $PY extract_hf.py llm --model EleutherAI/pythia-$2 --revision step$st --name $n \
      --data dbpedia --dtype float32 --batch 32 > $L/$n.extract.log 2>&1 || { echo "FAILED extract $n" >> $L/failures.txt; continue; }
    $CPU4 $PY select_layers.py --model $n --primary l2 --secondary l3 > $L/$n.select.log 2>&1 \
      || echo "FAILED select $n" >> $L/failures.txt &
  done
  wait
  echo done > $L/$3.DONE
}

gpu0() {
  for s in 1 2 3 4; do pythia_run 0 410m-seed$s pythia410m_seed$s; done
  [ -f f4_run.sh ] && { sh f4_run.sh 0 > $L/f4.log 2>&1 || echo "FAILED f4" >> $L/failures.txt; }
  pythia_run 0 410m-deduped pythia410m_dedup
  pythia_run 0 1.4b-deduped pythia14b_dedup
  echo done > $L/GPU0_DONE
}

gpu1() {
  for spec in "Qwen/Qwen3-1.7B-Base qwen3_17b_base float32" "Qwen/Qwen3.5-2B-Base qwen35_2b_base float32" \
              "HuggingFaceTB/SmolLM2-1.7B-Instruct smollm2_17b_it float32" "google/gemma-4-E2B-it gemma4_e2b_it float16"; do
    set -- $spec
    for d in dbpedia 20ng; do
      n=$2; [ $d = 20ng ] && n=${2}_20ng
      CUDA_VISIBLE_DEVICES=1 $PY extract_hf.py llm --model $1 --name $n --data $d --dtype $3 --batch 32 \
        > $L/$n.extract.log 2>&1 || { echo "FAILED extract $n" >> $L/failures.txt; continue; }
      if [ $d = dbpedia ]; then p="--primary l2 --secondary l3"; else p="--primary fine --secondary coarse"; fi
      $CPU4 $PY select_layers.py --model $n $p > $L/$n.select.log 2>&1 || echo "FAILED select $n" >> $L/failures.txt &
    done
  done
  wait
  echo done > $L/F3_EXTRACT_DONE
  pythia_run 1 2.8b pythia28b
  echo done > $L/GPU1_DONE
}

gpu0 & gpu1 & wait
echo done > $L/FLOW_GPU_DONE
