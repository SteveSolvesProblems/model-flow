#!/bin/sh
# research/PREREG_flow.md Amendment 2 (F5): base ModernBERT-large vs Laya's fine-tuned encoder, DBpedia and 20NG.
cd "$(dirname "$0")"
PY=${PY:-${PY:-python}}
export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache HF_HUB_CACHE=data/hf
L=results/flow; mkdir -p $L
CPU4="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 DIAG_THREADS=4 CUDA_VISIBLE_DEVICES="
for spec in "mbert_large -" "laya convaiinnovations/laya:model.safetensors:encoder."; do
  set -- $spec
  w=""; [ "$2" != "-" ] && w="--weights-from $2"
  for d in dbpedia 20ng; do
    n=$1; [ $d = 20ng ] && n=${1}_20ng
    CUDA_VISIBLE_DEVICES=${GPU:-0} $PY extract_hf.py encoder --model answerdotai/ModernBERT-large $w --name $n --data $d \
      --dtype float32 --batch 32 > $L/$n.extract.log 2>&1 || { echo "FAILED extract $n" >> $L/failures.txt; continue; }
    if [ $d = dbpedia ]; then p="--primary l2 --secondary l3"; else p="--primary fine --secondary coarse"; fi
    $CPU4 $PY select_layers.py --model $n $p > $L/$n.select.log 2>&1 || echo "FAILED select $n" >> $L/failures.txt
  done
done
for r in "mbert_large laya" "mbert_large_20ng laya_20ng"; do
  set -- $r
  $CPU4 $PY quant_lens.py --ref $1 --variants $2 --ks 512 --tag ftflips > $L/ftflips_$1.log 2>&1 || echo "FAILED flips $1" >> $L/failures.txt
done
echo done > $L/LAYA_DONE
