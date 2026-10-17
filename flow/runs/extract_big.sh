#!/bin/sh
# PREREG_large_k.md set B: 120k-train DBpedia caches for the two dev models. Run from diag/ on gpu-host.
# Sequential: each extraction holds all layers' pooled features in RAM (~12-13 GB) until it finishes.
export HF_HUB_DISABLE_PROGRESS_BARS=1 DIAG_THREADS=6
PY=${PY:-python}
L=results/extract_big; mkdir -p $L
[ -n "$WAIT_PID" ] && while kill -0 $WAIT_PID 2>/dev/null; do sleep 60; done
grep -q "done$" $L/qwen35_2b_big.log 2>/dev/null || CUDA_VISIBLE_DEVICES=0 $PY extract_hf.py llm --model Qwen/Qwen3.5-2B \
  --name qwen35_2b_big --data dbpedia --dtype float32 --batch 32 --n-train 120000 > $L/qwen35_2b_big.log 2>&1 \
  || echo "FAILED qwen35_2b_big" >> $L/failures.txt
CUDA_VISIBLE_DEVICES=1 $PY extract_hf.py llm --model google/gemma-4-E2B --name gemma4_e2b_big --data dbpedia \
  --dtype float16 --batch 32 --n-train 120000 > $L/gemma4_e2b_big.log 2>&1 || echo "FAILED gemma4_e2b_big" >> $L/failures.txt
echo DONE > $L/done
