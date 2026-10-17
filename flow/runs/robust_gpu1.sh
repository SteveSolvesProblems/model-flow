#!/bin/sh
# research/PREREG_robustness.md B4-b and B4-c on GPU 0 (Gemma 4 E2B-it GGUFs; reference logits on the external drive).
cd "$(dirname "$0")"
PY=${PY:-python}
L=results/robust; mkdir -p $L
SC=data/qkl_scratch
CUDA_VISIBLE_DEVICES=1 $PY qkl.py --phase ref --scratch $SC --dev cuda:0 > $L/qkl_ref.log 2>&1 || { echo "FAILED ref" >> $L/failures.txt; exit 1; }
CUDA_VISIBLE_DEVICES=1 $PY qkl.py --phase mixed --extra single --scratch $SC --dev cuda:0 --out results/robust_b4b.json > $L/qkl_single.log 2>&1 || echo "FAILED single" >> $L/failures.txt
CUDA_VISIBLE_DEVICES=1 $PY qkl.py --phase mixed --extra random20 --scratch $SC --dev cuda:0 --out results/robust_b4c.json --per-doc results/robust_b4c_perdoc.npz > $L/qkl_random20.log 2>&1 || echo "FAILED random20" >> $L/failures.txt
echo done > $L/GPU_DONE
