#!/bin/sh
# After gemma_bos_queue.sh (LADDER_DONE, QKL_REF_DONE): quantization study re-run with <bos>.
# 1) copy gq_* caches to gpu-host (old ones moved to cache_nobos/), run the CPU lens/selection/scoring there
#    (quant_ladder.sh, robustness.py b4a, qat_shift.py); 2) pull the lens choices; 3) mixed-precision KL on the local
#    3060 (qkl.py: main, deterministic re-run with per-doc sums, single blocks, 20 random sets); 4) CIs.
cd "$(dirname "$0")"
L=results/gemma_bos; E=data
export HF_HOME=$HOME/.cache/huggingface HF_HUB_DISABLE_PROGRESS_BARS=1 DIAG_LOW_VRAM=1 DIAG_ATTN=eager
PY=${PY:-python}; SC=data/scratch/qkl_scratch_bos
while [ ! -f $L/QKL_REF_DONE ]; do [ -f $L/failures.txt ] && exit 1; sleep 60; done
[ -f $L/failures.txt ] && exit 1
V="gq_bf16 gq_q8_0 gq_q6k gq_q5km gq_q4km gq_q4_0 gq_qat4_0 gq_q3km gq_q2k gq_iq2m"
ssh -o BatchMode=yes gpu-host "mkdir -p $E/cache_nobos && for v in $V; do for d in $E/cache/\${v}_L??; do [ -d \$d ] && mv \$d $E/cache_nobos/; done; done; true" || exit 1
for v in $V; do rsync -a cache/${v}_L?? gpu-host:$E/cache/ || exit 1; done
ssh -o BatchMode=yes gpu-host 'cd . && export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache DIAG_THREADS=12
PY=${PY:-python}; L=results/gemma_bos; C="nice -n 10 env CUDA_VISIBLE_DEVICES="
$C $PY quant_lens.py --ref gq_bf16 --variants gq_q4km --tag quantlens > $L/quantlens_q4km.log 2>&1 || { echo FAILED > $L/QUANT_CPU_FAILED; exit 1; }
touch $L/QUANTLENS_Q4KM_DONE
$C $PY quant_lens.py --ref gq_bf16 --variants gq_q8_0 gq_q6k gq_q5km gq_q4km gq_q4_0 gq_qat4_0 gq_q3km gq_q2k gq_iq2m --tag quantlens_all > $L/quantlens_all.log 2>&1
for n in gq_bf16 gq_q8_0 gq_q6k gq_q5km gq_q4km gq_q4_0 gq_qat4_0 gq_q3km gq_q2k gq_iq2m; do $C $PY select_layers.py --model $n --primary l2 --secondary l3 > $L/select_$n.log 2>&1; done
$C $PY score_quant.py > $L/score_quant.log 2>&1; $C $PY qat_shift.py > $L/qat_shift.log 2>&1; $C $PY robustness.py b4a > $L/b4a.log 2>&1
touch $L/QUANT_CPU_DONE' &
until ssh -o BatchMode=yes gpu-host "test -f ./results/gemma_bos/QUANTLENS_Q4KM_DONE || test -f ./results/gemma_bos/QUANT_CPU_FAILED"; do sleep 60; done
rsync -a gpu-host:flow/results/gq_bf16__quantlens.json results/ || exit 1
$PY qkl.py --phase mixed --scratch $SC --dev cuda:0 > $L/qkl_mixed.log 2>&1 || { echo "FAILED qkl mixed" >> $L/failures.txt; exit 1; }
$PY qkl.py --phase mixed --scratch $SC --dev cuda:0 --out results/gq__qkl_rerun.json --per-doc results/gq__qkl_perdoc.npz > $L/qkl_rerun.log 2>&1
$PY qkl.py --phase mixed --extra single --scratch $SC --dev cuda:0 --out results/robust_b4b.json > $L/qkl_single.log 2>&1
$PY qkl.py --phase mixed --extra random20 --scratch $SC --dev cuda:0 --out results/robust_b4c.json --per-doc results/robust_b4c_perdoc.npz > $L/qkl_random20.log 2>&1
wait
until ssh -o BatchMode=yes gpu-host "test -f ./results/gemma_bos/QUANT_CPU_DONE"; do sleep 60; done
for f in gq_bf16__quantlens_all.json quant_score.json qat_shift.json robust_b4a.json; do rsync -a gpu-host:flow/results/$f results/; done
rsync -a gpu-host:flow/results/gemma_bos/ $L/
for n in $V; do rsync -a gpu-host:flow/results/${n}__select.json results/; done
python3 score_qkl_ci.py > $L/score_qkl_ci.log 2>&1; python3 score_b4bc.py > $L/score_b4bc.log 2>&1
rsync -a results/gq__qkl*.json results/gq__qkl_perdoc.npz results/robust_b4*.json results/robust_b4c_perdoc.npz results/gq__qkl_ci.json results/robust_b4bc.json gpu-host:flow/results/
touch $L/QUANT_DONE
