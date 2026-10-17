#!/bin/sh
# Pre-registered quantization study (research/PREREG_quantization_lens.md, Amendment 1: fp16 on 2x P100).
# Every variant, reference included, goes through the same transformers GGUF dequant path. Run from diag/ on gpu-host.
# Sequential on purpose: a GGUF load peaks ~21 GB RAM, so two loads at once OOM the 30 GB host.
# Order: reference, Q4_K_M, then Q3 (lens flips + mixed-precision KL, the decisive test), then the rest of the ladder.
export HF_HUB_DISABLE_PROGRESS_BARS=1 DIAG_THREADS=12
PY=${PY:-python}
L=results/quant; mkdir -p $L
R=bartowski/google_gemma-4-E2B-it-GGUF
run() { name=$1; repo=$2; file=$3
  [ -f cache/${name}_L00/meta.json ] && grep -q "done$" $L/$name.log 2>/dev/null && return
  CUDA_VISIBLE_DEVICES=1 $PY extract_hf.py llm --model $repo --gguf-file $file --tokenizer google/gemma-4-E2B-it \
      --name $name --data dbpedia --dtype float16 --batch 32 > $L/$name.log 2>&1 || echo "FAILED $name" >> $L/failures.txt
}
[ -n "$WAIT_PID" ] && while kill -0 $WAIT_PID 2>/dev/null; do sleep 30; done
run gq_bf16   $R google_gemma-4-E2B-it-bf16.gguf
run gq_q4km   $R google_gemma-4-E2B-it-Q4_K_M.gguf
CUDA_VISIBLE_DEVICES="" $PY quant_lens.py --ref gq_bf16 --variants gq_q4km --tag quantlens > $L/quantlens_q4km.log 2>&1
CUDA_VISIBLE_DEVICES=1 $PY qkl.py --phase ref --dev cuda:0 > $L/qkl_ref.log 2>&1 &&
CUDA_VISIBLE_DEVICES=1 $PY qkl.py --phase mixed --dev cuda:0 > $L/qkl_mixed.log 2>&1
run gq_q8_0   $R google_gemma-4-E2B-it-Q8_0.gguf
run gq_q4_0   $R google_gemma-4-E2B-it-Q4_0.gguf
run gq_qat4_0 google/gemma-4-E2B-it-qat-q4_0-gguf gemma-4-E2B_q4_0-it.gguf
run gq_q2k    $R google_gemma-4-E2B-it-Q2_K.gguf
run gq_q6k    $R google_gemma-4-E2B-it-Q6_K.gguf
run gq_q5km   $R google_gemma-4-E2B-it-Q5_K_M.gguf
run gq_q3km   $R google_gemma-4-E2B-it-Q3_K_M.gguf
run gq_iq2m   $R google_gemma-4-E2B-it-IQ2_M.gguf
V=$(for n in gq_q8_0 gq_q6k gq_q5km gq_q4km gq_q4_0 gq_qat4_0 gq_q3km gq_q2k gq_iq2m; do [ -f cache/${n}_L00/meta.json ] && printf "%s " $n; done)
CUDA_VISIBLE_DEVICES="" $PY quant_lens.py --ref gq_bf16 --variants $V --tag quantlens_all > $L/quantlens_all.log 2>&1
for n in gq_bf16 $V; do CUDA_VISIBLE_DEVICES="" $PY select_layers.py --model $n --primary l2 --secondary l3 > $L/select_$n.log 2>&1; done
echo DONE > $L/ladder.done
