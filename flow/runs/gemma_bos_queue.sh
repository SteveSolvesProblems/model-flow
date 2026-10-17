#!/bin/sh
# Local RTX 3060 queue: every Gemma 4 E2B-it run re-done with <bos> (the -it tokenizer adds none; NOTE_geometry.md).
# 1) token caches (PREREG_token_lens.md), 2) quantization ladder (quant_ladder.sh), 3) mixed-precision KL (qkl.py,
# robust_gpu.sh). Same files, dtype and settings as the originals; tokenizer = base (same vocabulary, adds <bos>);
# GGUF config and layer_scalar still from the official -it repo. CPU analyses run on gpu-host afterwards.
cd "$(dirname "$0")"
export HF_HOME=$HOME/.cache/huggingface HF_HUB_DISABLE_PROGRESS_BARS=1 DIAG_LOW_VRAM=1 DIAG_ATTN=eager   # shared desktop 3060
PY=${PY:-python}
L=results/gemma_bos; mkdir -p $L
SC=data/scratch/qkl_scratch_bos
R=bartowski/google_gemma-4-E2B-it-GGUF; T=google/gemma-4-E2B; O=google/gemma-4-E2B-it
while [ ! -f results/flow_bos/F3_BOS_EXTRACT_DONE ]; do [ -f results/flow_bos/failures.txt ] && exit 1; sleep 60; done
tx() { grep -q "done$" $L/$1.log 2>/dev/null && return   # caches moved to data/scratch/cache_mirror
  $PY token_extract.py --name $1 --model $R --gguf-file $2 --tokenizer $T --official $O --meta tok_meta_bos.npz \
      --batch 8 > $L/$1.log 2>&1 || echo "FAILED $1" >> $L/failures.txt; }
tx tok_it_bos google_gemma-4-E2B-it-bf16.gguf
tx tok_it2bit_bos google_gemma-4-E2B-it-IQ2_M.gguf
run() { name=$1; repo=$2; file=$3
  [ -f cache/${name}_L00/meta.json ] && grep -q "done$" $L/$name.log 2>/dev/null && return
  $PY extract_hf.py llm --model $repo --gguf-file $file --tokenizer $T --official $O \
      --name $name --data dbpedia --dtype float16 --batch 8 > $L/$name.log 2>&1 || echo "FAILED $name" >> $L/failures.txt
}
run gq_bf16   $R google_gemma-4-E2B-it-bf16.gguf
run gq_q4km   $R google_gemma-4-E2B-it-Q4_K_M.gguf
run gq_q8_0   $R google_gemma-4-E2B-it-Q8_0.gguf
run gq_q4_0   $R google_gemma-4-E2B-it-Q4_0.gguf
run gq_qat4_0 google/gemma-4-E2B-it-qat-q4_0-gguf gemma-4-E2B_q4_0-it.gguf
run gq_q2k    $R google_gemma-4-E2B-it-Q2_K.gguf
run gq_q6k    $R google_gemma-4-E2B-it-Q6_K.gguf
run gq_q5km   $R google_gemma-4-E2B-it-Q5_K_M.gguf
run gq_q3km   $R google_gemma-4-E2B-it-Q3_K_M.gguf
run gq_iq2m   $R google_gemma-4-E2B-it-IQ2_M.gguf
touch $L/LADDER_DONE
# qkl needs the lens choices (quant_lens on gpu-host, CPU) first: see gemma_bos_cpu.sh; mixed phase runs after that.
$PY qkl.py --phase ref --scratch $SC --dev cuda:0 > $L/qkl_ref.log 2>&1 || echo "FAILED qkl ref" >> $L/failures.txt
touch $L/QKL_REF_DONE
