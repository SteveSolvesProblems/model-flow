#!/bin/sh
# Re-extract chat-tuned Gemma 4 E2B with <bos> on the local RTX 3060 (the -it tokenizer adds none; NOTE_geometry.md).
# Same texts, dtype and pooling as flow_gpu.sh; base tokenizer (same vocabulary, adds <bos>).
cd "$(dirname "$0")"
export HF_HOME=$HOME/.cache/huggingface DIAG_LOW_VRAM=1 DIAG_ATTN=eager   # shared desktop 3060: ~6 GB; see extract_hf.py
PY=${PY:-python}
L=results/flow_bos; mkdir -p $L
for d in dbpedia 20ng; do
  n=gemma4_e2b_it; [ $d = 20ng ] && n=${n}_20ng
  $PY extract_hf.py llm --model google/gemma-4-E2B-it --tokenizer google/gemma-4-E2B --name $n --data $d \
      --dtype float16 --batch ${BATCH:-8} > $L/$n.extract.log 2>&1 || { echo "FAILED extract $n" >> $L/failures.txt; exit 1; }
done
touch $L/F3_BOS_EXTRACT_DONE
