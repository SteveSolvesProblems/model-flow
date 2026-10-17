#!/bin/sh
# PREREG_baselines.md phase B: two GPU lanes, one model at a time per lane.
cd "$(dirname "$0")"
PY=${PY:-${PY:-python}}
L=results/token_metrics; mkdir -p $L
lane() { gpu=$1; shift
  while [ $# -gt 0 ]; do m=$1; n=$2; dt=$3; ex=$4; shift 4
    CUDA_VISIBLE_DEVICES=$gpu $PY token_metrics.py --model $m --name $n --dtype $dt $ex > $L/$n.log 2>&1 || echo "FAILED $n" >> $L/failures.txt
  done; }
lane 0 Qwen/Qwen3-1.7B qwen3_17b float32 "" Qwen/Qwen3.5-0.8B qwen35_08b float32 "" \
       Qwen/Qwen3.5-2B qwen35_2b float32 "" HuggingFaceTB/SmolLM2-1.7B smollm2_17b float32 "" &
lane 1 Qwen/Qwen3.5-4B qwen35_4b float16 "" google/gemma-4-E2B gemma4_e2b float16 "" \
       google/gemma-4-E4B gemma4_e4b float16 --offload &
wait
echo done > $L/ALL_DONE
