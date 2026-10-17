#!/bin/bash
# Replication and bias controls for the geometry claims (research/NOTE_geometry.md, "Replication"). CPU only, two
# streams of 6 threads. Seed 1 = a held-out sample disjoint from the original one (common.sample_idx).
cd "$(dirname "$0")"
export DIAG_CACHE=data/cache DIAG_CACHE_EXTRA=$PWD/cache
run() { nice -n 10 env OMP_NUM_THREADS=6 DIAG_THREADS=6 CUDA_VISIBLE_DEVICES= ${PY:-python} -u "$@" 2>&1 | grep --line-buffered -v Warn; }
L=results/geo_logs
streamA() {
  for p in "gemma4_e2b qwen3_17b" "gemma4_e2b gemma4_e4b" "qwen3_17b smollm2_17b" "qwen3_17b qwen35_2b"; do set -- $p
    run geometry_crossmodel.py --a $1 --b $2 > $L/cross_$1__$2.log; done
  for s in "gemma4_e2b test" "qwen3_17b test" "tok_base eval" "tok_qwen3 eval"; do set -- $s
    run geometry_probe.py --name $1 --split $2 --seed 1 > $L/probe_s1_$1.log; done
  for n in 1000 6000; do run geometry_probe.py --name gemma4_e2b --split test --n $n > $L/probe_n${n}_gemma4_e2b.log; done
  for m in smollm2_17b qwen35_2b; do run geometry_probe.py --name $m --split test > $L/probe_$m.log; done
  run geometry_readout.py --name tok_base --meta tok_meta.npz --model google/gemma-4-E2B --seed 1 > $L/readout_s1_tok_base.log
  run geometry_readout.py --name tok_qwen3 --meta tok_meta_qwen3.npz --model Qwen/Qwen3-1.7B --seed 1 > $L/readout_s1_tok_qwen3.log
  for m in smollm2_17b qwen35_2b; do run geometry_linmap.py --name $m > $L/linmap_$m.log; done
  touch $L/REPL_A_DONE
}
streamB() {
  for s in "gemma4_e2b train test" "qwen3_17b train test" "tok_base fit eval" "tok_qwen3 fit eval"; do set -- $s
    run geometry_linmap.py --name $1 --fit $2 --eval $3 --seed 1 > $L/linmap_s1_$1.log; done
  run geometry_linmap.py --name gemma4_e4b > $L/linmap_gemma4_e4b.log
  touch $L/REPL_B_DONE
}
streamA & streamB & wait
