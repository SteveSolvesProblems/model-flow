#!/bin/sh
# research/PREREG_flow.md F4 (Amendment 1): Gemma 4 E2B base with per-layer inputs zeroed. $1 = GPU index.
# Run from flow_gpu.sh (DIAG_CACHE = external drive; tok_meta.npz and tok_base on the internal disk via DIAG_CACHE_EXTRA).
PY=${PY:-${PY:-python}}
[ -f "$DIAG_CACHE/tok_meta.npz" ] || cp cache/tok_meta.npz "$DIAG_CACHE/"
GPU=$1
for spec in "tok_nople18 18" "tok_nople0 0"; do
  set -- $spec
  [ -d "$DIAG_CACHE/$1_L00" ] && [ -f "$DIAG_CACHE/$1_L35/meta.json" ] && continue   # resume: already extracted
  CUDA_VISIBLE_DEVICES=$GPU $PY token_extract.py --name $1 --model google/gemma-4-E2B --ablate-ple $2 || exit 1
done
env OMP_NUM_THREADS=8 DIAG_THREADS=8 CUDA_VISIBLE_DEVICES= $PY token_lens.py \
  --models base=tok_base nople18=tok_nople18 nople0=tok_nople0 --tag tok_ple || exit 1
$PY score_f4.py
