#!/bin/bash
# research/PREREG_stability_v2.md, extraction on the local RTX 3060. Caches land on gpu-host's external drive
# (data/cache) over a rate-limited rsync; pairs listed in LOCAL_CPU are also analysed here
# (stab_v2.py) before the local copy is deleted, the rest are analysed on gpu-host (stabv2_host.sh).
# Models come from gpu-host's HF caches at the revisions the earlier caches used (Qwen3.5-4B-Base from the hub).
# Run from diag/:  nohup bash stabv2_local.sh > results/stabv2/local.log 2>&1 &
set -u
PY=${PY:-python}
T=data/stabv2_tmp
EXT=data
BW=${BW:-8000}                                   # rsync KiB/s, leaves headroom on the WiFi link
LOCAL_CPU=" ${LOCAL_CPU:-} "
mkdir -p $T/hf/hub $T/cache results/stabv2
export HF_HOME=$HOME/.cache/huggingface HF_HUB_CACHE=$T/hf/hub HF_DATASETS_OFFLINE=1 DIAG_CACHE=$T/cache DIAG_THREADS=10
log() { echo "$(date +%H:%M:%S) $*"; }

fetch() {   # repo, sha, source hub dir on gpu-host ("" = already local / hub download)
  local repo=$1 sha=$2 src=$3 d=models--${1//\//--}
  [ -d $T/hf/hub/$d/snapshots/$sha ] && return
  if [ "$src" = local ]; then ln -sfn $HOME/.cache/huggingface/hub/$d $T/hf/hub/$d; return; fi
  if [ "$src" = hub ]; then
    HF_HUB_OFFLINE=0 $PY -c "from huggingface_hub import snapshot_download as s; s('$repo', revision='$sha')" || return
    mkdir -p $T/hf/hub/$d/refs; echo -n $sha > $T/hf/hub/$d/refs/main; return     # offline loads resolve refs/main
  fi
  mkdir -p $T/hf/hub/$d/refs $T/hf/hub/$d/snapshots
  rsync -aL --bwlimit=$BW gpu-host:$src/$d/snapshots/$sha $T/hf/hub/$d/snapshots/
  echo -n $sha > $T/hf/hub/$d/refs/main
}

drop() {    # free a fetched model once its extractions are done (symlinked local models are kept)
  local d=$T/hf/hub/models--${1//\//--}
  [ -L $d ] && rm $d || rm -rf $d
}

pending() { for n; do ssh gpu-host "test -e $EXT/stabv2/$n.landed" || return 0; done; return 1; }   # any not landed?

wait_disk() { while [ $(df --output=avail -BG /mnt/d | tail -1 | tr -dc 0-9) -lt 18 ]; do sleep 60; done; }

land() {    # push one pair's caches, mark landed, analyse here if assigned, then free the local copy
  local name=$1 label=$2
  rsync -a --bwlimit=$BW $T/cache/${name}_L* gpu-host:$EXT/cache/ && ssh gpu-host "touch $EXT/stabv2/$name.landed" \
    || { log "PUSH FAILED $name"; return; }
  log "landed $name"
  if [[ "$LOCAL_CPU" == *" $name "* ]]; then
    DIAG_RESULTS=results flock $T/cpu.lock $PY stab_v2.py --model $name --label $label > results/stabv2/$name.log 2>&1 \
      && rsync -a results/${name}__stabv2.json gpu-host:$EXT/stabv2/ && log "analysed $name"
  fi
  rm -rf $T/cache/${name}_L*
}

job() {     # name repo sha data dtype label [extra args]
  local name=$1 repo=$2 sha=$3 data=$4 dtype=$5 label=$6; shift 6
  if ssh gpu-host "test -e $EXT/stabv2/$name.landed"; then log "skip $name (landed)"; return; fi
  wait_disk
  log "extract $name"
  HF_HUB_OFFLINE=1 $PY extract_hf.py llm --model $repo --name $name --data $data --dtype $dtype --batch 32 "$@" \
    > results/stabv2/extract_$name.log 2>&1 || { log "EXTRACT FAILED $name"; return; }
  land $name $label &
}

ssh gpu-host "mkdir -p $EXT/stabv2"
# new models: DBpedia, 20NG, Yelp
pending llama32_1b llama32_1b_20ng llama32_1b_yelp && fetch meta-llama/Llama-3.2-1B 4e20de362430cd3b72f300e6b0f18e50e7166e08 $EXT/hf
job llama32_1b      meta-llama/Llama-3.2-1B 4e20de3 dbpedia float32 l2
job llama32_1b_20ng meta-llama/Llama-3.2-1B 4e20de3 20ng    float32 fine
job llama32_1b_yelp meta-llama/Llama-3.2-1B 4e20de3 yelp    float32 stars
drop meta-llama/Llama-3.2-1B
pending smollm2_360m smollm2_360m_20ng smollm2_360m_yelp && fetch HuggingFaceTB/SmolLM2-360M f8027fd0eaeea54caa13c31d31b9fdc459c38b49 $EXT/hf
job smollm2_360m      HuggingFaceTB/SmolLM2-360M f8027fd dbpedia float32 l2
job smollm2_360m_20ng HuggingFaceTB/SmolLM2-360M f8027fd 20ng    float32 fine
job smollm2_360m_yelp HuggingFaceTB/SmolLM2-360M f8027fd yelp    float32 stars
drop HuggingFaceTB/SmolLM2-360M
pending qwen35_4b_base qwen35_4b_base_20ng qwen35_4b_base_yelp && fetch Qwen/Qwen3.5-4B-Base 1001bb4d826a52d1f399e183466143f4da7b741b hub
job qwen35_4b_base      Qwen/Qwen3.5-4B-Base 1001bb4 dbpedia float16 l2
job qwen35_4b_base_20ng Qwen/Qwen3.5-4B-Base 1001bb4 20ng    float16 fine
job qwen35_4b_base_yelp Qwen/Qwen3.5-4B-Base 1001bb4 yelp    float16 stars
# earlier models on Yelp, same revision and dtype as their earlier caches
drop Qwen/Qwen3.5-4B-Base
pending qwen3_17b_yelp && fetch Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e local
job qwen3_17b_yelp Qwen/Qwen3-1.7B 70d244c yelp float32 stars
drop Qwen/Qwen3-1.7B
pending gemma4_e2b_yelp && fetch google/gemma-4-E2B d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f local
job gemma4_e2b_yelp google/gemma-4-E2B d29ff6b yelp float16 stars
drop google/gemma-4-E2B
pending qwen35_2b_yelp && fetch Qwen/Qwen3.5-2B 15852e8c16360a2fea060d615a32b45270f8a8fc $EXT/hf
job qwen35_2b_yelp Qwen/Qwen3.5-2B 15852e8 yelp float32 stars
drop Qwen/Qwen3.5-2B
pending smollm2_17b_yelp && fetch HuggingFaceTB/SmolLM2-1.7B effd688a12921b4cc83e3312b6feb579f70f9c71 $EXT/hf
job smollm2_17b_yelp HuggingFaceTB/SmolLM2-1.7B effd688 yelp float32 stars
drop HuggingFaceTB/SmolLM2-1.7B
pending qwen35_4b_yelp && fetch Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a ~/.cache/huggingface/hub
job qwen35_4b_yelp Qwen/Qwen3.5-4B 851bf6e yelp float16 stars
# Pythia-2.8B step143000: float16 (float32 weights do not fit the 3060's 12 GB); pytorch_model.bin as before
drop Qwen/Qwen3.5-4B
pending pythia28b_yelp && fetch EleutherAI/pythia-2.8b dbe7ae300a54abcdc475a33907b3dff81d25709f $EXT/hf
job pythia28b_yelp EleutherAI/pythia-2.8b dbe7ae3 yelp float16 stars --revision dbe7ae300a54abcdc475a33907b3dff81d25709f --no-safetensors
drop EleutherAI/pythia-2.8b
wait
log "ALL DONE"
