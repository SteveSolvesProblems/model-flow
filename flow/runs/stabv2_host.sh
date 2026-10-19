#!/bin/bash
# research/PREREG_stability_v2.md, CPU half on gpu-host: analyse each pair not assigned to the local machine as soon
# as stabv2_local.sh marks its caches landed on the external drive. One pair at a time (stab_v2.py resumes per layer).
# Run from diag/ on gpu-host:  nohup bash stabv2_host.sh > data/stabv2/gpu-host.log 2>&1 &
set -u
PY=${PY:-python}
EXT=data
export DIAG_CACHE=$EXT/cache DIAG_RESULTS=$EXT/stabv2 DIAG_THREADS=${DIAG_THREADS:-12}
log() { echo "$(date +%H:%M:%S) $*"; }
PAIRS=${PAIRS:?space-separated name:label list}
for p in $PAIRS; do
  name=${p%%:*} label=${p##*:}
  until [ -e $EXT/stabv2/$name.landed ]; do sleep 120; done
  if [ -e $EXT/stabv2/$name.done ]; then log "skip $name"; continue; fi
  log "analyse $name"
  $PY stab_v2.py --model $name --label $label > $EXT/stabv2/$name.log 2>&1 && touch $EXT/stabv2/$name.done \
    || log "FAILED $name"
done
log "ALL DONE"
