#!/bin/bash
# Watch the training block for the three quiet failures:
#   STALL   alive but the step count is frozen
#   DIED    gone without writing summary.json -- xargs moves on, so a
#           crashed run leaves a GAP while the script still prints
#           ALLDONE. That is the worst mode and it has happened here.
#   WEDGED  the driver script is gone before the work finished
set -u
cd "$(dirname "$0")"
EXPECT="land_reveal land_minimal reveal_s0 minimal_s0 reveal_s1 minimal_s1 reveal_s2 minimal_s2"
declare -A LAST
while true; do
  ts=$(date +%H:%M:%S)
  done_n=0
  for n in $EXPECT; do [ -f "runs/$n/summary.json" ] && done_n=$((done_n+1)); done
  live=$(ps -eo args --no-headers | grep -c "[t]rain_roll.py --rig" || true)
  drv=$(ps -eo args --no-headers | grep -c "[r]un_stages.sh" || true)
  msg="$ts done=$done_n/8 live=$((live/2)) load=$(cut -d' ' -f1 /proc/loadavg)"
  for d in runs/*/; do
    n=$(basename "$d")
    [ -f "$d/summary.json" ] && continue
    [ -f "$d/history.json" ] || continue
    st=$(python3 -c "
import json
try: print(json.load(open('$d/history.json'))[-1]['step'])
except Exception: print(0)" 2>/dev/null)
    prev=${LAST[$n]:-}
    [ -n "$prev" ] && [ "$st" = "$prev" ] && msg="$msg STALL:$n@$st"
    LAST[$n]=$st
  done
  grep -q "FAILED " stages.log 2>/dev/null && msg="$msg DIED:see-stages.log"
  [ "$drv" -eq 0 ] && ! grep -q ALLDONE_MARKER stages.log 2>/dev/null \
      && msg="$msg WEDGED:driver-gone"
  echo "$msg" >> watchdog.log
  grep -q ALLDONE_MARKER stages.log 2>/dev/null && { echo "$ts FINISHED" >> watchdog.log; break; }
  sleep 300
done
