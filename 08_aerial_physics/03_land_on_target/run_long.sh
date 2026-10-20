#!/usr/bin/env bash
# Does the widest arm CATCH UP, or is it capped?
#
# At 4M every arm was still climbing, so "patch scores 0%" is a
# statement about sample efficiency at a fixed budget and NOT about
# a ceiling. Both arms are continued to 10M total so the comparison
# stays matched-budget; running only the losing arm longer would
# answer a different question than the one the page asks.
set -u
cd "$(dirname "$0")"
fail=0
for arm in blind patch; do
  name="${arm}_s0_10M"
  [ -f "runs/$name/summary.json" ] && { echo "SKIP $name"; continue; }
  echo "START $name  $(date -u +%H:%M:%S)"
  if uv run train_target.py --name "$name" --sense "$arm" --seed 0 \
       --rig dense --preset badlands --total-steps 6000000 \
       --init-from "runs/${arm}_s0" --eval-episodes 96 \
       > "logs/$name.log" 2>&1; then
    echo "OK    $name  $(date -u +%H:%M:%S)"
  else
    echo "FAIL  $name rc=$?"; fail=$((fail+1))
  fi
done
if [ "$fail" -eq 0 ]; then echo "LONG RUNS COMPLETE"; else echo "LONG RUNS FINISHED WITH $fail FAILURE(S)"; fi
