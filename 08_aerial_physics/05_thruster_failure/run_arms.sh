#!/usr/bin/env bash
# Healthy first, then robust warm-started from it.
#
# THE CURRICULUM IS NOT OPTIONAL. Trained from scratch with a
# failure in every episode, the policy never learned to land at all:
# 0.8% landings at 1.35M steps. On `reveal` about half of all
# (nozzle, mode) draws leave no hover reachable, so half the
# training signal came from episodes nothing could save, and the
# base skill never formed. Learn to land, then break the vehicle.
#
# The healthy arm is also the CONTROL: it is scored with failures
# on, having never seen one.
set -u
cd "$(dirname "$0")"
mkdir -p logs runs
fail=0
run_one () {
  local name="$1"; shift
  if [ -f "runs/$name/summary.json" ]; then echo "SKIP  $name"; return 0; fi
  echo "START $name  $(date -u +%H:%M:%S)"
  if uv run train_fail.py --name "$name" "$@" > "logs/$name.log" 2>&1; then
    echo "OK    $name  $(date -u +%H:%M:%S)"
  else
    echo "FAIL  $name rc=$? -- see logs/$name.log"; fail=$((fail+1))
  fi
}
for sd in 0 1; do
  run_one "healthy_s${sd}" --fail-p 0.0 --seed "$sd" \
          --total-steps 2500000 --eval-episodes 128
  run_one "robust_s${sd}"  --fail-p 1.0 --seed "$sd" \
          --init-from "runs/healthy_s${sd}" \
          --total-steps 2500000 --eval-episodes 128
done
# What a perfect fault detector would be worth -- a different
# question from whether the policy can cope without one.
run_one "told_s0" --fail-p 1.0 --tell --seed 0 \
        --init-from "runs/healthy_s0" --total-steps 2500000 --eval-episodes 128
if [ "$fail" -eq 0 ]; then echo "ALL ARMS COMPLETE"; else echo "SWEEP FINISHED WITH $fail FAILURE(S)"; fi
