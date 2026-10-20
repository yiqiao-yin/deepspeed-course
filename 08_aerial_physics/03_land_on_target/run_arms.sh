#!/usr/bin/env bash
# Three senses x two seeds on the `dense` rig, then the engineered
# baselines on the SAME rig.
#
# Rig choice is load-bearing and was measured, not guessed: `dense`
# and `fore10` are the only two layouts with no dead wrench axis,
# and on this task `fore10` reaches 6% on-target at 2.8M steps
# where `dense` reaches 37%. Lab 1's own run history says the same
# thing -- fore10 does not break out until 3.9M on the EASIER task.
#
# Serial on purpose: two 16-worker runs on 20 cores is slower than
# either alone. run_one PROPAGATES exit status; a sweep that echoes
# "done" regardless once printed SWEEP COMPLETE with 9 of 14 runs
# dead.
set -u
cd "$(dirname "$0")"
mkdir -p logs runs
fail=0
run_one () {
  local name="$1"; shift
  if [ -f "runs/$name/summary.json" ]; then
    echo "SKIP  $name (already has summary.json)"; return 0
  fi
  echo "START $name  $(date -u +%H:%M:%S)"
  if uv run train_target.py --name "$name" "$@" > "logs/$name.log" 2>&1; then
    echo "OK    $name  $(date -u +%H:%M:%S)"
  else
    echo "FAIL  $name  rc=$? -- see logs/$name.log"; fail=$((fail+1))
  fi
}
for seed in 0 1; do
  for sense in blind slope patch; do
    run_one "${sense}_s${seed}" --sense "$sense" --seed "$seed" \
            --rig dense --preset badlands --total-steps 4000000
  done
done
# The engineered arms, on the same rig and terrain as the learned
# ones. The divert budget is swept because "does terrain knowledge
# help" is a different question from "does THIS controller use it
# well", and only the sweep separates them.
for d in 0 2 4 6; do
  echo "CLASSICAL divert=${d}m  $(date -u +%H:%M:%S)"
  uv run classical.py --rig dense --preset badlands --episodes 64 \
      --max-divert "$d" > "logs/classical_d${d}.log" 2>&1 \
      || { echo "FAIL classical d=$d"; fail=$((fail+1)); }
done
if [ "$fail" -eq 0 ]; then echo "ALL ARMS COMPLETE"; else echo "SWEEP FINISHED WITH $fail FAILURE(S)"; fi
