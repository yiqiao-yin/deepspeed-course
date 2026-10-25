#!/usr/bin/env bash
# The residual arms with BEST-CHECKPOINT selection, then the
# scheduled baselines on the same courses and seeds.
#
# WHY RESIDUAL. From scratch, end-to-end RL converged at 1.14 of 3
# gates across 7.4M steps and two action spaces -- flat, not slowly
# rising. The policy therefore starts from the guidance law: at
# action 0.5 the residual is exactly zero, so episode one IS the
# scheduled controller and any gain is a gain over a real baseline.
#
# WHY BEST-CHECKPOINT. Both Mars runs peaked near 88% around 1.5M
# steps and drifted back to 77% by 4M. Scoring the final weights
# measured a policy that had already degraded. Selection uses the
# TRAINING window; the published number is still held-out seeds.
set -u
cd "$(dirname "$0")"
mkdir -p logs runs
fail=0
run_one () {
  local name="$1"; shift
  if [ -f "runs/$name/summary.json" ]; then echo "SKIP  $name"; return 0; fi
  echo "START $name  $(date -u +%H:%M:%S)"
  if uv run train_flight.py --name "$name" "$@" > "logs/$name.log" 2>&1; then
    echo "OK    $name  $(date -u +%H:%M:%S)"
  else
    echo "FAIL  $name rc=$? -- see logs/$name.log"; fail=$((fail+1))
  fi
}
run_one mars_res_s0  --world mars  --seed 0 --total-steps 4000000 --eval-episodes 96
run_one mars_res_s1  --world mars  --seed 1 --total-steps 4000000 --eval-episodes 96
run_one earth_res_s0 --world earth --seed 0 --total-steps 4000000 --eval-episodes 96
for W in mars earth; do
  [ -f "logs/classical_${W}.log" ] && { echo "SKIP classical $W"; continue; }
  echo "CLASSICAL $W  $(date -u +%H:%M:%S)"
  uv run classical.py --world "$W" --episodes 96 > "logs/classical_${W}.log" 2>&1 \
    || { echo "FAIL classical $W"; fail=$((fail+1)); }
done
if [ "$fail" -eq 0 ]; then echo "ALL ARMS COMPLETE"; else echo "SWEEP FINISHED WITH $fail FAILURE(S)"; fi
