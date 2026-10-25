#!/usr/bin/env bash
# The two distillation arms of lab 7, two student seeds each.
#
#   ./run_distill.sh                       # bc and dagger, seeds 0 and 1
#   TEACHER=runs/oracle_s1 ./run_distill.sh
#
# Run AFTER ./run_arms.sh -- both arms need a trained teacher.
#
# BOTH ARMS DISTIL FROM THE SAME TEACHER, deliberately. The seeds
# here vary the STUDENT's initialisation and the rollout stream, not
# the teacher, so the comparison isolates the method. Varying the
# teacher too would confound "DAgger beats BC" with "this teacher
# was better", and the oracle arm's own seed spread is 60% against
# 31% -- more than large enough to swamp the effect being measured.
#
# The teacher is the BEST oracle seed rather than the average one.
# A teacher is not a measurement, it is an input: the lab's question
# is what a student can recover from a good one.
set -euo pipefail
cd "$(dirname "$0")"

# export WANDB_API_KEY="your_key_here"      # optional, and must stay commented

TEACHER="${TEACHER:-runs/oracle_s0}"
EPISODES="${EPISODES:-600}"
ITERS="${ITERS:-10}"
EPOCHS="${EPOCHS:-12}"
SEEDS="${SEEDS:-0 1}"
METHODS="${1:-bc dagger}"

mkdir -p logs
echo "teacher: $TEACHER   methods: $METHODS   seeds: $SEEDS"
echo "budget:  $EPISODES labelled episodes, $EPOCHS epochs per round"
echo "         bc spends all of it in one round; dagger splits it over $ITERS"

for method in $METHODS; do
  for seed in $SEEDS; do
    name="${method}_s${seed}"
    echo "START $name  $(date +%T)"
    uv run distill.py \
      --teacher "$TEACHER" \
      --method "$method" \
      --seed "$seed" \
      --name "$name" \
      --episodes "$EPISODES" \
      --iters "$ITERS" \
      --epochs "$EPOCHS" \
      > "logs/${name}.log" 2>&1
    echo "OK    $name  $(date +%T)  $(grep -o 'HELD-OUT.*' "logs/${name}.log" | head -1)"
  done
done
echo "ALL DISTILLATION ARMS COMPLETE"
