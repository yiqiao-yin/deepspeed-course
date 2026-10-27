#!/usr/bin/env bash
# The three PPO arms of lab 7, two seeds each.
#
#   ./run_arms.sh                 # all six, sequentially
#   ./run_arms.sh oracle          # one arm, both seeds
#
# SEQUENTIAL ON PURPOSE. Each run owns a 16-process pool and the box
# has 20 cores, so two concurrent runs would oversubscribe and both
# would report throughput that says nothing about either. Topic 7
# lab 5 measured the pool at 2.5x a serial loop; that number only
# holds when the pool has the cores to itself.
#
# The DAgger and behaviour-cloning arms are not here -- they need a
# trained teacher first. Run `./run_distill.sh` after this.
set -euo pipefail
cd "$(dirname "$0")"

# export WANDB_API_KEY="your_key_here"      # optional, and must stay commented

STEPS="${STEPS:-4000000}"
ENVS="${ENVS:-16}"
SEEDS="${SEEDS:-0 1}"
ARMS="${1:-oracle onboard deaf}"

mkdir -p logs
echo "arms: $ARMS   seeds: $SEEDS   steps: $STEPS   workers: $ENVS"

for sense in $ARMS; do
  for seed in $SEEDS; do
    name="${sense}_s${seed}"
    echo "START $name  $(date +%T)"
    uv run train_policy.py \
      --sense "$sense" \
      --seed "$seed" \
      --name "$name" \
      --total-steps "$STEPS" \
      --n-envs "$ENVS" \
      > "logs/${name}.log" 2>&1
    echo "OK    $name  $(date +%T)  $(grep -o 'HELD-OUT.*' "logs/${name}.log" | head -1)"
  done
done
echo "ALL PPO ARMS COMPLETE"
