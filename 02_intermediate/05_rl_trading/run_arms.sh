#!/usr/bin/env bash
# Every learned arm of this lab, three seeds each.
#
#   ./run_arms.sh                  # all of them
#   SEEDS="0" ./run_arms.sh        # one seed, faster
#
# THREE SEEDS IS THE MINIMUM HERE and it is still thin. A single
# seed on a financial time series tells you about one optimisation
# trajectory over one history, and the spread between seeds is
# routinely larger than the effect being measured. Every published
# figure in the README is a median across these three with the
# spread quoted beside it.
set -euo pipefail
cd "$(dirname "$0")"

# export WANDB_API_KEY="your_key_here"      # optional, and must stay commented

SEEDS="${SEEDS:-0 1 2}"
ITERS="${ITERS:-120}"

mkdir -p logs
echo "seeds: $SEEDS   iters: $ITERS"

for seed in $SEEDS; do
  # PPO: a learned critic supplies the baseline.
  uv run train_rl.py --method ppo  --seed "$seed" --iters "$ITERS" \
      > "logs/ppo_s${seed}.log" 2>&1
  echo "OK ppo_s${seed}  $(grep -o 'HELD OUT.*' logs/ppo_s${seed}.log)"

  # GRPO: the basket mean supplies it, and that mean is the market.
  uv run train_rl.py --method grpo --seed "$seed" --iters "$ITERS" \
      > "logs/grpo_s${seed}.log" 2>&1
  echo "OK grpo_s${seed} $(grep -o 'HELD OUT.*' logs/grpo_s${seed}.log)"

  # GRPO with the net market exposure removed. The strictest test of
  # whether the group baseline learned anything RELATIVE: this book
  # cannot earn from the market going up, only from picking which
  # sector does better than the others.
  uv run train_rl.py --method grpo --neutral --seed "$seed" --iters "$ITERS" \
      > "logs/grpo_neutral_s${seed}.log" 2>&1
  echo "OK grpo_neutral_s${seed} $(grep -o 'HELD OUT.*' logs/grpo_neutral_s${seed}.log)"
done
echo "ALL ARMS COMPLETE"
