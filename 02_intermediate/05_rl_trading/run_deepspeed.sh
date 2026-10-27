#!/usr/bin/env bash
#SBATCH --job-name=rl_trading
#SBATCH --partition=h200-low           # CoreWeave queue; change to yours
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # ONE task. A launcher spawns its own
                                       # workers; N tasks x N GPUs gives N^2
                                       # processes and usually a hang.
#SBATCH --gres=gpu:1                   # nearly decoration -- see below
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rl_trading_%j.out
#SBATCH --error=logs/rl_trading_%j.err
#
# Train the trading arms, score them, draw the figures.
#
#   sbatch run_deepspeed.sh
#   sbatch run_deepspeed.sh --dry-run      # ~1 min, proves it assembles
#
# RunPod and other single-pod platforms: the #SBATCH lines are inert
# comments, so this runs as a plain shell script.
#
# THIS JOB DOES NOT NEED THE GPU, and `capacity.py` is the argument.
# The policies are 4,866 and 9,731 parameters -- 0.08 and 0.16 MB of
# weights, gradients and Adam state. ZeRO partitions exactly those
# three things, so sharding them across eight cards saves 0.01 MB a
# card and costs an all-gather every step. The real bottleneck is a
# SEQUENTIAL rollout over 2,992 trading days, because the position on
# day t depends on the action on day t-1, and that does not shard at
# all. Ask for cores.
#
# The card earns its place only at `--width 1024` and above, which
# builds the cross-sectional model `capacity.py` sizes. That is the
# arm `ds_config.json` exists for, and even there the measurement
# says activations are 3 to 6 times the thing ZeRO can touch.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

# Secrets stay COMMENTED and QUOTED. An uncommented
# `export WANDB_API_KEY=<KEY>` is a bash syntax error -- `<` is a
# redirection operator -- so the script would abort on that line and
# never reach training.
# export WANDB_API_KEY="your_key_here"

EXTRA="${*:-}"

echo "=============================================================="
echo "  RL TRADING"
echo "  node   $(hostname)"
echo "  cores  $(nproc)        <- the number that sets the wall clock"
echo "  gpu    $(command -v nvidia-smi >/dev/null 2>&1 \
                && nvidia-smi --query-gpu=name --format=csv,noheader \
                || echo 'none visible — fine, this lab does not need one')"
echo "=============================================================="

uv sync

# The panel, the bias it avoids, and the proof nothing reads forward.
uv run data.py
uv run data.py --bias
uv run data.py --leakage

# Everything the learned arms have to beat.
uv run baselines.py

# The two methods. One file, one flag.
uv run train_rl.py --method ppo  --seed 0 --name ppo_s0  $EXTRA
uv run train_rl.py --method grpo --seed 0 --name grpo_s0 $EXTRA
uv run train_rl.py --method grpo --neutral --seed 0 --name grpo_neutral_s0 $EXTRA

# Score them by regime, against the falsifiers.
uv run evaluate.py --json results.json

echo "done: $(date)"
