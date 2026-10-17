#!/bin/bash
#SBATCH --job-name=perception_pays
#SBATCH --partition=h200-low
#SBATCH --gres=gpu:1                   # animation only; the sweep is CPU
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/perception_pays_%j.out
#SBATCH --error=logs/perception_pays_%j.err

# ---------------------------------------------------------------------------
# Lab 10: how good must perception be before it pays? (It does not.)
#
# THERE IS NO `deepspeed` CALL IN THIS FILE, AND THAT IS DELIBERATE.
#
# Fifteenth `launcher="python"` example. NOTHING TRAINS HERE -- the
# lab replays lab 9's policies and varies a router. No parameters,
# no gradients, nothing for ZeRO to shard.
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs

# Credentials stay COMMENTED and QUOTED; an uncommented
# `export WANDB_API_KEY=<KEY>` is a bash syntax error.
# export WANDB_API_KEY="your_key_here"

echo "=============================================================="
echo "  Does perception pay?"
echo "=============================================================="
nproc && free -g | head -2

cd "$(dirname "$0")"
uv sync --frozen

# The decidability check FIRST. A sweep over tasks where belief
# cannot change the route produces a clean curve of nothing.
uv run perception_sweep.py --check "$@"

uv run perception_sweep.py --seeds "${SEEDS:-16}" "$@"

uv run render_beliefs.py || \
    echo "  (no OpenGL on this node — the measurements above still stand)"

echo "  done."
