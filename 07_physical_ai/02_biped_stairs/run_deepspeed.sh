#!/bin/bash
#SBATCH --job-name=biped_stairs      # shows in squeue
#SBATCH --nodes=1                       # one node
#SBATCH --ntasks-per-node=1             # one process: nothing to distribute
#SBATCH --cpus-per-task=16              # MuJoCo stepping is the bottleneck
#SBATCH --gres=gpu:0                    # NO GPU: see the README. CPU is 2.3x faster
#SBATCH --mem=16G                       # the whole world is an XML string
#SBATCH --time=01:00:00                 # ~8 min a run; 6 runs fits easily
#SBATCH --partition=cpu                 # adjust to your cluster
#SBATCH --output=logs/biped_stairs_%j.out
#SBATCH --error=logs/biped_stairs_%j.err

# PPO on a 3D MuJoCo staircase.
#
# NOTE: no `deepspeed` launcher, deliberately. The policy is ~10k
# parameters -- there is nothing for ZeRO to shard, and the bottleneck is
# physics on the CPU. See "Why there is no deepspeed launcher" in README.md.
# Requesting zero GPUs is not an oversight; it is the measured right answer.

mkdir -p logs

# Optional experiment tracking. Keep this COMMENTED and QUOTED: an
# uncommented `export WANDB_API_KEY=<KEY>` is a bash syntax error, because
# `<` redirects, and the script would abort before reaching training.
# export WANDB_API_KEY="your_key_here"

set -euo pipefail
cd "$(dirname "$0")" || exit 1
uv sync

# Cheap dry run first: proves the pipeline assembles before anything long.
uv run train_ppo.py --dry-run

# The whole 2x2: four cells x three seeds. THREE SEEDS IS THE MINIMUM --
# lab 1's finding was that one run per arm supports either conclusion.
uv run train_ppo.py --sweep --seeds 3 --jobs 6 --total-steps 800000

uv run make_figures.py
