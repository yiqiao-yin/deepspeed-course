#!/bin/bash
#SBATCH --job-name=obstacle_hopper      # shows in squeue
#SBATCH --nodes=1                       # one node
#SBATCH --ntasks-per-node=1             # one process: nothing to distribute
#SBATCH --cpus-per-task=16              # MuJoCo stepping is the bottleneck
#SBATCH --gres=gpu:0                    # NO GPU: CPU is 2.0x faster at 4 envs, 1.7x at 16
#SBATCH --mem=16G                       # the whole world is an XML string
#SBATCH --time=01:00:00                 # ~8 min a run; 6 runs fits easily
#SBATCH --partition=cpu                 # adjust to your cluster
#SBATCH --output=logs/obstacle_hopper_%j.out
#SBATCH --error=logs/obstacle_hopper_%j.err

# PPO on a 3D MuJoCo obstacle course.
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

# Three seeds per arm. One run is not evidence -- the seed spread on this
# task is larger than the effect anybody wants to report.
for seed in 0 1 2; do
    uv run train_ppo.py --total-steps 600000 --seed "$seed" --name "seeing_s${seed}"
    uv run train_ppo.py --total-steps 600000 --seed "$seed" --blind-to-height \
        --name "blind_s${seed}"
done

uv run make_figures.py
