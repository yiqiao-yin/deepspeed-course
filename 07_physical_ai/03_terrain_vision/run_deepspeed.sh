#!/bin/bash
#SBATCH --job-name=terrain_vision    # shows in squeue
#SBATCH --nodes=1                       # one node
#SBATCH --ntasks-per-node=1             # one process: nothing to distribute
#SBATCH --cpus-per-task=16              # the RL half is MuJoCo on the CPU
#SBATCH --gres=gpu:1                    # ONE GPU, and only the student uses it
#SBATCH --mem=24G                       # the BC dataset is 23 MB; this is slack
#SBATCH --time=02:00:00                 # ~13 min a teacher, ~2 min the student
#SBATCH --partition=h200-low            # adjust to your cluster
#SBATCH --output=logs/terrain_vision_%j.out
#SBATCH --error=logs/terrain_vision_%j.err

# Teacher-student terrain locomotion with a depth camera.
#
# NOTE: no `deepspeed` launcher, deliberately -- the NINTH such exception
# in this course, and the first for a new reason. The other eight skip it
# because a GPU buys nothing. Here a GPU buys 13.6x +/- 0.9, measured
# over five repeats (see README.md), and the launcher is STILL wrong:
# the student is 193k parameters, so there is nothing for ZeRO to shard. "A GPU helps" and
# "DeepSpeed helps" are different claims, and this lab is the one place
# in the category where they come apart.
#
# A GPU is requested above because the student genuinely wants one. The
# two PPO teachers do NOT -- they run faster on the CPU, same as labs 1
# and 2 -- so they are simply left on it.

mkdir -p logs

# Optional experiment tracking. Keep this COMMENTED and QUOTED: an
# uncommented `export WANDB_API_KEY=<KEY>` is a bash syntax error, because
# `<` redirects, and the script would abort before reaching training.
# export WANDB_API_KEY="your_key_here"

set -euo pipefail
cd "$(dirname "$0")" || exit 1
uv sync

# Cheap dry run first: proves the pipeline assembles before anything long.
uv run train_teacher.py --dry-run

# --- the RL half, on the CPU ------------------------------------------------
# THREE SEEDS IS THE MINIMUM. Lab 1's finding was that one run per arm
# supports either conclusion, and this lab's whole claim is a difference
# BETWEEN arms -- the blind arm's ascent score across seeds is
# 0.375 / 0.000 / 0.000, which no single run would have told you.
for s in 0 1 2; do
  uv run train_teacher.py                 --seed "$s" --name "v3_priv_s$s"  --quiet
  uv run train_teacher.py --no-privileged --seed "$s" --name "v3_blind_s$s" --quiet
done

# --- the vision half, on the GPU --------------------------------------------
# Rendering costs 250x, so it happens ONCE here rather than inside a
# training loop. See the header of collect.py.
uv run collect.py --episodes 120
uv run train_student.py --device cuda --epochs 40

# The CPU-vs-GPU measurement this lab exists to make concrete.
uv run train_student.py --bench

uv run make_figures.py
uv run render.py --all
