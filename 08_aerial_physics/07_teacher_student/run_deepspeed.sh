#!/usr/bin/env bash
#SBATCH --job-name=ts_distill          # shows in squeue
#SBATCH --partition=h200-low           # CoreWeave queue; change to yours
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # ONE task. A launcher spawns its own
                                       # workers, so N tasks x N GPUs gives
                                       # N^2 processes and usually a hang.
#SBATCH --gres=gpu:1                   # one card is enough -- see below
#SBATCH --cpus-per-task=20             # THE RESOURCE THAT ACTUALLY MATTERS
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ts_distill_%j.out
#SBATCH --error=logs/ts_distill_%j.err
#
# Lab 7: train a teacher, then distil it into a student that only has
# a rangefinder.
#
#   sbatch run_deepspeed.sh                 # the full pipeline
#   sbatch run_deepspeed.sh --dry-run       # ~3 min, proves it assembles
#
# RunPod and other single-pod platforms: the #SBATCH lines above are
# inert comments, so this runs as a plain shell script.
#
# ASK FOR CORES, NOT CARDS, AND THE REASON IS MEASURED
# -----------------------------------------------------
# Every PPO arm here is a ~15k-parameter MLP stepped against MuJoCo
# on the CPU. Topic 7 measured a policy that size running 1.7x
# SLOWER on a GPU than on the CPU, because the rollout batch never
# fills a card, and measured a 16-process pool at 2.5x a serial
# loop. So --cpus-per-task is the line that decides how long this
# takes and --gres=gpu:1 is nearly decoration.
#
# It is not ENTIRELY decoration: `distill.py --size deep|xl` builds a
# student large enough to want the card, and that is the arm
# `ds_config.json` exists for. The default `--size match` is 17k
# parameters and runs on the CPU, deliberately -- the headline
# comparison is between METHODS, and a student with more capacity
# than the teacher would confound it.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

# Secrets stay COMMENTED and QUOTED. An uncommented
# `export WANDB_API_KEY=<KEY>` is a bash syntax error -- `<` is a
# redirection operator -- so the script would abort on that line and
# never reach training. Seven scripts in this repo shipped that way
# and could never run.
# export WANDB_API_KEY="your_key_here"

EXTRA="${*:-}"

echo "=============================================================="
echo "  LAB 7 — teacher and student"
echo "  node    $(hostname)"
echo "  cores   $(nproc)        <- the number that sets the wall clock"
echo "  gpu     $(command -v nvidia-smi >/dev/null 2>&1 \
                 && nvidia-smi --query-gpu=name --format=csv,noheader \
                 || echo 'none visible')"
echo "=============================================================="

uv sync

# 1. The teacher. Exact altitude and clearance -- information no
#    vehicle could carry. This is the ceiling everything else is
#    measured against.
uv run train_policy.py --sense oracle  --seed 0 --name oracle_s0  $EXTRA

# 2. The two controls, so the student's score is readable. Without
#    them a student at 48% could be most of what is achievable or
#    barely above what you get for free.
uv run train_policy.py --sense onboard --seed 0 --name onboard_s0 $EXTRA
uv run train_policy.py --sense deaf    --seed 0 --name deaf_s0    $EXTRA

# 3. The two distilled arms, from the SAME teacher and on the SAME
#    labelled budget. BC spends it all in one teacher-driven round;
#    DAgger splits it over ten and lets the student fly.
uv run distill.py --teacher runs/oracle_s0 --method bc     --name bc_s0     $EXTRA
uv run distill.py --teacher runs/oracle_s0 --method dagger --name dagger_s0 $EXTRA

# 4. Score them as MATCHED PAIRS and print the falsifier's verdict.
uv run compare.py --json compare.json

echo "done: $(date)"
