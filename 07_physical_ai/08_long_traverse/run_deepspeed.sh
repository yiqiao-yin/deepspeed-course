#!/bin/bash
#SBATCH --job-name=long_traverse
#SBATCH --partition=h200-low
#SBATCH --gres=gpu:1                   # only the animation needs it
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20             # the physics is CPU-bound
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/long_traverse_%j.out
#SBATCH --error=logs/long_traverse_%j.err

# ---------------------------------------------------------------------------
# Lab 8: five bodies, one task — the body is the bottleneck.
#
# THERE IS NO `deepspeed` CALL IN THIS FILE, AND THAT IS DELIBERATE.
#
# Thirteenth `launcher="python"` example in the course. Nothing here
# trains: the lab REPLAYS policies trained in lab 5 (which owns the
# robot) and compares five morphologies on one task. The policies are
# 15-22k parameters and the bottleneck is MuJoCo physics on the CPU,
# so there is nothing for ZeRO to shard. Lab 6 measured GPU physics at
# 1.50x against a 5x bar; lab 7 is where ZeRO-3 earns its place.
#
# `--cpus-per-task=20` matters more than the GPU request: the
# comparison is 5 bodies x 36 episodes of physics.
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs

# Credentials stay COMMENTED and QUOTED. An uncommented
# `export WANDB_API_KEY=<KEY>` is a bash syntax error -- `<` is a
# redirection operator -- so the script would abort on that line.
# export WANDB_API_KEY="your_key_here"

echo "=============================================================="
echo "  Long traverse — does the body decide?"
echo "=============================================================="
nproc && free -g | head -2

cd "$(dirname "$0")"
uv sync --frozen

# 1. The cheap checks first: a broken metric makes every number below
#    meaningless, which is exactly what happened the first time.
uv run traverse.py --check "$@"
uv run ../../tests/test_morphology.py

# 2. The comparison. Needs the lab 5 policies; says so if they are
#    missing rather than reporting a shorter table as if complete.
uv run compare_bodies.py --episodes "${EPISODES:-36}" "$@"

# 3. The animation, if this node can open a GL context at all.
uv run render_morphology.py || \
    echo "  (no OpenGL on this node — the measurements above still stand)"

echo "  done."
