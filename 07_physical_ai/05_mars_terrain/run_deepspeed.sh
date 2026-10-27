#!/bin/bash
#SBATCH --job-name=mars_terrain
#SBATCH --partition=h200-low           # any partition; this lab needs no GPU
#SBATCH --gres=gpu:0                   # ZERO. See the note below.
#SBATCH --ntasks-per-node=1            # one task; nothing here spawns workers
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mars_terrain_%j.out
#SBATCH --error=logs/mars_terrain_%j.err

# ---------------------------------------------------------------------------
# Lab 5, Part 1: build a Mars-like surface and MEASURE it. No robot.
#
# THERE IS NO `deepspeed` CALL IN THIS FILE, AND THAT IS DELIBERATE.
#
# The file is named `run_deepspeed.sh` because every example in this
# course carries one and a reader coming from lab 4 should find the
# launcher where they expect it. But this lab trains nothing: it
# generates height fields with numpy and renders them with MuJoCo.
# There are no parameters, no gradients and no optimizer state, so
# there is nothing for ZeRO to shard. Invoking a distributed launcher
# here would be cargo cult, exactly as it would be for the five other
# examples in this repository that skip it for stated reasons.
#
# `--gres=gpu:0` is likewise deliberate. The generator is pure numpy.
# The renderer needs an OpenGL context, which on a headless node means
# EGL or OSMesa -- and `render_mars.py` probes for one and exits with a
# clear message if none is available, rather than dying inside MuJoCo.
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs

# Credentials stay COMMENTED and QUOTED. An uncommented
# `export WANDB_API_KEY=<KEY>` is a bash syntax error -- `<` is a
# redirection operator -- so the script would abort on that line and
# never reach the work. Seven SLURM scripts in this repo once shipped
# exactly that way and could not run at all.
# export WANDB_API_KEY="your_key_here"

echo "=============================================================="
echo "  Mars terrain — generate and measure"
echo "=============================================================="
nproc && free -g | head -2

cd "$(dirname "$0")"
uv sync --frozen

# The cheap path first: does the surface hold its properties? This is
# seconds, needs no graphics, and is the thing worth failing on.
uv run mars.py --check 12

# A look at one map, in text, so a log file is enough to see it.
uv run mars.py --scale rover --ascii

# Then the animations, if this node can open a GL context at all.
uv run render_mars.py --all || \
    echo "  (no OpenGL on this node — the measurements above still stand)"

echo "  done. Part 2 adds the robot."
