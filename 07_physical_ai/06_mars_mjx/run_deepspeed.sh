#!/bin/bash
#SBATCH --job-name=mars_mjx
#SBATCH --partition=h200-low
#SBATCH --gres=gpu:1                   # a real GPU: it is what is on trial
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20             # the CPU pool is the BASELINE here
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/mars_mjx_%j.out
#SBATCH --error=logs/mars_mjx_%j.err

# ---------------------------------------------------------------------------
# Lab 6: does GPU physics pay for the Mars course?
#
# THERE IS NO `deepspeed` CALL IN THIS FILE, AND THAT IS DELIBERATE.
#
# This is the twelfth `launcher="python"` example in the course and the
# second-clearest after lab 5 part 1: nothing here trains. The script
# steps a physics model on three backends and times it. No parameters,
# no gradients, no optimizer state, so there is nothing for ZeRO to
# shard and a distributed launcher would be cargo cult. The lab's
# subject is precisely whether the hardware DeepSpeed usually runs on
# helps at all here, and the measured answer is that it does not.
#
# `--cpus-per-task=20` matters more than the GPU request. The baseline
# is lab 5's 16-process worker pool, and starving it would manufacture
# a GPU win out of a thin CPU allocation. A benchmark whose baseline is
# handicapped is not a benchmark.
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs

# Credentials stay COMMENTED and QUOTED. An uncommented
# `export WANDB_API_KEY=<KEY>` is a bash syntax error -- `<` is a
# redirection operator -- so the script would abort on that line and
# never reach the work.
# export WANDB_API_KEY="your_key_here"

echo "=============================================================="
echo "  Mars on GPU physics — measure before renting"
echo "=============================================================="
nproc && free -g | head -2
nvidia-smi --query-gpu=name,memory.total --format=csv || \
    echo "  (no GPU visible — the CPU baselines below still run)"

cd "$(dirname "$0")"
uv sync --frozen

# The feasibility gate first: does MJX accept our height field at all,
# and does it produce the same physics? Cheap, and the thing worth
# failing on before any timing is believed.
uv run probe_mjx.py

# The benchmark proper, with the ablation that explains the result.
uv run bench_backends.py --json runs/bench.json

# The animation. Needs an OpenGL context, which a headless node may
# not have; the numbers above do not depend on it.
uv run render_swarm.py --tiles 16 --rough || \
    echo "  (no OpenGL on this node — the measurements above still stand)"

echo "  done."
