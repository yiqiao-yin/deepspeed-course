#!/bin/bash
#SBATCH --job-name=body_search
#SBATCH --partition=h200-low
#SBATCH --gres=gpu:1                   # animations only; the search is CPU
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20             # the search IS cpu-bound
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/body_search_%j.out
#SBATCH --error=logs/body_search_%j.err

# ---------------------------------------------------------------------------
# Lab 9: search the space of bodies instead of guessing five of them.
#
# THERE IS NO `deepspeed` CALL IN THIS FILE, AND THAT IS DELIBERATE.
#
# Fourteenth `launcher="python"` example in the course. The policies
# are 15-45k parameters and the bottleneck is MuJoCo physics on the
# CPU -- measured, not assumed: lab 6 puts physics at 49-56% of a
# training step and GPU physics at 2.06x against a 5x bar. There is
# nothing for ZeRO to shard.
#
# The free win here was not hardware. Switching the integrator from
# RK4 to `implicitfast` -- which is what MuJoCo recommends for
# contact-rich models -- made training 1.7-2.6x faster end to end,
# MORE than the ~1.40x eight rented GPUs would have bought.
#
# `--cpus-per-task=20` matters far more than the GPU request.
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs

# Credentials stay COMMENTED and QUOTED. An uncommented
# `export WANDB_API_KEY=<KEY>` is a bash syntax error -- `<` is a
# redirection operator -- so the script would abort on that line.
# export WANDB_API_KEY="your_key_here"

echo "=============================================================="
echo "  Body search — which shape walks furthest?"
echo "=============================================================="
nproc && free -g | head -2

cd "$(dirname "$0")"
uv sync --frozen

# 1. Free checks first. A body generator that silently drops a
#    feature makes every number below meaningless.
uv run ../../tests/test_bodyspec.py
uv run search_bodies.py --stage count --plan "$@"

# 2. The speed measurement that makes the search affordable.
uv run speed_physics.py "$@"

# 3. The search itself. Bodies already trained are skipped, so this
#    is resumable after an interruption.
uv run search_bodies.py --stage count "$@"

# 4. Rank on fresh episodes with intervals, then film it.
uv run rank_bodies.py --episodes "${EPISODES:-36}"
uv run render_search.py || \
    echo "  (no OpenGL on this node — the measurements above still stand)"

echo "  done."
