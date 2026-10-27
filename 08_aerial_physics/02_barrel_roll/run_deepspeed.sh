#!/bin/bash
#SBATCH --job-name=barrel_roll
#SBATCH --partition=h200-low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/barrel_roll_%j.out
#SBATCH --error=logs/barrel_roll_%j.err

# ---------------------------------------------------------------------------
# Topic 8 Lab 2 — a full 360 degree barrel roll, then a landing.
#
# NO `deepspeed` CALL AND NO --gres=gpu, both deliberate and both
# measured. The policy is 13,397 parameters: ZeRO shards optimizer
# state, gradients and parameters, and all three are negligible.
# Topic 7 measured a policy this size running 1.7x SLOWER on a GPU
# because the rollout batch never fills a card. This asks for CORES.
#
# THE CURRICULUM ORDER IS LOAD-BEARING. Stage A learns the ROLL from
# scratch; stage B warm starts from it and adds the landing. Taught
# the other way round -- landing first -- the policy reaches 0% roll
# completion and stays there: a landing prior is the reflex "null the
# body rates", which destroys the launch rotation before any roll
# accumulates. Do not swap these stages.
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs

# Credentials stay COMMENTED and QUOTED. An uncommented
# `export WANDB_API_KEY=<KEY>` is a bash syntax error -- `<` is a
# redirection operator -- so the script would abort on that line.
# export WANDB_API_KEY="your_key_here"

echo "=============================================================="
echo "  Barrel roll — cold-gas car, PPO, CPU"
echo "=============================================================="
nproc && free -g | head -2

cd "$(dirname "$0")"
uv sync --frozen

# Free checks first, cheapest to most expensive.
uv run thrusters.py --all        # can the rig produce a pure couple?
uv run roll_env.py --budget      # can the tank afford a turn?
uv run verify.py                 # conservation residuals
uv run ppo.py                    # the learner's properties

uv run train_roll.py --dry-run

RIG="${RIG:-reveal}"
N_ENVS="${N_ENVS:-$(( $(nproc) < 16 ? $(nproc) : 16 ))}"

# Stage A: the roll, from scratch.
uv run train_roll.py --rig "$RIG" --name roller \
    --total-steps "${STEPS_A:-3000000}" --n-envs "$N_ENVS"

# Stage B: add the landing, warm started.
uv run train_roll.py --rig "$RIG" --name roll2land \
    --init-from runs/roller \
    --total-steps "${STEPS_B:-3000000}" --n-envs "$N_ENVS" \
    --eval-episodes 96 "$@"

uv run evaluate.py runs/roll2land --episodes 96 --compare

uv run render_roll.py runs/roll2land --episodes 4 || \
    echo "  (no OpenGL on this node — the measurements above still stand)"

echo "  done."
