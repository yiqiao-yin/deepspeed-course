#!/bin/bash
#SBATCH --job-name=fly_on_mars
#SBATCH --partition=h200-low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/fly_on_mars_%j.out
#SBATCH --error=logs/fly_on_mars_%j.err

# ---------------------------------------------------------------------------
# Topic 8 Lab 4 — fly a gate course on Mars, then land.
#
# NO `deepspeed` CALL AND NO --gres=gpu, both deliberate and both
# measured. The policy is ~15k parameters: ZeRO shards optimizer
# state, gradients and parameters, and all three are negligible.
# Topic 7 measured a policy this size running 1.7x SLOWER on a GPU
# because the rollout batch never fills a card. This asks for CORES.
#
# THE TWO ARMS DIFFER ONLY IN --world. Gravity, ambient pressure and
# air density change; the vehicle, rig, tank, reward, clock and
# course generator do not, and the generator is seeded identically
# so both worlds fly the SAME course. That is what makes Earth a
# control rather than a second experiment.
#
# TRAINING SAMPLES SHORTER LEGS THAN THE TASK CONTAINS. --leg-min
# defaults below the task's shortest leg purely to bootstrap
# exploration; `evaluate()` is called with the task range, so the
# published number is never inflated by episodes the task does not
# contain. Do not "tidy" that asymmetry away.
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs

# Credentials stay COMMENTED and QUOTED. An uncommented
# `export WANDB_API_KEY=<KEY>` is a bash syntax error -- `<` is a
# redirection operator -- so the script would abort on that line.
# export WANDB_API_KEY="your_key_here"

echo "=============================================================="
echo "  Land on target — cold-gas car, PPO, CPU"
echo "=============================================================="
nproc && free -g | head -2

cd "$(dirname "$0")"
uv sync --frozen

# Free checks first, cheapest to most expensive.
uv run flight_env.py --budget     # can the tank afford the course?
uv run thrusters.py --all         # what the rig can and cannot do
uv run verify.py                  # conservation residuals
uv run ppo.py                     # the learner's properties

uv run train_flight.py --dry-run

RIG="${RIG:-dense}"
N_ENVS="${N_ENVS:-$(( $(nproc) < 16 ? $(nproc) : 16 ))}"
STEPS="${STEPS:-4000000}"

# The two arms. Mars is the lab; Earth is the control.
for WORLD in mars earth; do
    uv run train_flight.py --rig "$RIG" --world "$WORLD" \
        --name "${WORLD}_s0" --seed 0 \
        --total-steps "$STEPS" --n-envs "$N_ENVS" \
        --eval-episodes 96 "$@"
done

# The scheduled guidance law on both worlds, same course, same seeds.
for WORLD in mars earth; do
    uv run classical.py --world "$WORLD" --episodes 96
done

uv run render_flight.py runs/mars_s0 --episodes 3 || \
    echo "  (no OpenGL on this node — the measurements above still stand)"

echo "  done."
