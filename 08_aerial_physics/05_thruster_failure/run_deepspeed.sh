#!/bin/bash
#SBATCH --job-name=thruster_failure
#SBATCH --partition=h200-low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/thruster_failure_%j.out
#SBATCH --error=logs/thruster_failure_%j.err

# ---------------------------------------------------------------------------
# Topic 8 Lab 5 — land the car after a thruster breaks in mid-air.
#
# NO `deepspeed` CALL AND NO --gres=gpu, both deliberate and both
# measured. The policy is ~14k parameters: ZeRO shards optimizer
# state, gradients and parameters, and all three are negligible.
# Topic 7 measured a policy this size running 1.7x SLOWER on a GPU
# because the rollout batch never fills a card. This asks for CORES.
#
# THE CURRICULUM IS NOT OPTIONAL. Trained from scratch with a
# failure in every episode the policy never learns to land at all --
# 0.8% at 1.35M steps -- because on `reveal` about half of all
# (nozzle, mode) draws leave no hover reachable, so half the signal
# comes from episodes nothing could save. Learn the clean landing
# first, then warm start with failures on. Do not reorder these.
#
# THE ARMS DIFFER ONLY IN --fail-p AT TRAINING TIME. Both are scored
# with failures on, on the same held-out seeds, and the fault is
# drawn from the seed so both meet the same breakages.
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
uv run fault.py                   # which failures are survivable, by geometry
uv run thrusters.py --all         # what the rig can and cannot do
uv run verify.py                  # conservation residuals
uv run ppo.py                     # the learner's properties

uv run train_fail.py --dry-run

RIG="${RIG:-reveal}"
N_ENVS="${N_ENVS:-$(( $(nproc) < 16 ? $(nproc) : 16 ))}"
STEPS="${STEPS:-2500000}"

for SD in 0 1; do
    # The control: learn to land, no failures. Also the baseline.
    uv run train_fail.py --rig "$RIG" --fail-p 0.0 --seed "$SD" \
        --name "healthy_s${SD}" --total-steps "$STEPS" \
        --n-envs "$N_ENVS" --eval-episodes 128

    # The robust arm, warm started from it.
    uv run train_fail.py --rig "$RIG" --fail-p 1.0 --seed "$SD" \
        --name "robust_s${SD}" --init-from "runs/healthy_s${SD}" \
        --total-steps "$STEPS" --n-envs "$N_ENVS" \
        --eval-episodes 128 "$@"
done

# Did the cheap geometric check predict any of it?
uv run analyze.py runs/healthy_s0 runs/robust_s0 runs/robust_s1

uv run render_fail.py runs/robust_s0 --episodes 3 || \
    echo "  (no OpenGL on this node — the measurements above still stand)"

echo "  done."
