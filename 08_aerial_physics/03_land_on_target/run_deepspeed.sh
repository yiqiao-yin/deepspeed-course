#!/bin/bash
#SBATCH --job-name=land_on_target
#SBATCH --partition=h200-low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/land_on_target_%j.out
#SBATCH --error=logs/land_on_target_%j.err

# ---------------------------------------------------------------------------
# Topic 8 Lab 3 — put the car down on a marked spot, and ask whether
# seeing the ground helps.
#
# NO `deepspeed` CALL AND NO --gres=gpu, both deliberate and both
# measured. The policy is ~13k parameters: ZeRO shards optimizer
# state, gradients and parameters, and all three are negligible.
# Topic 7 measured a policy this size running 1.7x SLOWER on a GPU
# because the rollout batch never fills a card. This asks for CORES.
#
# TWO THINGS HERE ARE LOAD-BEARING AND BOTH WERE MEASURED.
#
# THE RIG. `dense` and `fore10` are the only two layouts in the set
# with no dead wrench axis, so either can translate in any direction.
# They are NOT interchangeable for learning: on this task `fore10`
# reaches 6% on-target at 2.8M steps where `dense` reaches 37%, and
# lab 1's own run history shows fore10 failing to break out until
# 3.9M steps on the EASIER task. Do not swap the rig to save
# nozzles.
#
# THE ARMS DIFFER ONLY IN --sense. blind / slope / patch change the
# observation and nothing else: same rig, same terrain, same reward,
# same budget, same seeds. Anything else that moves between arms
# invalidates the comparison the lab exists to make.
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
uv run thrusters.py --all        # what the rig can and cannot do
uv run target_env.py --decidable # is there a DECISION in this ground?
uv run verify.py                 # conservation residuals
uv run ppo.py                    # the learner's properties

uv run train_target.py --dry-run

RIG="${RIG:-dense}"
N_ENVS="${N_ENVS:-$(( $(nproc) < 16 ? $(nproc) : 16 ))}"
STEPS="${STEPS:-4000000}"

# The three arms. One seed here; the published numbers use two.
for SENSE in blind slope patch; do
    uv run train_target.py --rig "$RIG" --sense "$SENSE" \
        --name "${SENSE}_s0" --seed 0 \
        --total-steps "$STEPS" --n-envs "$N_ENVS" \
        --eval-episodes 96 "$@"
done

# The engineered baseline on the SAME rig and terrain, swept over
# how far it will divert to reach survivable ground. "Does terrain
# knowledge help" and "does THIS controller use it well" are
# different questions, and only the sweep separates them.
for D in 0 2 4 6; do
    uv run classical.py --rig "$RIG" --preset badlands \
        --episodes 64 --max-divert "$D"
done

uv run render_target.py runs/blind_s0 --episodes 4 || \
    echo "  (no OpenGL on this node — the measurements above still stand)"

echo "  done."
