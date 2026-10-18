#!/bin/bash
#SBATCH --job-name=aerial_stab
#SBATCH --partition=h200-low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/aerial_stab_%j.out
#SBATCH --error=logs/aerial_stab_%j.err

# ---------------------------------------------------------------------------
# Topic 8 Lab 1 — a cold-gas car that must land on its wheels.
#
# THERE IS NO `deepspeed` CALL IN THIS FILE, AND NO --gres=gpu EITHER.
# Both omissions are deliberate and both are MEASURED rather than assumed.
#
#   * The policy is 13,013 parameters. ZeRO shards optimizer state,
#     gradients and parameters; all three are negligible here, so
#     there is nothing to shard.
#   * Topic 7 measured a policy this size running 1.7x SLOWER on a
#     GPU than on the CPU, because the rollout batch never fills a
#     card. What paid was a process pool -- 4.70x on 20 cores,
#     asserted bit-identical to the serial loop first.
#
# So this asks for CORES, not a card. If you have a GPU node spare,
# `--device cuda` exists so you can check the claim rather than
# believe it.
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs

# Credentials stay COMMENTED and QUOTED. An uncommented
# `export WANDB_API_KEY=<KEY>` is a bash syntax error -- `<` is a
# redirection operator -- so the script would abort on that line and
# never reach training. Seven SLURM scripts in this repo shipped that
# way and could never run.
# export WANDB_API_KEY="your_key_here"

echo "=============================================================="
echo "  Aerial stabilization — cold-gas car, PPO, CPU"
echo "=============================================================="
nproc && free -g | head -2

cd "$(dirname "$0")"
uv sync --frozen

# FREE CHECKS FIRST, in increasing order of cost. Each can fail the
# run before any compute is spent on it.
uv run thrusters.py --all          # can the rig even do the job?
uv run ppo.py                      # does the learner still have its properties?
uv run vecenv.py                   # is the pool bit-identical to the loop?

# The cheap dry run the contract asks for.
uv run train_aerial.py --dry-run

N_ENVS="${N_ENVS:-$(( $(nproc) < 16 ? $(nproc) : 16 ))}"
STEPS="${STEPS:-4000000}"

uv run train_aerial.py --rig "${RIG:-reveal}" --name "${NAME:-r_reveal}" \
    --total-steps "$STEPS" --n-envs "$N_ENVS" "$@"

uv run evaluate.py "runs/${NAME:-r_reveal}" --episodes 96 --compare

uv run render_flight.py "runs/${NAME:-r_reveal}" || \
    echo "  (no OpenGL on this node — the measurements above still stand)"

echo "  done."
