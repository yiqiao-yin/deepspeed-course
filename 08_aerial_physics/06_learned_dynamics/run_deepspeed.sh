#!/bin/bash
#SBATCH --job-name=learned_dynamics
#SBATCH --partition=h200-low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/learned_dynamics_%j.out
#SBATCH --error=logs/learned_dynamics_%j.err

# ---------------------------------------------------------------------------
# Topic 8 Lab 6 — can a big model learn this vehicle's physics?
#
# THE FIRST --gres=gpu IN TOPIC 8, and the first real require_gpu().
# Labs 1 to 5 train a ~14k-parameter policy against MuJoCo and are
# measurably FASTER on a CPU, so they ask for cores and carry no
# guard. The largest rung here is 705M parameters -- 11.3 GB of
# weights and Adam state before a single activation, measured at
# 15.97 GB of a 16 GB card -- so the card is real.
#
# STILL NO `deepspeed` LAUNCHER, and that is the lab's RESULT rather
# than an omission. ZeRO-2 at world size 1 was measured here at
# 1.81x the memory and 1.12x the wall clock of plain torch, for the
# same accuracy: with one card there is nobody to partition with.
# `--deepspeed` exists so the reader can reproduce that.
#
# The guard is SIZE-AWARE. tiny/small/medium run on a CPU and are
# not blocked, because a guard that fires for every size would make
# Clawdeck advertise a CPU-runnable command that cannot run.
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
uv run dynamics.py                # the size ladder and its memory
uv run baseline.py || true        # needs data/; harmless before collection

# Data. Disjoint seeds for the validation split, deliberately.
[ -f data/train.npz ] || uv run collect.py --episodes 1200 --out data/train.npz
[ -f data/val.npz ] || uv run collect.py --episodes 300 --seed0 500000 \
    --out data/val.npz

uv run baseline.py                # what "learned nothing" scores

EPOCHS="${EPOCHS:-8}"
BATCH="${BATCH:-256}"

# The ladder. An OOM is recorded rather than fatal.
EPOCHS="$EPOCHS" BATCH="$BATCH" bash run_ladder.sh

# Is "bigger is worse" the optimiser rather than the problem?
bash lr_check.sh

# DeepSpeed, actually run, so the verdict is measured not estimated.
uv pip install "deepspeed>=0.14" && \
  uv run train_dyn.py --kind mlp --size xxl --lr 1e-4 --epochs "$EPOCHS" \
      --batch "$BATCH" --deepspeed --zero-stage 2 --name mlp_xxl_ds2 "$@"

uv run ladder.py                  # collate and judge against the bar

echo "  done."
