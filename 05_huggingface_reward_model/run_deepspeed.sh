#!/bin/bash
# SLURM batch script — Reward model training (Bradley-Terry on preference pairs)
#
# CoreWeave / any SLURM cluster:
#     sbatch run_deepspeed.sh
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 05_huggingface_reward_model \
#         --dry-run --collect --wait --terminate --yes
#
# No GPU? the Bradley-Terry objective runs on CPU:
#     uv run 05_huggingface_reward_model/reward_modeling.py
#     uv run tests/test_reward_model.py          # no GPU, no download

#SBATCH --gres=gpu:1
# ONE GPU. A reward model is a sequence classifier with a scalar head —
# far lighter than the policy it will later score.

#SBATCH --partition=h200-low
# Update to match your cluster's partitions (check with: sinfo)

#SBATCH --time=03:00:00

#SBATCH --job-name=reward_model

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --output=logs/reward_model_%j.out
#SBATCH --error=logs/reward_model_%j.err

set -euo pipefail

mkdir -p logs

echo "=================================================="
echo "Job ID:   ${SLURM_JOB_ID:-none}"
echo "Node:     ${SLURM_NODELIST:-local}"
echo "GPUs:     ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start:    $(date)"
echo "=================================================="

# Environment, built ONCE on a LOGIN node with uv. Compute nodes usually have
# no network egress, so building it inside the job fails.
#   uv venv ~/myenv && source ~/myenv/bin/activate
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu128
#   uv pip install deepspeed transformers trl peft accelerate datasets
if [ -f ~/myenv/bin/activate ]; then
    # shellcheck disable=SC1090
    source ~/myenv/bin/activate
fi

# $HOME is usually a small NFS quota and a multi-GB model download into it
# fails slowly. Point the cache at scratch.
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

# Optional experiment tracking. Leave COMMENTED — an uncommented
# `export WANDB_API_KEY=<KEY>` is a bash syntax error, because `<` redirects.
# export WANDB_API_KEY="your_value_here"

nvidia-smi

NUM_GPUS="${NUM_GPUS:-1}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

deepspeed --num_gpus="${NUM_GPUS}" train_reward_model.py \
    --deepspeed ds_config.json \
    --model "${MODEL}" \
    "$@"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
