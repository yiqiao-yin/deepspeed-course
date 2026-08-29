#!/bin/bash
# SLURM batch script — Online preference optimization (Online DPO / Nash-MD / XPO)
#
# CoreWeave / any SLURM cluster:
#     sbatch run_deepspeed.sh
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 06_huggingface_online_dpo \
#         --dry-run --collect --wait --terminate --yes
#
# No GPU? offline DPO's objectives are the place to start, and they need none:
#     uv run 05_huggingface_dpo/preference_losses.py

#SBATCH --gres=gpu:2
# 2 GPUs. Online methods GENERATE during training and hold a judge as
# well as a reference — budget like GRPO, not like offline DPO.

#SBATCH --partition=h200-low
# Update to match your cluster's partitions (check with: sinfo)

#SBATCH --time=06:00:00

#SBATCH --job-name=online_dpo

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=8

#SBATCH --mem=96G

#SBATCH --output=logs/online_dpo_%j.out
#SBATCH --error=logs/online_dpo_%j.err

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
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu121
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

NUM_GPUS="${NUM_GPUS:-2}"
METHOD="${METHOD:-online_dpo}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

deepspeed --num_gpus="${NUM_GPUS}" train_online_dpo.py \
    --deepspeed ds_config.json \
    --method "${METHOD}" \
    --model "${MODEL}" \
    "$@"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
