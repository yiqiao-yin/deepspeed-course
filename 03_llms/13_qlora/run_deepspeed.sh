#!/bin/bash
# SLURM batch script — QLoRA vs LoRA
#
# CoreWeave / any SLURM cluster:
#     sbatch run_deepspeed.sh
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 03_llms/13_qlora \
#         --collect --wait --terminate --yes

#SBATCH --gres=gpu:1
# One GPU is enough to compare per-GPU memory; more ranks each need 24 GB.

#SBATCH --partition=h200-low
# Update to match your cluster's partitions (check with: sinfo)

#SBATCH --time=02:00:00
# Wall-clock ceiling. The job is killed at this point, so overestimate.

#SBATCH --job-name=qlora

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=8
# Cores for the data pipeline. Too few starves the dataloader and the GPU
# idles between batches — which looks like a slow model and is not.

#SBATCH --mem=48G

#SBATCH --output=logs/qlora_%j.out
#SBATCH --error=logs/qlora_%j.err

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs

echo "=================================================="
echo "Job ID:   ${SLURM_JOB_ID:-none}"
echo "Node:     ${SLURM_NODELIST:-local}"
echo "GPUs:     ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start:    $(date)"
echo "=================================================="

# Run uv sync on the login node before submitting; compute nodes may have
# no download access. uv run uses the project's locked environment.

# $HOME is usually a small NFS quota and a multi-GB model download into it
# fails slowly. Point the cache at scratch.
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

# Credentials, if your example needs them. KEEP THESE COMMENTED AND QUOTED.
# An uncommented `export HF_TOKEN=<ENTER_KEY_HERE>` is a bash SYNTAX ERROR —
# `<` is a redirection operator — so the script aborts on that line and never
# reaches the training command. Seven scripts shipped that way once and could
# never run. tests/test_runpod_ctl.py runs `bash -n` over every shell script
# to stop it recurring.
# export HF_TOKEN="your_value_here"
# export WANDB_API_KEY="your_value_here"

nvidia-smi

NUM_GPUS="${NUM_GPUS:-1}"

uv run --locked deepspeed --num_gpus="${NUM_GPUS}" train_qlora.py \
    --deepspeed ds_config.json \
    "$@"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
