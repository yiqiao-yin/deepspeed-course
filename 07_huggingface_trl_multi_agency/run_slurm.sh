#!/bin/bash
# SLURM batch script for multi-agent GRPO (TRL, no DeepSpeed)
# This example drives TRL's GRPOTrainer directly — no deepspeed launcher

#SBATCH --gres=gpu:1
# Single GPU; the model is Qwen-1.5B.

#SBATCH --partition=h200-low
# Update this to match your cluster's partition names (check with: sinfo)

#SBATCH --time=04:00:00
# Maximum wall-clock time. RL rollouts dominate; scale with dataset size.

#SBATCH --job-name=multi_agent_grpo

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=8
# CPU cores for the data pipeline. Too few starves the dataloader and leaves
# the GPU idle between batches.

#SBATCH --mem=48G
# Host memory. Model plus rollout buffers.

#SBATCH --output=logs/multi_agent_grpo_%j.out
#SBATCH --error=logs/multi_agent_grpo_%j.err

mkdir -p logs

echo "=================================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
echo "Start:    $(date)"
echo "=================================================="

# Activate your environment (built once on a LOGIN node with uv):
#   uv venv ~/myenv && source ~/myenv/bin/activate
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu121
#   uv pip install deepspeed transformers datasets trl
source ~/myenv/bin/activate

# Point the HuggingFace cache at scratch. $HOME is usually a small NFS quota,
# and a multi-GB model download into it fails slowly.
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

# Pre-fetch model and dataset on a LOGIN node if compute nodes are air-gapped.
# export HF_HUB_OFFLINE=1

# Optional experiment tracking. Leave unset to skip it; the scripts handle that.
# export WANDB_API_KEY=<ENTER_KEY_HERE>

nvidia-smi

python train_grpo_math.py

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
