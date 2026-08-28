#!/bin/bash
# SLURM batch script for HuggingFace LLM fine-tuning with DeepSpeed
# Fine-tunes a HuggingFace causal LM using ZeRO

#SBATCH --gres=gpu:2
# Request 2 GPUs. Adjust to match your ds_config.json.

#SBATCH --partition=h200-low
# Update this to match your cluster's partition names (check with: sinfo)

#SBATCH --time=04:00:00
# Maximum wall-clock time. Dominated by the model download on a cold cache.

#SBATCH --job-name=hf_train

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=16
# CPU cores for the data pipeline. Too few starves the dataloader and leaves
# the GPU idle between batches.

#SBATCH --mem=64G
# Host memory. Model init, dataset, and optimizer states.

#SBATCH --output=logs/hf_train_%j.out
#SBATCH --error=logs/hf_train_%j.err

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
#   uv pip install deepspeed transformers datasets accelerate
source ~/myenv/bin/activate

# Point the HuggingFace cache at scratch. $HOME is usually a small NFS quota,
# and a multi-GB model download into it fails slowly.
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

# Compute nodes are often air-gapped. Pre-fetch the model on a LOGIN node,
# then uncomment these so a cache miss fails fast instead of hanging.
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# Optional experiment tracking. Leave unset to skip it; the scripts handle that.
# export WANDB_API_KEY=<ENTER_KEY_HERE>

nvidia-smi

deepspeed --num_gpus=2 train_ds.py

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
