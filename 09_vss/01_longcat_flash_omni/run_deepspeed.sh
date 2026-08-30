#!/bin/bash
# SLURM batch script for video-speech-to-speech, LongCat-Flash-Omni 560B
# ZeRO-3 with full CPU offload. Gated on HOST RAM, not GPUs.

#SBATCH --gres=gpu:2
# 2 GPUs, but see the memory note — RAM is the real constraint.

#SBATCH --partition=h200-low
# Update this to match your cluster's partition names (check with: sinfo)

#SBATCH --time=24:00:00
# Maximum wall-clock time. Expect 30-60 min/epoch; offload traffic crosses PCIe.

#SBATCH --job-name=vss_longcat

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=32
# CPU cores for the data pipeline. Too few starves the dataloader and leaves
# the GPU idle between batches.

#SBATCH --mem=3000G
# Host memory. ~3 TB. This is NOT optional: 1.1 TB of BF16 weights live in host
#   memory under ZeRO-3 offload. Under-provisioning does not degrade
#   gracefully — the host swaps and throughput effectively stops.

#SBATCH --output=logs/vss_longcat_%j.out
#SBATCH --error=logs/vss_longcat_%j.err

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
#   uv pip install deepspeed transformers datasets accelerate trl peft torchaudio opencv-python-headless
source ~/myenv/bin/activate

# Point the HuggingFace cache at scratch. $HOME is usually a small NFS quota,
# and a multi-GB model download into it fails slowly.
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

# ~1.1 TB of weights. Pre-fetch on a LOGIN node and point HF_HOME at a
# volume with 2 TB+ free, or the download fails late.

# Optional experiment tracking. Leave unset to skip it; the scripts handle that.
# export WANDB_API_KEY="your_value_here"

# Verify storage before starting; the script refuses if short.
./check_storage.sh || exit 1

nvidia-smi

# "$@" forwards sbatch's extra arguments to the training script, so a
# cluster user can dry-run without burning a full allocation.
deepspeed --num_gpus=2 train_ds_2xB200.py "$@"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
