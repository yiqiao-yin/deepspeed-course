#!/bin/bash
# SLURM batch script for CIFAR-10 CNN training with DeepSpeed
# This script trains an enhanced CNN on CIFAR-10 dataset (real-world images)
# Achieves 81% accuracy (Excellent tier) with BatchNorm + SGD optimization

#SBATCH --gres=gpu:2
# Request 2 GPUs for faster training (CIFAR-10 has 50K training samples)

#SBATCH --partition=h200-low
# Submit to the "h200-low" partition/queue

#SBATCH --time=02:00:00
# Maximum wall-clock time: 2 hours
# Training 50 epochs takes ~30-40 minutes on 2 GPUs

#SBATCH --job-name=cifar10_cnn_ds
# Job name: CIFAR-10 CNN with DeepSpeed

#SBATCH --ntasks-per-node=1
# Number of tasks per node (DeepSpeed handles multi-GPU internally)

#SBATCH --cpus-per-task=8
# Number of CPU cores (8 cores for efficient data loading)

#SBATCH --mem=32G
# Total memory per node: 32 GB
# CIFAR-10 dataset ~200 MB, model ~300K params, but need space for caching

#SBATCH --output=logs/cifar10_cnn_%j.out
# Standard output log

#SBATCH --error=logs/cifar10_cnn_%j.err
# Standard error log

# Create logs directory if it doesn't exist
mkdir -p logs

# Export HuggingFace token (commented out - not needed for CIFAR-10)
# export HF_TOKEN=<ENTER_KEY_HERE>

# Export Weights & Biases API key for experiment tracking
# REQUIRED: This script uses comprehensive W&B logging
# Get your API key from: https://wandb.ai/authorize
export WANDB_API_KEY=<ENTER_KEY_HERE>

# Activate Python virtual environment
# Update this path to your actual virtual environment location
source ~/myenv/bin/activate

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=================================================="

# Clean up previous training artifacts (optional)
# Uncomment if you want a fresh start each run
# echo "Cleaning up previous training data..."
# rm -rf ./data ./checkpoints ./wandb
# echo "Cleanup complete."

# Launch CIFAR-10 training with DeepSpeed
# Features:
# - BatchNormalization for stability
# - SGD optimizer (lr=0.01, momentum=0.9)
# - Gradient clipping (1.0)
# - FP32 precision for numerical stability
# - Early stopping with patience=15
# - Comprehensive W&B tracking
echo "Starting DeepSpeed CIFAR-10 training..."
echo "Expected result: 81% accuracy (Excellent tier)"
deepspeed --num_gpus=2 cifar10_deepspeed.py

echo "=================================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "Check W&B dashboard for detailed metrics and visualizations"
echo "=================================================="
