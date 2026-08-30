#!/bin/bash
# SLURM batch script for basic CNN training with DeepSpeed
# This script trains a simple CNN on synthetic MNIST-like data

#SBATCH --gres=gpu:1
# Request 1 GPU (sufficient for small CNN ~33K parameters)

#SBATCH --partition=h200-low
# Submit to the "h200-low" partition/queue

#SBATCH --time=01:00:00
# Maximum wall-clock time: 1 hour
# CNN training with 50 epochs takes ~30-45 minutes

#SBATCH --job-name=basic_cnn_ds
# Job name: basic CNN with DeepSpeed

#SBATCH --ntasks-per-node=1
# Number of tasks per node

#SBATCH --cpus-per-task=4
# Number of CPU cores for data loading

#SBATCH --mem=20G
# Total memory per node: 20 GB (includes model + synthetic data)

#SBATCH --output=logs/basic_cnn_%j.out
# Standard output log

#SBATCH --error=logs/basic_cnn_%j.err
# Standard error log

# Create logs directory if it doesn't exist
mkdir -p logs

# Export HuggingFace token (commented out - not needed for this example)
# export HF_TOKEN="your_value_here"

# Export Weights & Biases API key for experiment tracking
# Get your API key from: https://wandb.ai/authorize
# Set this to enable WANDB_API_KEY (optional; scripts skip it when unset):
# export WANDB_API_KEY="your_value_here"

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

# Launch CNN training with DeepSpeed
# Uses train_ds.py with enhanced features (Kaiming init, LR scheduling, etc.)
echo "Starting DeepSpeed CNN training..."
# "$@" forwards sbatch's extra arguments to the training script, which
# is what makes `sbatch run_deepspeed.sh --max-steps 20` a real dry run.
# Without it the flag is silently swallowed and the full job runs.
deepspeed --num_gpus=1 train_ds.py "$@"

echo "=================================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=================================================="
