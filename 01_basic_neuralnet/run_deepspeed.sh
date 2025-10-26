#!/bin/bash
# SLURM batch script for basic neural network training with DeepSpeed
# This script trains a simple MLP on synthetic linear regression data

#SBATCH --gres=gpu:1
# Request 1 GPU (sufficient for small model ~150K parameters)

#SBATCH --partition=h200-low
# Submit to the "h200-low" partition/queue

#SBATCH --time=00:30:00
# Maximum wall-clock time: 30 minutes
# Basic neural net trains quickly (50-100 epochs)

#SBATCH --job-name=basic_nn_ds
# Job name: basic neural network with DeepSpeed

#SBATCH --ntasks-per-node=1
# Number of tasks per node (1 for single-process training)

#SBATCH --cpus-per-task=4
# Number of CPU cores (4 is sufficient for small datasets)

#SBATCH --mem=16G
# Total memory per node: 16 GB (more than enough for synthetic data)

#SBATCH --output=logs/basic_nn_%j.out
# Standard output log: logs/basic_nn_<job_id>.out

#SBATCH --error=logs/basic_nn_%j.err
# Standard error log: logs/basic_nn_<job_id>.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Export HuggingFace token (commented out - not needed for this example)
# export HF_TOKEN=<ENTER_KEY_HERE>

# Export Weights & Biases API key for experiment tracking
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

# Launch enhanced training with DeepSpeed
# Uses train_ds_enhanced.py with W&B tracking and early stopping
echo "Starting DeepSpeed training..."
deepspeed --num_gpus=1 train_ds_enhanced.py

# Alternative: Run basic version without W&B
# deepspeed --num_gpus=1 train_ds.py

echo "=================================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=================================================="
