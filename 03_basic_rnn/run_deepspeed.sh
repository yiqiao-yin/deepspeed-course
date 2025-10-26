#!/bin/bash
# SLURM batch script for LSTM/RNN training with DeepSpeed
# This script trains a 2-layer LSTM on synthetic time series data (sine waves)
# Uses ZeRO-2 optimization and FP16 mixed precision

#SBATCH --gres=gpu:2
# Request 2 GPUs (efficient for ZeRO-2 gradient partitioning)

#SBATCH --partition=h200-low
# Submit to the "h200-low" partition/queue

#SBATCH --time=01:00:00
# Maximum wall-clock time: 1 hour
# LSTM training 50 epochs takes ~5-10 minutes per GPU

#SBATCH --job-name=lstm_rnn_ds
# Job name: LSTM with DeepSpeed

#SBATCH --ntasks-per-node=1
# Number of tasks per node (DeepSpeed handles multi-GPU coordination)

#SBATCH --cpus-per-task=8
# Number of CPU cores (8 for efficient sequence processing)

#SBATCH --mem=24G
# Total memory per node: 24 GB
# Includes LSTM hidden states and sequence batches

#SBATCH --output=logs/lstm_rnn_%j.out
# Standard output log

#SBATCH --error=logs/lstm_rnn_%j.err
# Standard error log

# Create logs directory if it doesn't exist
mkdir -p logs

# Export HuggingFace token (commented out - not needed for synthetic data)
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

# Launch LSTM training with DeepSpeed
# Features:
# - Proper LSTM initialization (Xavier + Orthogonal)
# - ZeRO-2 optimization for memory efficiency
# - FP16 mixed precision training
# - Gradient clipping (1.0) for RNN stability
# - Validation set evaluation
# - Early stopping with patience=10
# - Comprehensive W&B tracking
echo "Starting DeepSpeed LSTM training..."
echo "Task: Time series prediction (multi-frequency sine wave)"
echo "Expected result: MSE < 0.05 (Excellent tier)"
deepspeed --num_gpus=2 train_rnn_deepspeed.py

echo "=================================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "Check W&B dashboard for loss curves and gradient tracking"
echo "=================================================="
