#!/bin/bash
# SLURM batch script for Stock Price RNN training with DeepSpeed
# This script trains a 2-layer SimpleRNN on stock price delta data (moving averages)
# Uses ZeRO-2 optimization and FP16 mixed precision

#SBATCH --gres=gpu:2
# Request 2 GPUs (efficient for ZeRO-2 gradient partitioning)

#SBATCH --partition=h200-low
# Submit to the "h200-low" partition/queue

#SBATCH --time=02:00:00
# Maximum wall-clock time: 2 hours
# Includes stock data download and 50 epochs of training (~10-15 minutes per GPU)

#SBATCH --job-name=stock_rnn_ds
# Job name: Stock RNN with DeepSpeed

#SBATCH --ntasks-per-node=1
# Number of tasks per node (DeepSpeed handles multi-GPU coordination)

#SBATCH --cpus-per-task=8
# Number of CPU cores (8 for efficient stock data processing and sequence creation)

#SBATCH --mem=32G
# Total memory per node: 32 GB
# Includes stock data, moving average calculations, RNN hidden states, and sequence batches

#SBATCH --output=logs/stock_rnn_%j.out
# Standard output log

#SBATCH --error=logs/stock_rnn_%j.err
# Standard error log

# Create logs directory if it doesn't exist
mkdir -p logs

# Export HuggingFace token (commented out - not needed for stock data)
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

# Launch Stock RNN training with DeepSpeed
# Features:
# - Downloads AAPL stock data from Yahoo Finance (2015-2025)
# - Calculates moving averages: [14, 26, 50, 100, 200] periods
# - Predicts average delta (price - MA)
# - SimpleRNN initialization (Xavier + Orthogonal)
# - ZeRO-2 optimization for memory efficiency
# - FP16 mixed precision training
# - Gradient clipping (1.0) for RNN stability
# - Train/val/test split (70/15/15)
# - Early stopping with patience=10
# - Comprehensive W&B tracking with visualizations
echo "Starting DeepSpeed Stock RNN training..."
echo "Task: Stock price delta prediction"
echo "Ticker: AAPL (configurable in script)"
echo "Expected result: Test RMSE depends on market volatility"
echo "Note: Requires internet access for yfinance data download"
deepspeed --num_gpus=2 train_rnn_stock_data_ds.py

echo "=================================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "Check W&B dashboard for:"
echo "  - Loss curves (train/val/test)"
echo "  - Time series plots (price, MAs, deltas)"
echo "  - Distribution plots"
echo "  - Prediction results"
echo "  - Gradient norms"
echo "=================================================="
