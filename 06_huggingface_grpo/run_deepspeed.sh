#!/bin/bash
# SLURM batch script for GRPO training with DeepSpeed
# Trains Qwen 1.5B model on GSM8K using LoRA and ZeRO-2 optimization

#SBATCH --gres=gpu:2
# Request 2 GPUs (tested on 2x RTX 3070 8GB)
# With LoRA + ZeRO-2, each GPU uses ~6-7GB VRAM

#SBATCH --partition=h200-low
# Submit to the "h200-low" partition/queue
# Update this based on your CoreWeave partition names

#SBATCH --time=02:00:00
# Maximum wall-clock time: 2 hours
# Typical runtime: 30-45 minutes for 3 epochs on 8K examples
# Allocating extra time for dataset download and model initialization

#SBATCH --job-name=grpo_gsm8k_lora
# Job name: GRPO training with LoRA on GSM8K

#SBATCH --ntasks-per-node=1
# Number of tasks per node (1 for multi-GPU training with DeepSpeed)

#SBATCH --cpus-per-task=16
# Number of CPU cores for data loading and preprocessing
# 16 cores ensures smooth data pipeline for 8K training examples

#SBATCH --mem=64G
# Total memory per node: 64 GB
# Needed for:
#   - Model initialization (~4GB)
#   - Dataset loading (~2GB)
#   - DeepSpeed optimizer states (~4GB with ZeRO-2)
#   - System overhead (~2GB)

#SBATCH --output=logs/grpo_gsm8k_%j.out
# Standard output log: logs/grpo_gsm8k_<job_id>.out

#SBATCH --error=logs/grpo_gsm8k_%j.err
# Standard error log: logs/grpo_gsm8k_<job_id>.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Export Weights & Biases API key for experiment tracking
# Get your API key from: https://wandb.ai/authorize
# IMPORTANT: Set this before running the job!
export WANDB_API_KEY=<ENTER_KEY_HERE>

# Optional: Export HuggingFace token if using gated models
# export HF_TOKEN=<ENTER_KEY_HERE>

# Activate Python virtual environment
# Update this path to your actual virtual environment location
# If using uv: source ~/myenv/bin/activate
source ~/myenv/bin/activate

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "GPUs per Node: $SLURM_GPUS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=================================================="

# Print GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "=================================================="

# Launch GRPO training with DeepSpeed
# Configuration:
#   - Model: Qwen 1.5B (eagle0504/qwen-distilled-scout-1.5b-instruct-gen2)
#   - Dataset: GSM8K enhanced (8K train, 1K test)
#   - Optimization: LoRA (r=16, alpha=32) + ZeRO-2
#   - Batch size: 4 per GPU, gradient accumulation: 8
#   - Effective batch size: 4 * 2 GPUs * 8 = 64
#   - Training: 3 epochs, ~375 total steps
echo "Starting GRPO training with DeepSpeed..."
echo "Configuration:"
echo "  - LoRA: r=16, alpha=32, dropout=0.1"
echo "  - DeepSpeed: ZeRO-2 (optimizer + gradient partitioning)"
echo "  - Batch size: 4 per GPU, gradient accumulation: 8"
echo "  - Effective batch size: 64"
echo "  - Epochs: 3"
echo "  - Learning rate: 5e-5"
echo "=================================================="

deepspeed --num_gpus=2 grpo_gsm8k_train.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "=================================================="
    echo "Training completed successfully!"
    echo "End Time: $(date)"
    echo "=================================================="
    echo "Output location: ./grpo-trained-qwen-gsm8k-lora/"
    echo "To use the model, load both:"
    echo "  1. Base model: eagle0504/qwen-distilled-scout-1.5b-instruct-gen2"
    echo "  2. LoRA adapter: ./grpo-trained-qwen-gsm8k-lora/"
    echo "=================================================="
else
    echo "=================================================="
    echo "Training failed with exit code: $?"
    echo "End Time: $(date)"
    echo "Check logs for errors:"
    echo "  - stdout: logs/grpo_gsm8k_${SLURM_JOB_ID}.out"
    echo "  - stderr: logs/grpo_gsm8k_${SLURM_JOB_ID}.err"
    echo "=================================================="
    exit 1
fi
